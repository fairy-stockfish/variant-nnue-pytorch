import numpy as np
import ctypes
import torch
import os
import sys
import glob
import operator
from torch.utils.data import Dataset

if sys.platform == 'win32':
    library_extension = '.dll'
elif sys.platform == 'darwin':
    library_extension = '.dylib'
else:
    library_extension = '.so'

module_directory = os.path.dirname(os.path.abspath(__file__))
local_dllpath = sorted(glob.glob(os.path.join(module_directory, '*training_data_loader*' + library_extension)))
if not local_dllpath:
    raise RuntimeError('Cannot find the training data loader shared library; run compile_data_loader.bat first')
if len(local_dllpath) > 1:
    raise RuntimeError('Multiple training data loader libraries found: {}'.format(', '.join(local_dllpath)))
dllpath = os.path.abspath(local_dllpath[0])
dll = ctypes.cdll.LoadLibrary(dllpath)

class SparseBatch(ctypes.Structure):
    _fields_ = [
        ('num_inputs', ctypes.c_int),
        ('size', ctypes.c_int),
        ('is_white', ctypes.POINTER(ctypes.c_float)),
        ('outcome', ctypes.POINTER(ctypes.c_float)),
        ('score', ctypes.POINTER(ctypes.c_float)),
        ('num_active_white_features', ctypes.c_int),
        ('num_active_black_features', ctypes.c_int),
        ('max_active_features', ctypes.c_int),
        ('white', ctypes.POINTER(ctypes.c_int)),
        ('black', ctypes.POINTER(ctypes.c_int)),
        ('white_values', ctypes.POINTER(ctypes.c_float)),
        ('black_values', ctypes.POINTER(ctypes.c_float)),
        ('psqt_indices', ctypes.POINTER(ctypes.c_int)),
        ('layer_stack_indices', ctypes.POINTER(ctypes.c_int)),
    ]

    def get_tensors(self, device):
        device = torch.device(device)

        def owned_tensor(pointer, shape, dtype=None):
            # ctypes/Numpy only provide a view over SparseBatch memory. The C++
            # batch is destroyed as soon as this method returns, so CPU tensors
            # must take their own copy too (a GPU transfer already does so).
            tensor = torch.from_numpy(np.ctypeslib.as_array(pointer, shape=shape))
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            if device.type == 'cuda':
                tensor = tensor.pin_memory().to(device=device, non_blocking=True)
            else:
                tensor = tensor.clone().to(device=device)
            return tensor

        shape = (self.size, self.max_active_features)
        white_values = owned_tensor(self.white_values, shape)
        black_values = owned_tensor(self.black_values, shape)
        white_indices = owned_tensor(self.white, shape)
        black_indices = owned_tensor(self.black, shape)
        us = owned_tensor(self.is_white, (self.size, 1))
        them = 1.0 - us
        outcome = owned_tensor(self.outcome, (self.size, 1))
        score = owned_tensor(self.score, (self.size, 1))
        psqt_indices = owned_tensor(self.psqt_indices, (self.size,), torch.long)
        layer_stack_indices = owned_tensor(self.layer_stack_indices, (self.size,), torch.long)
        return us, them, white_indices, white_values, black_indices, black_values, outcome, score, psqt_indices, layer_stack_indices

SparseBatchPtr = ctypes.POINTER(SparseBatch)


INT32_MAX = 2**31 - 1
UINT64_MAX = 2**64 - 1


def _bounded_integer(name, value, minimum, maximum):
    try:
        value = operator.index(value)
    except TypeError as error:
        raise TypeError('{} must be an integer'.format(name)) from error
    if isinstance(value, bool) or value < minimum or value > maximum:
        raise ValueError('{} must be in the range [{}, {}]'.format(name, minimum, maximum))
    return value


class TrainingDataProvider:
    def __init__(
        self,
        feature_set,
        create_stream,
        destroy_stream,
        fetch_next,
        destroy_part,
        filename,
        cyclic,
        num_workers,
        batch_size=None,
        filtered=False,
        random_fen_skipping=0,
        device='cpu',
        seed=None,
        get_error=None):

        self.feature_set = feature_set.encode('utf-8')
        self.create_stream = create_stream
        self.destroy_stream = destroy_stream
        self.fetch_next = fetch_next
        self.destroy_part = destroy_part
        self.filename = os.fsencode(filename)
        self.cyclic = cyclic
        self.num_workers = _bounded_integer('num_workers', num_workers, 1, INT32_MAX)
        self.batch_size = _bounded_integer('batch_size', batch_size, 1, INT32_MAX)
        self.filtered = filtered
        self.random_fen_skipping = _bounded_integer('random_fen_skipping', random_fen_skipping, 0, INT32_MAX)
        self.seed = None if seed is None else _bounded_integer('seed', seed, 0, UINT64_MAX)
        self.device = device
        self.get_error = get_error

        stream_arguments = [
            self.feature_set,
            self.num_workers,
            self.filename,
            self.batch_size,
            cyclic,
            filtered,
            self.random_fen_skipping,
        ]
        if self.seed is not None:
            stream_arguments.append(self.seed)
        self.stream = self.create_stream(*stream_arguments)
        if not self.stream:
            raise RuntimeError('Native data loader rejected the stream configuration')

    def __iter__(self):
        return self

    def __next__(self):
        v = self.fetch_next(self.stream)

        if v:
            try:
                return v.contents.get_tensors(self.device)
            finally:
                self.destroy_part(v)
        else:
            if self.get_error is not None:
                native_error = self.get_error(self.stream)
                if native_error:
                    if isinstance(native_error, bytes):
                        native_error = native_error.decode('utf-8', errors='replace')
                    raise RuntimeError('Native data loader failed: {}'.format(native_error))
            raise StopIteration

    def __del__(self):
        if getattr(self, "stream", None):
            self.destroy_stream(self.stream)
            self.stream = None

create_sparse_batch_stream = dll.create_sparse_batch_stream
create_sparse_batch_stream.restype = ctypes.c_void_p
create_sparse_batch_stream.argtypes = [
    ctypes.c_char_p,  # feature_set
    ctypes.c_int,     # num_workers
    ctypes.c_char_p,  # filename
    ctypes.c_int,     # batch_size
    ctypes.c_bool,    # cyclic
    ctypes.c_bool,    # filtered
    ctypes.c_int      # random_fen_skipping
]
create_sparse_batch_stream_with_seed = dll.create_sparse_batch_stream_with_seed
create_sparse_batch_stream_with_seed.restype = ctypes.c_void_p
create_sparse_batch_stream_with_seed.argtypes = [
    ctypes.c_char_p,  # feature_set
    ctypes.c_int,     # num_workers
    ctypes.c_char_p,  # filename
    ctypes.c_int,     # batch_size
    ctypes.c_bool,    # cyclic
    ctypes.c_bool,    # filtered
    ctypes.c_int,     # random_fen_skipping
    ctypes.c_uint64   # deterministic random-skip seed
]
destroy_sparse_batch_stream = dll.destroy_sparse_batch_stream
destroy_sparse_batch_stream.argtypes = [ctypes.c_void_p]
destroy_sparse_batch_stream.restype = None

fetch_next_sparse_batch = dll.fetch_next_sparse_batch
fetch_next_sparse_batch.restype = SparseBatchPtr
fetch_next_sparse_batch.argtypes = [ctypes.c_void_p]
get_sparse_batch_stream_error = dll.get_sparse_batch_stream_error
get_sparse_batch_stream_error.restype = ctypes.c_char_p
get_sparse_batch_stream_error.argtypes = [ctypes.c_void_p]
destroy_sparse_batch = dll.destroy_sparse_batch
destroy_sparse_batch.argtypes = [SparseBatchPtr]
destroy_sparse_batch.restype = None


class SparseBatchProvider(TrainingDataProvider):
    def __init__(self, feature_set, filename, batch_size, cyclic=True, num_workers=1, filtered=False, random_fen_skipping=0, device='cpu', seed=0):
        super(SparseBatchProvider, self).__init__(
            feature_set,
            create_sparse_batch_stream_with_seed,
            destroy_sparse_batch_stream,
            fetch_next_sparse_batch,
            destroy_sparse_batch,
            filename,
            cyclic,
            num_workers,
            batch_size,
            filtered,
            random_fen_skipping,
            device,
            seed,
            get_sparse_batch_stream_error)

class SparseBatchDataset(torch.utils.data.IterableDataset):
  def __init__(self, feature_set, filename, batch_size, cyclic=True, num_workers=1, filtered=False, random_fen_skipping=0, device='cpu', seed=0):
    super().__init__()
    self.feature_set = feature_set
    self.filename = filename
    self.batch_size = batch_size
    self.cyclic = cyclic
    self.num_workers = num_workers
    self.filtered = filtered
    self.random_fen_skipping = random_fen_skipping
    self.seed = seed
    self.device = device

  def __iter__(self):
    return SparseBatchProvider(self.feature_set, self.filename, self.batch_size, cyclic=self.cyclic, num_workers=self.num_workers, filtered=self.filtered, random_fen_skipping=self.random_fen_skipping, device=self.device, seed=self.seed)

class FixedNumBatchesDataset(Dataset):
  def __init__(self, dataset, num_batches, restart_on_zero=False):
    super().__init__()
    self.dataset = dataset
    self.iter = None
    self.num_batches = num_batches
    self.restart_on_zero = restart_on_zero

  def __len__(self):
    return self.num_batches

  def __getitem__(self, idx):
    if idx < 0 or idx >= self.num_batches:
      raise IndexError(idx)
    if self.iter is None or (self.restart_on_zero and idx == 0):
      self.iter = iter(self.dataset)
    return next(self.iter)
