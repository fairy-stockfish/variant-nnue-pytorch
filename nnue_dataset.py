import numpy as np
import ctypes
import torch
import os
import sys
import glob
from torch.utils.data import Dataset

def _get_dll_path():
    """Get the path to the training data loader shared library."""
    local_dllpath = [n for n in glob.glob('./*training_data_loader.*') if n.endswith('.so') or n.endswith('.dll') or n.endswith('.dylib')]
    if not local_dllpath:
        print('Cannot find data_loader shared library.')
        sys.exit(1)
    return os.path.abspath(local_dllpath[0])

def _load_dll():
    """Load the training data loader shared library. Called per-process to avoid pickling issues."""
    dllpath = _get_dll_path()
    return ctypes.cdll.LoadLibrary(dllpath)

def _setup_dll_functions(dll):
    """Setup ctypes function signatures for the dll. Called per-process to avoid pickling issues."""
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
    
    destroy_sparse_batch_stream = dll.destroy_sparse_batch_stream
    destroy_sparse_batch_stream.argtypes = [ctypes.c_void_p]
    
    fetch_next_sparse_batch = dll.fetch_next_sparse_batch
    fetch_next_sparse_batch.restype = SparseBatchPtr
    fetch_next_sparse_batch.argtypes = [ctypes.c_void_p]
    
    destroy_sparse_batch = dll.destroy_sparse_batch
    
    return create_sparse_batch_stream, destroy_sparse_batch_stream, fetch_next_sparse_batch, destroy_sparse_batch

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
        white_values = torch.from_numpy(np.ctypeslib.as_array(self.white_values, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
        black_values = torch.from_numpy(np.ctypeslib.as_array(self.black_values, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
        white_indices = torch.from_numpy(np.ctypeslib.as_array(self.white, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
        black_indices = torch.from_numpy(np.ctypeslib.as_array(self.black, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
        us = torch.from_numpy(np.ctypeslib.as_array(self.is_white, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
        them = 1.0 - us
        outcome = torch.from_numpy(np.ctypeslib.as_array(self.outcome, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
        score = torch.from_numpy(np.ctypeslib.as_array(self.score, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
        psqt_indices = torch.from_numpy(np.ctypeslib.as_array(self.psqt_indices, shape=(self.size,))).long().pin_memory().to(device=device, non_blocking=True)
        layer_stack_indices = torch.from_numpy(np.ctypeslib.as_array(self.layer_stack_indices, shape=(self.size,))).long().pin_memory().to(device=device, non_blocking=True)
        return us, them, white_indices, white_values, black_indices, black_values, outcome, score, psqt_indices, layer_stack_indices

SparseBatchPtr = ctypes.POINTER(SparseBatch)


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
        device='cpu'):

        self.feature_set = feature_set.encode('utf-8')
        self.filename = filename.encode('utf-8')
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.filtered = filtered
        self.random_fen_skipping = random_fen_skipping
        self.device = device

        # Store function references but don't use them yet - they'll be set up when needed
        self._create_stream_func = None
        self._destroy_stream_func = None
        self._fetch_next_func = None
        self._destroy_part_func = None
        self.stream = None
        
        # Initialize the dll and stream
        self._setup_dll()

    def _setup_dll(self):
        """Setup the DLL and create the stream. This is called per-instance to avoid pickling issues."""
        dll = _load_dll()
        self._create_stream_func, self._destroy_stream_func, self._fetch_next_func, self._destroy_part_func = _setup_dll_functions(dll)
        
        self.stream = self._create_stream_func(
            self.feature_set,
            self.num_workers,
            self.filename,
            self.batch_size,
            self.cyclic,
            self.filtered,
            self.random_fen_skipping
        )

    def __iter__(self):
        return self

    def __next__(self):
        if self.stream is None:
            # If stream is None (e.g., after unpickling), recreate it
            self._setup_dll()
            
        v = self._fetch_next_func(self.stream)

        if v:
            tensors = v.contents.get_tensors(self.device)
            self._destroy_part_func(v)
            return tensors
        else:
            raise StopIteration

    def __del__(self):
        if hasattr(self, "stream") and self.stream is not None and hasattr(self, "_destroy_stream_func"):
            self._destroy_stream_func(self.stream)
            
    def __getstate__(self):
        """Custom pickling - exclude unpicklable ctypes objects."""
        state = self.__dict__.copy()
        # Remove unpicklable ctypes objects
        state['_create_stream_func'] = None
        state['_destroy_stream_func'] = None
        state['_fetch_next_func'] = None
        state['_destroy_part_func'] = None
        state['stream'] = None
        return state
        
    def __setstate__(self, state):
        """Custom unpickling - restore the object and reinitialize ctypes objects."""
        self.__dict__.update(state)
        # Don't setup dll here - it will be done lazily in __next__ when needed

class SparseBatchProvider(TrainingDataProvider):
    def __init__(self, feature_set, filename, batch_size, cyclic=True, num_workers=1, filtered=False, random_fen_skipping=0, device='cpu'):
        # Pass None for the function parameters since we'll set them up in _setup_dll
        super(SparseBatchProvider, self).__init__(
            feature_set,
            None,  # create_stream - will be set in _setup_dll
            None,  # destroy_stream - will be set in _setup_dll
            None,  # fetch_next - will be set in _setup_dll
            None,  # destroy_part - will be set in _setup_dll
            filename,
            cyclic,
            num_workers,
            batch_size,
            filtered,
            random_fen_skipping,
            device)

class SparseBatchDataset(torch.utils.data.IterableDataset):
  def __init__(self, feature_set, filename, batch_size, cyclic=True, num_workers=1, filtered=False, random_fen_skipping=0, device='cpu'):
    super(SparseBatchDataset).__init__()
    self.feature_set = feature_set
    self.filename = filename
    self.batch_size = batch_size
    self.cyclic = cyclic
    self.num_workers = num_workers
    self.filtered = filtered
    self.random_fen_skipping = random_fen_skipping
    self.device = device

  def __iter__(self):
    return SparseBatchProvider(self.feature_set, self.filename, self.batch_size, cyclic=self.cyclic, num_workers=self.num_workers, filtered=self.filtered, random_fen_skipping=self.random_fen_skipping, device=self.device)

class FixedNumBatchesDataset(Dataset):
  def __init__(self, dataset, num_batches):
    super(FixedNumBatchesDataset, self).__init__()
    self.dataset = dataset
    self.iter = iter(self.dataset)
    self.num_batches = num_batches

  def __len__(self):
    return self.num_batches

  def __getitem__(self, idx):
    return next(self.iter)
