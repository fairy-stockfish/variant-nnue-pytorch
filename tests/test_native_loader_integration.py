import ctypes
import hashlib

import pytest
import torch

import nnue_dataset


def batch_digest(batch):
    digest = hashlib.sha256()
    for tensor in batch:
        tensor = tensor.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def collect(path, workers, seed, random_skipping=1, batch_size=5):
    provider = nnue_dataset.SparseBatchProvider(
        'HalfKAv2',
        str(path),
        batch_size,
        cyclic=False,
        num_workers=workers,
        filtered=False,
        random_fen_skipping=random_skipping,
        device='cpu',
        seed=seed,
    )
    return [(batch[0].shape[0], batch_digest(batch)) for batch in provider]


def test_seeded_loader_order_is_independent_of_worker_completion(atomic_legacy_32):
    single_worker = collect(atomic_legacy_32, workers=1, seed=20260710)
    four_workers = collect(atomic_legacy_32, workers=4, seed=20260710)
    repeated = collect(atomic_legacy_32, workers=4, seed=20260710)

    assert single_worker
    assert four_workers == single_worker
    assert repeated == single_worker


def test_loader_retains_all_records_and_final_partial_batch(atomic_legacy_32):
    batches = collect(
        atomic_legacy_32,
        workers=4,
        seed=0,
        random_skipping=0,
        batch_size=5,
    )
    assert batches == [
        (5, '468b008d16dcc79adb7ec9c4351a9f1bd5b2f1bda739dd882bc39a04616bd78f'),
        (5, 'acd00d1881c9eb91e8372d5c1ed035915fefca6e5f48bc6ca4b05eae6440a319'),
        (5, 'ef8a94cdfcc66ef59e19775bda28bebd89491566e1c2d23d06e3d9c2d6f47bd7'),
        (5, 'e2a901f99be8cbf74c04def8b8f58919dff5779bca58695d0442c15fc02fdc69'),
        (5, '0b1878933dce5ebb9a0a159fd1ae69ad336d44b8de87a32ae12a5b351362d96d'),
        (5, '41b105f867f385db6707f4ac869bc046880e7f1f32903f05831a516fef875fba'),
        (2, '565a456b98235927d040a77bb497fae07a4c9edf096e6f9413e15572998b7fa1'),
    ]


def test_legacy_and_seeded_c_abis_fetch_real_batches(atomic_legacy_32):
    common = (
        b'HalfKAv2',
        2,
        str(atomic_legacy_32).encode(),
        4,
        False,
        False,
        0,
    )
    streams = [
        nnue_dataset.create_sparse_batch_stream(*common),
        nnue_dataset.create_sparse_batch_stream_with_seed(*common, 7),
    ]
    try:
        for stream in streams:
            assert stream
            batch = nnue_dataset.fetch_next_sparse_batch(stream)
            assert batch
            try:
                assert batch.contents.size == 4
                tensors = batch.contents.get_tensors('cpu')
                assert tensors[2].dtype == torch.int32
            finally:
                nnue_dataset.destroy_sparse_batch(batch)
    finally:
        for stream in streams:
            if stream:
                nnue_dataset.destroy_sparse_batch_stream(stream)


def test_cpu_tensors_own_memory_after_native_storage_changes():
    floats = [(ctypes.c_float * 1)(value) for value in (1.0, 0.75, 12.0, 1.0, 1.0)]
    ints = [(ctypes.c_int * 1)(value) for value in (3, 4, 0, 0)]
    batch = nnue_dataset.SparseBatch(
        16,
        1,
        floats[0],
        floats[1],
        floats[2],
        1,
        1,
        1,
        ints[0],
        ints[1],
        floats[3],
        floats[4],
        ints[2],
        ints[3],
    )
    tensors = batch.get_tensors('cpu')
    ints[0][0] = 15
    floats[0][0] = 0.0

    assert tensors[0].item() == 1.0
    assert tensors[2].item() == 3


def test_native_batch_is_destroyed_when_tensor_conversion_fails(monkeypatch):
    batch = nnue_dataset.SparseBatch()
    pointer = ctypes.pointer(batch)
    destroyed = []
    monkeypatch.setattr(nnue_dataset.SparseBatch, 'get_tensors', lambda self, device: (_ for _ in ()).throw(RuntimeError('boom')))
    provider = nnue_dataset.TrainingDataProvider(
        'HalfKAv2',
        lambda *args: object(),
        lambda value: None,
        lambda value: pointer,
        lambda value: destroyed.append(value),
        'unused.bin',
        cyclic=False,
        num_workers=1,
        batch_size=1,
    )

    try:
        with pytest.raises(RuntimeError, match='boom'):
            next(provider)
    finally:
        provider.__del__()
    assert destroyed == [pointer]


def test_full_size_corrupt_record_propagates_native_error(atomic_legacy_32, tmp_path):
    record = bytearray(atomic_legacy_32.read_bytes()[:72])
    record[70] = 2  # Legacy game_result must be -1, 0, or 1.
    corrupt = tmp_path / 'corrupt-result.bin'
    corrupt.write_bytes(record)

    provider = nnue_dataset.SparseBatchProvider(
        'HalfKAv2',
        str(corrupt),
        1,
        cyclic=False,
        num_workers=2,
        filtered=False,
        random_fen_skipping=0,
        device='cpu',
        seed=0,
    )
    try:
        with pytest.raises(RuntimeError, match='result'):
            next(provider)
    finally:
        provider.__del__()
