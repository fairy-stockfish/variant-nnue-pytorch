import ctypes

import pytest

import nnue_dataset


def test_seeded_native_loader_binding_is_available():
    assert hasattr(nnue_dataset.dll, "create_sparse_batch_stream")
    assert hasattr(nnue_dataset.dll, "create_sparse_batch_stream_with_seed")
    assert hasattr(nnue_dataset.dll, "get_sparse_batch_stream_error")
    assert len(nnue_dataset.create_sparse_batch_stream.argtypes) == 7
    assert nnue_dataset.create_sparse_batch_stream_with_seed.argtypes[-1] is ctypes.c_uint64


def test_training_data_provider_forwards_seed_to_native_stream():
    calls = []
    stream = object()

    def create_stream(*args):
        calls.append(args)
        return stream

    provider = nnue_dataset.TrainingDataProvider(
        "HalfKAv2",
        create_stream,
        lambda value: None,
        lambda value: None,
        lambda value: None,
        "unused.bin",
        cyclic=True,
        num_workers=2,
        batch_size=32,
        filtered=False,
        random_fen_skipping=3,
        seed=0x123456789ABCDEF0,
        device="cpu",
    )

    assert provider.stream is stream
    assert calls[0][-1] == 0x123456789ABCDEF0


def test_training_data_provider_preserves_legacy_seven_argument_call():
    calls = []
    provider = nnue_dataset.TrainingDataProvider(
        'HalfKAv2',
        lambda *args: calls.append(args) or object(),
        lambda value: None,
        lambda value: None,
        lambda value: None,
        'unused.bin',
        False,
        1,
        32,
        False,
        0,
        'cpu',
    )
    assert len(calls[0]) == 7
    provider.__del__()


def test_sparse_dataset_forwards_seed_to_provider(monkeypatch):
    captured = {}
    provider = object()

    def make_provider(*args, **kwargs):
        captured.update(kwargs)
        return provider

    monkeypatch.setattr(nnue_dataset, "SparseBatchProvider", make_provider)
    dataset = nnue_dataset.SparseBatchDataset(
        "HalfKAv2",
        "unused.bin",
        32,
        random_fen_skipping=3,
        seed=99,
    )

    assert dataset.__iter__() is provider
    assert captured["random_fen_skipping"] == 3
    assert captured["seed"] == 99


@pytest.mark.parametrize(
    ('argument', 'value'),
    [
        ('num_workers', 0),
        ('batch_size', 0),
        ('random_fen_skipping', -1),
        ('random_fen_skipping', 2**31),
        ('seed', -1),
        ('seed', 2**64),
    ],
)
def test_provider_rejects_out_of_range_native_arguments(argument, value):
    arguments = dict(
        feature_set='HalfKAv2',
        create_stream=lambda *args: object(),
        destroy_stream=lambda value: None,
        fetch_next=lambda value: None,
        destroy_part=lambda value: None,
        filename='unused.bin',
        cyclic=False,
        num_workers=1,
        batch_size=1,
        random_fen_skipping=0,
        seed=0,
    )
    arguments[argument] = value
    with pytest.raises(ValueError):
        nnue_dataset.TrainingDataProvider(**arguments)


def test_native_stream_rejection_is_reported_before_fetch():
    with pytest.raises(RuntimeError, match='rejected'):
        nnue_dataset.TrainingDataProvider(
            'HalfKAv2',
            lambda *args: None,
            lambda value: None,
            lambda value: pytest.fail('fetch must not be called'),
            lambda value: None,
            'unused.bin',
            cyclic=False,
            num_workers=1,
            batch_size=1,
        )


class CountingIterable:
    def __init__(self):
        self.starts = 0

    def __iter__(self):
        self.starts += 1
        return iter(range(self.starts * 100, self.starts * 100 + 100))


def test_fixed_validation_dataset_restarts_at_batch_zero():
    source = CountingIterable()
    dataset = nnue_dataset.FixedNumBatchesDataset(source, 2, restart_on_zero=True)

    assert dataset[0] == 100
    assert dataset[1] == 101
    assert dataset[0] == 200
    assert source.starts == 2


def test_fixed_training_dataset_keeps_stream_across_epoch_indices():
    source = CountingIterable()
    dataset = nnue_dataset.FixedNumBatchesDataset(source, 2)

    assert dataset[0] == 100
    assert dataset[1] == 101
    assert dataset[0] == 102
    assert source.starts == 1

    with pytest.raises(IndexError):
        dataset[2]
