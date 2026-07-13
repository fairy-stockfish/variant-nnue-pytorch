from types import SimpleNamespace
import os

import pytest
import torch

import train


class FakeModel:
    def cuda(self):
        raise AssertionError('load_or_create_model must not force CUDA')


def test_seed_is_applied_before_model_construction(monkeypatch):
    samples = []

    def make_model(**kwargs):
        samples.append(torch.rand(4))
        return FakeModel()

    monkeypatch.setattr(train.M, "NNUE", make_model)
    feature_set = SimpleNamespace()

    train.load_or_create_model(feature_set, lambda_=1.0, seed=1234)
    train.load_or_create_model(feature_set, lambda_=1.0, seed=1234)

    assert torch.equal(samples[0], samples[1])


def test_checkpoint_is_loaded_on_cpu(monkeypatch):
    calls = []
    checkpoint = FakeModel()
    checkpoint.set_feature_set = lambda value: None

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return checkpoint

    monkeypatch.setattr(train.torch, 'load', load)
    train.load_or_create_model(SimpleNamespace(), lambda_=0.5, seed=7, resume_from_model='model.pt')

    assert calls[0][1]['map_location'] == 'cpu'
    assert calls[0][1]['weights_only'] is False


def test_resume_model_rejects_unknown_file_type():
    with pytest.raises(ValueError, match='accepts only'):
        train.load_or_create_model(SimpleNamespace(), lambda_=1.0, seed=7, resume_from_model='model.ckpt')


def test_default_batch_size_depends_on_trainer_device():
    assert train.default_batch_size('cpu') == 128
    assert train.default_batch_size('cuda:0') == 16384


def test_legacy_trainer_rejects_multi_device_or_multi_node_worlds():
    train.require_single_device_trainer(SimpleNamespace(world_size=1))
    with pytest.raises(ValueError, match='exactly one'):
        train.require_single_device_trainer(SimpleNamespace(world_size=2))


def test_same_training_and_validation_file_is_rejected(tmp_path):
    data = tmp_path / 'data.bin'
    data.write_bytes(b'x')

    with pytest.raises(ValueError, match='separate files'):
        train.validate_data_paths(str(data), str(data))
    train.validate_data_paths(str(data), str(data), allow_train_as_validation=True)


def test_relative_absolute_and_hardlink_aliases_are_rejected(tmp_path, monkeypatch):
    data = tmp_path / 'data.bin'
    alias = tmp_path / 'alias.bin'
    data.write_bytes(b'x')
    try:
        os.link(data, alias)
    except OSError as error:
        pytest.skip('hard links unavailable: {}'.format(error))

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match='separate files'):
        train.validate_data_paths('data.bin', str(data.resolve()))
    with pytest.raises(ValueError, match='separate files'):
        train.validate_data_paths(str(data), str(alias))


def test_missing_training_or_validation_file_is_rejected(tmp_path):
    existing = tmp_path / 'existing.bin'
    existing.write_bytes(b'x')
    with pytest.raises(FileNotFoundError):
        train.validate_data_paths(str(tmp_path / 'missing.bin'), str(existing))
    with pytest.raises(FileNotFoundError):
        train.validate_data_paths(str(existing), str(tmp_path / 'missing.bin'))


@pytest.mark.parametrize('value', ['-1', str(2**31)])
def test_random_skip_cli_range_rejects_invalid_values(value):
    with pytest.raises(Exception):
        train.non_negative_skip_count(value)


@pytest.mark.parametrize('value', ['0', '-1', str(2**31)])
def test_worker_cli_range_rejects_invalid_values(value):
    with pytest.raises(Exception):
        train.positive_int32(value)


@pytest.mark.parametrize('value', ['0', '-2', str(2**31)])
def test_batch_size_cli_range_rejects_invalid_values(value):
    with pytest.raises(Exception):
        train.batch_size_argument(value)


@pytest.mark.parametrize('value', ['-1', str(2**32)])
def test_seed_cli_range_rejects_invalid_values(value):
    with pytest.raises(Exception):
        train.deterministic_seed(value)


def test_automatic_batch_size_sentinel_is_accepted():
    assert train.batch_size_argument('-1') == -1


@pytest.mark.parametrize('value', ['-0.01', '1.01', 'nan', 'inf'])
def test_lambda_cli_range_rejects_invalid_values(value):
    with pytest.raises(Exception):
        train.lambda_argument(value)


def test_validation_loader_does_not_use_random_training_skips(monkeypatch):
    datasets = []
    fixed_datasets = []

    class FakeSparseBatchDataset:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            datasets.append(self)

    monkeypatch.setattr(train.nnue_dataset, "SparseBatchDataset", FakeSparseBatchDataset)
    def make_fixed_dataset(dataset, num_batches, **kwargs):
        fixed_datasets.append((dataset, num_batches, kwargs))
        return dataset, num_batches, kwargs

    monkeypatch.setattr(train.nnue_dataset, "FixedNumBatchesDataset", make_fixed_dataset)
    monkeypatch.setattr(train, "DataLoader", lambda dataset, **kwargs: dataset)

    train.make_data_loaders(
        "train.bin",
        "validation.bin",
        SimpleNamespace(name="HalfKAv2"),
        num_workers=4,
        batch_size=32,
        filtered=True,
        random_fen_skipping=7,
        main_device="cpu",
        epoch_size=64,
        val_size=32,
        seed=2026,
    )

    assert datasets[0].kwargs["random_fen_skipping"] == 7
    assert datasets[0].kwargs["seed"] == 2026
    assert datasets[1].kwargs["random_fen_skipping"] == 0
    assert datasets[1].kwargs["seed"] == 2026
    assert fixed_datasets[0][2].get("restart_on_zero", False) is False
    assert fixed_datasets[1][2]["restart_on_zero"] is True
