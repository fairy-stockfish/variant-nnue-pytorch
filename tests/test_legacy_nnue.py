import contextlib
import gc
import hashlib
import io
import struct

import pytest
import torch

import features
import model
import serialize
import train


def test_nnue_header_detection_rejects_unknown_formats():
    with pytest.raises(serialize.NNUEFormatError, match='Unsupported NNUE version'):
        serialize.NNUEReader(io.BytesIO(struct.pack('<II', 0, 0)))

    with pytest.raises(serialize.NNUEFormatError, match='Could not uniquely identify'):
        serialize.NNUEReader(io.BytesIO(struct.pack('<II', serialize.VERSION, 0)))


def test_legacy_halfkav2_roundtrip_and_factorized_training_resume(tmp_path):
    source_features = features.get_feature_set_from_name('HalfKAv2')
    target_features = features.get_feature_set_from_name('HalfKAv2^')

    torch.manual_seed(20260711)
    source = model.NNUE(source_features)
    source.nnue_description = 'Legacy Atomic V1 integration fixture'
    with contextlib.redirect_stdout(io.StringIO()):
        writer = serialize.NNUEWriter(source)

    network_path = tmp_path / 'legacy-atomic-v1.nnue'
    network_path.write_bytes(writer.buf)
    expected_size = len(writer.buf)
    expected_hash = hashlib.sha256(writer.buf).hexdigest()
    del writer, source
    gc.collect()

    resumed = train.load_or_create_model(
        target_features,
        lambda_=0.75,
        seed=20260711,
        resume_from_model=network_path,
    )
    assert resumed.feature_set.name == 'HalfKAv2^'
    assert resumed.lambda_ == 0.75
    assert resumed.nnue_description == 'Legacy Atomic V1 integration fixture'
    assert resumed.input.weight.shape[0] == target_features.num_features
    assert resumed.input.num_inputs == target_features.num_features
    assert torch.count_nonzero(resumed.input.weight[source_features.num_features:]).item() == 0

    # Serializing a freshly resumed factorized model must coalesce its zeroed
    # virtual rows back to the original real-only legacy payload exactly.
    with contextlib.redirect_stdout(io.StringIO()):
        roundtrip = serialize.NNUEWriter(resumed)
    assert len(roundtrip.buf) == expected_size
    assert hashlib.sha256(roundtrip.buf).hexdigest() == expected_hash
    del roundtrip

    # The newly added factor rows must participate in autograd immediately.
    indices = torch.tensor([[0, source_features.num_features, -1]], dtype=torch.int32)
    values = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)
    transformed = resumed.input(indices, values, indices, values)
    (transformed[0].sum() + transformed[1].sum()).backward()
    assert torch.count_nonzero(resumed.input.weight.grad[source_features.num_features]).item() > 0

    del transformed, resumed
    gc.collect()
