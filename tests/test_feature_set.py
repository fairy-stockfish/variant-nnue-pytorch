from collections import OrderedDict

from feature_block import FeatureBlock
from feature_set import FeatureSet


class StubFeature(FeatureBlock):
    def __init__(self, name, hash_value):
        super().__init__(name, hash_value, OrderedDict([(name, 1)]))

    def get_active_features(self, board):
        raise NotImplementedError


def combined_hash(head, tail):
    return (head ^ (tail << 1) ^ (tail >> 1)) & 0xFFFFFFFF


def test_single_feature_hash_is_unchanged():
    feature = StubFeature("A", 0x12345678)

    assert FeatureSet([feature]).hash == feature.hash


def test_composed_feature_hash_recurses_and_stays_uint32():
    features = [
        StubFeature("A", 0x5D69D5B8),
        StubFeature("B", 0x5F134CB8),
        StubFeature("C", 0x5F234CB8),
    ]
    expected = combined_hash(features[0].hash, combined_hash(features[1].hash, features[2].hash))

    actual = FeatureSet(features).hash

    assert actual == expected
    assert 0 <= actual <= 0xFFFFFFFF
