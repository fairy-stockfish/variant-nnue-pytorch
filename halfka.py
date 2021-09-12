import chess
import torch
import feature_block
from collections import OrderedDict
from feature_block import *

import variant

NUM_SQ = variant.SQUARES
NUM_KSQ = variant.KING_SQUARES
NUM_PT = variant.PIECES
NUM_PLANES = (NUM_SQ * NUM_PT + 1)

def orient(is_white_pov: bool, sq: int):
  return sq % variant.FILES + (variant.RANKS - 1 - (sq // variant.FILES)) * variant.FILES if not is_white_pov else sq

def halfka_idx(is_white_pov: bool, king_sq: int, sq: int, piece_type: int, color: bool):
  p_idx = (piece_type - 1) * 2 + (color != is_white_pov)
  return 1 + orient(is_white_pov, sq) + p_idx * NUM_SQ + king_sq * NUM_PLANES

def halfka_psqts():
  values = [0] * (NUM_PLANES * NUM_SQ)

  for ksq in range(NUM_KSQ):
    for s in range(NUM_SQ):
      for pt, val in variant.PIECE_VALUES.items():
        idxw = halfka_idx(True, ksq, s, pt, chess.WHITE)
        idxb = halfka_idx(True, ksq, s, pt, chess.BLACK)
        values[idxw] = val
        values[idxb] = -val

  return values

class Features(FeatureBlock):
  def __init__(self):
    super(Features, self).__init__('HalfKA', 0x5f134cb8, OrderedDict([('HalfKA', NUM_PLANES * NUM_SQ)]))

  def get_active_features(self, board: chess.Board):
    def piece_features(turn):
      indices = torch.zeros(NUM_PLANES * NUM_SQ)
      for sq, p in board.piece_map().items():
        indices[halfka_idx(turn, orient(turn, board.king(turn)), sq, p)] = 1.0
      return indices
    return (piece_features(chess.WHITE), piece_features(chess.BLACK))

  def get_initial_psqt_features(self):
    return halfka_psqts()

class FactorizedFeatures(FeatureBlock):
  def __init__(self):
    super(FactorizedFeatures, self).__init__('HalfKA^', 0x5f134cb8, OrderedDict([('HalfKA', NUM_PLANES * NUM_SQ), ('A', NUM_SQ * NUM_PT)]))

  def get_active_features(self, board: chess.Board):
    raise Exception('Not supported yet, you must use the c++ data loader for factorizer support during training')

  def get_feature_factors(self, idx):
    if idx >= self.num_real_features:
      raise Exception('Feature must be real')

    a_idx = idx % NUM_PLANES - 1

    return [idx, self.get_factor_base_feature('A') + a_idx]

  def get_initial_psqt_features(self):
    return halfka_psqts() + [0] * (NUM_SQ * NUM_PT)

'''
This is used by the features module for discovery of feature blocks.
'''
def get_feature_block_clss():
  return [Features, FactorizedFeatures]
