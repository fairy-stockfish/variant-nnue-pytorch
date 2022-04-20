import chess
import torch
import feature_block
from collections import OrderedDict
from feature_block import *

import variant

NUM_SQ = variant.SQUARES
NUM_KSQ = variant.KING_SQUARES
NUM_PT_REAL = variant.PIECES - (NUM_KSQ != 1)
NUM_PT_VIRTUAL = variant.PIECES
NUM_PLANES_REAL = NUM_SQ * NUM_PT_REAL + (NUM_PT_REAL - (NUM_KSQ != 1)) * variant.POCKETS
NUM_PLANES_VIRTUAL = NUM_SQ * NUM_PT_VIRTUAL + (NUM_PT_REAL - (NUM_KSQ != 1)) * variant.POCKETS
NUM_INPUTS = NUM_PLANES_REAL * NUM_KSQ

def orient(is_white_pov: bool, sq: int):
  return sq % variant.FILES + (variant.RANKS - 1 - (sq // variant.FILES)) * variant.FILES if not is_white_pov else sq

def halfka_idx(is_white_pov: bool, king_sq: int, sq: int, piece_type: int, color: bool):
  p_idx = (piece_type - 1) * 2 + (color != is_white_pov)
  if NUM_PT_REAL % 2 and p_idx == NUM_PT_REAL:
    # merge kings into one plane
    p_idx -= 1
  return orient(is_white_pov, sq) + p_idx * NUM_SQ + king_sq * NUM_PLANES_REAL

def halfka_hand_idx(is_white_pov: bool, king_sq: int, handCount: int, piece_type: int, color: bool):
  p_idx = (piece_type - 1) * 2 + (color != is_white_pov)
  return handCount + p_idx * variant.POCKETS + NUM_SQ * NUM_PT_REAL + king_sq * NUM_PLANES_REAL

def map_king(sq: int):
  # palace squares for Xiangi/Janggi
  if NUM_KSQ == 9 and NUM_KSQ != NUM_SQ:
    if sq > variant.FILES * ((variant.RANKS + 1) // 2):
      # in order to allow unambiguously detecting opposing kings, just return value out of range
      return sq
    # map accessible king squares skipping the gaps
    return (sq - 6 * (sq // variant.FILES) - 3) % NUM_KSQ
  return sq % NUM_KSQ

def halfka_psqts():
  values = [0] * (NUM_PLANES_REAL * NUM_KSQ)

  for ksq in range(NUM_KSQ):
    for s in range(NUM_SQ):
      for pt, val in variant.PIECE_VALUES.items():
        idxw = halfka_idx(True, ksq, s, pt, chess.WHITE)
        idxb = halfka_idx(True, ksq, s, pt, chess.BLACK)
        values[idxw] = val
        values[idxb] = -val
    for i in range(variant.POCKETS):
      for pt, val in variant.PIECE_VALUES.items():
        idxw = halfka_hand_idx(True, ksq, i, pt, chess.WHITE)
        idxb = halfka_hand_idx(True, ksq, i, pt, chess.BLACK)
        values[idxw] = val
        values[idxb] = -val

  return values

class Features(FeatureBlock):
  def __init__(self):
    super(Features, self).__init__('HalfKAv2', 0x5f234cb8, OrderedDict([('HalfKAv2', NUM_PLANES_REAL * NUM_KSQ)]))

  def get_active_features(self, board: chess.Board):
    def piece_features(turn):
      indices = torch.zeros(NUM_PLANES_REAL * NUM_KSQ)
      for sq, p in board.piece_map().items():
        indices[halfka_idx(turn, orient(turn, board.king(turn)), sq, p)] = 1.0
      return indices
    return (piece_features(chess.WHITE), piece_features(chess.BLACK))

  def get_initial_psqt_features(self):
    return halfka_psqts()

class FactorizedFeatures(FeatureBlock):
  def __init__(self):
    super(FactorizedFeatures, self).__init__('HalfKAv2^', 0x5f234cb8, OrderedDict([('HalfKAv2', NUM_PLANES_REAL * NUM_KSQ), ('A', NUM_PLANES_VIRTUAL)]))

  def get_active_features(self, board: chess.Board):
    raise Exception('Not supported yet, you must use the c++ data loader for factorizer support during training')

  def get_feature_factors(self, idx):
    if idx >= self.num_real_features:
      raise Exception('Feature must be real')

    a_idx = idx % NUM_PLANES_REAL
    k_idx = idx // NUM_PLANES_REAL

    if NUM_PT_VIRTUAL != NUM_PT_REAL:
      if a_idx // NUM_SQ == NUM_PT_REAL - 1 and k_idx != map_king(a_idx % NUM_SQ):
        # is king piece, but not ours
        a_idx += NUM_SQ
      elif a_idx >= NUM_SQ * NUM_PT_REAL:
        # pockets
        a_idx += NUM_SQ

    return [idx, self.get_factor_base_feature('A') + a_idx]

  def get_initial_psqt_features(self):
    return halfka_psqts() + [0] * NUM_PLANES_VIRTUAL

'''
This is used by the features module for discovery of feature blocks.
'''
def get_feature_block_clss():
  return [Features, FactorizedFeatures]
