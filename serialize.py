import argparse
import features
import model as M
import numpy
import os
import struct
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from functools import reduce
import operator

def ascii_hist(name, x, bins=6):
  N,X = numpy.histogram(x, bins=bins)
  total = 1.0*len(x)
  width = 50
  nmax = N.max()

  print(name)
  for (xi, n) in zip(X,N):
    bar = '#'*int(n*1.0*width/nmax)
    xi = '{0: <8.4g}'.format(xi).ljust(10)
    print('{0}| {1}'.format(xi,bar))

# hardcoded for now
VERSION = 0x7AF32F20
DEFAULT_DESCRIPTION = "Network trained with the https://github.com/ianfab/variant-nnue-pytorch trainer."
MAX_DESCRIPTION_LENGTH = 1024 * 1024


class NNUEFormatError(ValueError):
  pass


def get_unfactorized_feature_set(feature_set):
  """Return the real feature layout stored by an NNUE file.

  Virtual factorizer rows only exist while training and are coalesced away by
  the writer. Consequently an on-disk HalfKAv2 net always contains HalfKAv2,
  never HalfKAv2^, even when it was trained with factorization enabled.
  """
  names = [feature.get_main_factor_name() for feature in feature_set.features]
  return features.get_feature_set_from_name('+'.join(names))

class NNUEWriter():
  """
  All values are stored in little endian.
  """
  def __init__(self, model, description=None):
    if description is None:
        saved_description = getattr(model, 'nnue_description', None)
        description = DEFAULT_DESCRIPTION if saved_description is None else saved_description
    if not isinstance(description, str):
        raise TypeError('NNUE description must be text')

    self.buf = bytearray()

    fc_hash = self.fc_hash(model)
    self.write_header(model, fc_hash, description)
    self.int32(model.feature_set.hash ^ (M.L1*2)) # Feature transformer hash
    self.write_feature_transformer(model)
    for l1, l2, output in model.layer_stacks.get_coalesced_layer_stacks():
      self.int32(fc_hash) # FC layers hash
      self.write_fc_layer(l1)
      self.write_fc_layer(l2)
      self.write_fc_layer(output, is_output=True)

  @staticmethod
  def fc_hash(model):
    return NNUEWriter.fc_hash_for_num_ls_buckets(model.num_ls_buckets)

  @staticmethod
  def fc_hash_for_num_ls_buckets(num_ls_buckets):
    # The legacy hash describes one stack; the bucket count is represented by
    # repeating that hashed payload in the file rather than mixing the count.
    _ = num_ls_buckets
    # InputSlice hash
    prev_hash = 0xEC42E90D
    prev_hash ^= (M.L1 * 2)

    # Fully connected layers
    for output_size in (M.L2, M.L3, 1):
      layer_hash = 0xCC03DAE4
      layer_hash += output_size
      layer_hash ^= prev_hash >> 1
      layer_hash ^= (prev_hash << 31) & 0xFFFFFFFF
      if output_size != 1:
        # Clipped ReLU hash
        layer_hash = (layer_hash + 0x538D24C7) & 0xFFFFFFFF
      prev_hash = layer_hash
    return layer_hash

  def write_header(self, model, fc_hash, description):
    self.int32(VERSION) # version
    self.int32(fc_hash ^ model.feature_set.hash ^ (M.L1*2)) # halfkp network hash
    encoded_description = description.encode('utf-8')
    if len(encoded_description) > MAX_DESCRIPTION_LENGTH:
      raise ValueError('NNUE description exceeds {} bytes'.format(MAX_DESCRIPTION_LENGTH))
    self.int32(len(encoded_description)) # Network definition
    self.buf.extend(encoded_description)

  def write_feature_transformer(self, model):
    # int16 bias = round(x * 127)
    # int16 weight = round(x * 127)
    layer = model.input
    bias = layer.bias.data[:M.L1]
    bias = bias.mul(127).round().to(torch.int16).cpu()
    ascii_hist('ft bias:', bias.numpy())
    self.buf.extend(bias.flatten().numpy().tobytes())

    weight = M.coalesce_ft_weights(model, layer)
    weight0 = weight[:, :M.L1]
    psqtweight0 = weight[:, M.L1:]
    weight = weight0.mul(127).round().to(torch.int16).cpu()
    psqtweight = psqtweight0.mul(9600).round().to(torch.int32).cpu() # kPonanzaConstant * FV_SCALE = 9600
    ascii_hist('ft weight:', weight.numpy())
    # weights stored as [41024][256]
    self.buf.extend(weight.flatten().numpy().tobytes())
    self.buf.extend(psqtweight.flatten().numpy().tobytes())

  def write_fc_layer(self, layer, is_output=False):
    # FC layers are stored as int8 weights, and int32 biases
    kWeightScaleBits = 6
    kActivationScale = 127.0
    if not is_output:
      kBiasScale = (1 << kWeightScaleBits) * kActivationScale # = 8128
    else:
      kBiasScale = 9600.0 # kPonanzaConstant * FV_SCALE = 600 * 16 = 9600
    kWeightScale = kBiasScale / kActivationScale # = 64.0 for normal layers
    kMaxWeight = 127.0 / kWeightScale # roughly 2.0

    # int32 bias = round(x * kBiasScale)
    # int8 weight = round(x * kWeightScale)
    bias = layer.bias.data
    bias = bias.mul(kBiasScale).round().to(torch.int32).cpu()
    ascii_hist('fc bias:', bias.numpy())
    self.buf.extend(bias.flatten().numpy().tobytes())
    weight = layer.weight.data
    clipped = int(torch.count_nonzero(weight.clamp(-kMaxWeight, kMaxWeight) - weight).item())
    total_elements = torch.numel(weight)
    clipped_max = float(torch.max(torch.abs(weight.clamp(-kMaxWeight, kMaxWeight) - weight)).item())
    print("layer has {}/{} clipped weights. Exceeding by {} the maximum {}.".format(clipped, total_elements, clipped_max, kMaxWeight))
    weight = weight.clamp(-kMaxWeight, kMaxWeight).mul(kWeightScale).round().to(torch.int8)
    ascii_hist('fc weight:', weight.cpu().numpy())
    # FC inputs are padded to 32 elements for simd.
    num_input = weight.shape[1]
    if num_input % 32 != 0:
      num_input += 32 - (num_input % 32)
      new_w = weight.new_zeros((weight.shape[0], num_input))
      new_w[:, :weight.shape[1]] = weight
      weight = new_w
    # Stored as [outputs][inputs], so we can flatten
    self.buf.extend(weight.flatten().cpu().numpy().tobytes())

  def int32(self, v):
    self.buf.extend(struct.pack("<I", v))

class NNUEReader():
  def __init__(self, f, feature_set=None):
    self.f = f
    if feature_set is None:
      feature_set = self.detect_feature_set()
    else:
      feature_set = get_unfactorized_feature_set(feature_set)

    self.feature_set = feature_set
    self.model = M.NNUE(feature_set)
    fc_hash = NNUEWriter.fc_hash(self.model)

    self.description = self.read_header(feature_set, fc_hash)
    self.model.nnue_description = self.description
    self.read_int32(feature_set.hash ^ (M.L1*2)) # Feature transformer hash
    self.read_feature_transformer(self.model.input, self.model.num_psqt_buckets)
    for i in range(self.model.num_ls_buckets):
      l1 = nn.Linear(2*M.L1, M.L2)
      l2 = nn.Linear(M.L2, M.L3)
      output = nn.Linear(M.L3, 1)
      self.read_int32(fc_hash) # FC layers hash
      self.read_fc_layer(l1)
      self.read_fc_layer(l2)
      self.read_fc_layer(output, is_output=True)
      self.model.layer_stacks.l1.weight.data[i*M.L2:(i+1)*M.L2, :] = l1.weight
      self.model.layer_stacks.l1.bias.data[i*M.L2:(i+1)*M.L2] = l1.bias
      self.model.layer_stacks.l2.weight.data[i*M.L3:(i+1)*M.L3, :] = l2.weight
      self.model.layer_stacks.l2.bias.data[i*M.L3:(i+1)*M.L3] = l2.bias
      self.model.layer_stacks.output.weight.data[i:(i+1), :] = output.weight
      self.model.layer_stacks.output.bias.data[i:(i+1)] = output.bias

    if self.f.read(1):
      raise NNUEFormatError('Unexpected trailing bytes after the NNUE payload')

  def detect_feature_set(self):
    try:
      position = self.f.tell()
    except (AttributeError, OSError) as error:
      raise NNUEFormatError('NNUE input must be a seekable binary file') from error

    header = self.f.read(8)
    self.f.seek(position)
    if len(header) != 8:
      raise NNUEFormatError('Truncated NNUE header')

    version, network_hash = struct.unpack('<II', header)
    if version != VERSION:
      raise NNUEFormatError('Unsupported NNUE version: expected {:08x}, got {:08x}'.format(VERSION, version))

    matches = []
    for name in features.get_available_feature_blocks_names():
      if '^' in name:
        continue
      candidate = features.get_feature_set_from_name(name)
      fc_hash = NNUEWriter.fc_hash_for_num_ls_buckets(candidate.num_ls_buckets)
      expected = fc_hash ^ candidate.hash ^ (M.L1 * 2)
      if network_hash == expected:
        matches.append(candidate)

    if len(matches) != 1:
      raise NNUEFormatError(
        'Could not uniquely identify the NNUE feature set from network hash {:08x}'.format(network_hash)
      )
    return matches[0]

  def read_header(self, feature_set, fc_hash):
    self.read_int32(VERSION) # version
    self.read_int32(fc_hash ^ feature_set.hash ^ (M.L1*2)) # halfkp network hash
    desc_len = self.read_int32() # Network definition
    if desc_len > MAX_DESCRIPTION_LENGTH:
      raise NNUEFormatError('NNUE description is unreasonably large: {} bytes'.format(desc_len))
    description = self.read_exact(desc_len, 'network description')
    try:
      return description.decode('utf-8')
    except UnicodeDecodeError as error:
      raise NNUEFormatError('NNUE description is not valid UTF-8') from error

  def read_exact(self, size, label):
    data = self.f.read(size)
    if len(data) != size:
      raise NNUEFormatError('Truncated NNUE {}: expected {} bytes, got {}'.format(label, size, len(data)))
    return data

  def tensor(self, dtype, shape):
    count = reduce(operator.mul, shape, 1)
    item_size = numpy.dtype(dtype).itemsize
    raw = self.read_exact(count * item_size, 'tensor')
    d = numpy.frombuffer(raw, dtype, count)
    d = torch.from_numpy(d.astype(numpy.float32))
    d = d.reshape(shape)
    return d

  def read_feature_transformer(self, layer, num_psqt_buckets):
    bias = self.tensor(numpy.int16, [layer.bias.shape[0]-num_psqt_buckets]).divide(127.0)
    layer.bias.data = torch.cat([bias, torch.tensor([0]*num_psqt_buckets)])
    # weights stored as [41024][256], so we need to transpose the pytorch [256][41024]
    shape = layer.weight.shape
    weights = self.tensor(numpy.int16, [shape[0], shape[1]-num_psqt_buckets])
    psqtweights = self.tensor(numpy.int32, [shape[0], num_psqt_buckets])
    weights = weights.divide(127.0)
    psqtweights = psqtweights.divide(9600.0)
    layer.weight.data = torch.cat([weights, psqtweights], dim=1)

  def read_fc_layer(self, layer, is_output=False):
    # FC layers are stored as int8 weights, and int32 biases
    kWeightScaleBits = 6
    kActivationScale = 127.0
    if not is_output:
      kBiasScale = (1 << kWeightScaleBits) * kActivationScale # = 8128
    else:
      kBiasScale = 9600.0 # kPonanzaConstant * FV_SCALE = 600 * 16 = 9600
    kWeightScale = kBiasScale / kActivationScale # = 64.0 for normal layers

    # FC inputs are padded to 32 elements for simd.
    non_padded_shape = layer.weight.shape
    padded_shape = (non_padded_shape[0], ((non_padded_shape[1]+31)//32)*32)

    layer.bias.data = self.tensor(numpy.int32, layer.bias.shape).divide(kBiasScale)
    layer.weight.data = self.tensor(numpy.int8, padded_shape).divide(kWeightScale)

    # Strip padding.
    layer.weight.data = layer.weight.data[:non_padded_shape[0], :non_padded_shape[1]]

  def read_int32(self, expected=None):
    v = struct.unpack("<I", self.read_exact(4, 'integer'))[0]
    if expected is not None and v != expected:
      raise NNUEFormatError("Expected: %x, got %x" % (expected, v))
    return v


def load_nnue_for_training(f, target_feature_set):
  """Load the serialized real features, then add zeroed training factors."""
  reader = NNUEReader(f)
  model = reader.model
  model.set_feature_set(target_feature_set)
  return model

def main():
  parser = argparse.ArgumentParser(description="Converts files between ckpt and nnue format.")
  parser.add_argument("source", help="Source file (can be .ckpt, .pt, .pth or .nnue)")
  parser.add_argument("target", help="Target file (can be .pt, .pth or .nnue)")
  parser.add_argument("--description", default=None, type=str, dest='description', help="The description string to include in the network. Only works when serializing into a .nnue file.")
  features.add_argparse_args(parser, default=None)
  args = parser.parse_args()

  feature_set = features.get_feature_set_from_name(args.features) if args.features else None
  source_suffix = os.path.splitext(args.source)[1].lower()
  target_suffix = os.path.splitext(args.target)[1].lower()

  print('Converting %s to %s' % (args.source, args.target))

  if source_suffix == '.ckpt':
    if feature_set is None:
      feature_set = features.get_feature_set_from_name('HalfKAv2^')
    nnue = M.NNUE.load_from_checkpoint(args.source, feature_set=feature_set)
    nnue.eval()
  elif source_suffix in ('.pt', '.pth'):
    # Load with weights_only=False to avoid safe_globals complexity
    # This is safe since we trust the checkpoint source
    nnue = torch.load(args.source, map_location='cpu', weights_only=False)
  elif source_suffix == '.nnue':
    with open(args.source, 'rb') as f:
      reader = NNUEReader(f, feature_set)
      nnue = reader.model
  else:
    raise Exception('Invalid network input format.')

  if target_suffix == '.ckpt':
    raise Exception('Cannot convert into .ckpt')
  elif target_suffix in ('.pt', '.pth'):
    torch.save(nnue, args.target)
  elif target_suffix == '.nnue':
    writer = NNUEWriter(nnue, args.description)
    with open(args.target, 'wb') as f:
      f.write(writer.buf)
  else:
    raise Exception('Invalid network output format.')

if __name__ == '__main__':
  main()
