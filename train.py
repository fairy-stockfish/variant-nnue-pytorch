import argparse
import model as M
import nnue_dataset
import pytorch_lightning as pl
import features
import math
import os
import torch
from torch import set_num_threads as t_set_num_threads
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset


INT32_MAX = 2**31 - 1
UINT32_MAX = 2**32 - 1


def non_negative_skip_count(value):
  value = int(value)
  if value < 0 or value > INT32_MAX:
    raise argparse.ArgumentTypeError('must be between 0 and {}'.format(INT32_MAX))
  return value


def positive_int32(value):
  value = int(value)
  if value < 1 or value > INT32_MAX:
    raise argparse.ArgumentTypeError('must be between 1 and {}'.format(INT32_MAX))
  return value


def batch_size_argument(value):
  value = int(value)
  if value != -1 and not 1 <= value <= INT32_MAX:
    raise argparse.ArgumentTypeError('must be -1 (automatic) or between 1 and {}'.format(INT32_MAX))
  return value


def deterministic_seed(value):
  value = int(value)
  if value < 0 or value > UINT32_MAX:
    raise argparse.ArgumentTypeError('must be between 0 and {}'.format(UINT32_MAX))
  return value


def lambda_argument(value):
  value = float(value)
  if not math.isfinite(value) or value < 0.0 or value > 1.0:
    raise argparse.ArgumentTypeError('must be a finite value between 0 and 1')
  return value


def default_batch_size(device):
  return 16384 if torch.device(device).type == 'cuda' else 128


def require_single_device_trainer(trainer):
  """Fail closed until the stateful native stream has rank-aware sharding."""
  world_size = int(trainer.world_size)
  if world_size != 1:
    raise ValueError(
      'Legacy Atomic V1 training supports exactly one Lightning process/device; '
      'multi-device and multi-node strategies would duplicate the stateful native stream.'
    )


def validate_data_paths(train_filename, val_filename, allow_train_as_validation=False):
  if not os.path.isfile(train_filename):
    raise FileNotFoundError('{} is not a training-data file'.format(train_filename))
  if not os.path.isfile(val_filename):
    raise FileNotFoundError('{} is not a validation-data file'.format(val_filename))

  try:
    same_file = os.path.samefile(train_filename, val_filename)
  except OSError:
    same_file = os.path.normcase(os.path.realpath(train_filename)) == os.path.normcase(os.path.realpath(val_filename))

  if same_file and not allow_train_as_validation:
    raise ValueError(
      'Training and validation must be separate files. '
      'Use --allow-train-as-validation only for an explicit smoke/debug run.'
    )

def make_data_loaders(train_filename, val_filename, feature_set, num_workers, batch_size, filtered, random_fen_skipping, main_device, epoch_size, val_size, seed):
  features_name = feature_set.name
  train_infinite = nnue_dataset.SparseBatchDataset(features_name, train_filename, batch_size, num_workers=num_workers,
                                                   filtered=filtered, random_fen_skipping=random_fen_skipping, device=main_device, seed=seed)
  val_infinite = nnue_dataset.SparseBatchDataset(features_name, val_filename, batch_size, filtered=filtered,
                                                   random_fen_skipping=0, device=main_device, seed=seed)
  # num_workers has to be 0 for sparse, and 1 for dense
  # it currently cannot work in parallel mode but it shouldn't need to
  train = DataLoader(nnue_dataset.FixedNumBatchesDataset(train_infinite, (epoch_size + batch_size - 1) // batch_size), batch_size=None, batch_sampler=None)
  val = DataLoader(nnue_dataset.FixedNumBatchesDataset(val_infinite, (val_size + batch_size - 1) // batch_size, restart_on_zero=True), batch_size=None, batch_sampler=None)
  return train, val

def load_or_create_model(feature_set, lambda_, seed, resume_from_model=None):
  # Model initialization consumes torch RNG state, so seed before constructing
  # the model (or loading a checkpoint that may initialize Python objects).
  seed = deterministic_seed(seed)
  lambda_ = lambda_argument(lambda_)
  pl.seed_everything(seed, workers=True)

  if resume_from_model is None:
    nnue = M.NNUE(feature_set=feature_set, lambda_=lambda_)
  else:
    suffix = os.path.splitext(os.fspath(resume_from_model))[1].lower()
    if suffix == '.nnue':
      import serialize
      with open(resume_from_model, 'rb') as network_file:
        nnue = serialize.load_nnue_for_training(network_file, feature_set)
    elif suffix in ('.pt', '.pth'):
      # Load with weights_only=False to avoid safe_globals complexity.
      # This is safe since we trust the checkpoint source.
      nnue = torch.load(resume_from_model, map_location='cpu', weights_only=False)
      nnue.set_feature_set(feature_set)
    else:
      raise ValueError('--resume-from-model accepts only .pt, .pth, or .nnue files')
    nnue.lambda_ = lambda_

  return nnue

def main():
  parser = argparse.ArgumentParser(description="Trains the network.")
  parser.add_argument("train", help="Training data (.bin)")
  parser.add_argument("val", help="Validation data (.bin)")
  parser = pl.Trainer.add_argparse_args(parser)
  parser.add_argument("--lambda", default=1.0, type=lambda_argument, dest='lambda_', help="lambda=1.0 = train on evaluations, lambda=0.0 = train on game results, interpolates between (default=1.0).")
  parser.add_argument("--num-workers", default=1, type=positive_int32, dest='num_workers', help="Number of native worker threads to use for data loading.")
  parser.add_argument("--batch-size", default=-1, type=batch_size_argument, dest='batch_size', help="Number of positions per batch / per iteration. Default on GPU = 16384, on CPU = 128.")
  parser.add_argument("--threads", default=-1, type=int, dest='threads', help="Number of torch threads to use. Default automatic (cores) .")
  parser.add_argument("--seed", default=42, type=deterministic_seed, dest='seed', help="Deterministic model and native-loader seed (0..2^32-1).")
  parser.add_argument("--smart-fen-skipping", action='store_true', dest='smart_fen_skipping_deprecated', help="If enabled positions that are bad training targets will be skipped during loading. Default: True, kept for backwards compatibility. This option is ignored")
  parser.add_argument("--no-smart-fen-skipping", action='store_true', dest='no_smart_fen_skipping', help="If used then no smart fen skipping will be done. By default smart fen skipping is done.")
  parser.add_argument("--random-fen-skipping", default=3, type=non_negative_skip_count, dest='random_fen_skipping', help="Skip this many positions on average before retaining one (0..INT_MAX).")
  parser.add_argument("--allow-train-as-validation", action='store_true', help="Permit the same file for training and validation. Intended only for explicit smoke/debug runs.")
  parser.add_argument("--resume-from-model", dest='resume_from_model', help="Initialize from a trusted .pt/.pth model or a legacy HalfKAv2 .nnue network.")
  parser.add_argument("--epoch-size", type=positive_int32, default=20000000, dest='epoch_size', help="Number of positions per epoch.")
  parser.add_argument("--validation-size", type=positive_int32, default=1000000, dest='validation_size', help="Number of positions per validation step.")
  features.add_argparse_args(parser)
  args = parser.parse_args()

  validate_data_paths(args.train, args.val, args.allow_train_as_validation)

  feature_set = features.get_feature_set_from_name(args.features)
  nnue = load_or_create_model(feature_set, args.lambda_, args.seed, args.resume_from_model)

  print("Feature set: {}".format(feature_set.name))
  print("Num real features: {}".format(feature_set.num_real_features))
  print("Num virtual features: {}".format(feature_set.num_virtual_features))
  print("Num features: {}".format(feature_set.num_features))

  print("Training with {} validating with {}".format(args.train, args.val))

  print("Seed {}".format(args.seed))

  if args.threads > 0:
    print('limiting torch to {} threads.'.format(args.threads))
    t_set_num_threads(args.threads)

  logdir = args.default_root_dir if args.default_root_dir else 'logs/'
  print('Using log dir {}'.format(logdir), flush=True)

  tb_logger = pl_loggers.TensorBoardLogger(logdir)
  checkpoint_callback = pl.callbacks.ModelCheckpoint(save_last=True, every_n_epochs=1, save_top_k=-1)
  trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback], logger=tb_logger)
  require_single_device_trainer(trainer)

  main_device = torch.device(trainer.strategy.root_device)

  batch_size = args.batch_size
  if batch_size <= 0:
    batch_size = default_batch_size(main_device)
  print('Using batch size {}'.format(batch_size))

  print('Smart fen skipping: {}'.format(not args.no_smart_fen_skipping))
  print('Random fen skipping: {}'.format(args.random_fen_skipping))

  print('Using c++ data loader')
  train, val = make_data_loaders(args.train, args.val, feature_set, args.num_workers, batch_size, not args.no_smart_fen_skipping, args.random_fen_skipping, main_device, args.epoch_size, args.validation_size, args.seed)

  trainer.fit(nnue, train, val)

if __name__ == '__main__':
  main()
