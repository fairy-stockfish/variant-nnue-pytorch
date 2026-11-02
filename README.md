# Variant NNUE trainer

This is the chess variant NNUE training code for [Fairy-Stockfish](https://github.com/ianfab/Fairy-Stockfish). See the documentation in the [wiki](https://github.com/fairy-stockfish/variant-nnue-pytorch/wiki) for more details on the training process and [join our discord](https://discord.gg/FYUGgmCFB4) to ask questions. This project is derived from the [trainer for standard chess](https://github.com/glinscott/nnue-pytorch) used for official Stockfish.

# Setup

A [CUDA capable GPU](https://developer.nvidia.com/cuda-gpus) is strongly recommended.
Note: you don't need to download the CUDA Toolkit

Training on CPU only is also possible (see below), but much slower. In that case CuPy is not required,
and a CPU build of PyTorch is sufficient (`pip install torch --index-url https://download.pytorch.org/whl/cpu`).

#### Package dependencies

At least Python 3.9 needed.

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

PyTorch with CUDA 11.8 will be automatically installed, along with the matching CuPy version
If your GPU is CUDA 12.8 capable you can use requirements-CUDA128.txt instead of requirements.txt

#### Build the fast DataLoader
This requires a C++17 compiler and cmake.

Windows:
```
compile_data_loader.bat
```

Linux/Mac:
```
sh compile_data_loader.bat
```

# Train a network

```
source env/bin/activate
python train.py train_data.bin val_data.bin
```

## Resuming from a checkpoint
```
python train.py --resume_from_checkpoint <path> ...
```

## Training on GPU
```
python train.py --gpus 1 ...
```

## Training on CPU
```
python train.py --accelerator cpu --batch-size 1024 --threads 8 ...
```
Without a GPU (or without CuPy) the feature transformer falls back to a pure PyTorch implementation.
## Feature set selection
By default the trainer uses a factorized HalfKAv2 feature set (named "HalfKAv2^")
If you wish to change the feature set used then you can use the `--features=NAME` option. For the list of available features see `--help`
The default is:
```
python train.py ... --features="HalfKAv2^"
```

# Logging

```
pip install tensorboard
tensorboard --logdir=logs
```
Then, go to http://localhost:6006/

