# Legacy Atomic V1 contract

This trainer keeps the deployed Fairy-Stockfish Atomic NNUE format usable while
the engine, generator, and trainer are specialized. The compatibility baseline
is the repository commit `b15df38a9aae8ab9b40b2378020b3099c7c5d179`.

## Network format

- Version: `0x7AF32F20`.
- Real feature set: `HalfKAv2`, hash `0x5F234CB8`.
- Real features: 45,056.
- Training feature set: `HalfKAv2^`, with 768 additional virtual rows.
- Feature-transformer width: 512 plus eight PSQT buckets.
- Layer stacks: eight copies of `1024 -> 16 -> 32 -> 1`.
- Integers and tensor payloads: little-endian legacy layout.

Virtual rows are a training aid and never appear in `.nnue`. The reader first
validates the version, network hash, transformer hash, dimensions, description,
complete payload, and EOF. `train.py --resume-from-model NET.nnue` then pads a
compatible factorized target with zero rows. A read followed by an immediate
write preserves every byte, including the description.

The current strongest compatibility fixture is identified by SHA-256, not by
its filename:

```
99dc67eabf26a64faeeca3a88b4c38597a840b8d4a874b9f2cf658c6f92a04a6
```

## Dataset format

The legacy reader consumes fixed 72-byte `PackedSfenValue` records. It rejects
missing, empty, wrong-extension, and non-integral files before worker threads
start. The checked-in test fixture contains 32 records (2,304 bytes) with
SHA-256:

```
9749b7673746e51177327081b8982a7962e4f7fbeb80561f69dd16c953970b91
```

The seed is applied before model construction and is forwarded to the native
stream. Input reads receive monotonically increasing sequence numbers while
feature extraction may run concurrently, so yielded batch order is identical
for one or many workers. Random skipping is enabled only for training;
validation keeps the same smart-filter setting, never uses random skipping,
and restarts its dedicated stream for each validation pass so epoch metrics use
the same positions.

Smart filtering preserves the historical target-quality rule: discard teacher
captures (including en passant) and positions where the moving king is under a
geometric orthodox attack. It is deliberately not an Atomic legality oracle.

## Supported continuation flow

```
python train.py train.bin validation.bin \
  --resume-from-model atomic.nnue \
  --features="HalfKAv2^" \
  --seed 42

python serialize.py model.pt candidate.nnue
```

Training and validation must be distinct filesystem objects unless
`--allow-train-as-validation` is explicitly supplied for a smoke test. This
legacy path rejects a Lightning world size other than one before constructing
the stateful native streams. Multi-device or multi-node training remains
unsupported until rank-aware deterministic data sharding is implemented.

The loader bounds-checks the packed bit stream and all board indices before
constructing a position. A structurally valid 72-byte record with an invalid
king square, Huffman code, teacher move, or result fails the batch fetch; worker
exceptions are returned to Python as `RuntimeError` rather than being mistaken
for end-of-file.
