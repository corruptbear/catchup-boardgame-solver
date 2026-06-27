# Model Training

This is the entry point for Catchup neural training notes. Keep this file short:
current status, canonical commands, and links to the detailed notes. Historical
tables belong in `neural_experiments.md`.

## Current Status

- Training is still policy/value learning from generated Catchup positions, not a complete AlphaZero loop.
- The active model family in the latest notes is the directional-CNN h64 no-player model.
- The preferred Apple Silicon inference path is native MLX when a safetensors export is available.
- AOTI/MPS works, but measured runtime was dominated by model inference rather than feature construction or postprocessing.
- Direct neural-vs-neural arena checks should use visit-count sampling, not deterministic max-visit action selection.
- Older checkpoints and generated shards should not be overwritten.

## Detail Notes

- [model_architectures.md](model_architectures.md): MLP, graph, and directional-CNN model designs and supervised results.
- [model_export_and_inference.md](model_export_and_inference.md): MLX/AOTI export, neural PUCT runtime, batching, and backend timings.
- [neural_self_play.md](neural_self_play.md): bootstrap data, iterative neural self-play, losses and metrics, replay-buffer coverage, and exploration controls.
- [neural_experiments.md](neural_experiments.md): historical self-play branches, arena checks, and artifact lists.

## Canonical Commands

Generate bootstrap data:

```sh
catchup/cpp/build/catchup_self_play --games 50 --simulations 10000 --threads 12 --early-win false --out data/bootstrap/shard_0001_50g_10k.jsonl
```

Train a supervised directional-CNN checkpoint:

```sh
python3.10 -m catchup.training.torch_policy_value --architecture directional-cnn --cnn-layers 4 --data-glob 'data/bootstrap/shard_*_50g_10k.jsonl' --validation-shards 3 --epochs 20 --batch-size 1024 --hidden-size 64 --symmetry-copies 3 --device mps --out data/models/directional_cnn_h64_noplayer_30shards_3sym_20ep.pt --metrics-out data/models/directional_cnn_h64_noplayer_30shards_3sym_20ep_metrics.json
```

Export an MLX safetensors file:

```sh
python3.10 -m catchup.training.export_mlx_weights --checkpoint data/models/directional_cnn_h64_noplayer_adamw_wd1e4_iter_0014_npuct200_replay.pt --out data/models/directional_cnn_h64_noplayer_adamw_wd1e4_iter_0014_npuct200_replay_mlx.safetensors
```

Run a neural arena check with MLX:

```sh
catchup/cpp/build/catchup_arena --agent-a neural-puct:200:data/models/directional_cnn_h64_noplayer_adamw_wd1e4_iter_0014_npuct200_replay_mlx.safetensors --agent-b puct:10000:prior=heuristic:rollout=biased --neural-backend mlx --threads 128 --neural-batch-size 128
```

Generate neural self-play data:

```sh
catchup/cpp/build/catchup_self_play --teacher neural-puct --model data/models/gnn_policy_value_30shards_3sym_20ep_aoti_mps_b32.pt2 --games 50 --simulations 100 --threads 12 --neural-batch-size 32 --early-win false --out data/neural_self_play/example_50g.jsonl
```

## Operating Rules

- Do not use TorchScript for new work; use `torch.export` / AOTInductor or MLX safetensors.
- AOTInductor packages have a fixed batch size; match the package batch size with `--neural-batch-size`.
- Commands involving MPS should be run unsandboxed.
- For neural self-play data, use root Dirichlet noise plus visit-count sampling; the saved policy target remains raw normalized visits.
- Count raw positions for replay-buffer coverage. Symmetry augmentation changes views; it does not create independent positions.
- Arena game seeds are hashed from `(base_seed, pair_index, game_in_pair)`, so adjacent base seeds no longer just shift by one game. Older arena rows generated before this fix used consecutive game seeds; separated base seeds such as `1`, `100001`, and `200001` were used there to avoid overlap.
