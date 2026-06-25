# Model Export And Inference

This note covers checkpoint export, neural PUCT runtime behavior, backend selection, and measured inference bottlenecks.

## Model Export

All model export starts from a trained PyTorch checkpoint:

```text
data/models/...pt
```

Do not use TorchScript for new work. PyTorch's current docs say TorchScript is
deprecated and recommend `torch.export` instead.

### MLX Safetensors

The native C++ MLX backend uses safetensors:

```text
trained PyTorch checkpoint
-> safetensors weight file
-> C++ MLX loader
```

Convert `.pt` checkpoints to `.safetensors` for the MLX C++ backend.

Export command:

```sh
python3.10 -m catchup.training.export_mlx_weights --checkpoint data/models/directional_cnn_h64_noplayer_adamw_wd1e4_iter_0014_npuct200_replay.pt --out data/models/directional_cnn_h64_noplayer_adamw_wd1e4_iter_0014_npuct200_replay_mlx.safetensors
```

Use the exported file with the MLX backend:

```sh
catchup/cpp/build/catchup_arena --agent-a neural-puct:200:data/models/directional_cnn_h64_noplayer_adamw_wd1e4_iter_0014_npuct200_replay_mlx.safetensors --agent-b puct:10000:prior=heuristic:rollout=biased --neural-backend mlx
```

### AOTI Package

AOTI is the PyTorch/MPS package path:

```text
trained PyTorch checkpoint
-> torch.export exported program
-> AOTInductor .pt2 package
-> C++ AOTIModelPackageLoader
-> MPS tensors
```

Export command:

```sh
python3.10 -m catchup.training.export_aoti --checkpoint data/models/gnn_policy_value_30shards_3sym_20ep.pt --exported-program data/models/gnn_policy_value_30shards_3sym_20ep_exported_b32.pt2 --package data/models/gnn_policy_value_30shards_3sym_20ep_aoti_mps_b32.pt2 --device mps --batch-size 32
```

AOTInductor packages are exported for a fixed input batch size, so the package
batch size must match `--neural-batch-size`.

The installed PyTorch 2.12 build currently writes the AOTInductor package but
then raises an internal `AssertionError`. The export helper treats that as a
warning only when the requested package file exists. If the package is missing,
it fails.

Commands involving MPS should be run unsandboxed.

The AOTI loader should use device index `-1` for MPS packages. Passing explicit
device index `0` failed with:

```text
Incorrect device passed to aoti_runner_mps
```

## Neural PUCT

For each evaluated state, neural PUCT needs:

```text
policy priors over legal actions
value from current_player perspective
```

The policy guides tree selection. The value replaces random rollout at newly
reached non-terminal leaves.

### Python Version

`catchup/neural_puct.py` loads checkpoints through checkpoint metadata, so both
MLP and graph checkpoints use the same evaluator.

Example:

```sh
python3.10 -m catchup.neural_puct --checkpoint data/models/gnn_policy_value_30shards_3sym_3ep.pt --simulations 100 --device mps
```

### C++ Version

The C++ neural search is split into:

```text
catchup/cpp/puct_neural.hpp
catchup/cpp/puct_neural.cpp
```

The C++ arena accepts a neural agent in this form:

```text
neural-puct:N:MODEL
```

Example:

```sh
catchup/cpp/build/catchup_arena --agent-a neural-puct:200:data/models/directional_cnn_h64_noplayer_adamw_wd1e4_iter_0014_npuct200_replay_mlx.safetensors --agent-b puct:10000:prior=heuristic:rollout=biased --neural-backend mlx --threads 128 --neural-batch-size 128
```

The backend selects how `MODEL` is loaded: `.safetensors` for MLX, or `.pt2`
for AOTI. Both paths build the same feature vector and normalize policy logits
only over legal actions.

The arena uses `BatchedNeuralEvaluator` when either side is a neural PUCT agent.
If both neural agents use the same model path, they share one batcher. If they
use different model paths, each model gets its own batcher.

## Batched Neural Evaluation

Batched evaluation is only about grouping model calls. It is not the training
loop.

The C++ neural evaluator batches requests across active games, not inside one
game's search tree. Each game thread still runs its PUCT loop in order. When a
search needs a model evaluation, it sends the leaf state to one shared evaluator
and waits for the result. The evaluator collects requests from multiple game
threads, runs one backend batch, then returns each result to the game that
requested it.

That gives this shape:

```text
game worker 0 -> leaf state ----\
game worker 1 -> leaf state ----- shared BatchedNeuralEvaluator -> model batch
game worker 2 -> leaf state ----/
```

This does not use virtual loss or any other within-tree synchronization trick.
The order of decisions inside each game is preserved.

The search code can run with either evaluator shape:

```text
NeuralEvaluator         -> one model call for one state
BatchedNeuralEvaluator  -> one model call for several queued states
```

Keep active game workers near the neural batch size; otherwise the batcher is
underfed.

For the directional-CNN h64 package, batch 128 was faster than batch 64 on the
same 256-game stochastic same-model arena workload: 0.602s/game vs 0.706s/game,
a 1.17x throughput speedup.

Batch wait profiling on the directional-CNN h64 package (128 games, 100
simulations, 128 threads, batch size 128, seed 123):

```text
wait_ms  real_s  avg_batch  full_batch  total_fill_wait_ms  avg_model_ms  avg_request_ms
2.00      92.36    100.34      64.5%             5630.82        13.54          14.25
1.00      88.99    100.32      64.0%             2910.05        13.47          13.86
0.50      84.64    100.29      63.9%             1470.30        13.05          13.41
0.25      85.61    100.21      62.1%              806.77        13.25          13.66
0.00      90.47     91.59      55.4%                0.00        12.99          13.76
```

The old `2.0ms` wait spent measurable time waiting without improving batch
size over `0.5ms`. No deliberate wait (`0.0ms`) made batches smaller and was
slower. The current C++ default is therefore `--neural-batch-wait-ms 0.5`.

Internal timing for the `0.5ms` run, with an explicit MPS synchronization after
`loader.run()`:

```text
avg_model_ms        13.302
avg_feature_ms       0.049
avg_input_ms         0.667
avg_inference_ms    12.063
avg_output_ms        0.488
avg_postprocess_ms   0.027
```

`avg_inference_ms` is the real bottleneck. Without the explicit MPS synchronization,
this time appeared under output copy because copying tensors back to CPU forced
the queued MPS work to finish.

CPU AOTI inference was much slower than MPS in the same self-play profile:

```text
device  package suffix   real_s  requests  batches  avg_batch  avg_model_ms  avg_inference_ms  avg_request_ms
mps     aoti_mps_b128     76.37    604524     5804    104.16        12.30        11.08          12.71
cpu     aoti_cpu_b128    739.25    607804     5866    103.62       125.24       125.05         125.38
```

Inference on CPU is slow: about 9.7x slower in wall time, with model calls about
10.2x slower.

MLX is for inference on Apple Silicon. Its directional-CNN output matches the
PyTorch checkpoint numerically:

```text
checkpoint data/models/directional_cnn_h64_noplayer_iter_0008_npuct100cont_replay.pt
max_abs_diff policy ~= 5e-5
max_abs_diff value  ~= 3e-6
```

Native C++ MLX integration uses the same `neural-puct` search engine with
`--neural-backend mlx`.

```text
backend  real_s  requests  batches  avg_batch  avg_model_ms  avg_inference_ms  avg_request_ms
aoti-mps  76.37    604524     5804    104.16        12.30             11.08          12.71
mlx       33.56    607804     5867    103.60         4.94              4.84           5.13
```

The current native MLX path is about 2.1x faster end-to-end than the AOTI MPS
path in the same C++ self-play profile: h64 directional-CNN model, 128 games,
100 simulations per move, 128 threads, neural batch size 128, `wait_ms = 0.5`,
seed 123.
