# Neural Experiments

This note is the historical experiment log. Keep raw result tables here, but put stable operating rules in the shorter reference notes.

## Experiment 1: Warm-Up And Search Increase

Setup:

```text
base model
data/models/directional_cnn_h64_noplayer_30shards_3sym_20ep_aoti_mps_b128.pt2

self-play settings
iter_0001 through iter_0005  neural-puct:100
iter_0006                    neural-puct:200
iter_0007 through iter_0009  neural-puct:400
games per generation      640
threads                   128
neural batch size         128
root noise                default decayed schedule
seed                      random_device, not fixed
output directory          data/neural_self_play_noplayer/
```

Self-play and replay training summary:

```text
generation  sims  samples  Blue  White  avg filled  avg turns  train batches  train samples  train sec  loss    policy  value
iter_0001    100    32691   342    298      49.919    22.462             26          13312      2.360  2.9713  2.2708  0.7005
iter_0002    100    33083   304    336      50.427    22.356             52          26624      4.636  2.8225  2.1564  0.6661
iter_0003    100    32360   297    343      49.652    21.977             76          38912      6.330  2.8079  2.1334  0.6744
iter_0004    100    32351   311    329      49.567    21.995            102          52224      8.933  2.7023  2.0867  0.6156
iter_0005    100    32216   300    340      49.475    22.092            126          64512     10.268  2.6565  2.0482  0.6082
iter_0006    200    33613   314    326      51.645    22.972            132          67584     11.649  2.6399  2.0061  0.6338
iter_0007    400    33454   325    315      51.625    23.150            131          67072     11.519  2.6948  1.9853  0.7095
iter_0008    400    33465   303    337      51.642    23.225            131          67072     11.818  2.6952  1.9392  0.7560
iter_0009    400    33728   313    327      51.875    23.278            132          67584     11.562  2.7173  1.9253  0.7920
```

Saved shard path pattern:

```text
data/neural_self_play_noplayer/iter_*_directional_h64_noplayer_npuct*_640g_b128_root_noise_decay.jsonl
```

Saved replay checkpoints and batch-128 packages:

```text
data/models/directional_cnn_h64_noplayer_iter_0001_replay.pt
data/models/directional_cnn_h64_noplayer_iter_0001_replay_aoti_mps_b128.pt2
data/models/directional_cnn_h64_noplayer_iter_0002_replay.pt
data/models/directional_cnn_h64_noplayer_iter_0002_replay_aoti_mps_b128.pt2
data/models/directional_cnn_h64_noplayer_iter_0003_replay.pt
data/models/directional_cnn_h64_noplayer_iter_0003_replay_aoti_mps_b128.pt2
data/models/directional_cnn_h64_noplayer_iter_0004_replay.pt
data/models/directional_cnn_h64_noplayer_iter_0004_replay_aoti_mps_b128.pt2
data/models/directional_cnn_h64_noplayer_iter_0005_replay.pt
data/models/directional_cnn_h64_noplayer_iter_0005_replay_aoti_mps_b128.pt2
data/models/directional_cnn_h64_noplayer_iter_0006_npuct200_replay.pt
data/models/directional_cnn_h64_noplayer_iter_0006_npuct200_replay_aoti_mps_b128.pt2
data/models/directional_cnn_h64_noplayer_iter_0007_npuct400_replay.pt
data/models/directional_cnn_h64_noplayer_iter_0007_npuct400_replay_aoti_mps_b128.pt2
data/models/directional_cnn_h64_noplayer_iter_0008_npuct400_replay.pt
data/models/directional_cnn_h64_noplayer_iter_0008_npuct400_replay_aoti_mps_b128.pt2
data/models/directional_cnn_h64_noplayer_iter_0009_npuct400_replay.pt
data/models/directional_cnn_h64_noplayer_iter_0009_npuct400_replay_aoti_mps_b128.pt2
```

Deterministic max-visit neural-vs-neural arena matches are the wrong way to
assess model strength; they can collapse into repeated color-dominated lines.
Use stochastic visit-count sampling for neural-vs-neural checks.

Arena checks against heuristic `puct:10000:prior=heuristic:rollout=biased`:

```text
generation  search            seed  games  result  score rate  Blue result  White result
iter_0006   neural-puct:200      1    128   36-92       28.1%      20-44         16-48
iter_0007   neural-puct:200      1    128   41-87       32.0%      21-43         20-44
iter_0007   neural-puct:400      1    128   58-70       45.3%      26-38         32-32
iter_0008   neural-puct:200      1    128   35-93       27.3%      21-43         14-50
iter_0008   neural-puct:400      1    128   47-81       36.7%      20-44         27-37
iter_0009   neural-puct:400      1     40   12-28       30.0%       5-15          7-13
```

These checks suggest the post-`iter_0007` regression is real enough to
investigate before continuing the same loop.

## Experiment 2: 100-Simulation Continuation

100-simulation continuation branch from `iter_0005`:

```text
data directory  data/neural_self_play_noplayer_npuct100_cont/
checkpoint tag  npuct100cont
iter_0006 through iter_0014 all use neural-puct:100 for self-play generation.
```

Self-play and replay training summary:

```text
generation  sims  samples  Blue  White  avg filled  avg turns  train batches  train samples  train sec  loss    policy  value
iter_0006    100    32656   317    323      50.078    22.269            128          65536     10.685  2.6323  2.0087  0.6236
iter_0007    100    32343   335    305      49.598    21.973            127          65024     10.458  2.6001  1.9809  0.6192
iter_0008    100    32794   306    334      50.264    22.397            129          66048     10.206  2.5470  1.9397  0.6074
iter_0009    100    32166   321    319      49.438    22.042            126          64512     11.156  2.5263  1.9319  0.5944
iter_0010    100    32914   294    346      50.542    22.559            129          66048     10.990  2.5225  1.9003  0.6223
iter_0011    100    32625   330    310      49.944    22.372            128          65536     10.444  2.5256  1.8960  0.6296
iter_0012    100    32773   309    331      50.217    22.473            129          66048     11.580  2.4889  1.8719  0.6170
iter_0013    100    32805   330    310      50.147    22.414            129          66048     11.031  2.4901  1.8775  0.6126
iter_0014    100    32688   311    329      50.072    22.381            128          65536     11.179  2.4776  1.8616  0.6161
```

Arena checks against heuristic `puct:10000:prior=heuristic:rollout=biased`:

```text
generation  search            seed  games  result  score rate  Blue result  White result
iter_0006   neural-puct:100      1    128   27-101      21.1%      15-49         12-52
iter_0006   neural-puct:200      1    128   41-87       32.0%      25-39         16-48
iter_0006   neural-puct:400      1    128   54-74       42.2%      28-36         26-38
iter_0006   neural-puct:800      1    128   43-85       33.6%      22-42         21-43
iter_0007   neural-puct:100      1    128   31-97       24.2%      11-53         20-44
iter_0007   neural-puct:200      1    128   35-93       27.3%      15-49         20-44
iter_0007   neural-puct:400      1    128   41-87       32.0%      23-41         18-46
iter_0007   neural-puct:800      1    128   46-82       35.9%      25-39         21-43
iter_0008   neural-puct:100      1    128   39-89       30.5%      22-42         17-47
iter_0008   neural-puct:200      1    128   36-92       28.1%      17-47         19-45
iter_0008   neural-puct:400      1    128   46-82       35.9%      25-39         21-43
iter_0008   neural-puct:800      1    128   53-75       41.4%      29-35         24-40
iter_0009   neural-puct:100      1    128   30-98       23.4%      12-52         18-46
iter_0009   neural-puct:200      1    128   35-93       27.3%      16-48         19-45
iter_0009   neural-puct:400      1    128   46-82       35.9%      23-41         23-41
iter_0009   neural-puct:800      1    128   55-73       43.0%      26-38         29-35
iter_0014   neural-puct:100      1    128   27-101      21.1%      13-51         14-50
```

The 100-simulation branch did not show the same monotonic decline as the
400-simulation branch, and its replay value loss improved instead of worsening.
It also did not clearly beat the best 400-simulation checkpoint in arena.

Direct neural-vs-neural arena checks must use visit-count sampling. Deterministic
max-visit action selection can collapse into repeated lines and is not model
strength evidence for these neural checkpoints. `catchup_arena` now defaults to
visit-count sampling for both agents when both agents are `neural-puct`.

## Experiment 3: AdamW npuct200

Status: incomplete in the source note; only setup and the weight-decay reminder
were recorded.

Setup:

```text
optimizer                       AdamW
weight_decay                    1e-4
self-play search                neural-puct:200
self-play games per generation  640
evaluator backend               MLX
neural batch size               128
```

The AdamW weight decay is optimizer regularization. It is not added to the
reported training loss.
