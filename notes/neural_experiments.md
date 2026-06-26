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
generation  arena search        seed  games  result  score rate  Blue result  White result
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

Increasing the self-play sim count from 100 to higher value since iter_0006 in experiment 1 might lead to self-play data distribution shift, resulting in higher loss.

So in experiment 2 the sim count is kept at 100.

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
generation  arena search       seed  games  result  score rate  Blue result  White result
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

Previous experiments suggest that 100 simulations per move might be too low. Also noticed that there was no L2 regularization earlier.

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

Replay buffer settings:

```text
replay window generations       10
replay gamma                    0.85
target lifetime coverage        2.0
training minibatch size         512
symmetry copies                 3
replay data glob                data/neural_self_play_noplayer_adamw_npuct200/iter_*.jsonl
```

Self-play and replay training summary:

```text
generation  sims  samples  Blue  White  avg filled  avg turns  train batches  train samples  train sec  loss    policy  value
iter_0001    200    34626   333    307      53.133    24.048             14           7168      1.254  3.2978  2.2718  1.0260
iter_0002    200    35363   311    329      53.998    23.917             28          14336      2.403  3.1748  2.2526  0.9221
iter_0003    200    35094   277    363      53.969    24.142             42          21504      3.559  3.1241  2.2410  0.8831
iter_0004    200    35473   332    308      54.362    24.387             56          28672      4.980  3.1078  2.2184  0.8893
iter_0005    200    35261   324    316      54.009    23.859             69          35328      5.752  3.0643  2.1916  0.8727
iter_0006    200    35231   316    324      54.133    23.970             83          42496      7.723  3.0469  2.1744  0.8725
iter_0007    200    34855   325    315      53.545    24.295             96          49152      8.258  3.0463  2.1731  0.8732
iter_0008    200    34971   308    332      53.636    24.178            110          56320      9.498  3.0036  2.1572  0.8464
iter_0009    200    35380   316    324      54.339    24.728            125          64000     11.217  2.9893  2.1484  0.8409
iter_0010    200    35222   309    331      54.159    24.480            138          70656     11.268  2.9599  2.1211  0.8388
iter_0011    200    35257   316    324      54.234    24.181            138          70656     11.345  2.9441  2.1067  0.8373
iter_0012    200    35046   297    343      54.070    24.080            137          70144     11.934  2.9372  2.0950  0.8421
iter_0013    200    34932   325    315      53.506    24.252            137          70144     12.037  2.9080  2.0744  0.8336
iter_0014    200    35222   312    328      54.066    24.481            138          70656     11.479  2.9066  2.0670  0.8396
```

Arena checks against heuristic `puct:10000:prior=heuristic:rollout=biased`:

```text
generation  search            seed  games  result  score rate  Blue result  White result  real
bootstrap   neural-puct:200      1    128   75-53       58.6%      36-28         39-25      142.77s
iter_0006   neural-puct:200      1    128   71-57       55.5%      35-29         36-28      145.93s
iter_0007   neural-puct:200      1    128   64-64       50.0%      28-36         36-28      149.52s
iter_0008   neural-puct:200      1    128   74-54       57.8%      30-34         44-20      150.65s
iter_0009   neural-puct:200      1    128   73-55       57.0%      38-26         35-29      142.05s
iter_0010   neural-puct:200      1    128   65-63       50.8%      31-33         34-30      145.26s
iter_0011   neural-puct:200      1    128   76-52       59.4%      43-21         33-31      146.81s
iter_0012   neural-puct:200      1    128   89-39       69.5%      45-19         44-20      151.96s
iter_0012   neural-puct:400      1    128   95-33       74.2%      50-14         45-19      195.64s
iter_0013   neural-puct:200      1    128   88-40       68.8%      43-21         45-19      152.70s
iter_0014   neural-puct:200      1    128   89-39       69.5%      45-19         44-20      144.40s
iter_0012   neural-puct:200      2    128   83-45       64.8%      39-25         44-20      145.07s
iter_0013   neural-puct:200      2    128   81-47       63.3%      39-25         42-22      148.01s
iter_0014   neural-puct:200      2    128   80-48       62.5%      45-19         35-29      140.82s
```

The seed-2 rerun reduced the score rates for `iter_0012` through `iter_0014`,
but all three remained clearly above 50% against the same `puct:10000` opponent.

Direct neural-vs-neural check:

```text
A model    B model    search           seed  games  A result  A score rate  A Blue  A White  real
bootstrap  iter_0014  neural-puct:200     1    128   31-97          24.2%   17-47   14-50    193.27s
bootstrap  iter_0014  neural-puct:200     2    128   42-86          32.8%   22-42   20-44    191.85s
```

Both agents used visit-count sampling in this neural-vs-neural arena check.
`iter_0014` was clearly stronger than the bootstrap model in this run.

## Experiment 4: AdamW npuct400

This branch starts again from the AdamW bootstrap model, but uses 400 neural
PUCT simulations per self-play move. Other settings match Experiment 3.

Setup:

```text
optimizer                       AdamW
weight_decay                    1e-4
self-play search                neural-puct:400
self-play games per generation  640
evaluator backend               MLX
neural batch size               128
```

Replay buffer settings:

```text
replay window generations       10
replay gamma                    0.85
target lifetime coverage        2.0
training minibatch size         512
symmetry copies                 3
replay data glob                data/neural_self_play_noplayer_adamw_npuct400/iter_*.jsonl
```

Self-play and replay training summary:

```text
generation  sims  samples  Blue  White  avg filled  avg turns  train batches  train samples  train sec  loss    policy  value
iter_0001    400    34954   311    329      53.895    24.358             14           7168      1.414  3.2541  2.2813  0.9728
iter_0002    400    35169   320    320      53.625    24.705             28          14336      2.664  3.0848  2.1678  0.9169
iter_0003    400    35382   328    312      54.536    24.422             42          21504      3.863  3.1139  2.1916  0.9223
iter_0004    400    35701   295    345      55.062    24.372             56          28672      5.444  3.1038  2.1896  0.9141
iter_0005    400    35904   334    306      55.320    25.053             71          36352      6.117  3.0715  2.1725  0.8990
iter_0006    400    35799   324    316      55.167    24.959             84          43008      7.274  3.0475  2.1581  0.8894
iter_0007    400    35898   309    331      55.431    25.127             99          50688      8.372  3.0398  2.1520  0.8878
iter_0008    400    36121   338    302      55.539    25.378            113          57856      9.183  3.0471  2.1455  0.9016
iter_0009    400    35414   311    329      54.742    24.925            125          64000     10.310  3.0194  2.1359  0.8834
iter_0010    400    35701   320    320      55.109    24.977            140          71680     11.071  3.0069  2.1272  0.8797
iter_0011    400    35828   314    326      55.044    25.409            140          71680     11.090  2.9940  2.1032  0.8908
iter_0012    400    35779   339    301      55.223    25.072            140          71680     13.053  2.9746  2.0939  0.8807
```

Saved shard path pattern:

```text
data/neural_self_play_noplayer_adamw_npuct400/iter_*_directional_h64_noplayer_adamw_wd1e4_npuct400_640g_b128_tau005_gamma085.jsonl
```

Saved replay checkpoint pattern:

```text
data/models/directional_cnn_h64_noplayer_adamw_wd1e4_iter_*_npuct400_replay.pt
data/models/directional_cnn_h64_noplayer_adamw_wd1e4_iter_*_npuct400_replay_mlx.safetensors
```

Arena checks against heuristic `puct:10000:prior=heuristic:rollout=biased`:

```text
generation  search            seed  games  result  score rate  Blue result  White result  real
iter_0006   neural-puct:200      1    128   72-56       56.2%      35-29         37-27      152.09s
iter_0006   neural-puct:400      1    128   64-64       50.0%      34-30         30-34      196.33s
iter_0007   neural-puct:200      1    128   67-61       52.3%      31-33         36-28      148.64s
iter_0007   neural-puct:400      1    128   86-42       67.2%      42-22         44-20      208.35s
```

Direct neural-vs-neural check:

```text
A model             B model             search           seed  games  A result  A score rate  A Blue  A White  real
iter_0007 npuct400  iter_0007 npuct200  neural-puct:400     1    128   64-64          50.0%   32-32   32-32    369.88s
iter_0012 npuct400  iter_0012 npuct200  neural-puct:200     1    128   60-68          46.9%   28-36   32-32    173.19s
iter_0012 npuct400  iter_0012 npuct200  neural-puct:400     1    128   54-74          42.2%   29-35   25-39    378.96s
```

Both agents used visit-count sampling in the direct neural-vs-neural check.
