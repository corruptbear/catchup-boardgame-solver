# Neural Experiments

This note is the historical experiment log. Keep raw result tables here, but put stable operating rules in the shorter reference notes.

Unless a table explicitly says otherwise, the recorded `loss`, `policy`, and
`value` numbers are replay-sampled training losses. They are not validation
losses.

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

Self-play and replay training summary. The loss columns are training losses,
not validation losses:

```text
generation  sims  samples  Blue  White  avg filled  avg turns  train batches  train samples  train sec  train total loss  train policy loss  train value loss
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

Self-play and replay training summary. The loss columns are training losses,
not validation losses:

```text
generation  sims  samples  Blue  White  avg filled  avg turns  train batches  train samples  train sec  train total loss  train policy loss  train value loss
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

Self-play and replay training summary. The loss columns are training losses,
not validation losses:

```text
generation  sims  samples  Blue  White  avg filled  avg turns  train batches  train samples  train sec  train total loss  train policy loss  train value loss
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

Self-play and replay training summary. The loss columns are training losses,
not validation losses:

```text
generation  sims  samples  Blue  White  avg filled  avg turns  train batches  train samples  train sec  train total loss  train policy loss  train value loss
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

## Experiment 5: Random Init Tanh-Margin

This branch starts from random weights instead of a bootstrap teacher-trained
checkpoint. It also trains the value head on the terminal tanh-margin target
instead of the win/loss target.

Setup:

```text
base checkpoint                  data/models/directional_cnn_h64_tanh_margin_random_init.pt
architecture                     directional-cnn-tanh-margin
value target                     tanh-margin-scale6
weight initialization            Kaiming normal trunk, zero biases, small output heads
optimizer                        AdamW
weight_decay                     1e-4
self-play search                 neural-puct:200
self-play games per generation   640
evaluator backend                MLX
self-play neural batch size      128 through iter_0040, 640 from iter_0041
early win during self-play        false
```

Boundary note: `iter_0001` through `iter_0010` were generated before the
terminal-leaf value convention was corrected. In those generations,
non-terminal leaves used the model's tanh-margin output, but terminal leaves
backed up exact win/loss `+1/-1`. Starting with `iter_0011`, terminal leaves
also back up `tanh-margin-scale6`. Self-play remains full-board play with
`--early-win false`.

Replay buffer settings:

```text
replay window generations        10
replay gamma                     0.85
target lifetime coverage         2.0
training minibatch size          512
symmetry copies                  3
replay data glob                 data/neural_self_play_random_init_tanh_margin/iter_*.jsonl
```

Self-play and replay training summary. The loss columns are training losses,
not validation losses:

```text
generation  sims  samples  Blue  White  avg filled  avg turns  train batches  train samples  train sec  train total loss  train policy loss  train value loss
iter_0001    200    50808   323    317      61.000    39.584             20          10240      1.954  3.6729  2.7377  0.9351
iter_0002    200    50870   312    328      61.000    37.334             40          20480      3.734  2.9235  2.1783  0.7452
iter_0003    200    45886   416    224      61.000    30.166             54          27648      5.170  2.7376  2.1348  0.6028
iter_0004    200    44221   351    289      61.000    28.688             70          35840      6.594  2.6603  2.1533  0.5070
iter_0005    200    44130   337    303      61.000    29.341             87          44544      7.706  2.6515  2.1677  0.4838
iter_0006    200    43673   318    322      61.000    29.161            103          52736      9.201  2.6514  2.1920  0.4594
iter_0007    200    44649   337    303      61.000    30.753            123          62976     11.242  2.6547  2.2073  0.4474
iter_0008    200    42168   343    297      61.000    28.741            132          67584     10.816  2.6792  2.2502  0.4290
iter_0009    200    40657   324    316      61.000    26.156            143          73216     11.781  2.6939  2.2825  0.4114
iter_0010    200    41153   301    339      61.000    27.075            161          82432     14.417  2.6727  2.2865  0.3861
iter_0011    200    40428   302    338      61.000    26.453            158          80896     14.409  2.6716  2.3163  0.3553
iter_0012    200    40628   327    313      61.000    26.781            159          81408     13.121  2.6553  2.3180  0.3373
iter_0013    200    40783   307    333      61.000    27.528            160          81920     14.284  2.6658  2.3273  0.3385
iter_0014    200    40470   330    310      61.000    26.691            159          81408     13.245  2.6387  2.3125  0.3262
iter_0015    200    40465   338    302      61.000    27.009            159          81408     13.528  2.6083  2.2976  0.3108
iter_0016    200    40293   306    334      61.000    26.622            158          80896     14.189  2.5958  2.2887  0.3071
iter_0017    200    40272   326    314      61.000    26.884            158          80896     14.437  2.5550  2.2743  0.2808
iter_0018    200    40556   324    316      61.000    27.036            159          81408     13.287  2.5264  2.2438  0.2826
iter_0019    200    40075   349    291      61.000    26.897            157          80384     12.999  2.4987  2.2266  0.2721
iter_0020    200    40454   332    308      61.000    27.209            159          81408     12.822  2.4715  2.2003  0.2712
iter_0021    200    40372   317    323      61.000    27.225            158          80896     12.863  2.4424  2.1719  0.2704
iter_0022    200    40528   339    301      61.000    27.414            159          81408     13.565  2.4017  2.1388  0.2629
iter_0023    200    40448   331    309      61.000    27.712            158          80896     12.367  2.3932  2.1289  0.2643
iter_0024    200    40478   323    317      61.000    27.397            159          81408     13.110  2.3729  2.1156  0.2573
iter_0025    200    40492   315    325      61.000    27.466            159          81408     13.830  2.3545  2.0965  0.2581
iter_0026    200    40271   334    306      61.000    27.236            158          80896     12.368  2.3506  2.0973  0.2533
iter_0027    200    40525   323    317      61.000    27.986            159          81408     13.375  2.3337  2.0766  0.2570
iter_0028    200    40318   312    328      61.000    27.663            158          80896     14.206  2.3156  2.0663  0.2493
iter_0029    200    40372   310    330      61.000    27.753            158          80896     13.769  2.2952  2.0515  0.2437
iter_0030    200    39991   327    313      61.000    27.212            157          80384     13.966  2.2874  2.0479  0.2395
iter_0031    200    40219   309    331      61.000    27.505            158          80896     13.635  2.2903  2.0514  0.2389
iter_0032    200    40282   305    335      61.000    27.802            158          80896     12.428  2.2892  2.0553  0.2339
iter_0033    200    40139   326    314      61.000    27.436            157          80384     12.961  2.2785  2.0490  0.2296
iter_0034    200    40319   324    316      61.000    27.892            158          80896     12.367  2.2525  2.0271  0.2254
iter_0035    200    40264   315    325      61.000    27.927            158          80896     12.249  2.2456  2.0214  0.2242
iter_0036    200    40315   320    320      61.000    27.913            158          80896     12.248  2.2375  2.0138  0.2237
iter_0037    200    40242   324    316      61.000    27.778            158          80896     12.239  2.2307  2.0151  0.2156
iter_0038    200    40427   331    309      61.000    28.150            158          80896     12.265  2.2138  1.9993  0.2145
iter_0039    200    40085   307    333      61.000    27.863            157          80384     13.737  2.2149  1.9991  0.2159
iter_0040    200    40273   311    329      61.000    27.973            158          80896     13.217  2.2110  1.9995  0.2115
iter_0041    200    40356   326    314      61.000    27.961            158          80896     14.196  2.1996  1.9916  0.2080
iter_0042    200    40455   317    323      61.000    28.391            159          81408     14.456  2.1817  1.9676  0.2141
iter_0043    200    40512   300    340      61.000    28.270            159          81408     12.959  2.1735  1.9595  0.2140
iter_0044    200    40231   346    294      61.000    28.078            158          80896     13.933  2.1626  1.9538  0.2088
iter_0045    200    40353   327    313      61.000    28.102            158          80896     12.792  2.1382  1.9345  0.2037
iter_0046    200    40546   316    324      61.000    28.636            159          81408     14.398  2.1481  1.9360  0.2121
iter_0047    200    40313   329    311      61.000    28.089            158          80896     12.509  2.1312  1.9174  0.2137
iter_0048    200    40300   342    298      61.000    28.086            158          80896     14.300  2.1188  1.9174  0.2013
iter_0049    200    40352   314    326      61.000    28.170            158          80896     12.797  2.1255  1.9220  0.2035
iter_0050    200    40296   319    321      61.000    28.053            158          80896     12.904  2.1103  1.9068  0.2035
iter_0051    200    40035   333    307      61.000    28.012            157          80384     14.185  2.0959  1.8971  0.1989
iter_0052    200    40235   356    284      61.000    27.947            158          80896     14.032  2.0922  1.8996  0.1926
iter_0053    200    40188   316    324      61.000    28.038            157          80384     13.566  2.0856  1.8983  0.1872
iter_0054    200    40204   313    327      61.000    28.025            158          80896     13.816  2.0718  1.8843  0.1876
iter_0055    200    40089   319    321      61.000    27.831            157          80384     13.693  2.0732  1.8890  0.1843
iter_0056    200    39979   310    330      61.000    27.788            157          80384     13.690  2.0741  1.8891  0.1850
iter_0057    200    40065   319    321      61.000    28.020            157          80384     13.776  2.0719  1.8858  0.1862
iter_0058    200    40236   303    337      61.000    28.255            158          80896     13.862  2.0739  1.8827  0.1912
iter_0059    200    40089   334    306      61.000    28.038            157          80384     13.628  2.0684  1.8822  0.1861
iter_0060    200    40368   344    296      61.000    28.361            158          80896     13.743  2.0741  1.8864  0.1877
iter_0061    200    40137   332    308      61.000    28.108            157          80384     13.637  2.0666  1.8773  0.1893
iter_0062    200    40020   323    317      61.000    27.747            157          80384     13.627  2.0851  1.8946  0.1906
iter_0063    200    39978   316    324      61.000    27.756            157          80384     13.621  2.0806  1.8926  0.1880
iter_0064    200    40165   324    316      61.000    27.997            157          80384     13.727  2.0697  1.8853  0.1843
iter_0065    200    40210   295    345      61.000    28.061            158          80896     13.732  2.0600  1.8774  0.1826
iter_0066    200    40149   326    314      61.000    28.147            157          80384     13.752  2.0632  1.8801  0.1831
iter_0067    200    40130   329    311      61.000    28.038            157          80384     13.686  2.0561  1.8759  0.1801
iter_0068    200    40305   316    324      61.000    28.581            158          80896     13.851  2.0557  1.8727  0.1830
iter_0069    200    40015   315    325      61.000    27.934            157          80384     13.744  2.0514  1.8730  0.1784
iter_0070    200    40265   338    302      61.000    28.144            158          80896     13.620  2.0394  1.8650  0.1744
iter_0071    200    39919   305    335      61.000    27.669            156          79872     13.606  2.0549  1.8781  0.1768
iter_0072    200    40097   311    329      61.000    27.959            157          80384     13.612  2.0570  1.8800  0.1770
iter_0073    200    39969   323    317      61.000    27.897            157          80384     13.705  2.0578  1.8822  0.1755
iter_0074    200    40007   312    328      61.000    28.084            157          80384     13.630  2.0621  1.8881  0.1740
iter_0075    200    40045   315    325      61.000    27.969            157          80384     13.651  2.0441  1.8720  0.1720
iter_0076    200    39975   321    319      61.000    27.781            157          80384     13.567  2.0408  1.8712  0.1696
iter_0077    200    40096   347    293      61.000    28.017            157          80384     13.784  2.0393  1.8684  0.1709
iter_0078    200    40262   323    317      61.000    28.255            158          80896     13.595  2.0412  1.8708  0.1704
iter_0079    200    40218   321    319      61.000    28.248            158          80896     13.780  2.0279  1.8579  0.1700
iter_0080    200    39963   332    308      61.000    27.939            157          80384     13.755  2.0260  1.8598  0.1661
```


Saved shard path pattern:

```text
data/neural_self_play_random_init_tanh_margin/iter_*_directional_h64_tanh_margin_random_init_npuct200_640g_b*.jsonl
```

Saved replay checkpoint pattern:

```text
data/models/directional_cnn_h64_tanh_margin_random_init_adamw_wd1e4_iter_*_npuct200_replay.pt
data/models/directional_cnn_h64_tanh_margin_random_init_adamw_wd1e4_iter_*_npuct200_replay_mlx.safetensors
```

Arena checks:

```text
model matchup                     search           seed  games  action selection  result  score rate  Blue result  White result  real
iter_0010 vs random               neural-puct:200      1     40  max/max           40-0       100.0%     20-0         20-0       72.46s
iter_0009 vs heuristic puct:1000  neural-puct:200      1     80  max/max            0-80        0.0%      0-40         0-40      60.50s
iter_0010 vs iter_0005            neural-puct:200      1     80  sample/sample     73-7        91.2%     38-2         35-5      227.51s
iter_0020 vs heuristic puct:1000  neural-puct:200      1     80  max/max           13-67       16.2%      8-32         5-35      80.80s
iter_0020 vs iter_0010            neural-puct:200      1     80  sample/sample     74-6        92.5%     36-4         38-2      221.87s
iter_0030 vs heuristic puct:1000  neural-puct:200      1     80  max/max           43-37       53.8%     23-17        20-20      76.74s
iter_0030 vs heuristic puct:10000 neural-puct:200      1     80  max/max           16-64       20.0%      8-32         8-32     113.28s
iter_0040 vs heuristic puct:1000  neural-puct:200      1     80  max/max           55-25       68.8%     27-13        28-12      79.07s
iter_0040 vs heuristic puct:10000 neural-puct:200      1     80  max/max           26-54       32.5%     11-29        15-25     113.48s
iter_0040 vs iter_0030            neural-puct:200      1    128  sample/sample     95-33       74.2%     53-11        42-22     153.75s
iter_0050 vs heuristic puct:1000  neural-puct:200      1    128  max/max          107-21       83.6%     54-10        53-11      70.59s
iter_0050 vs heuristic puct:10000 neural-puct:200      1    128  max/max           67-61       52.3%     37-27        30-34     140.33s
iter_0050 vs exp3 bootstrap       neural-puct:200      1    128  sample/sample     74-54       57.8%     41-23        33-31     165.73s
iter_0060 vs heuristic puct:1000  neural-puct:200      1    128  max/max          113-15       88.3%     57-7         56-8       77.19s
iter_0060 vs heuristic puct:10000 neural-puct:200      1    128  max/max           78-50       60.9%     41-23        37-27     147.02s
iter_0070 vs heuristic puct:1000  neural-puct:200      1    128  max/max          114-14       89.1%     55-9         59-5       62.90s
iter_0070 vs heuristic puct:10000 neural-puct:200      1    128  max/max           72-56       56.2%     31-33        41-23     149.29s
iter_0080 vs heuristic puct:1000  neural-puct:200      1    128  max/max          119-9        93.0%     58-6         61-3       65.69s
iter_0080 vs heuristic puct:10000 neural-puct:200      1    128  max/max           84-44       65.6%     37-27        47-17     137.67s
iter_0080 vs heuristic puct:10000 neural-puct:200 100001    128  max/max           80-48       62.5%     42-22        38-26     146.54s
iter_0080 vs heuristic puct:10000 neural-puct:200 200001    128  max/max           83-45       64.8%     45-19        38-26     139.80s
```

The direct neural-vs-neural checks above used full-board arena play with
`--early-win false`; average filled cells were `61.0`.

## Experiment 6: H128 Random Init Tanh-Margin

This branch uses the same random-init tanh-margin setup as Experiment 5, but
changes the directional-CNN hidden size from 64 to 128.

Setup:

```text
architecture                     directional-cnn-tanh-margin
hidden size                      128
parameters                       528131
value target                     tanh-margin-scale6
optimizer                        AdamW
weight_decay                     1e-4
self-play search                 neural-puct:200
self-play games per generation   640
evaluator backend                MLX
self-play neural batch size      640
early win during self-play        false
seed replay window               Experiment 5 iter_0041 through iter_0050
```

Replay buffer settings:

```text
replay window generations        10
replay gamma                     0.85
target lifetime coverage         2.0
training minibatch size          512
symmetry copies                  3
replay data glob                 data/neural_self_play_h128_tanh_margin_random_init/iter_*.jsonl
```

`iter_0050` is the seed checkpoint trained from random h128 weights on the
Experiment 5 `iter_0041` through `iter_0050` replay window. `iter_0060` is the
first checkpoint whose replay window is entirely h128-generated data.

Self-play and replay training summary. The loss columns are training losses,
not validation losses:

```text
generation  sims  samples  Blue  White  avg filled  avg turns  train batches  train samples  train sec  train total loss  train policy loss  train value loss
iter_0050   seed        -     -      -           -         -            158          80896     29.587  3.6214  2.3175  1.3039
iter_0051    200    50751   338    302      61.000    36.144            199         101888     38.261  2.6169  2.1661  0.4508
iter_0052    200    40457   318    322      61.000    26.939            159          81408     29.621  2.3783  2.0693  0.3090
iter_0053    200    40597   318    322      61.000    27.925            159          81408     29.627  2.3394  2.0457  0.2936
iter_0054    200    40054   310    330      61.000    26.733            157          80384     29.963  2.3149  2.0353  0.2795
iter_0055    200    40296   310    330      61.000    27.331            158          80896     30.548  2.2912  2.0262  0.2650
iter_0056    200    40672   332    308      61.000    28.230            159          81408     31.890  2.2911  2.0173  0.2738
iter_0057    200    40154   328    312      61.000    27.414            157          80384     29.594  2.2605  2.0063  0.2542
iter_0058    200    40640   321    319      61.000    28.175            159          81408     30.266  2.2656  2.0137  0.2519
iter_0059    200    40565   339    301      61.000    27.827            159          81408     28.266  2.2541  2.0091  0.2450
iter_0060    200    40482   313    327      61.000    27.859            159          81408     30.431  2.2435  2.0027  0.2408
iter_0061    200    40545   319    321      61.000    27.845            159          81408     30.776  2.2335  2.0065  0.2271
iter_0062    200    40582   310    330      61.000    28.045            159          81408     28.739  2.2004  1.9805  0.2200
iter_0063    200    40634   334    306      61.000    28.127            159          81408     32.642  2.1765  1.9522  0.2242
iter_0064    200    40401   320    320      61.000    27.639            158          80896     29.534  2.1535  1.9433  0.2102
iter_0065    200    40350   310    330      61.000    27.967            158          80896     28.780  2.1371  1.9297  0.2075
iter_0066    200    40821   334    306      61.000    28.627            160          81920     32.213  2.1194  1.9137  0.2057
iter_0067    200    40464   353    287      61.000    28.042            159          81408     30.202  2.1257  1.9241  0.2016
iter_0068    200    40197   334    306      61.000    27.709            158          80896     29.581  2.1163  1.9244  0.1919
iter_0069    200    40420   301    339      61.000    28.195            158          80896     28.892  2.1036  1.9150  0.1886
iter_0070    200    40582   315    325      61.000    28.298            159          81408     28.583  2.0838  1.8965  0.1873
iter_0071    200    40357   336    304      61.000    28.047            158          80896     28.192  2.0698  1.8853  0.1845
iter_0072    200    40429   310    330      61.000    28.261            158          80896     28.575  2.0579  1.8717  0.1862
iter_0073    200    40659   308    332      61.000    28.648            159          81408     28.517  2.0438  1.8565  0.1873
iter_0074    200    40130   337    303      61.000    27.770            157          80384     28.123  2.0500  1.8677  0.1823
iter_0075    200    40318   317    323      61.000    28.181            158          80896     29.667  2.0309  1.8560  0.1749
iter_0076    200    40233   314    326      61.000    28.269            158          80896     28.152  2.0522  1.8806  0.1716
iter_0077    200    40367   321    319      61.000    28.386            158          80896     28.742  2.0447  1.8729  0.1718
iter_0078    200    40463   317    323      61.000    28.634            159          81408     28.729  2.0405  1.8657  0.1747
iter_0079    200    40360   322    318      61.000    28.538            158          80896     28.371  2.0310  1.8549  0.1761
iter_0080    200    40210   320    320      61.000    28.623            158          80896     31.511  2.0300  1.8577  0.1724
```

Saved shard path pattern:

```text
data/neural_self_play_h128_tanh_margin_random_init/iter_*_directional_h128_tanh_margin_random_init_npuct200_640g_b640.jsonl
```

Saved replay checkpoint pattern:

```text
data/models/directional_cnn_h128_tanh_margin_random_init_adamw_wd1e4_iter_*_npuct200_replay.pt
data/models/directional_cnn_h128_tanh_margin_random_init_adamw_wd1e4_iter_*_npuct200_replay_mlx.safetensors
```

Arena checks:

```text
model matchup                     search           seed  games  action selection  result  score rate  Blue result  White result  real
iter_0060 vs heuristic puct:1000  neural-puct:200      1    128  max/max           96-32       75.0%     49-15        47-17      79.78s
iter_0060 vs heuristic puct:10000 neural-puct:200      1    128  max/max           37-91       28.9%     21-43        16-48     163.08s
iter_0070 vs heuristic puct:1000  neural-puct:200      1    128  max/max          114-14       89.1%     57-7         57-7       79.22s
iter_0070 vs heuristic puct:10000 neural-puct:200      1    128  max/max           74-54       57.8%     41-23        33-31     164.67s
iter_0075 vs heuristic puct:1000  neural-puct:200      1    128  max/max          115-13       89.8%     58-6         57-7       86.40s
iter_0075 vs heuristic puct:10000 neural-puct:200      1    128  max/max           96-32       75.0%     47-17        49-15     157.72s
iter_0075 vs heuristic puct:10000 neural-puct:200 100001    128  max/max           82-46       64.1%     44-20        38-26     159.17s
iter_0075 vs heuristic puct:10000 neural-puct:200 200001    128  max/max           88-40       68.8%     46-18        42-22     154.63s
iter_0080 vs heuristic puct:10000 neural-puct:200      1    128  max/max           73-55       57.0%     35-29        38-26     154.74s
iter_0080 vs heuristic puct:10000 neural-puct:200 100001    128  max/max           89-39       69.5%     45-19        44-20     154.61s
iter_0080 vs heuristic puct:10000 neural-puct:200 200001    128  max/max           91-37       71.1%     51-13        40-24     154.20s
```

Direct neural-vs-neural check after arena seed hashing fix:

```text
model matchup                           search           seed  games  action selection  result   score rate  A Blue  A White  real
Exp6 h128 iter_0080 vs Exp5 h64 iter_0080 neural-puct:200      1    256  sample/sample    123-133      48.0%   59-69   64-64   158.62s
Exp6 h128 iter_0080 vs Exp3 h64 iter_0012 neural-puct:200      1    256  sample/sample    142-114      55.5%   67-61   75-53   169.81s
```

## Experiment 7: H128 Dual-Value Redo

This branch redoes the Experiment 6 h128 run with two value heads. The model
predicts both win/loss and tanh-margin. Neural PUCT backs up both values and
uses `Q = win_q + 0.1 * margin_q` during selection.

Setup:

```text
architecture                     directional-cnn-dual-value
hidden size                      128
parameters                       561156
value target                     win-loss + tanh-margin-scale6
optimizer                        AdamW
weight_decay                     1e-4
self-play search                 neural-puct:200
self-play games per generation   640
evaluator backend                MLX
self-play neural batch size      640
early win during self-play        false
seed replay window               Experiment 5 iter_0041 through iter_0050
```

Replay buffer settings:

```text
replay window generations        10
replay gamma                     0.85
target lifetime coverage         2.0
training minibatch size          512
symmetry copies                  3
replay data glob                 data/neural_self_play_h128_dual_value/iter_*.jsonl
```

`iter_0050` is the seed checkpoint trained from random h128 dual-value weights
on the Experiment 5 `iter_0041` through `iter_0050` replay window. `iter_0060`
is the first checkpoint whose replay window is entirely Exp7-generated data.

Self-play and replay training summary. The loss columns are training losses,
not validation losses:

```text
generation  sims  samples  Blue  White  avg filled  avg turns  train batches  train samples  train sec  train total loss  train policy loss  train value loss  train win value loss  train margin value loss
iter_0050   seed        -     -      -           -         -            158          80896     30.769  3.5318  2.3297  1.2020  0.9368  0.2652
iter_0051    200    40215   329    311      61.000    26.719            158          80896     29.427  3.3133  2.1428  1.1705  0.9064  0.2641
iter_0052    200    41102   311    329      61.000    27.197            161          82432     30.677  3.2380  2.0976  1.1404  0.8807  0.2598
iter_0053    200    41190   330    310      61.000    28.080            161          82432     31.125  3.1753  2.0677  1.1076  0.8470  0.2606
iter_0054    200    40981   303    337      61.000    28.514            161          82432     30.052  3.1716  2.0763  1.0953  0.8323  0.2630
iter_0055    200    41363   318    322      61.000    28.967            162          82944     30.162  3.1626  2.0651  1.0975  0.8208  0.2767
iter_0056    200    40796   314    326      61.000    28.233            160          81920     29.619  3.1589  2.0785  1.0804  0.8132  0.2672
iter_0057    200    40875   316    324      61.000    27.619            160          81920     30.013  3.0856  2.0482  1.0373  0.7813  0.2561
iter_0058    200    41573   319    321      61.000    29.334            163          83456     31.605  3.0838  2.0509  1.0329  0.7731  0.2599
iter_0059    200    41063   323    317      61.000    28.000            161          82432     33.439  3.0521  2.0547  0.9974  0.7491  0.2483
iter_0060    200    41003   345    295      61.000    28.530            161          82432     29.653  3.0575  2.0566  1.0009  0.7529  0.2480
iter_0061    200    41353   320    320      61.000    28.934            162          82944     30.336  3.0473  2.0475  0.9999  0.7545  0.2454
iter_0062    200    41289   308    332      61.000    28.892            162          82944     30.207  3.0395  2.0424  0.9971  0.7512  0.2459
iter_0063    200    40862   315    325      61.000    28.934            160          81920     30.295  3.0295  2.0402  0.9893  0.7470  0.2423
iter_0064    200    40812   298    342      61.000    28.752            160          81920     29.155  3.0179  2.0324  0.9855  0.7499  0.2356
iter_0065    200    40814   324    316      61.000    27.783            160          81920     29.460  3.0107  2.0208  0.9899  0.7621  0.2278
iter_0066    200    40979   331    309      61.000    28.831            161          82432     31.189  2.9846  2.0099  0.9747  0.7505  0.2242
iter_0067    200    40861   317    323      61.000    28.442            160          81920     29.061  2.9601  2.0059  0.9542  0.7389  0.2154
iter_0068    200    40540   321    319      61.000    28.244            159          81408     29.617  2.9688  2.0122  0.9566  0.7437  0.2130
iter_0069    200    40532   316    324      61.000    28.370            159          81408     30.299  2.9687  2.0014  0.9673  0.7538  0.2134
iter_0070    200    40893   318    322      61.000    28.803            160          81920     32.141  2.9471  1.9847  0.9624  0.7503  0.2121
```

Saved shard path pattern:

```text
data/neural_self_play_h128_dual_value/iter_*_directional_h128_dual_value_npuct200_640g_b640_beta0p1.jsonl
```

Saved replay checkpoint pattern:

```text
data/models/directional_cnn_h128_dual_value_adamw_wd1e4_iter_*_npuct200_replay.pt
data/models/directional_cnn_h128_dual_value_adamw_wd1e4_iter_*_npuct200_replay_mlx.safetensors
```

Arena checks:

```text
model matchup                     search           seed  games  action selection  result  score rate  Blue result  White result  real
iter_0060 vs heuristic puct:1000  neural-puct:200      1    128  max/max           90-38       70.3%     48-16        42-22      74.91s
iter_0060 vs heuristic puct:10000 neural-puct:200      1    128  max/max           41-87       32.0%     15-49        26-38     164.45s
iter_0070 vs heuristic puct:1000  neural-puct:200      1    128  max/max          108-20       84.4%     52-12        56-8       76.95s
iter_0070 vs heuristic puct:10000 neural-puct:200      1    128  max/max           73-55       57.0%     32-32        41-23     166.63s
iter_0070 vs heuristic puct:10000 neural-puct:200 100001    128  max/max           72-56       56.2%     36-28        36-28     157.76s
```
