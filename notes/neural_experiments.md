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
iter_0071    200    40789   322    318      61.000    28.552            160          81920     29.773  2.9379  1.9747  0.9632  0.7538  0.2095
iter_0072    200    40658   324    316      61.000    28.286            159          81408     29.811  2.9096  1.9555  0.9541  0.7507  0.2033
iter_0073    200    40462   338    302      61.000    28.456            159          81408     29.755  2.8868  1.9390  0.9478  0.7499  0.1979
iter_0074    200    40472   309    331      61.000    27.906            159          81408     29.610  2.8556  1.9238  0.9319  0.7416  0.1903
iter_0075    200    40384   329    311      61.000    28.464            158          80896     29.581  2.8870  1.9314  0.9556  0.7606  0.1950
iter_0076    200    40637   315    325      61.000    28.344            159          81408     30.447  2.8759  1.9282  0.9477  0.7590  0.1887
iter_0077    200    40568   317    323      61.000    28.856            159          81408     29.801  2.8711  1.9238  0.9473  0.7588  0.1884
iter_0078    200    40479   326    314      61.000    28.595            159          81408     29.861  2.8724  1.9216  0.9508  0.7626  0.1882
iter_0079    200    40601   330    310      61.000    28.863            159          81408     29.992  2.8305  1.9026  0.9279  0.7420  0.1859
iter_0080    200    40641   334    306      61.000    28.983            159          81408     29.661  2.8390  1.8948  0.9442  0.7554  0.1888
iter_0081    200    40530   302    338      61.000    28.673            159          81408     29.667  2.8053  1.8766  0.9287  0.7428  0.1860
iter_0082    200    40466   326    314      61.000    29.081            159          81408     29.734  2.7910  1.8679  0.9231  0.7352  0.1879
iter_0083    200    40501   334    306      61.000    28.825            159          81408     30.456  2.7876  1.8654  0.9222  0.7346  0.1876
iter_0084    200    40380   327    313      61.000    28.747            158          80896     29.522  2.7767  1.8607  0.9160  0.7308  0.1853
iter_0085    200    40450   313    327      61.000    28.939            159          81408     29.583  2.7658  1.8579  0.9079  0.7248  0.1831
iter_0086    200    40352   323    317      61.000    28.414            158          80896     31.668  2.7506  1.8463  0.9044  0.7247  0.1796
iter_0087    200    40439   319    321      61.000    28.784            158          80896     29.504  2.7477  1.8351  0.9126  0.7312  0.1814
iter_0088    200    40369   334    306      61.000    28.483            158          80896     31.348  2.7232  1.8176  0.9056  0.7279  0.1777
iter_0089    200    40545   310    330      61.000    29.127            159          81408     29.649  2.7074  1.7989  0.9085  0.7305  0.1780
iter_0090    200    40368   325    315      61.000    28.636            158          80896     30.657  2.7093  1.8007  0.9086  0.7322  0.1764
iter_0091    200    40563   323    317      61.000    28.952            159          81408     30.949  2.7113  1.7893  0.9220  0.7441  0.1779
iter_0092    200    40576   319    321      61.000    29.109            159          81408     31.027  2.6919  1.7818  0.9101  0.7360  0.1740
iter_0093    200    40424   328    312      61.000    28.695            158          80896     30.087  2.6581  1.7662  0.8919  0.7256  0.1663
iter_0094    200    40468   344    296      61.000    28.673            159          81408     32.317  2.6351  1.7518  0.8832  0.7197  0.1635
iter_0095    200    40445   318    322      61.000    28.828            158          80896     29.404  2.6119  1.7337  0.8782  0.7181  0.1601
iter_0096    200    40570   293    347      61.000    29.166            159          81408     29.249  2.5972  1.7121  0.8851  0.7209  0.1643
iter_0097    200    40474   318    322      61.000    29.125            159          81408     30.329  2.5618  1.6933  0.8686  0.7070  0.1616
iter_0098    200    40323   310    330      61.000    28.839            158          80896     30.293  2.5546  1.6941  0.8605  0.7020  0.1585
iter_0099    200    40498   316    324      61.000    29.169            159          81408     30.368  2.5294  1.6597  0.8698  0.7061  0.1637
iter_0100    200    40436   329    311      61.000    29.002            158          80896     29.267  2.5188  1.6545  0.8642  0.7017  0.1625
iter_0101    200    40406   331    309      61.000    28.916            158          80896     29.906  2.5208  1.6470  0.8738  0.7110  0.1628
iter_0102    200    40309   298    342      61.000    28.877            158          80896     29.354  2.4867  1.6327  0.8540  0.6951  0.1589
iter_0103    200    40500   327    313      61.000    28.952            159          81408     32.535  2.4583  1.6119  0.8464  0.6895  0.1569
iter_0104    200    40361   315    325      61.000    28.709            158          80896     29.795  2.4650  1.6110  0.8540  0.6937  0.1603
iter_0105    200    40487   303    337      61.000    29.041            159          81408     29.652  2.4426  1.5995  0.8431  0.6852  0.1579
iter_0106    200    40297   318    322      61.000    28.692            158          80896     30.742  2.4387  1.5889  0.8498  0.6909  0.1589
iter_0107    200    40256   329    311      61.000    28.777            158          80896     29.101  2.4138  1.5753  0.8385  0.6846  0.1539
iter_0108    200    40264   344    296      61.000    28.712            158          80896     30.102  2.4027  1.5640  0.8387  0.6862  0.1525
iter_0109    200    40407   320    320      61.000    28.886            158          80896     30.182  2.4090  1.5649  0.8441  0.6890  0.1552
iter_0110    200    40315   314    326      61.000    28.953            158          80896     30.103  2.3881  1.5582  0.8299  0.6796  0.1503
iter_0111    200    40381   321    319      61.000    29.145            158          80896     31.674  2.3899  1.5531  0.8368  0.6842  0.1526
iter_0112    200    40372   329    311      61.000    28.741            158          80896     28.820  2.3820  1.5432  0.8388  0.6836  0.1552
iter_0113    200    40509   313    327      61.000    28.944            159          81408     29.840  2.3672  1.5403  0.8269  0.6746  0.1523
iter_0114    200    40400   320    320      61.000    29.000            158          80896     28.355  2.3410  1.5321  0.8090  0.6604  0.1486
iter_0115    200    40272   297    343      61.000    28.766            158          80896     31.570  2.3265  1.5156  0.8110  0.6618  0.1491
iter_0116    200    40277   302    338      61.000    28.773            158          80896     29.268  2.3301  1.5221  0.8080  0.6585  0.1495
iter_0117    200    40277   285    355      61.000    28.855            158          80896     28.990  2.3117  1.5106  0.8010  0.6511  0.1499
iter_0118    200    40315   286    354      61.000    28.847            158          80896     29.214  2.3127  1.5085  0.8043  0.6529  0.1514
iter_0119    200    40292   290    350      61.000    28.777            158          80896     28.604  2.3089  1.5108  0.7981  0.6486  0.1495
iter_0120    200    40300   268    372      61.000    28.817            158          80896     28.965  2.2804  1.4893  0.7911  0.6411  0.1500
```

The faster policy-loss drop near `iter_0100` is mostly explained by sharper
teacher visit distributions, not by a replay-window bug. Manifest checks for
`iter_0081` through `iter_0100` had 10 unique existing shards per window. The
replay-weighted policy-target entropy fell from `1.5917` at `iter_0080` to
`1.3617` at `iter_0100`, while the KL distance stayed roughly flat:

```text
iter_0080  target entropy 1.5917  policy loss 1.8948  KL distance 0.3032
iter_0090  target entropy 1.4981  policy loss 1.8007  KL distance 0.3026
iter_0100  target entropy 1.3617  policy loss 1.6545  KL distance 0.2928
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
iter_0079 vs heuristic puct:10000 neural-puct:200      1    128  max/max           92-36       71.9%     46-18        46-18          -
iter_0079 vs heuristic puct:10000 neural-puct:200 100001    128  max/max           85-43       66.4%     43-21        42-22          -
iter_0079 vs heuristic puct:10000 neural-puct:200 200001    128  max/max           80-48       62.5%     38-26        42-22          -
iter_0090 vs heuristic puct:1000  neural-puct:200      1    128  max/max          120-8        93.8%     59-5         61-3       79.20s
iter_0090 vs heuristic puct:10000 neural-puct:200      1    128  max/max          100-28       78.1%     49-15        51-13     172.08s
iter_0090 vs heuristic puct:10000 neural-puct:200 100001    128  max/max           90-38       70.3%     46-18        44-20     182.56s
iter_0090 vs heuristic puct:10000 neural-puct:200 200001    128  max/max           96-32       75.0%     46-18        50-14     184.16s
iter_0100 vs heuristic puct:1000  neural-puct:200      1    128  max/max          122-6        95.3%     62-2         60-4       78.00s
iter_0100 vs heuristic puct:10000 neural-puct:200      1    128  max/max           96-32       75.0%     44-20        52-12     159.77s
iter_0100 vs heuristic puct:10000 neural-puct:200 100001    128  max/max          109-19       85.2%     55-9         54-10     160.74s
iter_0100 vs heuristic puct:10000 neural-puct:200 200001    128  max/max          100-28       78.1%     50-14        50-14     168.24s
iter_0110 vs heuristic puct:1000  neural-puct:200      1    128  max/max          121-7        94.5%     59-5         62-2       77.52s
iter_0110 vs heuristic puct:10000 neural-puct:200      1    128  max/max          112-16       87.5%     56-8         56-8      154.56s
iter_0110 vs heuristic puct:10000 neural-puct:200 100001    128  max/max          121-7        94.5%     60-4         61-3      157.99s
iter_0110 vs heuristic puct:10000 neural-puct:200 200001    128  max/max          113-15       88.3%     54-10        59-5      153.52s
iter_0120 vs heuristic puct:10000 neural-puct:200      1    128  max/max          111-17       86.7%     55-9         56-8      150.09s
iter_0120 vs heuristic puct:10000 neural-puct:200 100001    128  max/max          112-16       87.5%     57-7         55-9      145.61s
iter_0120 vs heuristic puct:10000 neural-puct:200 200001    128  max/max          118-10       92.2%     60-4         58-6      149.27s
```

## Experiment 8: H128 Dual-Value Restart With Root Policy Temperature

Motivation: Exp7 gained arena strength, but its policy targets also sharpened
rapidly and the opening policy showed self-reinforcing feedback. In particular,
some opening moves could be selected often because of prior/visit feedback even
when their value was questionable. Exp7 also developed a suspicious Blue/White
balance drift in later self-play generations. Exp8 tests whether root policy
temperature can reduce that feedback while preserving the useful value-learning
trend.

Exp8 restarts from the Exp7 `iter_0101` checkpoint. The only search-policy
change is root policy temperature before root Dirichlet noise.

```text
self-play search                  neural-puct:200
self-play games per generation    640
evaluator backend                 MLX
self-play neural batch size       640
early win during self-play         false
neural margin beta                 0.1
optimizer                          AdamW
weight_decay                       1e-4
replay window generations          10
replay gamma                       0.85
target lifetime coverage           2.0
training minibatch size            512
symmetry copies                    3
```

Root policy temperature settings:

```text
start                              1.25
end                                1.10
halflife in completed turns        9.0
schedule                           tau(t) = 1.10 + 0.15 * 2^(-t / 9)
visit temperature                  max(0.05, empty_count / 61)
root noise                         existing decayed schedule
```

`halflife = 9` came from the board-diameter heuristic. On this board, full-board
self-play averages about 29 completed turns, so the schedule still applies
noticeable flattening late in the game. The permanent floor `end = 1.10` may be
more important than the halflife.

Self-play and replay training summary. The loss columns are training losses,
not validation losses:

```text
generation  sims  samples  Blue  White  avg filled  avg turns  train batches  train samples  train sec  train total loss  train policy loss  train value loss  train win value loss  train margin value loss
iter_0102    200    40272   298    342      61.000    28.803            158          80896     29.494  2.5049  1.6519  0.8530  0.6932  0.1598
iter_0103    200    40513   307    333      61.000    28.694            159          81408     33.287  2.5274  1.6618  0.8656  0.7020  0.1636
iter_0104    200    40338   328    312      61.000    28.808            158          80896     29.595  2.5069  1.6637  0.8432  0.6819  0.1613
iter_0105    200    40269   312    328      61.000    28.870            158          80896     31.329  2.5014  1.6629  0.8385  0.6783  0.1602
iter_0106    200    40433   349    291      61.000    28.805            158          80896     29.463  2.5097  1.6683  0.8414  0.6805  0.1609
iter_0107    200    40411   360    280      61.000    28.939            158          80896     29.235  2.4985  1.6621  0.8365  0.6793  0.1572
iter_0108    200    40346   332    308      61.000    28.736            158          80896     29.276  2.4872  1.6585  0.8286  0.6746  0.1541
iter_0109    200    40379   312    328      61.000    29.009            158          80896     29.503  2.4724  1.6527  0.8196  0.6663  0.1533
iter_0110    200    40360   298    342      61.000    28.791            158          80896     29.453  2.4646  1.6542  0.8103  0.6582  0.1521
iter_0111    200    40458   358    282      61.000    28.770            159          81408     32.374  2.4386  1.6503  0.7883  0.6371  0.1512
iter_0112    200    40364   292    348      61.000    28.692            158          80896     30.841  2.4491  1.6565  0.7926  0.6403  0.1523
iter_0113    200    40467   319    321      61.000    28.778            159          81408     30.678  2.4511  1.6521  0.7991  0.6446  0.1545
iter_0114    200    40537   316    324      61.000    28.945            159          81408     30.363  2.4329  1.6467  0.7862  0.6317  0.1546
iter_0115    200    40292   294    346      61.000    28.720            158          80896     30.014  2.4264  1.6444  0.7820  0.6283  0.1537
iter_0116    200    40662   331    309      61.000    28.809            159          81408     31.300  2.4233  1.6492  0.7740  0.6205  0.1535
iter_0117    200    40399   273    367      61.000    28.781            158          80896     30.782  2.4098  1.6426  0.7672  0.6155  0.1516
iter_0118    200    40378   297    343      61.000    28.664            158          80896     29.457  2.4054  1.6386  0.7668  0.6138  0.1531
iter_0119    200    40463   281    359      61.000    28.861            159          81408     29.905  2.3960  1.6327  0.7632  0.6092  0.1540
iter_0120    200    40479   298    342      61.000    28.925            159          81408     30.271  2.3968  1.6338  0.7630  0.6070  0.1560
```

Exp8 overall through `iter_0120` is mildly White-skewed. The late window is
more concerning than the full-run average:

```text
iter_0102 through iter_0114  Blue 4181  White 4139  Blue rate 50.3%
iter_0102 through iter_0119  Blue 5657  White 5863  Blue rate 49.1%
iter_0102 through iter_0120  Blue 5955  White 6205  Blue rate 49.0%
iter_0115 through iter_0120  Blue 1774  White 2066  Blue rate 46.2%
iter_0107                   Blue 360   White 280   Blue rate 56.2%
iter_0111                   Blue 358   White 282   Blue rate 55.9%
iter_0112                   Blue 292   White 348   Blue rate 45.6%
iter_0117                   Blue 273   White 367   Blue rate 42.7%
iter_0119                   Blue 281   White 359   Blue rate 43.9%
iter_0120                   Blue 298   White 342   Blue rate 46.6%
```

Opening-root checks. `chosen` is the sampled self-play first move. `argmax` is
the largest entry in the MCTS visit target. `avg` is the average target
probability for that cell across the 640 opening positions.

```text
generation  #30 chosen  #30 argmax  #30 avg  #21 chosen  #21 argmax  #21 avg  note
iter_0102      29/640      57/640   0.0405      -           -        0.0452  flat opening cluster
iter_0103      91/640      75/640   0.1161    63/640     372/640    0.1203  #21/#30/#29 cluster
iter_0104      34/640      31/640   0.0711    59/640     440/640    0.0825  #30 reduced, #21 argmax high
iter_0105      41/640     254/640   0.0816    37/640     114/640    0.0670  #30 argmax returns
iter_0106      52/640       7/640   0.0698    62/640     527/640    0.1004  #21 argmax dominates
iter_0107      57/640       0/640   0.0789    68/640     593/640    0.1053  #21 argmax dominates
iter_0108     110/640     366/640   0.1699    82/640     111/640    0.1169  #30 concentration returns
iter_0109      57/640      78/640   0.0881    50/640     481/640    0.0895  #21 argmax dominates
iter_0110     114/640     555/640   0.1860    51/640      18/640    0.0784  #30 argmax dominates
iter_0111      55/640     352/640   0.1041    61/640     228/640    0.1008  #split #30/#21
iter_0112      53/640     540/640   0.0997    50/640      43/640    0.0863  #30 argmax dominates
iter_0113      67/640     536/640   0.0990    41/640      21/640    0.0638  #30 argmax dominates
iter_0114      59/640     540/640   0.0910    57/640      87/640    0.0858  #30 argmax dominates
iter_0115     161/640     632/640   0.2470    31/640       1/640    0.0544  #30 concentration jumps
iter_0116      75/640     640/640   0.1343    56/640       0/640    0.0908  #30 argmax saturates
iter_0117     228/640     639/640   0.3434    32/640       0/640    0.0480  #30 chosen often
iter_0118     239/640     639/640   0.4113    29/640       1/640    0.0505  #30 target keeps sharpening
iter_0119     289/640     640/640   0.4799    22/640       0/640    0.0359  #30 dominates hard
iter_0120     188/640     640/640   0.2983    44/640       0/640    0.0611  #30 argmax still saturated
```

Policy-loss interpretation:

```text
policy loss = target entropy + KL(target || model)
KL(target || model) = policy loss - target entropy
```

Exp7 policy loss dropped mostly because the teacher targets sharpened. Exp8
kept the target entropy roughly flat, so the raw policy loss stopped receiving
that entropy-driven drop.

```text
Direct same-generation comparison:

generation  Exp7 entropy  Exp7 policy  Exp7 KL  Exp7 value  Exp8 entropy  Exp8 policy  Exp8 KL  Exp8 value
iter_0102      1.3427        1.6327    0.2901     0.8540     1.3573        1.6519    0.2947     0.8530
iter_0103      1.3236        1.6119    0.2883     0.8464     1.3661        1.6618    0.2957     0.8656
iter_0104      1.3228        1.6110    0.2881     0.8540     1.3717        1.6637    0.2920     0.8432
iter_0105      1.3143        1.5995    0.2852     0.8431     1.3746        1.6629    0.2882     0.8385
iter_0106      1.3025        1.5889    0.2864     0.8498     1.3745        1.6683    0.2938     0.8414
iter_0107      1.2901        1.5753    0.2853     0.8385     1.3709        1.6621    0.2912     0.8365
iter_0108      1.2760        1.5640    0.2880     0.8387     1.3661        1.6585    0.2925     0.8286
iter_0109      1.2784        1.5649    0.2865     0.8441     1.3587        1.6527    0.2940     0.8196
iter_0110      1.2704        1.5582    0.2878     0.8299     1.3580        1.6542    0.2962     0.8103
iter_0111      1.2629        1.5531    0.2902     0.8368     1.3627        1.6503    0.2876     0.7883
iter_0112      1.2562        1.5432    0.2870     0.8388     1.3638        1.6565    0.2928     0.7926
iter_0113      1.2589        1.5403    0.2814     0.8269     1.3588        1.6521    0.2932     0.7991
iter_0114      1.2447        1.5321    0.2874     0.8090     1.3626        1.6467    0.2841     0.7862
```

Current read:

```text
Exp7:
  arena strength improved, but much of the policy-loss drop came from target
  sharpening. This is the same mechanism that can create opening feedback.

Exp8:
  policy temperature delayed the #30-specific collapse, but did not prevent
  opening feedback. The hard argmax shifted between #21 and #30, then #30
  dominated again by iter_0110 through iter_0114. From iter_0115 through
  iter_0120, #30 becomes much more concentrated: its opening target average
  rises as high as 0.4799 at iter_0119, and it is the target argmax in
  essentially every opening position from iter_0116 onward. The same late
  window is White-skewed in self-play: Blue 1774, White 2066, Blue rate 46.2%.
  Raw policy loss moves down slowly, but mostly without KL-distance improvement.
  Value loss does improve.
```

Arena checks:

```text
model matchup                     search           seed  games  action selection  result  score rate  Blue result  White result  real
iter_0120 vs heuristic puct:10000 neural-puct:200      1    128  max/max          116-12       90.6%     55-9         61-3           -
iter_0120 vs heuristic puct:10000 neural-puct:200 100001    128  max/max          116-12       90.6%     57-7         59-5      160.95s
iter_0120 vs heuristic puct:10000 neural-puct:200 200001    128  max/max          111-17       86.7%     51-13        60-4      170.07s
```

The external arena result is strong despite the late self-play White skew. This
means the skew is still a training-loop warning signal, but it does not by
itself show that the current model is weak against the heuristic PUCT baseline.

Saved shard and checkpoint path patterns:

```text
data/neural_self_play_exp8_h128_dual_value/iter_*_directional_h128_dual_value_exp8_npuct200_640g_b640_beta0p1_policytau.jsonl
data/models/directional_cnn_h128_dual_value_exp8_adamw_wd1e4_iter_*_npuct200_policytau_replay.pt
data/models/directional_cnn_h128_dual_value_exp8_adamw_wd1e4_iter_*_npuct200_policytau_replay_mlx.safetensors
```
