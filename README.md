# MessySMAC - A Modified StarCraft MultiAgent Challenge with Configurable State Uncertainty

Based on PyMARL. Please refer to the repository for more documentation, e.g., regarding StarCraft II.

For more information regarding MessySMAC, check out our [accompanying paper](https://arxiv.org/abs/2301.01649).

## 1. Featured algorithms:

- Attention-based Embeddings of Recurrence In multi-Agent Learning ([AERIAL](https://arxiv.org/abs/2301.01649))

## 2. Evaluation domains

All available domains used in the paper are listed in the table below. The labels are used for the command in 5.

| Domain   	   	| Label            | Description                                                                     |
|---------------|------------------|---------------------------------------------------------------------------------|
| Dec-Tiger     | `dec_tiger`      | Dec-Tiger Problem with default horizon of 4	                                 |
| SMAC          | `sc2`            | StarCraft Multi-Agent Challenge                      					         |
| MessySMAC     | `messy_sc2`      | SMAC extension with stochastic observations and more variance in initial states |

## 3. MARL algorithms

The MARL algorithms used in the paper (see 6.) are listed in the table below. The labels are used for the command  in 5.

| Algorithm            | Label                  |
|----------------------|------------------------|
| AERIAL               | `aerial`               |
| AERIAL (no attention)| `aerial_no_att`        |
| AERIAL (raw history) | `aerial_raw_history1`  |
| QPLEX                | `qplex`                |
| CW-QMIX              | `cw_qmix`              |
| OW-QMIX              | `ow_qmix`              |
| QMIX                 | `qmix`                 |
| QTRAN                | `qtran`                |


## 4. Experiment parameters

Default experiment parameters like the learning rate, the exploration schedule, or batch sizes, etc. are specified in the respective `.yaml`-files in the `src/config/`-folder.

All default hyperparameters can be adjusted in the respective `.yaml`-files in the `src/config/algs`-folder.

## 5. Training

To train a MARL algorithm `A` (see table in 3.) in domain `D` (see table in 2.), run the following command:

    python3 src/main.py --config=A --env-config=D with env_args.map_name=M

`M` specifies the SMAC map (e.g., `10m_vs_11m`, `3s_vs_5z`) and can be set to `dec_tiger` if `D == dec_tiger`.

To configure stochasticity of observations and initial states, the parameters `env_args.failure_obs_prob` and `env_args.randomize_initial_state` can be set for `D == messy_sc2` respectively.

`train.sh` is an example script for running all settings as specified in the paper.

## 6. Citation
If you use MessySMAC or AERIAL in your work, please cite:

```
@misc{phan2023attention,
    author	    = {Thomy Phan and Fabian Ritz and Philipp Altmann and Maximilian Zorn and Jonas Nüßlein and Michael Kölle and Thomas Gabor and Claudia Linnhoff-Popien},
    title	    = {Attention-Based Recurrence for Multi-Agent Reinforcement Learning under Stochastic Partial Observability},
    year	    = {2023},
    publisher	= {PMLR},
    booktitle	= {Proceedings of the 40th International Conference on Machine Learning (ICML)},
    keywords	= {Dec-POMDP, stochastic partial observability, multi-agent learning, recurrence, self-attention},
    location    = {Hawaii, USA}
}
```