# Configuration

`load_config` merges a JSON file into the defaults in `config.py`, then validates the result. The training CLI saves the fully resolved configuration with every run.

The `configs/` directory contains example settings. The seven `train_*` files provide presets grouped by network: five core-periphery variants, a random ten-bank network and a random 200-bank network; `train_fixed_horizon_adam.json` combines Adam, leave-one-out baselines, advantage standardization, and gradient clipping.

## Environment

`environment.network_type` selects the network. Supported choices are `core_periphery`, `star`, `inverse_star`, `cycle`, `chain`, `pairs`, `asymmetric_stars`, `symmetric_stars`, `mirrored_cycles`, `random`, and `mixed_networks`. Fixed topologies and `mixed_networks` use ten banks, `random` accepts other network sizes.

The environment settings specify network size, initial capital and net worth, risky-asset dynamics, clearing parameters, and the capital-injection penalty. `environment.action_space_architecture` is `weighted_limit`: one global budget index followed by one local weight index per bank. The selected budget is distributed among solvent banks according to their weights. The capital cost uses that selected budget. The environment provides a flat mask over available budget and bank-weight choices.

The heuristic evaluation uses this same action space. `predict()` returns positions in `env.bins_action`, the list of available capital amounts. `_heuristic_action` in `evaluation.py` sets the full global budget and weight 1 for each bank with a positive index. Other banks get weight 0. If no bank is selected, the whole action is zero.

`initial_state_mode` is `fixed` or `distribution`. The fixed mode reuses an initial state and stores its snapshot in checkpoints. Distribution mode draws initial states according to the configured environment.

`capture_initial_state()` copies the network, balance sheets, risky-asset prices, and any bank permutation. `restore_initial_state(snapshot, seed=...)` restores these values and resets the episode counters. It does not restore a random generator. Ordinary `reset(seed=...)` remains available when a fresh initial state is wanted.

## Policy

`policy.family` selects `shared`, `routed`, or `flat`. Shared graph policies use one encoder for both action heads. Routed policies have separate budget and node encoders, configured under `policy.routed`. Flat policies use an MLP over the flattened observation.

Graph policies accept `bidirectional_mpnn` or `bidirectional_gatv2` as their encoder. `mlp` is used with the flat family. The settings control layers, embedding size, aggregation, readout, activation, and GATv2 attention heads. Node heads use local embeddings, the global budget head uses a graph readout. The default encoder activation is `tanh`. It can be changed explicitly through `policy.activation`.

## Training

`rollout.scheme` selects `geometric` or `fixed_horizon`. Fixed-horizon rollouts use `rollout.horizon`. Geometric rollouts use the discount-based stopping rule. `transition_budget` controls the number of sampled transitions. The default and all training presets use 100,000.

`mode: exact` uses fixed-state geometric rollouts, parameter projection, and manual SGD without the optional extensions. `mode: experimental` also permits fixed-horizon rollouts, distributed initial states, and the extensions below.

`step_size.type` accepts `constant`, `polynomial`, or `exponential` schedules. An exponential schedule can specify its decay factor directly or use `final_step_ratio`. `batch_size.type` is `constant` or `polynomial`.

`projection.enabled`, `default_bound`, and optional `group_bounds` control the parameter constraints. Exact mode uses the common bound.

These options are implemented in `financial_networks/trainer.py`: `_build_optimizer` selects the optimizer, `_advantages` and `_fixed_horizon_advantages` compute the baselines and standardization, `_apply_update` handles gradients and parameter updates, and `train_one_update` assembles the score-estimator objective. The rollout estimators are `_one_geometric_gradient_sample` and `_one_fixed_horizon_trajectory`. Geometric sampling is in `distributions.py`. `seeds.model`, `seeds.environment`, and `seeds.initial_state` control the random streams within one training run. `checkpoint_interval_updates` and `metrics_interval_updates` control local checkpoint and training-log frequency. `device` selects the PyTorch device.

## Evaluation

The evaluation CLI accepts a checkpoint, evaluation episode seeds, horizon, and deterministic/stochastic mode. It uses the saved policy and configuration. `--environment-config` supplies another compatible environment configuration. `--reference-checkpoint` supplies another environment together with its fixed initial-state snapshot.

Evaluation returns per-episode records and summaries for the selected policy. Use `--output` to save them to a local JSON file.