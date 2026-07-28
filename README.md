# Financial networks

This repository implements a financial-network environment and projected REINFORCE for a controller that chooses a global capital budget and local bank allocation weights.

If you use or cite this code, please cite the accompanying paper: *Reinforcement Learning for Global-Local Control of Dynamic Networks with Applications to Systemic Risk* by Nils Detering, Florian Grell, and Thilo Meyer-Brandis.

## Contents

- `financial_networks/environment/`: network construction, balance-sheet dynamics, clearing, action masks, and absorbing-state handling.
- `financial_networks/encoders/` and `financial_networks/policies.py`: shared and separate MPNN/GATv2 graph encoders and a flat MLP policy.
- `financial_networks/trainer.py`, `financial_networks/distributions.py`, and `financial_networks/config.py`: projected REINFORCE, rollout sampling, optional training extensions, and configuration validation.
- `financial_networks/cli.py` and `financial_networks/evaluation.py`: training, checkpoint recovery, and evaluation of an individual checkpoint.
- `heuristic_agents.py`: Zero, Uniform, and Greedy baseline agents.
- `configs/`: training presets for fixed and random networks.
- `demo_notebook.ipynb` and `financial_networks/notebook_ui.py`: a widget interface for network previews, training settings, local checkpoints, and individual evaluations.
- [configuration.md](configuration.md): configuration fields and available options.

## Installation

Use Python 3.10 or newer in a separate environment. For example, with Conda:

```bash
conda create -n financial-networks python=3.10
conda activate financial-networks
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

To use the notebook:

```bash
python -m pip install -r requirements-dev.txt
python -m ipykernel install --user --name financial-networks --display-name "Python (financial-networks)"
python -m jupyterlab
```

Open `demo_notebook.ipynb`, select **Python (financial-networks)**, and run its cells. The form provides presets and editable settings, an environment preview, and buttons to start training or evaluation. Running the cells only creates the interface. Training starts when you press **Start training**.

## Training and evaluation

Open `demo_notebook.ipynb` and run its cells to display the interface.

1. Select a training preset and adjust the environment, policy, and training settings. Use **Preview environment** to inspect the initial network.
2. Choose **New run** and press **Start training**, or select **Resume selected checkpoint** to continue an existing run.
3. For evaluation, select a trained checkpoint or one of the included heuristics: Zero, Uniform, or Greedy. Set the evaluation horizon and episode count, then press **Run evaluation**.

Training saves checkpoints and logs under `runs/`. Evaluation results are displayed as tables.

## Training presets

The supplied presets use a training budget of 100,000 transitions.

| Configuration | Network and training setup |
| --- | --- |
| `train_fixed_horizon.json` | Core-periphery, shared MPNN |
| `train_fixed_horizon_adam.json` | Core-periphery, MPNN with Adam, LOO, standardization, and clipping |
| `train_gatv2.json` | Core-periphery, shared GATv2 |
| `train_mlp.json` | Core-periphery, flat MLP |
| `train_separate_encoders.json` | Core-periphery, separate budget and bank encoders |
| `train_random_ten.json` | Random networks with 10 banks |
| `train_geometric.json` | Random networks with 200 banks and geometric rollouts |

See [configuration.md](configuration.md) for configuration details and command-line usage.