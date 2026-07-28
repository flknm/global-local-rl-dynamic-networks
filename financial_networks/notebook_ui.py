"""Notebook controls for one training run and individual policy evaluations."""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output, display

from .config import NETWORK_TYPES, load_config, validate_config
from .environment import FinancialNetworkEnv
from .evaluation import HEURISTIC_LABELS, evaluate_checkpoint, evaluate_heuristic
from .trainer import ProjectedReinforceTrainer, _environment_kwargs


# The form edits the same configuration used by the command-line trainer.
CHOICES = {
    "device": ("cpu", "cuda"),
    "environment.network_type": sorted(NETWORK_TYPES),
    "policy.family": (("Shared graph encoder", "shared"), ("Separate budget and bank encoders", "routed"), ("Flat MLP", "flat")),
    "policy.encoder": (("Bidirectional MPNN", "bidirectional_mpnn"), ("Bidirectional GATv2", "bidirectional_gatv2"), ("MLP", "mlp")),
    "policy.activation": ("tanh", "relu", "leaky_relu", "sigmoid", "elu", "identity"),
    "policy.aggregation": ("sum", "mean", "min", "max"),
    "policy.readout": ("sum", "mean"),
    "rollout.scheme": (("Fixed horizon", "fixed_horizon"), ("Geometric horizons", "geometric")),
    "step_size.type": ("constant", "polynomial", "exponential"),
    "batch_size.type": ("constant", "polynomial"),
    "experimental.optimizer": (("Manual SGD", "manual_sgd"), ("PyTorch SGD", "sgd"), ("Adam", "adam")),
    "experimental.baseline": (("No baseline", "none"), ("Leave-one-out (LOO)", "leave_one_out"), ("Batch mean", "batch_mean")),
}
LABELS = {
    "mode": "Training mode", "discount_factor": "Discount factor",
    "transition_budget": "Training transitions", "device": "Device",
    "initial_state_mode": "Initial state", "network_type": "Network topology",
    "num_nodes": "Number of banks", "num_core_nodes": "Core banks (random network)",
    "price_risky_asset": "Initial risky-asset price", "mu": "Long-run price",
    "sigma": "Price volatility", "theta": "Mean reversion", "rho": "Risky-asset share",
    "delta": "Initial net-worth ratio", "alpha": "Asset recovery fraction",
    "beta": "Payment recovery fraction", "inverse_demand_slope": "Inverse-demand slope",
    "factor_networth": "Net-worth reward weight", "factor_capital": "Capital cost weight",
    "factor_penalty": "New-default penalty", "upper_limit_capital_injections": "Capital limit",
    "num_capital_injection_actions": "Capital grid points", "num_action_weights": "Local weight levels",
    "action_space_architecture": "Action space", "permute_on_reset": "Permute bank labels",
    "tolerance": "Clearing tolerance", "max_iter": "Clearing iteration limit",
    "family": "Policy family", "encoder": "Encoder", "num_layers": "Encoder layers",
    "embedding_dim": "Features per bank", "num_heads": "Attention heads",
    "aggregation": "Message aggregation", "readout": "Graph readout", "activation": "Activation",
    "budget_head_num_layers": "Budget-head layers", "budget_bias_nudge": "Initial budget bias",
    "scheme": "Rollout scheme", "horizon": "Training horizon",
    "checkpoint_interval_updates": "Checkpoint every n updates (0 = off)",
    "metrics_interval_updates": "Log every n updates",
    "standardize_advantages": "Standardize advantages",
    "gradient_clip_norm": "Gradient clipping", "default_bound": "Projection bound",
    "enabled": "Enable projection", "final_step_ratio": "Final / initial step size",
    "decay_factor": "Decay per update", "maximum": "Maximum batch size",
}

PRESET_LABELS = {
    "train_fixed_horizon.json": "Core–periphery · shared MPNN · fixed horizon",
    "train_fixed_horizon_adam.json": "Core–periphery · MPNN · Adam + LOO",
    "train_gatv2.json": "Core–periphery · shared GATv2 · fixed horizon",
    "train_mlp.json": "Core–periphery · flat MLP · fixed horizon",
    "train_separate_encoders.json": "Core–periphery · separate MPNNs for budget and banks",
    "train_random_ten.json": "Random network · 10 banks · MPNN · fixed horizon",
    "train_geometric.json": "Random network · 200 banks · MPNN · geometric horizons",
}
PLAIN_TRAINING = {
    "optimizer": "manual_sgd", "baseline": "none",
    "standardize_advantages": False,
    "gradient_clip_norm": None,
}


class NotebookDashboard:
    """Build a widget form without starting training or evaluation on import."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.config = load_config(self.root / "configs" / "train_fixed_horizon.json")
        self.fields = {}
        self.optional = {}
        self.rows = {}
        self.use_extensions = widgets.Checkbox(
            value=self._has_extensions(self.config), description="Use training extensions",
        )
        self.use_extensions.observe(lambda _: self._sync_extensions(), names="value")
        self.last_training_summary = None
        self.last_evaluation = None
        self.preview_output = widgets.Output()
        self.training_output = widgets.Output()
        self.evaluation_output = widgets.Output()
        self.progress = widgets.IntProgress(description="Transitions", min=0, max=1)
        self.presets = widgets.Dropdown(
            options=[(label, str(self.root / "configs" / name)) for name, label in PRESET_LABELS.items()],
            value=str(self.root / "configs" / "train_fixed_horizon.json"), description="Preset",
            layout=widgets.Layout(width="750px"),
        )
        self.new_run = widgets.Checkbox(value=True, description="New run")
        self.run_name = widgets.Label(value=self._new_run_name())
        self.checkpoint = widgets.Dropdown(options=[("No checkpoint selected", "")], description="Checkpoint", layout=widgets.Layout(width="650px"))
        self.custom_checkpoint = widgets.Text(description="Custom path", placeholder="Optional path to a checkpoint outside runs/", layout=widgets.Layout(width="650px"))
        self.resume = widgets.Checkbox(value=False, description="Resume selected checkpoint")
        self.resume_checkpoint = widgets.Dropdown(options=[("No checkpoint selected", "")], description="Checkpoint", layout=widgets.Layout(width="650px"))
        self.resume_custom_checkpoint = widgets.Text(description="Custom path", placeholder="Optional path to a checkpoint outside runs/", layout=widgets.Layout(width="650px"))
        self.new_run_details = widgets.HBox([widgets.Label("Run folder"), self.run_name])
        self.resume_details = widgets.VBox([
            self.resume_checkpoint, self.resume_custom_checkpoint,
            self._button("Refresh checkpoints", lambda _: self.refresh_checkpoints()),
        ])
        self.new_run.observe(self._sync_training_mode, names="value")
        self.resume.observe(self._sync_training_mode, names="value")
        self.eval_target = widgets.Dropdown(options=[("Trained policy", "policy"), *[(v, k) for k, v in HEURISTIC_LABELS.items()]], description="Evaluate")
        self.eval_mode = widgets.Dropdown(options=["both", "deterministic", "stochastic"], description="Policy mode")
        self.eval_episodes = widgets.BoundedIntText(value=3, min=1, max=100000, description="Episodes")
        self.eval_seed = widgets.BoundedIntText(value=101, min=0, max=2**31-1, description="First seed")
        self.eval_horizon = widgets.BoundedIntText(value=100, min=1, max=100000, description="Horizon")
        self.save_evaluation = widgets.Checkbox(value=True, description="Save evaluation JSON locally")
        self.preview_button = self._button("Preview environment", self._preview_clicked)
        self.train_button = self._button("Start training", self._train_clicked, "success")
        self.evaluate_button = self._button("Run evaluation", self._evaluate_clicked, "info")
        self._build_form()
        self.refresh_checkpoints()

    def _new_run_name(self):
        return "run_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def _sync_training_mode(self, change=None):
        if getattr(self, "_updating_training_mode", False):
            return
        self._updating_training_mode = True
        try:
            resume = self.resume.value
            if change is not None and change["owner"] is self.new_run:
                resume = not change["new"]
            self.resume.value = resume
            self.new_run.value = not resume
            self.new_run_details.layout.display = "none" if resume else ""
            self.resume_details.layout.display = "" if resume else "none"
        finally:
            self._updating_training_mode = False

    @staticmethod
    def _has_extensions(config):
        return any(config["experimental"].get(key) != value for key, value in PLAIN_TRAINING.items())

    def _bank_count(self):
        return int(self.fields["environment.num_nodes"].value) if self.fields["environment.network_type"].value == "random" else 10

    def _sync_extensions(self):
        for name, control in self.fields.items():
            if name.startswith("experimental."):
                control.disabled = not self.use_extensions.value
                if name in self.optional:
                    self.optional[name].disabled = not self.use_extensions.value

    def _sync_network(self, _=None):
        random_network = self.fields["environment.network_type"].value == "random"
        self.fields["environment.num_nodes"].disabled = not random_network
        self.fields["environment.num_core_nodes"].disabled = not random_network
        if not random_network:
            self.fields["environment.num_nodes"].value = 10

    def _sync_policy(self, source="family"):
        if getattr(self, "_updating_policy", False):
            return
        self._updating_policy = True
        try:
            family = self.fields["policy.family"]
            encoder = self.fields["policy.encoder"]
            if source == "encoder":
                if encoder.value == "mlp":
                    family.value = "flat"
                elif family.value == "flat":
                    family.value = "shared"
            elif family.value == "flat":
                encoder.value = "mlp"
            elif encoder.value == "mlp":
                encoder.value = "bidirectional_mpnn"
            graph = encoder.value != "mlp"
            if graph != self._graph_embeddings:
                width = int(self.fields["policy.embedding_dim"].value)
                banks = self._bank_count()
                self.fields["policy.embedding_dim"].value = max(1, (width + banks - 1) // banks) if graph else width * banks
            self._graph_embeddings = graph
            self.rows["policy.embedding_dim"].children[0].value = "Features per bank" if graph else "MLP hidden width"
            for name, row in self.rows.items():
                if name.startswith("policy.routed"):
                    row.layout.display = "" if family.value == "routed" else "none"
            self.rows["policy.embedding_dim"].layout.display = "none" if family.value == "routed" else ""
            for name in ("policy.aggregation", "policy.readout"):
                self.rows[name].layout.display = "" if graph else "none"
            self.rows["policy.num_heads"].layout.display = "" if encoder.value == "bidirectional_gatv2" else "none"
        finally:
            self._updating_policy = False

    def _button(self, label, callback, style=""):
        button = widgets.Button(description=label, button_style=style, layout=widgets.Layout(width="190px"))
        button.on_click(callback)
        return button

    def _form_fields(self, settings, prefix=""):
        rows = []
        for key, value in settings.items():
            path = f"{prefix}.{key}" if prefix else key
            # These settings are not edited by the notebook form.
            if key in {"group_bounds", "node_global_context", "action_space_architecture"}:
                continue
            if isinstance(value, dict):
                heading = widgets.HTML(f"<b>{key.replace('_', ' ').title()} encoder</b>")
                self.rows[path] = heading
                rows.append(heading)
                rows.extend(self._form_fields(value, path))
                continue
            if key == "embedding_dim" and (prefix.startswith("policy.routed") or self._graph_embeddings):
                value = int(value) // int(self.config["environment"]["num_nodes"])
            if path in CHOICES:
                control = widgets.Dropdown(options=CHOICES[path], value=value)
            elif isinstance(value, bool):
                control = widgets.Checkbox(value=value, indent=False)
            elif isinstance(value, int):
                control = widgets.BoundedIntText(value=value, min=1, max=2**31-1) if key in {"embedding_dim", "num_nodes", "num_layers", "num_heads"} else widgets.IntText(value=value)
            else:
                control = widgets.FloatText(value=0.0 if value is None else float(value))
            control.layout.width = "220px"
            self.fields[path] = control
            label = LABELS.get(key, key.replace("_", " ").title())
            if prefix == "seeds":
                label += " seed"
            if prefix in {"step_size", "batch_size"}:
                label = {"type": "Schedule", "initial": "Initial value", "exponent": "Exponent"}.get(key, label)
            children = [widgets.Label(label, layout=widgets.Layout(width="280px")), control]
            if value is None or key in {"gradient_clip_norm", "final_step_ratio", "maximum"}:
                enabled = widgets.Checkbox(value=value is not None, description="Use value", layout=widgets.Layout(width="120px"))
                self.optional[path] = enabled
                children.append(enabled)
            row = widgets.HBox(children)
            self.rows[path] = row
            rows.append(row)
        return rows

    def _build_form(self):
        self.fields, self.optional, self.rows = {}, {}, {}
        self._graph_embeddings = self.config["policy"]["encoder"] != "mlp"
        top = {k: self.config[k] for k in ("discount_factor", "device", "transition_budget")}
        sections = [widgets.VBox(self._form_fields(top))]
        titles = ["Run settings"]
        for key, title in (("environment", "Environment and clearing"), ("policy", "Policy and encoder"), ("rollout", "Rollouts"), ("step_size", "Step size"), ("batch_size", "Batch size"), ("projection", "Projection"), ("experimental", "Training extensions"), ("seeds", "Random seeds")):
            controls = self._form_fields(self.config[key], key)
            if key == "experimental":
                controls = [self.use_extensions, widgets.HTML("<p>Enable these controls to choose an optimizer, a baseline or other training extensions.</p>"), *controls]
            sections.append(widgets.VBox(controls))
            titles.append(title)
        logging = {k: self.config[k] for k in ("checkpoint_interval_updates", "metrics_interval_updates")}
        sections.append(widgets.VBox(self._form_fields(logging)))
        titles.append("Local checkpoint and log frequency")
        form = widgets.Accordion(children=sections, selected_index=0)
        for index, title in enumerate(titles):
            form.set_title(index, title)
        load_button = self._button("Load preset", self._load_clicked)
        refresh_button = self._button("Refresh checkpoints", lambda _: self.refresh_checkpoints())
        children = [
            widgets.HTML("<h3>Configure a run</h3><p>Load a preset, then edit the fields below. Training presets use 100,000 transitions. Loading a preset does not start training.</p>"),
            widgets.HBox([self.presets, load_button]), form,
            self.preview_button, self.preview_output,
            widgets.HTML("<h3>Train a policy</h3>"),
            widgets.HBox([self.new_run, self.resume]), self.new_run_details,
            self.resume_details, self.train_button, self.progress, self.training_output,
            widgets.HTML("<h3>Evaluate a policy or heuristic</h3><p>Checkpoints are found in local runs/ folders. After training, the final checkpoint is selected automatically. A custom path overrides that selection. A policy uses its saved environment; a heuristic uses the form settings and the same action conversion as the paper evaluation. For symmetric_stars and mirrored_cycles, use a horizon of 1 as in the paper. The initial evaluation settings are 3 episodes, seeds starting at 101 and horizon 100; all are editable.</p>"),
            self.checkpoint, self.custom_checkpoint, refresh_button,
            widgets.HBox([self.eval_target, self.eval_mode]),
            widgets.HBox([self.eval_episodes, self.eval_seed, self.eval_horizon]),
            self.save_evaluation, self.evaluate_button, self.evaluation_output,
        ]
        if hasattr(self, "widget"):
            self.widget.children = tuple(children)
        else:
            self.widget = widgets.VBox(children, layout=widgets.Layout(max_width="1000px"))
        self.fields["policy.family"].observe(lambda _: self._sync_policy("family"), names="value")
        self.fields["policy.encoder"].observe(lambda _: self._sync_policy("encoder"), names="value")
        self.fields["environment.network_type"].observe(self._sync_network, names="value")
        self.fields["step_size.type"].observe(lambda _: self._sync_schedules(), names="value")
        self.fields["rollout.scheme"].observe(lambda _: self._sync_schedules(), names="value")
        self._sync_network()
        self._sync_policy()
        self._sync_extensions()
        self._sync_schedules()
        self._sync_training_mode()

    def _sync_schedules(self):
        exponential = self.fields["step_size.type"].value == "exponential"
        for name in ("step_size.decay_factor", "step_size.final_step_ratio"):
            self.rows[name].layout.display = "" if exponential else "none"
        self.fields["rollout.horizon"].disabled = self.fields["rollout.scheme"].value != "fixed_horizon"

    def read_config(self):
        config = copy.deepcopy(self.config)
        for path, control in self.fields.items():
            target = config
            parts = path.split(".")
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = None if path in self.optional and not self.optional[path].value else control.value
        banks = self._bank_count()
        if self._graph_embeddings:
            config["policy"]["embedding_dim"] *= banks
        for branch in ("budget", "node"):
            config["policy"]["routed"][branch]["embedding_dim"] *= banks
        if config["step_size"]["type"] != "exponential":
            config["step_size"]["final_step_ratio"] = None
        if not self.use_extensions.value:
            config["experimental"] = copy.deepcopy(PLAIN_TRAINING)
        # Keep the trainer's existing mode requirements, but infer the choice
        # from the form rather than ask users to select a second label.
        fixed_start = (config["environment"]["network_type"] not in {"random", "mixed_networks"}
                       and not config["environment"]["permute_on_reset"])
        compatible = (fixed_start and config["rollout"]["scheme"] == "geometric"
                      and not self._has_extensions(config) and config["projection"]["enabled"]
                      and not config["projection"]["group_bounds"])
        config["mode"] = "exact" if compatible else "experimental"
        config["initial_state_mode"] = "fixed" if compatible else "distribution"
        return validate_config(config)

    def load_preset(self, path):
        self.config = load_config(path)
        self.use_extensions.value = self._has_extensions(self.config)
        self._build_form()

    def _load_clicked(self, _):
        self._with_output(self.preview_output, lambda: (self.load_preset(self.presets.value), print("Preset loaded. Edit the form or preview the environment.")))

    @staticmethod
    def _with_output(output, operation):
        with output:
            clear_output(wait=True)
            try:
                operation()
            except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
                print(f"Could not complete the action: {exc}")

    def preview_environment(self):
        config = self.read_config()
        network = config["environment"]["network_type"]
        if network == "mixed_networks":
            network = "core_periphery"
            print("Showing core_periphery as one member of mixed_networks.")
        env = FinancialNetworkEnv(**_environment_kwargs(config, network_type=network))
        try:
            env.reset(seed=config["seeds"]["initial_state"])
            banks = env._get_observed_node_indices(env.node_features)
            previous_figures = set(plt.get_fignums())
            try:
                env.render()
            finally:
                for number in set(plt.get_fignums()) - previous_figures:
                    plt.close(number)
            display(pd.DataFrame({n: env.node_features[n] for n in banks}).T.rename_axis("Bank"))
        finally:
            env.close()

    def _preview_clicked(self, _):
        self._with_output(self.preview_output, self.preview_environment)

    def selected_checkpoint(self, *, for_resume=False):
        custom = self.resume_custom_checkpoint if for_resume else self.custom_checkpoint
        selection = self.resume_checkpoint if for_resume else self.checkpoint
        value = custom.value.strip() or selection.value
        if not value:
            raise FileNotFoundError("Train a policy first or select an existing checkpoint.")
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    def refresh_checkpoints(self, selected=None):
        paths = sorted((self.root / "runs").rglob("*_checkpoint.pt"))
        options = [("No checkpoint selected", ""), *[(str(p.relative_to(self.root)), str(p)) for p in paths]]
        for dropdown in (self.checkpoint, self.resume_checkpoint):
            previous = str(selected) if selected else dropdown.value
            dropdown.options = options
            if previous in {value for _, value in options}:
                dropdown.value = previous

    def train(self):
        self.last_training_summary = None
        if self.resume.value:
            checkpoint = self.selected_checkpoint(for_resume=True)
            config = load_config(checkpoint.parent / "resolved_config.json")
            output_dir = checkpoint.parent
        else:
            checkpoint = None
            config = self.read_config()
            run_path = Path(self.run_name.value.strip())
            if not self.run_name.value.strip() or run_path.is_absolute() or ".." in run_path.parts:
                raise ValueError("The generated run folder must be a relative path below runs/.")
            output_dir = self.root / "runs" / run_path
        trainer = ProjectedReinforceTrainer(config, output_dir, resume=checkpoint is not None)
        try:
            if checkpoint is not None:
                trainer.load_checkpoint(checkpoint)
            self.progress.max = int(config["transition_budget"])
            print(f"Training in {output_dir}")
            # train() retains responsibility for updates, recovery and final files.
            # Calling it with successive targets lets the notebook show progress
            # without duplicating the training algorithm.
            while trainer.transitions < self.progress.max:
                target = min(self.progress.max, trainer.transitions + max(1, self.progress.max // 20))
                self.last_training_summary = trainer.train(target_transition_budget=target)
                self.progress.value = min(self.progress.max, trainer.transitions)
            if self.last_training_summary is None or not self.last_training_summary["complete"]:
                self.last_training_summary = trainer.train()
            print(f"Completed {trainer.update} updates and {trainer.transitions} sampled transitions.")
        finally:
            trainer.close()
        self.refresh_checkpoints(output_dir / "final_checkpoint.pt")
        self.custom_checkpoint.value = ""
        self.resume_custom_checkpoint.value = ""
        self.run_name.value = self._new_run_name()
        metrics = pd.read_json(output_dir / "metrics.jsonl", lines=True)
        display(metrics.tail(5))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(metrics["transitions"], metrics["mean_q_estimate"])
        ax.set(xlabel="Sampled transitions", ylabel="Mean sampled reward-to-go", title="Training batch statistic")
        display(fig)
        plt.close(fig)
        print("This statistic averages the reward-to-go estimates sampled for training. It is not the mean return of a separate policy evaluation.")
        return self.last_training_summary

    def _train_clicked(self, _):
        self.train_button.disabled = True
        try:
            self._with_output(self.training_output, self.train)
        finally:
            self.train_button.disabled = False

    def evaluate(self):
        seeds = range(self.eval_seed.value, self.eval_seed.value + self.eval_episodes.value)
        if self.eval_target.value == "policy":
            checkpoint = self.selected_checkpoint()
            modes = ("deterministic", "stochastic") if self.eval_mode.value == "both" else (self.eval_mode.value,)
            result = evaluate_checkpoint(checkpoint, seeds=seeds, horizon=self.eval_horizon.value, modes=modes)
            output_dir = checkpoint.parent
        else:
            config = self.read_config()
            result = evaluate_heuristic(config, self.eval_target.value, seeds=seeds, horizon=self.eval_horizon.value)
            output_dir = self.root / "runs" / "heuristic_evaluations"
        self.last_evaluation = result
        rows = []
        for mode, values in result["modes"].items():
            rows.append({"Mode": mode, "Mean discounted return": values["mean_discounted_return"], "Mean injected capital": values["mean_total_injected_capital"], "Mean final defaults": values["mean_final_default_count"]})
        display(pd.DataFrame(rows).set_index("Mode"))
        episodes = pd.DataFrame([{**episode, "mode": mode} for mode, summary in result["modes"].items() for episode in summary["episodes"]])
        display(episodes[["mode", "seed", "network_type", "discounted_return", "total_injected_capital", "new_default_count", "length"]])
        if self.save_evaluation.value:
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / ("evaluation_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"Saved locally: {path}")
        return result

    def _evaluate_clicked(self, _):
        self.evaluate_button.disabled = True
        try:
            self._with_output(self.evaluation_output, self.evaluate)
        finally:
            self.evaluate_button.disabled = False
