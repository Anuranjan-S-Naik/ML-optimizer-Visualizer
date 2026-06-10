"""
diagnostics.py — Rule-Based Diagnostic Engine
Scores model health, detects issues, and suggests optimal configurations.
Ported from the JavaScript rule engine in optisuggest-v2.html.
"""

from dataclasses import dataclass, field
from typing import Optional
from parser import ParsedModel


@dataclass
class Issue:
    id: str
    severity: str  # 'critical', 'warning', 'info', 'ok'
    icon: str
    title: str
    desc: str
    penalty: int = 0
    fix_type: str = ""
    fix_from: str = ""
    fix_to: str = ""

    def to_dict(self):
        return {
            "id": self.id,
            "severity": self.severity,
            "icon": self.icon,
            "title": self.title,
            "desc": self.desc,
            "penalty": self.penalty,
            "fix_type": self.fix_type,
            "fix_from": self.fix_from,
            "fix_to": self.fix_to,
        }


@dataclass
class DiagResult:
    issues: list = field(default_factory=list)
    score: int = 100

    def to_dict(self):
        return {
            "issues": [i.to_dict() for i in self.issues],
            "score": self.score,
        }


@dataclass
class Suggestion:
    optimizer_name: str = ""
    optimizer_params: str = ""
    bad_label: str = ""
    good_label: str = ""
    root_cause: str = ""
    root_body: str = ""
    fixes: list = field(default_factory=list)
    diff_lines: list = field(default_factory=list)
    bad_curve_type: str = "stable"
    good_curve_type: str = "converging"
    bad_score: int = 50
    good_score: int = 90
    # Actual suggested optimizer config for training
    suggested_optimizer: str = "Adam"
    suggested_lr: float = 0.001
    suggested_weight_decay: float = 0.0
    suggested_activations: list = field(default_factory=list)
    loss_function: str = "CrossEntropy"

    def to_dict(self):
        return {
            "optimizer_name": self.optimizer_name,
            "bad_label": self.bad_label,
            "good_label": self.good_label,
            "root_cause": self.root_cause,
            "root_body": self.root_body,
            "fixes": self.fixes,
            "diff_lines": self.diff_lines,
            "bad_curve_type": self.bad_curve_type,
            "good_curve_type": self.good_curve_type,
            "bad_score": self.bad_score,
            "good_score": self.good_score,
            "suggested_optimizer": self.suggested_optimizer,
            "suggested_lr": self.suggested_lr,
            "suggested_weight_decay": self.suggested_weight_decay,
            "suggested_activations": self.suggested_activations,
            "loss_function": self.loss_function,
        }


def run_diagnostics(parsed: ParsedModel) -> DiagResult:
    """Run all diagnostic rules on parsed model. Returns issues + health score."""
    issues = []
    penalty_total = 0

    # ── RULE 1: SGD without momentum
    if parsed.optimizer == "SGD" and not parsed.momentum:
        issue = Issue(
            id="sgd_no_momentum",
            severity="critical",
            icon="🚨",
            title="SGD without momentum",
            desc="Plain SGD has no velocity — it takes longer paths and gets stuck in local minima. Momentum or switching to Adam fixes this.",
            penalty=25,
            fix_type="optimizer",
            fix_from="SGD(lr=...)",
            fix_to="Adam(lr=0.001)",
        )
        issues.append(issue)
        penalty_total += 25

    # ── RULE 2: SGD with high LR
    if parsed.optimizer == "SGD" and parsed.lr is not None and parsed.lr > 0.05:
        issue = Issue(
            id="sgd_high_lr",
            severity="critical",
            icon="🚨",
            title=f"SGD learning rate {parsed.lr} is too high",
            desc=f"For SGD, lr > 0.05 causes the optimizer to overshoot the loss minimum repeatedly. Typical working range is 0.001–0.01.",
            penalty=30,
            fix_type="lr",
            fix_from=f"lr={parsed.lr}",
            fix_to="lr=0.001 (or switch to Adam)",
        )
        issues.append(issue)
        penalty_total += 30

    # ── RULE 3: Sigmoid in deep network
    if "sigmoid" in parsed.activations and parsed.layer_count >= 4:
        issue = Issue(
            id="sigmoid_deep",
            severity="critical",
            icon="🚨",
            title=f"Sigmoid in {parsed.layer_count}-layer network",
            desc="Sigmoid squashes gradients to near-zero. In networks deeper than 3 layers, gradients multiply near-zero values and vanish — early layers stop learning entirely.",
            penalty=35,
            fix_type="activation",
            fix_from="nn.Sigmoid()",
            fix_to="nn.ReLU()",
        )
        issues.append(issue)
        penalty_total += 35

    # ── RULE 4: Sigmoid in any network (milder warning)
    if "sigmoid" in parsed.activations and parsed.layer_count < 4:
        issue = Issue(
            id="sigmoid_shallow",
            severity="warning",
            icon="⚠️",
            title="Sigmoid activation detected",
            desc="Even in shallow networks, Sigmoid can slow convergence. ReLU converges 2–3× faster in most tasks.",
            penalty=12,
            fix_type="activation",
            fix_from="nn.Sigmoid()",
            fix_to="nn.ReLU()",
        )
        issues.append(issue)
        penalty_total += 12

    # ── RULE 5: No dropout or weight decay (overfitting risk)
    if not parsed.has_dropout and not parsed.weight_decay and parsed.layer_count >= 3:
        issue = Issue(
            id="no_regularization",
            severity="warning",
            icon="⚠️",
            title="No regularization found",
            desc="No Dropout or weight_decay detected. Without regularization, models memorize training data and fail on new inputs — especially on small datasets.",
            penalty=20,
            fix_type="regularization",
            fix_from="no dropout",
            fix_to="Dropout(0.3) + weight_decay=1e-4",
        )
        issues.append(issue)
        penalty_total += 20

    # ── RULE 6: RNN/LSTM without gradient clipping
    if parsed.is_rnn and not parsed.has_grad_clip:
        issue = Issue(
            id="rnn_no_clip",
            severity="critical",
            icon="🚨",
            title="RNN/LSTM without gradient clipping",
            desc="Recurrent models unroll through time, multiplying gradients at each step. Without clipping, a single large gradient can corrupt all weights instantly.",
            penalty=35,
            fix_type="clip",
            fix_from="optimizer.step()",
            fix_to="nn.utils.clip_grad_norm_(model.parameters(), 1.0)\noptimizer.step()",
        )
        issues.append(issue)
        penalty_total += 35

    # ── RULE 7: RMSProp with very low LR
    if parsed.optimizer == "RMSprop" and parsed.lr is not None and parsed.lr < 1e-5:
        issue = Issue(
            id="rmsprop_tiny_lr",
            severity="warning",
            icon="⚠️",
            title=f"RMSProp lr={parsed.lr} may be too small",
            desc="RMSProp typically works well with lr between 0.0001–0.001. Very small values cause extremely slow convergence.",
            penalty=15,
            fix_type="lr",
            fix_from=f"lr={parsed.lr}",
            fix_to="lr=0.0001",
        )
        issues.append(issue)
        penalty_total += 15

    # ── RULE 8: Very low learning rate (any optimizer)
    if parsed.lr is not None and parsed.lr < 0.00001:
        issue = Issue(
            id="lr_too_small",
            severity="warning",
            icon="⚠️",
            title=f"Learning rate {parsed.lr} is very small",
            desc="This LR will cause extremely slow convergence. In practice, training would be stopped before the model learns useful representations.",
            penalty=18,
            fix_type="lr",
            fix_from=f"lr={parsed.lr}",
            fix_to="lr=0.001",
        )
        issues.append(issue)
        penalty_total += 18

    # ── RULE 9: Adam without weight decay in deep nets
    if parsed.optimizer == "Adam" and not parsed.weight_decay and parsed.layer_count >= 5:
        issue = Issue(
            id="adam_no_wd",
            severity="info",
            icon="ℹ️",
            title="Consider AdamW for regularization",
            desc="AdamW decouples weight decay from gradient updates, providing better regularization than plain Adam on deep networks.",
            penalty=8,
            fix_type="optimizer",
            fix_from="Adam(lr=0.001)",
            fix_to="AdamW(lr=0.001, weight_decay=0.01)",
        )
        issues.append(issue)
        penalty_total += 8

    # ── RULE 10: No batch normalization in deep net
    if not parsed.has_batchnorm and parsed.layer_count >= 5:
        issue = Issue(
            id="no_batchnorm",
            severity="info",
            icon="ℹ️",
            title="No batch normalization in deep network",
            desc="BatchNorm stabilizes training in deep networks by normalizing layer inputs. It also acts as a mild regularizer.",
            penalty=6,
            fix_type="batchnorm",
            fix_from="no BatchNorm",
            fix_to="nn.BatchNorm1d(hidden_dim)",
        )
        issues.append(issue)
        penalty_total += 6

    # ── RULE 11: Positive findings
    if parsed.optimizer in ("Adam", "AdamW"):
        issues.append(Issue(
            id="good_optimizer", severity="ok", icon="✅",
            title="Good optimizer choice",
            desc="Adam/AdamW provides adaptive per-parameter learning rates.",
        ))
    if parsed.has_dropout:
        issues.append(Issue(
            id="good_dropout", severity="ok", icon="✅",
            title="Dropout found",
            desc="Regularization present — reduces overfitting risk.",
        ))
    if parsed.has_grad_clip:
        issues.append(Issue(
            id="good_clip", severity="ok", icon="✅",
            title="Gradient clipping found",
            desc="Prevents gradient explosions during training.",
        ))
    if any(a in parsed.activations for a in ("relu", "leaky_relu", "gelu")):
        act_name = next(a for a in parsed.activations if a in ("relu", "leaky_relu", "gelu"))
        issues.append(Issue(
            id="good_activation", severity="ok", icon="✅",
            title=f"{act_name.upper()} activation",
            desc="Good activation — gradients flow properly through layers.",
        ))

    score = max(0, 100 - penalty_total)
    return DiagResult(issues=issues, score=score)


def generate_suggestion(parsed: ParsedModel, diag: DiagResult) -> Suggestion:
    """Generate the optimizer suggestion based on diagnostics."""
    criticals = [i for i in diag.issues if i.severity == "critical"]
    warnings = [i for i in diag.issues if i.severity == "warning"]

    actionable = sorted(criticals + warnings, key=lambda i: i.penalty, reverse=True)
    primary = actionable[0] if actionable else None

    # Build descriptive labels
    lr_str = f" lr={parsed.lr}" if parsed.lr else ""
    loss_str = parsed.loss_function if parsed.loss_function != "Unknown" else "CrossEntropy"
    layers_str = f" [{len(parsed.layer_sizes)} layers]" if parsed.layer_sizes else ""

    suggestion = Suggestion(
        optimizer_name=parsed.optimizer,
        bad_label=f"{parsed.optimizer}{lr_str} ({loss_str})",
        bad_score=diag.score,
        good_score=min(97, diag.score + 45),
        loss_function=loss_str,
        # Defaults for training
        suggested_optimizer="Adam",
        suggested_lr=0.001,
        suggested_activations=["relu"],
    )

    # Determine bad curve type
    issue_ids = [i.id for i in diag.issues]
    if "sgd_high_lr" in issue_ids:
        suggestion.bad_curve_type = "oscillating"
    elif "sigmoid_deep" in issue_ids:
        suggestion.bad_curve_type = "plateau"
    elif "rnn_no_clip" in issue_ids:
        suggestion.bad_curve_type = "exploding"
    elif "no_regularization" in issue_ids:
        suggestion.bad_curve_type = "overfit"
    elif diag.score > 70:
        suggestion.bad_curve_type = "slow"

    if not primary:
        suggestion.good_label = f"{parsed.optimizer} + LR Scheduler"
        suggestion.root_cause = "Model looks healthy!"
        suggestion.root_body = "No major issues detected. Minor improvements possible with learning rate scheduling."
        suggestion.fixes = [["Add LR Scheduler", "CosineAnnealingLR improves final accuracy by ~2-3%."]]
        suggestion.good_curve_type = "fast"
        suggestion.diff_lines = [
            {"type": "neutral", "text": f"optimizer = {parsed.optimizer}(lr={parsed.lr or 0.001})"},
            {"type": "plus", "text": "scheduler = CosineAnnealingLR(optimizer, T_max=100)"},
        ]
        suggestion.suggested_optimizer = parsed.optimizer
        suggestion.suggested_lr = parsed.lr or 0.001
        return suggestion

    pid = primary.id

    if pid in ("sgd_high_lr", "sgd_no_momentum"):
        suggestion.good_label = "Adam lr=0.001"
        suggestion.root_cause = primary.title
        suggestion.root_body = primary.desc
        suggestion.fixes = [
            ["Switch to Adam optimizer", "Adaptive learning rates per parameter — handles noisy gradients automatically."],
            ["Lower learning rate to 0.001", "Adam's default lr=0.001 works for 90% of tasks without tuning."],
            ["Add lr scheduler (optional)", "CosineAnnealingLR decays LR smoothly after warmup."],
        ]
        suggestion.diff_lines = [
            {"type": "minus", "text": f"optimizer = SGD(model.parameters(), lr={parsed.lr or 0.1})"},
            {"type": "plus", "text": "optimizer = Adam(model.parameters(), lr=0.001)"},
        ]
        suggestion.good_curve_type = "converging"
        suggestion.suggested_optimizer = "Adam"
        suggestion.suggested_lr = 0.001

    elif pid in ("sigmoid_deep", "sigmoid_shallow"):
        suggestion.good_label = "ReLU + Adam + BatchNorm"
        suggestion.root_cause = primary.title
        suggestion.root_body = primary.desc
        suggestion.fixes = [
            ["Replace Sigmoid → ReLU", "ReLU gradients are either 0 or 1 — no squashing, early layers learn."],
            ["Switch to Adam optimizer", "Adam recovers slow-learning parameters via adaptive LR."],
            ["Add BatchNorm after each layer", "Stabilizes activations and acts as implicit regularizer."],
        ]
        suggestion.diff_lines = [
            {"type": "minus", "text": "nn.Sigmoid()  # ← vanishing gradient"},
            {"type": "plus", "text": "nn.ReLU()"},
            {"type": "plus", "text": "nn.BatchNorm1d(hidden_size)"},
            {"type": "minus", "text": f"optimizer = SGD(model.parameters(), lr={parsed.lr or 0.01})"},
            {"type": "plus", "text": "optimizer = Adam(model.parameters(), lr=0.001)"},
        ]
        suggestion.good_curve_type = "converging"
        suggestion.suggested_optimizer = "Adam"
        suggestion.suggested_lr = 0.001
        suggestion.suggested_activations = ["relu"]

    elif pid == "no_regularization":
        suggestion.good_label = "AdamW + Dropout + WeightDecay"
        suggestion.root_cause = primary.title
        suggestion.root_body = primary.desc
        suggestion.fixes = [
            ["Add Dropout(0.3) after ReLU layers", "Randomly zeroes 30% of neurons per forward pass — prevents co-adaptation."],
            ["Switch to AdamW with weight_decay=1e-4", "AdamW decouples weight decay from adaptive LR — better generalization."],
            ["Use early stopping", "Monitor val_loss and stop when it increases for 10 consecutive epochs."],
        ]
        suggestion.diff_lines = [
            {"type": "neutral", "text": "nn.ReLU(),"},
            {"type": "plus", "text": "nn.Dropout(0.3),  # ← add this"},
            {"type": "minus", "text": "optimizer = Adam(model.parameters(), lr=0.001)"},
            {"type": "plus", "text": "optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)"},
        ]
        suggestion.good_curve_type = "regularized"
        suggestion.suggested_optimizer = "AdamW"
        suggestion.suggested_lr = 0.001
        suggestion.suggested_weight_decay = 1e-4

    elif pid == "rnn_no_clip":
        suggestion.good_label = "RMSProp + GradClip"
        suggestion.root_cause = primary.title
        suggestion.root_body = primary.desc
        suggestion.fixes = [
            ["Add gradient clipping (max_norm=1.0)", "Clips gradient norm before optimizer.step() — prevents weight corruption."],
            ["Switch to RMSProp", "RMSProp adapts per-parameter LR — better for recurrent models than SGD."],
            ["Reduce LSTM layers if > 3", "Deep RNNs amplify gradient problems; 2 layers often sufficient."],
        ]
        suggestion.diff_lines = [
            {"type": "minus", "text": f"optimizer = SGD(model.parameters(), lr={parsed.lr or 0.01}, momentum=0.9)"},
            {"type": "plus", "text": "optimizer = RMSprop(model.parameters(), lr=0.001)"},
            {"type": "minus", "text": "optimizer.step()  # ← no clipping"},
            {"type": "plus", "text": "nn.utils.clip_grad_norm_(model.parameters(), 1.0)"},
            {"type": "plus", "text": "optimizer.step()"},
        ]
        suggestion.good_curve_type = "converging"
        suggestion.suggested_optimizer = "RMSprop"
        suggestion.suggested_lr = 0.001

    else:
        suggestion.good_label = "Optimized version"
        suggestion.root_cause = primary.title
        suggestion.root_body = primary.desc
        suggestion.fixes = [[primary.fix_type.upper(), f"Change {primary.fix_from} to {primary.fix_to}"]]
        suggestion.diff_lines = [
            {"type": "minus", "text": primary.fix_from},
            {"type": "plus", "text": primary.fix_to},
        ]

    return suggestion
