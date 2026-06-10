"""
parser.py — AST-based Python Code Parser
Extracts model architecture, optimizer, hyperparameters, and training patterns
from user-uploaded Python source code.
"""

import ast
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParsedModel:
    """Structured result from parsing a Python ML training script."""
    raw_code: str = ""
    optimizer: str = "Unknown"
    lr: Optional[float] = None
    momentum: Optional[float] = None
    weight_decay: Optional[float] = None
    activations: list = field(default_factory=list)
    has_dropout: bool = False
    has_batchnorm: bool = False
    has_grad_clip: bool = False
    is_rnn: bool = False
    layer_count: int = 0
    layer_sizes: list = field(default_factory=list)
    loss_function: str = "Unknown"
    framework: str = "unknown"
    task_type: str = "classification"  # classification or regression

    def to_dict(self):
        return {
            "optimizer": self.optimizer,
            "lr": self.lr,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "activations": self.activations,
            "has_dropout": self.has_dropout,
            "has_batchnorm": self.has_batchnorm,
            "has_grad_clip": self.has_grad_clip,
            "is_rnn": self.is_rnn,
            "layer_count": self.layer_count,
            "layer_sizes": self.layer_sizes,
            "loss_function": self.loss_function,
            "framework": self.framework,
            "task_type": self.task_type,
        }


class ModelCodeParser:
    """Parse Python ML code using AST + regex fallbacks."""

    # Optimizer name mapping
    OPTIMIZER_MAP = {
        "sgd": "SGD",
        "adam": "Adam",
        "adamw": "AdamW",
        "rmsprop": "RMSprop",
        "adagrad": "Adagrad",
        "adadelta": "Adadelta",
        "adamax": "Adamax",
        "lbfgs": "LBFGS",
    }

    # Activation name mapping
    ACTIVATION_MAP = {
        "relu": "relu",
        "sigmoid": "sigmoid",
        "tanh": "tanh",
        "leakyrelu": "leaky_relu",
        "leaky_relu": "leaky_relu",
        "gelu": "gelu",
        "elu": "elu",
        "selu": "selu",
        "softmax": "softmax",
    }

    def parse(self, code: str) -> ParsedModel:
        """Parse Python code and extract ML model information."""
        result = ParsedModel(raw_code=code)
        code_lower = code.lower()

        # Detect framework
        result.framework = self._detect_framework(code_lower)

        # Try AST parsing first, fall back to regex
        try:
            tree = ast.parse(code)
            self._extract_from_ast(tree, code, result)
        except SyntaxError:
            pass

        # Always run regex extraction to fill gaps
        self._extract_from_regex(code, code_lower, result)

        # Set defaults
        if not result.activations:
            result.activations = ["none"]

        return result

    def _detect_framework(self, code_lower: str) -> str:
        if "torch" in code_lower or "nn.module" in code_lower:
            return "pytorch"
        elif "tensorflow" in code_lower or "keras" in code_lower:
            return "tensorflow"
        elif "sklearn" in code_lower or "scikit" in code_lower:
            return "sklearn"
        return "unknown"

    def _extract_from_ast(self, tree: ast.AST, code: str, result: ParsedModel):
        """Walk AST to extract structured info."""
        for node in ast.walk(tree):
            # Detect optimizer calls: optim.SGD(...), optim.Adam(...)
            if isinstance(node, ast.Call):
                func_name = self._get_call_name(node)
                if func_name:
                    fn_lower = func_name.lower()

                    # --- sklearn MLPClassifier / MLPRegressor ---
                    if "mlpclassifier" in fn_lower or "mlpregressor" in fn_lower:
                        self._extract_sklearn_mlp(node, result)
                        if "mlpregressor" in fn_lower:
                            result.task_type = "regression"
                        else:
                            result.task_type = "classification"
                        continue

                    # Check if it's an optimizer call
                    for opt_key, opt_name in self.OPTIMIZER_MAP.items():
                        if opt_key in fn_lower and ("optim" in fn_lower or fn_lower.endswith(opt_key)):
                            if result.optimizer == "Unknown":
                                result.optimizer = opt_name
                            # Extract keyword args
                            self._extract_optimizer_kwargs(node, result)
                            break

                    # Check for activation layers
                    for act_key, act_name in self.ACTIVATION_MAP.items():
                        if act_key in fn_lower:
                            if act_name not in result.activations:
                                result.activations.append(act_name)

                    # Count layers
                    if any(layer in fn_lower for layer in
                           ["linear", "conv1d", "conv2d", "conv3d", "dense", "lstm", "gru", "rnn"]):
                        result.layer_count += 1
                        # Try to extract layer sizes
                        if "linear" in fn_lower or "dense" in fn_lower:
                            sizes = [
                                arg.value for arg in node.args
                                if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float))
                            ]
                            result.layer_sizes.extend(sizes)

                    # Detect special layers
                    if "dropout" in fn_lower:
                        result.has_dropout = True
                    if "batchnorm" in fn_lower or "layernorm" in fn_lower or "batch_norm" in fn_lower:
                        result.has_batchnorm = True
                    if "clip_grad" in fn_lower or "clip_norm" in fn_lower:
                        result.has_grad_clip = True
                    if any(rnn in fn_lower for rnn in ["lstm", "gru", "rnn"]):
                        result.is_rnn = True

                    # Loss functions
                    if "crossentropy" in fn_lower or "cross_entropy" in fn_lower:
                        result.loss_function = "CrossEntropy"
                        result.task_type = "classification"
                    elif "mseloss" in fn_lower or "mean_squared" in fn_lower:
                        result.loss_function = "MSE"
                        result.task_type = "regression"
                    elif "bceloss" in fn_lower or "binary_cross" in fn_lower:
                        result.loss_function = "BCE"
                        result.task_type = "classification"

    def _get_call_name(self, node: ast.Call) -> str:
        """Get the full dotted name of a function call."""
        parts = []
        func = node.func
        while isinstance(func, ast.Attribute):
            parts.append(func.attr)
            func = func.value
        if isinstance(func, ast.Name):
            parts.append(func.id)
        parts.reverse()
        return ".".join(parts) if parts else ""

    def _extract_optimizer_kwargs(self, node: ast.Call, result: ParsedModel):
        """Extract lr, momentum, weight_decay from optimizer call."""
        for kw in node.keywords:
            if kw.arg == "lr" or kw.arg == "learning_rate":
                if isinstance(kw.value, ast.Constant):
                    result.lr = float(kw.value.value)
            elif kw.arg == "momentum":
                if isinstance(kw.value, ast.Constant):
                    result.momentum = float(kw.value.value)
            elif kw.arg == "weight_decay":
                if isinstance(kw.value, ast.Constant):
                    result.weight_decay = float(kw.value.value)

    def _extract_sklearn_mlp(self, node: ast.Call, result: ParsedModel):
        """Extract architecture info from sklearn MLPClassifier / MLPRegressor."""
        # Solver → maps to our optimizer names
        SKLEARN_SOLVER_MAP = {
            "adam": "Adam",
            "sgd": "SGD",
            "lbfgs": "LBFGS",
        }
        # Activation → maps to our activation names
        SKLEARN_ACT_MAP = {
            "relu": "relu",
            "tanh": "tanh",
            "logistic": "sigmoid",
            "identity": "none",
        }

        for kw in node.keywords:
            if kw.arg == "solver" and isinstance(kw.value, ast.Constant):
                solver = str(kw.value.value).lower()
                if solver in SKLEARN_SOLVER_MAP:
                    result.optimizer = SKLEARN_SOLVER_MAP[solver]
            elif kw.arg == "learning_rate_init" and isinstance(kw.value, ast.Constant):
                result.lr = float(kw.value.value)
            elif kw.arg == "alpha" and isinstance(kw.value, ast.Constant):
                result.weight_decay = float(kw.value.value)
            elif kw.arg == "activation" and isinstance(kw.value, ast.Constant):
                act = str(kw.value.value).lower()
                mapped = SKLEARN_ACT_MAP.get(act, act)
                if mapped not in result.activations:
                    result.activations.append(mapped)
            elif kw.arg == "hidden_layer_sizes":
                sizes = self._extract_tuple_ints(kw.value)
                if sizes:
                    result.layer_sizes = sizes
                    result.layer_count = len(sizes)
            elif kw.arg == "max_iter" and isinstance(kw.value, ast.Constant):
                pass  # not needed but valid
            elif kw.arg == "early_stopping" and isinstance(kw.value, ast.Constant):
                if kw.value.value:
                    result.has_dropout = True  # treat early stopping as a form of regularization

        # sklearn MLPClassifier uses cross-entropy by default
        result.loss_function = "CrossEntropy"
        result.framework = "sklearn"

        # Default solver in sklearn is 'adam'
        if result.optimizer == "Unknown":
            result.optimizer = "Adam"
        # Default lr in sklearn is 0.001
        if result.lr is None:
            result.lr = 0.001
        # Default activation in sklearn is 'relu'
        if not result.activations:
            result.activations = ["relu"]
        # Default hidden_layer_sizes is (100,)
        if not result.layer_sizes:
            result.layer_sizes = [100]
            result.layer_count = 1

    def _extract_tuple_ints(self, node) -> list:
        """Extract integer values from a tuple AST node like (64, 32)."""
        if isinstance(node, ast.Tuple):
            return [
                int(elt.value) for elt in node.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, (int, float))
            ]
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return [int(node.value)]
        return []

    def _extract_from_regex(self, code: str, code_lower: str, result: ParsedModel):
        """Regex fallback for extracting info that AST might miss."""

        # Optimizer (if AST didn't find one)
        if result.optimizer == "Unknown":
            opt_patterns = [
                (r"optim\.adamw|adamw\s*\(", "AdamW"),
                (r"optim\.adam\b|adam\s*\(", "Adam"),
                (r"optim\.rmsprop|rmsprop\s*\(", "RMSprop"),
                (r"optim\.adagrad|adagrad\s*\(", "Adagrad"),
                (r"optim\.sgd\s*\(|sgd\s*\(", "SGD"),
                (r"keras\.optimizers\.sgd|compile.*sgd", "SGD"),
                (r"keras\.optimizers\.adam|compile.*adam", "Adam"),
                # sklearn solver regex fallback
                (r"solver\s*=\s*['\"]adam['\"]", "Adam"),
                (r"solver\s*=\s*['\"]sgd['\"]", "SGD"),
                (r"solver\s*=\s*['\"]lbfgs['\"]", "LBFGS"),
            ]
            for pattern, name in opt_patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    result.optimizer = name
                    break

        # Learning rate (if AST didn't find it)
        if result.lr is None:
            lr_match = re.search(r"lr\s*=\s*([\d.e\-]+)", code, re.IGNORECASE)
            if not lr_match:
                lr_match = re.search(r"learning_rate\s*=\s*([\d.e\-]+)", code, re.IGNORECASE)
            if lr_match:
                try:
                    result.lr = float(lr_match.group(1))
                except ValueError:
                    pass

        # Momentum
        if result.momentum is None:
            mom_match = re.search(r"momentum\s*=\s*([\d.e\-]+)", code, re.IGNORECASE)
            if mom_match:
                try:
                    result.momentum = float(mom_match.group(1))
                except ValueError:
                    pass

        # Weight decay
        if result.weight_decay is None:
            wd_match = re.search(r"weight_decay\s*=\s*([\d.e\-]+)", code, re.IGNORECASE)
            if wd_match:
                try:
                    result.weight_decay = float(wd_match.group(1))
                except ValueError:
                    pass

        # Activations from regex
        act_regex = {
            r"\bsigmoid\b": "sigmoid",
            r"\brelu\b": "relu",
            r"\btanh\b": "tanh",
            r"leaky_?relu": "leaky_relu",
            r"\bgelu\b": "gelu",
        }
        for pattern, name in act_regex.items():
            if re.search(pattern, code, re.IGNORECASE) and name not in result.activations:
                result.activations.append(name)

        # Architecture features
        if not result.has_dropout:
            result.has_dropout = bool(re.search(r"dropout", code, re.IGNORECASE))
        if not result.has_batchnorm:
            result.has_batchnorm = bool(
                re.search(r"batchnorm|batch_norm|layernorm", code, re.IGNORECASE)
            )
        if not result.has_grad_clip:
            result.has_grad_clip = bool(
                re.search(r"clip_grad|clip_norm|clipnorm|max_norm", code, re.IGNORECASE)
            )
        if not result.is_rnn:
            result.is_rnn = bool(re.search(r"lstm|gru|\brnn\b|recurrent", code, re.IGNORECASE))

        # Layer count from regex
        if result.layer_count == 0:
            layer_matches = re.findall(
                r"nn\.linear|nn\.conv|dense\s*\(|lstm\s*\(", code, re.IGNORECASE
            )
            result.layer_count = len(layer_matches)

        # Loss function
        if result.loss_function == "Unknown":
            if re.search(r"crossentropy|cross_entropy|log_loss", code, re.IGNORECASE):
                result.loss_function = "CrossEntropy"
                result.task_type = "classification"
            elif re.search(r"mseloss|mean_squared|mse_loss", code, re.IGNORECASE):
                result.loss_function = "MSE"
                result.task_type = "regression"
            elif re.search(r"bceloss|binary_cross", code, re.IGNORECASE):
                result.loss_function = "BCE"
                result.task_type = "classification"
            elif re.search(r"accuracy_score|accuracy", code, re.IGNORECASE):
                result.loss_function = "CrossEntropy"
                result.task_type = "classification"

        # sklearn hidden_layer_sizes regex fallback
        if result.layer_count == 0:
            hlm = re.search(r"hidden_layer_sizes\s*=\s*\(([\d\s,]+)\)", code)
            if hlm:
                sizes = [int(s.strip()) for s in hlm.group(1).split(',') if s.strip().isdigit()]
                if sizes:
                    result.layer_sizes = sizes
                    result.layer_count = len(sizes)
