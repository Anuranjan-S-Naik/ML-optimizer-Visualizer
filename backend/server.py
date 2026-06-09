"""
server.py — FastAPI Backend for ML Optimizer Real Diagnostics
Endpoints:
  POST /analyze    — Parse code, run diagnostics, return suggestions
  POST /upload     — Handle file upload, return code text
  GET  /train      — SSE stream of real training metrics (two models side by side)
"""

import json
import sys
import os
import uuid
import asyncio
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parser import ModelCodeParser
from diagnostics import run_diagnostics, generate_suggestion
from trainer import prepare_dataset, run_comparison

app = FastAPI(title="ML Optimizer Real Diagnostics", version="2.0")

# Allow CORS for local development (frontend served from file:// or different port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store for training jobs
sessions = {}

parser = ModelCodeParser()


# ─────────────────────────────────────────────
#  MODELS
# ─────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    code: str
    dataset_csv: Optional[str] = None
    epochs: Optional[int] = 30

class AnalyzeResponse(BaseModel):
    session_id: str
    parsed: dict
    diagnostics: dict
    suggestion: dict
    has_dataset: bool


# ─────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """
    Parse uploaded Python code, run diagnostics, and generate suggestions.
    If dataset_csv is provided, stores it for later training via /train.
    """
    if not req.code or len(req.code.strip()) < 10:
        raise HTTPException(status_code=400, detail="Code is too short to analyze.")

    # Parse the code
    parsed = parser.parse(req.code)

    # Run diagnostics
    diag = run_diagnostics(parsed)

    # Generate suggestion
    suggestion = generate_suggestion(parsed, diag)

    # Create session
    session_id = str(uuid.uuid4())[:8]
    sessions[session_id] = {
        "parsed": parsed,
        "suggestion": suggestion,
        "dataset_csv": req.dataset_csv,
        "epochs": req.epochs or 30,
    }

    return AnalyzeResponse(
        session_id=session_id,
        parsed=parsed.to_dict(),
        diagnostics=diag.to_dict(),
        suggestion=suggestion.to_dict(),
        has_dataset=bool(req.dataset_csv and len(req.dataset_csv.strip()) > 10),
    )


def reconstruct_code_from_pth(content: bytes, filename: str) -> str:
    import io
    import re
    import torch
    import torch.nn as nn

    try:
        # Load the checkpoint safely onto CPU
        checkpoint = torch.load(io.BytesIO(content), map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        return f"# Failed to load PyTorch checkpoint: {e}\n# Please make sure this is a valid PyTorch (.pth / .pt) save file."

    state_dict = None
    optimizer_info = {}
    other_info = {}

    if isinstance(checkpoint, dict):
        # Try to find state_dict in the keys
        possible_sd_keys = ['state_dict', 'model', 'model_state_dict', 'weights', 'state']
        for k in possible_sd_keys:
            if k in checkpoint and isinstance(checkpoint[k], dict):
                state_dict = checkpoint[k]
                break

        # Try to find optimizer state
        possible_opt_keys = ['optimizer', 'optimizer_state_dict', 'opt', 'opt_state']
        for k in possible_opt_keys:
            if k in checkpoint and isinstance(checkpoint[k], dict):
                opt_state = checkpoint[k]
                if 'param_groups' in opt_state and isinstance(opt_state['param_groups'], list) and len(opt_state['param_groups']) > 0:
                    pg = opt_state['param_groups'][0]
                    for hp in ['lr', 'weight_decay', 'momentum']:
                        if hp in pg:
                            optimizer_info[hp] = pg[hp]
                break

        # Extract other scalar metrics if any
        for k, v in checkpoint.items():
            if k not in possible_sd_keys and k not in possible_opt_keys:
                if isinstance(v, (int, float, str)) and len(str(v)) < 100:
                    other_info[k] = v

        # If no key was found but it's a flat dict of tensors, it is probably the state_dict itself
        if state_dict is None:
            is_sd = True
            for k, v in list(checkpoint.items())[:5]:
                if not hasattr(v, 'shape') and not hasattr(v, 'size'):
                    is_sd = False
                    break
            if is_sd and len(checkpoint) > 0:
                state_dict = checkpoint
    elif isinstance(checkpoint, nn.Module):
        state_dict = checkpoint.state_dict()
        other_info['class_name'] = checkpoint.__class__.__name__

    if state_dict is None:
        return f"# PyTorch checkpoint loaded successfully from '{filename}'\n# No state_dict (weights) found in the checkpoint. Content type: {type(checkpoint)}"

    # Natural sorting helper
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    keys = sorted(state_dict.keys(), key=natural_sort_key)

    layers_code = []
    forward_code = []
    processed_modules = set()

    for k in keys:
        if k.endswith('.weight') or '.weight_' in k:
            suffix = '.weight' if k.endswith('.weight') else k[k.find('.weight_'):]
            module_name = k[:-len(suffix)]
            if module_name in processed_modules:
                continue
            processed_modules.add(module_name)

            weight_tensor = state_dict[k]
            shape = list(weight_tensor.shape)

            # Check bias
            has_bias = False
            for b_key in [module_name + '.bias', module_name + '.bias_ih_l0']:
                if b_key in state_dict:
                    has_bias = True
                    break

            # Reconstruct layer definitions
            if len(shape) == 2:
                # Linear layer
                out_features, in_features = shape[0], shape[1]
                attr_name = module_name.replace('.', '_')
                layers_code.append(f"        self.{attr_name} = nn.Linear({in_features}, {out_features}, bias={has_bias})")
                forward_code.append(f"        x = self.{attr_name}(x)")
            elif len(shape) == 4:
                # Conv2d layer
                out_ch, in_ch, kh, kw = shape[0], shape[1], shape[2], shape[3]
                attr_name = module_name.replace('.', '_')
                layers_code.append(f"        self.{attr_name} = nn.Conv2d({in_ch}, {out_ch}, kernel_size=({kh}, {kw}), bias={has_bias})")
                forward_code.append(f"        x = self.{attr_name}(x)")
            elif len(shape) == 3:
                # Conv1d
                out_ch, in_ch, k_size = shape[0], shape[1], shape[2]
                attr_name = module_name.replace('.', '_')
                layers_code.append(f"        self.{attr_name} = nn.Conv1d({in_ch}, {out_ch}, kernel_size={k_size}, bias={has_bias})")
                forward_code.append(f"        x = self.{attr_name}(x)")
        elif '.weight_ih_l' in k:
            # Recurrent weights (LSTM/GRU)
            module_name = k.split('.weight_ih_l')[0]
            if module_name in processed_modules:
                continue
            processed_modules.add(module_name)

            weight_tensor = state_dict[k]
            shape = list(weight_tensor.shape)

            attr_name = module_name.replace('.', '_')
            if 'lstm' in module_name.lower():
                hidden_size = shape[0] // 4
                input_size = shape[1]
                layers_code.append(f"        self.{attr_name} = nn.LSTM({input_size}, {hidden_size}, batch_first=True)")
                forward_code.append(f"        x, _ = self.{attr_name}(x)")
            else:  # GRU
                hidden_size = shape[0] // 3
                input_size = shape[1]
                layers_code.append(f"        self.{attr_name} = nn.GRU({input_size}, {hidden_size}, batch_first=True)")
                forward_code.append(f"        x, _ = self.{attr_name}(x)")
        elif k.endswith('.running_mean'):
            # BatchNorm
            module_name = k[:-13]
            if module_name in processed_modules:
                continue
            processed_modules.add(module_name)

            mean_tensor = state_dict[k]
            num_features = mean_tensor.shape[0]
            attr_name = module_name.replace('.', '_')
            layers_code.append(f"        self.{attr_name} = nn.BatchNorm1d({num_features})")
            forward_code.append(f"        x = self.{attr_name}(x)")

    class_name = other_info.get('class_name', 'ModelFromCheckpoint')
    class_name = re.sub(r'[^a-zA-Z0-9_]', '', class_name)
    if not class_name or class_name[0].isdigit():
        class_name = "ReconstructedModel"

    code_lines = []
    code_lines.append(f"# Reconstructed PyTorch model from checkpoint: {filename}")
    code_lines.append("# Generated automatically for diagnostics and optimization analysis.")
    code_lines.append("")
    code_lines.append("import torch")
    code_lines.append("import torch.nn as nn")
    code_lines.append("import torch.optim as optim")
    code_lines.append("")
    code_lines.append(f"class {class_name}(nn.Module):")
    code_lines.append("    def __init__(self):")
    code_lines.append("        super().__init__()")

    if layers_code:
        code_lines.extend(layers_code)
    else:
        code_lines.append("        # No weight layers could be automatically parsed")
        code_lines.append("        self.fc1 = nn.Linear(10, 32)")
        code_lines.append("        self.fc2 = nn.Linear(32, 2)")
        forward_code = ["        x = self.fc1(x)", "        x = self.fc2(x)"]

    code_lines.append("")
    code_lines.append("    def forward(self, x):")

    if forward_code:
        for idx, f_line in enumerate(forward_code):
            code_lines.append(f_line)
            if idx < len(forward_code) - 1:
                curr_layer = layers_code[idx]
                if 'Linear' in curr_layer or 'Conv' in curr_layer:
                    code_lines.append("        x = nn.functional.relu(x)")
    else:
        code_lines.append("        return x")

    code_lines.append("")
    code_lines.append(f"model = {class_name}()")

    # Reconstruct optimizer
    lr = optimizer_info.get('lr', 0.001)
    weight_decay = optimizer_info.get('weight_decay', 0.0)
    momentum = optimizer_info.get('momentum', 0.0)

    opt_name = "Adam"
    if momentum > 0:
        opt_name = "SGD"

    opt_str = f"optimizer = optim.{opt_name}(model.parameters(), lr={lr}"
    if opt_name == "SGD" and momentum > 0:
        opt_str += f", momentum={momentum}"
    if weight_decay > 0:
        opt_str += f", weight_decay={weight_decay}"
    opt_str += ")"

    code_lines.append(opt_str)
    code_lines.append("criterion = nn.CrossEntropyLoss()")
    code_lines.append("")

    code_lines.append("# ---------------------------------------------")
    code_lines.append("#  Checkpoint Metadata:")
    for k, v in other_info.items():
        code_lines.append(f"#    {k}: {v}")
    for k, v in optimizer_info.items():
        code_lines.append(f"#    optimizer.{k}: {v}")
    code_lines.append("# ---------------------------------------------")

    return "\n".join(code_lines)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle file upload. Returns extracted code text."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""

    if ext in ("py", "txt", "ipynb", "pth", "pt"):
        content = await file.read()
        
        if ext in ("pth", "pt"):
            code = reconstruct_code_from_pth(content, file.filename)
        else:
            code = content.decode("utf-8", errors="replace")

            # For .ipynb, extract code cells
            if ext == "ipynb":
                try:
                    nb = json.loads(code)
                    cells = nb.get("cells", [])
                    code_cells = [
                        "".join(c.get("source", []))
                        for c in cells
                        if c.get("cell_type") == "code"
                    ]
                    code = "\n\n".join(code_cells)
                except json.JSONDecodeError:
                    pass

        return {"filename": file.filename, "code": code, "ext": ext}
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: .{ext}. Please upload .py, .txt, .ipynb, .pth, or .pt files."
        )


@app.get("/train")
async def train_stream(session_id: str):
    """
    Server-Sent Events endpoint.
    Trains two models side by side and streams metrics per epoch.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found. Call /analyze first.")

    session = sessions[session_id]
    parsed = session["parsed"]
    suggestion = session["suggestion"]
    dataset_csv = session["dataset_csv"]
    epochs = session["epochs"]

    if not dataset_csv or len(dataset_csv.strip()) < 10:
        raise HTTPException(status_code=400, detail="No dataset provided. Paste CSV/JSON data in the dataset field.")

    async def event_generator():
        try:
            # Run training in a thread to not block the event loop
            loop = asyncio.get_event_loop()

            def _train():
                results = []
                for metrics in run_comparison(parsed, suggestion, dataset_csv, epochs):
                    results.append(metrics)
                return results

            # Run synchronously and stream results
            # (For simplicity; a production version would use true async)
            for metrics in run_comparison(parsed, suggestion, dataset_csv, epochs):
                data = json.dumps(metrics)
                yield f"data: {data}\n\n"
                await asyncio.sleep(0.05)  # Small delay for smooth streaming

        except ValueError as e:
            error_data = json.dumps({"status": "error", "message": str(e)})
            yield f"data: {error_data}\n\n"
        except Exception as e:
            error_data = json.dumps({"status": "error", "message": f"Training error: {str(e)}"})
            yield f"data: {error_data}\n\n"
        finally:
            yield f"data: {json.dumps({'status': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "2.0"}


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("[*] Starting ML Optimizer Real Diagnostics Backend...")
    print("    Server: http://localhost:8000")
    print("    Docs:   http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
