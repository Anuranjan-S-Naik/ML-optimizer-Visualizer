# ⚡ ML Optimizer Visualizer

An interactive, educational platform designed to demystify Machine Learning optimization algorithms. Explore how different optimizers navigate complex mathematical landscapes, watch neural networks learn decision boundaries in real-time, and diagnose your own training scripts with a real PyTorch backend!

![ML Optimizer Visualizer Interface](https://img.shields.io/badge/Status-Active-brightgreen) ![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-teal) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)

## 🌟 Features

The application is divided into four main interactive modules:

### 1. 🗺️ Landscape Playground
Visualize classic optimization test functions (like the Rosenbrock "banana" or Rastrigin function) as beautiful, interactive contour maps. 
*   Drop multiple optimizers (SGD, Momentum, AdaGrad, RMSprop, Adam) onto the landscape simultaneously.
*   Watch in real-time as they race to find the global minimum.
*   Clearly see the differences between basic gradient descent and modern adaptive algorithms (e.g., how Momentum oscillates in narrow valleys while Adam adapts).

### 2. 🧠 Neural Network Playground
Watch a neural network learn to classify complex, non-linear data distributions.
*   Choose datasets that are notoriously difficult for linear models (like interlocked Spirals or concentric Circles).
*   Adjust the network architecture (hidden layers, activation functions like Tanh or ReLU).
*   Tweak optimizer hyperparameters (Learning Rate) and watch the decision boundary warp and twist live as the network learns to separate the classes.

### 3. 🎓 Lab Tour
An interactive, guided walkthrough designed for beginners. The Lab Tour explains the core concepts of loss landscapes, gradients, and the intuition behind advanced optimizers step-by-step.

### 4. 🔬 Real Diagnostics Engine (New!)
Upload your own actual Machine Learning code and test it against our diagnostic engine!
*   **AST Code Parsing:** The backend reads your pure Python/PyTorch script to extract your exact network architecture, optimizer choice, and hyperparameters without running arbitrary code.
*   **Rule-Based Diagnostics:** A 12-point health check identifies common pitfalls (e.g., missing momentum, learning rates that are too high, vanishing gradients from deep sigmoid networks, missing gradient clipping in RNNs).
*   **Live Side-by-Side Training:** Paste a sample of your dataset (CSV/JSON), and the backend will compile your current model alongside a "Recommended Fix" model. It uses PyTorch to train both simultaneously, streaming live loss and accuracy curves back to the browser via Server-Sent Events (SSE).

---

## 🛠️ Architecture

The project uses a modern hybrid architecture:

*   **Frontend:** Pure, dependency-free HTML, CSS, and Vanilla JavaScript. Features a responsive, glassmorphic dark-theme UI with custom 2D canvas rendering for the simulations.
*   **Backend (For Real Diagnostics):** A FastAPI server running locally that handles the heavy lifting.
    *   `parser.py`: Uses Python's built-in `ast` module to safely analyze uploaded model scripts.
    *   `diagnostics.py`: Evaluates the parsed data against known ML anti-patterns to generate a health score and suggestions.
    *   `trainer.py`: Uses `torch` to dynamically build and train models based on the parsed configurations.
    *   `server.py`: Serves the API endpoints and SSE streams.

---

## 🚀 Getting Started

If you only want to use the visual playgrounds (Tabs 1-3), you can simply open `index.html` in any modern web browser! No installation required.

### Setting up the Real Diagnostics Backend

To use the **Real Diagnostics** tab, you need to run the local Python backend:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Anuranjan-S-Naik/ML-optimizer-Visualizer.git
   cd ML-optimizer-Visualizer
   ```

2. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

3. **Install the required dependencies:**
   *(It is recommended to use a virtual environment)*
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the FastAPI server:**
   ```bash
   python server.py
   ```
   *The server will start on `http://localhost:8000`.*

5. **Open the Frontend:**
   Open the `index.html` file in your browser and click the **"🔬 Real Diagnostics"** tab!

---

## 📝 Example: Diagnosing a Bad Model

Want to test the diagnostic engine? Try pasting this intentionally flawed PyTorch code into the Real Diagnostics tab:

```python
import torch.nn as nn
import torch.optim as optim

class BadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 256)
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        # Using sigmoid in a deep network can cause vanishing gradients
        return self.fc2(torch.sigmoid(self.fc1(x)))

model = BadModel()
# Plain SGD with a very high learning rate and no momentum
optimizer = optim.SGD(model.parameters(), lr=0.1)
```

**What the engine will do:**
1. Detect that you are using basic SGD without Momentum (a critical issue for complex landscapes).
2. Warn you that a learning rate of `0.1` might cause wild oscillations.
3. Suggest switching to `Adam` with a standard learning rate (e.g., `0.001`).
4. Train both your original script and the Adam-powered version live in the browser, proving that the fixed version converges much faster!

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Anuranjan-S-Naik/ML-optimizer-Visualizer/issues).

## 📄 License

This project is open-source and available under standard open-source licenses.