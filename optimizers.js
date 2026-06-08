/**
 * optimizers.js — Loss Functions & Optimization Algorithms
 * ML Optimizer Visualizer
 *
 * Provides:
 *   - OPTIMIZER_INFO: Color/name map for each optimizer
 *   - LOSS_FUNCTIONS: 6 benchmark functions with analytical gradients
 *   - Optimizer class: Unified optimizer with per-index state (SGD/Momentum/AdaGrad/RMSprop/Adam)
 *   - createOptimizer(): Factory function
 */

// ==================== OPTIMIZER METADATA ====================

const OPTIMIZER_INFO = {
    sgd:      { name: 'SGD',       color: '#ef4444', colorRgb: '239,68,68'   },
    momentum: { name: 'Momentum',  color: '#3b82f6', colorRgb: '59,130,246'  },
    adagrad:  { name: 'AdaGrad',   color: '#f59e0b', colorRgb: '245,158,11'  },
    rmsprop:  { name: 'RMSprop',   color: '#a855f7', colorRgb: '168,85,247'  },
    adam:     { name: 'Adam',      color: '#22d3ee', colorRgb: '34,211,238'  }
};

// ==================== LOSS FUNCTIONS ====================

const LOSS_FUNCTIONS = {

    rosenbrock: {
        name: 'Rosenbrock',
        f: (x, y) => (1 - x) ** 2 + 100 * (y - x * x) ** 2,
        grad: (x, y) => [
            -2 * (1 - x) - 400 * x * (y - x * x),
            200 * (y - x * x)
        ],
        bounds: { x: [-2, 2], y: [-1, 3] },
        minimum: [1, 1],
        minValue: 0,
        defaultStart: [-1.5, 2.0],
        description: 'A narrow curved valley (banana function). Tests navigation of ill-conditioned surfaces where axes have very different curvatures.'
    },

    himmelblau: {
        name: 'Himmelblau',
        f: (x, y) => (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2,
        grad: (x, y) => [
            4 * x * (x * x + y - 11) + 2 * (x + y * y - 7),
            2 * (x * x + y - 11) + 4 * y * (x + y * y - 7)
        ],
        bounds: { x: [-5, 5], y: [-5, 5] },
        minimum: [3, 2],
        minValue: 0,
        defaultStart: [-4, 4],
        description: 'Four identical local minima at (3,2), (-2.81,3.13), (-3.78,-3.28), (3.58,-1.85). Tests multi-modal optimization.'
    },

    rastrigin: {
        name: 'Rastrigin',
        f: (x, y) => 20 + x * x - 10 * Math.cos(2 * Math.PI * x) + y * y - 10 * Math.cos(2 * Math.PI * y),
        grad: (x, y) => [
            2 * x + 20 * Math.PI * Math.sin(2 * Math.PI * x),
            2 * y + 20 * Math.PI * Math.sin(2 * Math.PI * y)
        ],
        bounds: { x: [-5.12, 5.12], y: [-5.12, 5.12] },
        minimum: [0, 0],
        minValue: 0,
        defaultStart: [4, 4],
        description: 'Highly multi-modal with many local minima arranged in a regular grid. Tests ability to escape local traps and find the global minimum.'
    },

    beale: {
        name: 'Beale',
        f: (x, y) => {
            const t1 = 1.5 - x + x * y;
            const t2 = 2.25 - x + x * y * y;
            const t3 = 2.625 - x + x * y * y * y;
            return t1 * t1 + t2 * t2 + t3 * t3;
        },
        grad: (x, y) => {
            const t1 = 1.5 - x + x * y;
            const t2 = 2.25 - x + x * y * y;
            const t3 = 2.625 - x + x * y * y * y;
            return [
                2 * t1 * (-1 + y) + 2 * t2 * (-1 + y * y) + 2 * t3 * (-1 + y * y * y),
                2 * t1 * x + 2 * t2 * (2 * x * y) + 2 * t3 * (3 * x * y * y)
            ];
        },
        bounds: { x: [-4.5, 4.5], y: [-4.5, 4.5] },
        minimum: [3, 0.5],
        minValue: 0,
        defaultStart: [-3, 3],
        description: 'Flat regions surrounding a sharp minimum. Tests optimizer precision and step-size adaptation in low-gradient zones.'
    },

    saddle: {
        name: 'Saddle Point',
        f: (x, y) => x * x - y * y,
        grad: (x, y) => [2 * x, -2 * y],
        bounds: { x: [-3, 3], y: [-3, 3] },
        minimum: [0, 0],
        minValue: 0,
        defaultStart: [0.5, 0.001],
        description: 'A saddle point at the origin where the gradient is zero but it is NOT a minimum. Tests escape from saddle points — a critical challenge in deep learning.'
    },

    quadratic: {
        name: 'Quadratic Bowl',
        f: (x, y) => x * x + 3 * y * y,
        grad: (x, y) => [2 * x, 6 * y],
        bounds: { x: [-4, 4], y: [-4, 4] },
        minimum: [0, 0],
        minValue: 0,
        defaultStart: [3.5, 3.5],
        description: 'Simple elliptical bowl. Baseline test — all optimizers should converge. Reveals differences in convergence speed and oscillation patterns.'
    }
};

// ==================== OPTIMIZER CLASS ====================

/**
 * Unified optimizer with per-index state tracking.
 * Works on arbitrary-length parameter arrays.
 *
 * For landscape optimization: pass grads = [∂f/∂x, ∂f/∂y]
 * For neural network training: pass grads = [∂L/∂w1, ∂L/∂w2, ..., ∂L/∂b1, ...]
 */
class Optimizer {
    constructor(type, lr, config = {}) {
        this.type = type;
        this.lr = lr;
        this.beta  = config.beta  !== undefined ? config.beta  : 0.9;
        this.beta1 = config.beta1 !== undefined ? config.beta1 : 0.9;
        this.beta2 = config.beta2 !== undefined ? config.beta2 : 0.999;
        this.epsilon = config.epsilon !== undefined ? config.epsilon : 1e-8;
        this.t = 0;
        this.state = {};
    }

    reset() {
        this.t = 0;
        this.state = {};
    }

    /** Perform one optimization step. Returns array of deltas to ADD to parameters. */
    step(grads) {
        this.t++;
        const deltas = new Array(grads.length);
        for (let i = 0; i < grads.length; i++) {
            deltas[i] = this._compute(i, grads[i]);
        }
        return deltas;
    }

    _ensureState(i) {
        if (!this.state[i]) {
            this.state[i] = { m: 0, v: 0, cache: 0 };
        }
    }

    _compute(i, g) {
        this._ensureState(i);
        const s = this.state[i];

        // Gradient clipping to prevent explosions
        g = Math.max(-10, Math.min(10, g));

        switch (this.type) {

            case 'sgd':
                return -this.lr * g;

            case 'momentum':
                s.v = this.beta * s.v + g;
                return -this.lr * s.v;

            case 'adagrad':
                s.cache += g * g;
                return -this.lr * g / (Math.sqrt(s.cache) + this.epsilon);

            case 'rmsprop':
                s.cache = this.beta * s.cache + (1 - this.beta) * g * g;
                return -this.lr * g / (Math.sqrt(s.cache) + this.epsilon);

            case 'adam': {
                s.m = this.beta1 * s.m + (1 - this.beta1) * g;
                s.v = this.beta2 * s.v + (1 - this.beta2) * g * g;
                const mHat = s.m / (1 - Math.pow(this.beta1, this.t));
                const vHat = s.v / (1 - Math.pow(this.beta2, this.t));
                return -this.lr * mHat / (Math.sqrt(vHat) + this.epsilon);
            }

            default:
                return -this.lr * g;
        }
    }
}

/** Factory function to create an optimizer instance. */
function createOptimizer(type, lr, config = {}) {
    return new Optimizer(type, lr, config);
}
