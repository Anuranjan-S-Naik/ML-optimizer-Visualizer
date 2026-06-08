/**
 * network.js — Feed-Forward Neural Network & Dataset Generators
 * ML Optimizer Visualizer
 *
 * Provides:
 *   - Activations: sigmoid, tanh, relu (with derivatives)
 *   - NeuralNetwork class: configurable layers, forward/backward, optimizer-agnostic training
 *   - Datasets: XOR, Spiral, Circle, Gaussian Clusters generators
 */

// ==================== ACTIVATION FUNCTIONS ====================

const Activations = {
    sigmoid: {
        name: 'Sigmoid',
        f: (x) => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x)))),
        df: (_raw, activated) => activated * (1 - activated)
    },
    tanh: {
        name: 'Tanh',
        f: (x) => Math.tanh(x),
        df: (_raw, activated) => 1 - activated * activated
    },
    relu: {
        name: 'ReLU',
        f: (x) => Math.max(0, x),
        df: (raw, _activated) => raw > 0 ? 1 : 0
    }
};

// ==================== NEURAL NETWORK ====================

class NeuralNetwork {
    /**
     * @param {number[]} layerSizes - e.g. [2, 8, 8, 1]
     * @param {string} activationType - 'tanh', 'relu', or 'sigmoid'
     */
    constructor(layerSizes, activationType = 'tanh') {
        this.layerSizes = layerSizes;
        this.activation = Activations[activationType];
        this.activationType = activationType;
        this.weights = [];
        this.biases = [];
        this.weightGrads = [];
        this.biasGrads = [];
        this.layerOutputs = [];
        this.layerRawInputs = [];
        this._initWeights();
    }

    /** Xavier initialization for weights, small random for biases. */
    _initWeights() {
        this.weights = [];
        this.biases = [];
        for (let l = 0; l < this.layerSizes.length - 1; l++) {
            const fanIn = this.layerSizes[l];
            const fanOut = this.layerSizes[l + 1];
            const scale = Math.sqrt(2.0 / (fanIn + fanOut));

            const w = [];
            for (let j = 0; j < fanOut; j++) {
                const row = [];
                for (let k = 0; k < fanIn; k++) {
                    row.push((Math.random() * 2 - 1) * scale);
                }
                w.push(row);
            }
            this.weights.push(w);

            const b = [];
            for (let j = 0; j < fanOut; j++) {
                b.push((Math.random() * 2 - 1) * 0.1);
            }
            this.biases.push(b);
        }
    }

    /** Re-initialize all weights (for reset). */
    reset() {
        this._initWeights();
    }

    /** Forward pass. Stores intermediate values for backprop. */
    forward(input) {
        this.layerOutputs = [input.slice()];
        this.layerRawInputs = [];

        let current = input.slice();

        for (let l = 0; l < this.weights.length; l++) {
            const w = this.weights[l];
            const b = this.biases[l];
            const rawInputs = [];
            const nextOutput = [];
            const isOutputLayer = (l === this.weights.length - 1);

            for (let j = 0; j < w.length; j++) {
                let sum = b[j];
                for (let k = 0; k < current.length; k++) {
                    sum += w[j][k] * current[k];
                }
                rawInputs.push(sum);

                // Output layer always uses sigmoid for binary classification
                if (isOutputLayer) {
                    nextOutput.push(Activations.sigmoid.f(sum));
                } else {
                    nextOutput.push(this.activation.f(sum));
                }
            }

            this.layerRawInputs.push(rawInputs);
            current = nextOutput;
            this.layerOutputs.push(current.slice());
        }

        return current;
    }

    /** Backward pass. Computes gradients for all weights and biases. */
    backward(target) {
        const numWeightLayers = this.weights.length;
        const targetArr = typeof target === 'number' ? [target] : target;

        // Initialize gradient arrays
        this.weightGrads = [];
        this.biasGrads = [];
        for (let l = 0; l < numWeightLayers; l++) {
            this.weightGrads.push(
                this.weights[l].map(row => new Array(row.length).fill(0))
            );
            this.biasGrads.push(
                new Array(this.biases[l].length).fill(0)
            );
        }

        // Output layer delta: (output - target) for BCE + sigmoid
        const output = this.layerOutputs[numWeightLayers];
        let deltas = [];
        for (let j = 0; j < output.length; j++) {
            deltas.push(output[j] - targetArr[j]);
        }

        // Propagate backwards through each weight layer
        for (let l = numWeightLayers - 1; l >= 0; l--) {
            const prevLayerOutput = this.layerOutputs[l];

            // Compute weight and bias gradients for this layer
            for (let j = 0; j < deltas.length; j++) {
                this.biasGrads[l][j] = deltas[j];
                for (let k = 0; k < prevLayerOutput.length; k++) {
                    this.weightGrads[l][j][k] = deltas[j] * prevLayerOutput[k];
                }
            }

            // Compute deltas for the previous layer (if not input layer)
            if (l > 0) {
                const newDeltas = [];
                const prevSize = this.layerSizes[l];
                for (let k = 0; k < prevSize; k++) {
                    let error = 0;
                    for (let j = 0; j < deltas.length; j++) {
                        error += deltas[j] * this.weights[l][j][k];
                    }
                    // Multiply by activation derivative of the previous layer
                    const raw = this.layerRawInputs[l - 1][k];
                    const activated = this.layerOutputs[l][k];
                    error *= this.activation.df(raw, activated);
                    newDeltas.push(error);
                }
                deltas = newDeltas;
            }
        }
    }

    /** Total number of trainable parameters. */
    getParamCount() {
        let count = 0;
        for (let l = 0; l < this.weights.length; l++) {
            count += this.weights[l].length * this.weights[l][0].length;
            count += this.biases[l].length;
        }
        return count;
    }

    /** Flatten all gradients into a single array (for Optimizer.step()). */
    flattenGradients() {
        const grads = [];
        for (let l = 0; l < this.weightGrads.length; l++) {
            for (let j = 0; j < this.weightGrads[l].length; j++) {
                for (let k = 0; k < this.weightGrads[l][j].length; k++) {
                    grads.push(this.weightGrads[l][j][k]);
                }
            }
            for (let j = 0; j < this.biasGrads[l].length; j++) {
                grads.push(this.biasGrads[l][j]);
            }
        }
        return grads;
    }

    /** Apply flat array of deltas to all weights and biases. */
    applyDeltas(deltas) {
        let idx = 0;
        for (let l = 0; l < this.weights.length; l++) {
            for (let j = 0; j < this.weights[l].length; j++) {
                for (let k = 0; k < this.weights[l][j].length; k++) {
                    this.weights[l][j][k] += deltas[idx++];
                }
            }
            for (let j = 0; j < this.biases[l].length; j++) {
                this.biases[l][j] += deltas[idx++];
            }
        }
    }

    /**
     * Train one epoch on the given data with the given optimizer.
     * Uses online (stochastic) updates — one sample at a time.
     * @returns {{ loss: number, accuracy: number }}
     */
    trainEpoch(inputs, targets, optimizer) {
        let totalLoss = 0;
        let correct = 0;

        // Shuffle indices for stochastic training
        const indices = Array.from({ length: inputs.length }, (_, i) => i);
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }

        for (const idx of indices) {
            const output = this.forward(inputs[idx]);
            const t = typeof targets[idx] === 'number' ? targets[idx] : targets[idx][0];
            const o = output[0];

            // Binary cross-entropy loss
            const loss = -(t * Math.log(o + 1e-10) + (1 - t) * Math.log(1 - o + 1e-10));
            totalLoss += loss;

            // Accuracy
            if ((o >= 0.5 && t === 1) || (o < 0.5 && t === 0)) correct++;

            // Backward pass
            this.backward(targets[idx]);

            // Flatten gradients, get optimizer deltas, apply
            const grads = this.flattenGradients();
            const deltas = optimizer.step(grads);
            this.applyDeltas(deltas);
        }

        return {
            loss: totalLoss / inputs.length,
            accuracy: correct / inputs.length
        };
    }

    /** Predict class label (0 or 1). */
    predict(input) {
        const output = this.forward(input);
        return output[0] >= 0.5 ? 1 : 0;
    }

    /** Get raw output probability [0, 1]. */
    predictRaw(input) {
        const output = this.forward(input);
        return output[0];
    }
}

// ==================== DATASET GENERATORS ====================

const Datasets = {

    /** XOR pattern with Gaussian noise. */
    xor(n = 200) {
        const inputs = [];
        const targets = [];
        for (let i = 0; i < n; i++) {
            const qx = Math.random() < 0.5 ? -1 : 1;
            const qy = Math.random() < 0.5 ? -1 : 1;
            const x = qx * (0.3 + Math.random() * 0.7) + (Math.random() - 0.5) * 0.15;
            const y = qy * (0.3 + Math.random() * 0.7) + (Math.random() - 0.5) * 0.15;
            const cls = (qx * qy < 0) ? 1 : 0;
            inputs.push([x, y]);
            targets.push(cls);
        }
        return { inputs, targets, name: 'XOR', range: [-1.8, 1.8] };
    },

    /** Two interleaved spirals. */
    spiral(n = 300) {
        const inputs = [];
        const targets = [];
        const half = Math.floor(n / 2);
        for (let cls = 0; cls < 2; cls++) {
            for (let i = 0; i < half; i++) {
                const r = (i / half) * 1.2;
                const angle = (i / half) * 2.5 * Math.PI + cls * Math.PI;
                const noise = 0.12;
                const x = r * Math.cos(angle) + (Math.random() - 0.5) * noise;
                const y = r * Math.sin(angle) + (Math.random() - 0.5) * noise;
                inputs.push([x, y]);
                targets.push(cls);
            }
        }
        return { inputs, targets, name: 'Spiral', range: [-1.8, 1.8] };
    },

    /** Concentric circles (inner vs outer ring). */
    circle(n = 300) {
        const inputs = [];
        const targets = [];
        for (let i = 0; i < n; i++) {
            const angle = Math.random() * 2 * Math.PI;
            const inner = Math.random() < 0.5;
            const r = inner
                ? Math.random() * 0.45
                : 0.75 + Math.random() * 0.4;
            const x = r * Math.cos(angle) + (Math.random() - 0.5) * 0.08;
            const y = r * Math.sin(angle) + (Math.random() - 0.5) * 0.08;
            inputs.push([x, y]);
            targets.push(inner ? 0 : 1);
        }
        return { inputs, targets, name: 'Circle', range: [-1.8, 1.8] };
    },

    /** Two Gaussian clusters. */
    clusters(n = 300) {
        const inputs = [];
        const targets = [];
        const centers = [[0.6, 0.6], [-0.6, -0.6]];
        for (let i = 0; i < n; i++) {
            const cls = Math.floor(Math.random() * 2);
            const cx = centers[cls][0];
            const cy = centers[cls][1];
            // Box-Muller for Gaussian noise
            const u1 = Math.random() || 1e-10;
            const u2 = Math.random();
            const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
            const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);
            const x = cx + z0 * 0.35;
            const y = cy + z1 * 0.35;
            inputs.push([x, y]);
            targets.push(cls);
        }
        return { inputs, targets, name: 'Clusters', range: [-1.8, 1.8] };
    }
};
