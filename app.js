/**
 * app.js — Main Controller for ML Optimizer Visualizer
 *
 * Manages:
 *   - Tab navigation
 *   - Landscape Playground: contour rendering, optimizer paths, metrics
 *   - Neural Network Playground: training, boundary, architecture, charts
 *   - Lab Tour: preset loading
 */

(function () {
    'use strict';

    // ==================== COLORMAP ====================

    /** Viridis-inspired colormap. t in [0, 1] → [r, g, b] */
    const VIRIDIS = [
        [68, 1, 84], [72, 26, 108], [71, 47, 126], [65, 68, 135],
        [57, 86, 140], [46, 105, 142], [35, 125, 142], [25, 146, 139],
        [34, 168, 132], [62, 188, 115], [109, 205, 89], [161, 218, 56],
        [210, 226, 27], [253, 231, 37]
    ];

    function viridis(t) {
        t = Math.max(0, Math.min(1, t));
        const idx = t * (VIRIDIS.length - 1);
        const i = Math.floor(idx);
        const f = idx - i;
        const c0 = VIRIDIS[Math.min(i, VIRIDIS.length - 1)];
        const c1 = VIRIDIS[Math.min(i + 1, VIRIDIS.length - 1)];
        return [
            Math.round(c0[0] + (c1[0] - c0[0]) * f),
            Math.round(c0[1] + (c1[1] - c0[1]) * f),
            Math.round(c0[2] + (c1[2] - c0[2]) * f)
        ];
    }

    // ==================== APP STATE ====================

    const App = {
        currentTab: 'landscape',

        // --- Landscape State ---
        ls: {
            canvas: null, ctx: null,
            fnKey: 'rosenbrock',
            startPoint: null,
            optimizers: {},         // { type: { optimizer, path, active, converged, diverged } }
            running: false,
            animId: null,
            stepCount: 0,
            maxSteps: 1500,
            speed: 3,
            lr: 0.01,
            contourImgData: null,
            contourValues: null,
            contourMin: 0,
            contourMax: 1,
        },

        // --- Network State ---
        nn: {
            net: null,
            dataset: null,
            optimizer: null,
            optType: 'adam',
            lr: 0.03,
            running: false,
            animId: null,
            epoch: 0,
            lossHistory: [],
            accHistory: [],
            epochsPerFrame: 5,
            hiddenConfig: '8',
            activation: 'tanh',
        },

        // ==================== INITIALIZATION ====================

        init() {
            this.ls.canvas = document.getElementById('landscape-canvas');
            this.ls.ctx = this.ls.canvas.getContext('2d');

            this.initTabs();
            this.initLandscapeControls();
            this.initNetworkControls();
            this.initTour();

            // Init diagnostics tab (from diagnostics-ui.js)
            if (typeof window.initDiagnostics === 'function') {
                window.initDiagnostics();
            }

            // Initial renders
            this.renderContour();
            this.initNetworkPlayground();
        },

        // ==================== TABS ====================

        initTabs() {
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    const tab = btn.dataset.tab;
                    this.switchTab(tab);
                });
            });
        },

        switchTab(tab) {
            this.currentTab = tab;
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
            document.querySelectorAll('.tab-content').forEach(s => s.classList.toggle('active', s.id === `section-${tab}`));
        },

        // ==================== LANDSCAPE: CONTROLS ====================

        initLandscapeControls() {
            const $ = id => document.getElementById(id);

            // Function selector
            $('landscape-function').addEventListener('change', e => {
                this.ls.fnKey = e.target.value;
                this.resetLandscape();
                this.renderContour();
                const fn = LOSS_FUNCTIONS[this.ls.fnKey];
                $('landscape-fn-title').textContent = fn.name + ' Function';
                $('landscape-fn-desc').textContent = fn.description;
            });

            // Learning rate
            $('landscape-lr').addEventListener('input', e => {
                this.ls.lr = Math.pow(10, parseFloat(e.target.value));
                $('lr-value').textContent = this.ls.lr.toFixed(4);
            });

            // Speed
            $('landscape-speed').addEventListener('input', e => {
                this.ls.speed = parseInt(e.target.value);
                $('speed-value').textContent = this.ls.speed;
            });

            // Max steps
            $('landscape-maxsteps').addEventListener('input', e => {
                this.ls.maxSteps = parseInt(e.target.value);
                $('maxsteps-value').textContent = this.ls.maxSteps;
            });

            // Canvas click — set start point
            this.ls.canvas.addEventListener('click', e => this.handleCanvasClick(e));

            // Mousemove — show coordinates
            this.ls.canvas.addEventListener('mousemove', e => {
                const fn = LOSS_FUNCTIONS[this.ls.fnKey];
                const rect = this.ls.canvas.getBoundingClientRect();
                const scaleX = this.ls.canvas.width / rect.width;
                const scaleY = this.ls.canvas.height / rect.height;
                const px = (e.clientX - rect.left) * scaleX;
                const py = (e.clientY - rect.top) * scaleY;
                const coord = this.pixelToCoord(px, py, fn);
                const val = fn.f(coord.x, coord.y);
                $('coord-display').textContent = `(${coord.x.toFixed(2)}, ${coord.y.toFixed(2)}) = ${val.toFixed(4)}`;
            });

            // Buttons
            $('btn-run').addEventListener('click', () => this.runLandscape());
            $('btn-pause').addEventListener('click', () => this.pauseLandscape());
            $('btn-step').addEventListener('click', () => this.stepLandscapeOnce());
            $('btn-reset').addEventListener('click', () => {
                this.resetLandscape();
                this.renderContour();
            });
        },

        // ==================== LANDSCAPE: COORDINATE TRANSFORMS ====================

        coordToPixel(x, y, fn) {
            const w = this.ls.canvas.width;
            const h = this.ls.canvas.height;
            const px = ((x - fn.bounds.x[0]) / (fn.bounds.x[1] - fn.bounds.x[0])) * w;
            const py = h - ((y - fn.bounds.y[0]) / (fn.bounds.y[1] - fn.bounds.y[0])) * h;
            return { x: px, y: py };
        },

        pixelToCoord(px, py, fn) {
            const w = this.ls.canvas.width;
            const h = this.ls.canvas.height;
            const x = fn.bounds.x[0] + (px / w) * (fn.bounds.x[1] - fn.bounds.x[0]);
            const y = fn.bounds.y[0] + ((h - py) / h) * (fn.bounds.y[1] - fn.bounds.y[0]);
            return { x, y };
        },

        // ==================== LANDSCAPE: CONTOUR RENDERING ====================

        renderContour() {
            const canvas = this.ls.canvas;
            const ctx = this.ls.ctx;
            const fn = LOSS_FUNCTIONS[this.ls.fnKey];
            const w = canvas.width;
            const h = canvas.height;

            const imageData = ctx.createImageData(w, h);
            const data = imageData.data;
            const values = new Float32Array(w * h);
            let minVal = Infinity, maxVal = -Infinity;

            // Compute function values
            for (let py = 0; py < h; py++) {
                for (let px = 0; px < w; px++) {
                    const coord = this.pixelToCoord(px, py, fn);
                    let v = fn.f(coord.x, coord.y);
                    v = Math.log1p(Math.abs(v)); // Log scale
                    const idx = py * w + px;
                    values[idx] = v;
                    if (v < minVal) minVal = v;
                    if (v > maxVal) maxVal = v;
                }
            }

            this.ls.contourValues = values;
            this.ls.contourMin = minVal;
            this.ls.contourMax = maxVal;

            const range = maxVal - minVal || 1;

            // Map to colors + add contour lines
            for (let py = 0; py < h; py++) {
                for (let px = 0; px < w; px++) {
                    const idx = py * w + px;
                    const t = (values[idx] - minVal) / range;
                    let [r, g, b] = viridis(t);

                    // Edge detection for contour lines
                    if (px > 0 && py > 0 && px < w - 1 && py < h - 1) {
                        const diff = Math.abs(values[idx + 1] - values[idx]) +
                                     Math.abs(values[idx - 1] - values[idx]) +
                                     Math.abs(values[(py + 1) * w + px] - values[idx]) +
                                     Math.abs(values[(py - 1) * w + px] - values[idx]);
                        if (diff > range * 0.04) {
                            r = Math.max(0, r - 25);
                            g = Math.max(0, g - 25);
                            b = Math.max(0, b - 25);
                        }
                    }

                    const i4 = idx * 4;
                    data[i4] = r;
                    data[i4 + 1] = g;
                    data[i4 + 2] = b;
                    data[i4 + 3] = 255;
                }
            }

            ctx.putImageData(imageData, 0, 0);
            this.ls.contourImgData = ctx.getImageData(0, 0, w, h);

            // Draw minimum marker
            const minPx = this.coordToPixel(fn.minimum[0], fn.minimum[1], fn);
            ctx.beginPath();
            ctx.arc(minPx.x, minPx.y, 8, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
            ctx.lineWidth = 2;
            ctx.setLineDash([3, 3]);
            ctx.stroke();
            ctx.setLineDash([]);

            // Star marker at minimum
            ctx.font = '16px serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
            ctx.fillText('★', minPx.x, minPx.y);

            // Save with minimum marker
            this.ls.contourImgData = ctx.getImageData(0, 0, w, h);
        },

        // ==================== LANDSCAPE: CLICK HANDLER ====================

        handleCanvasClick(e) {
            const fn = LOSS_FUNCTIONS[this.ls.fnKey];
            const rect = this.ls.canvas.getBoundingClientRect();
            const scaleX = this.ls.canvas.width / rect.width;
            const scaleY = this.ls.canvas.height / rect.height;
            const px = (e.clientX - rect.left) * scaleX;
            const py = (e.clientY - rect.top) * scaleY;
            const coord = this.pixelToCoord(px, py, fn);

            this.ls.startPoint = [coord.x, coord.y];
            this.setupLandscapeOptimizers();
            this.drawLandscape();

            // Enable run buttons
            document.getElementById('btn-run').disabled = false;
            document.getElementById('btn-step').disabled = false;
            document.getElementById('canvas-hint').classList.add('hidden');
        },

        // ==================== LANDSCAPE: OPTIMIZER SETUP ====================

        setupLandscapeOptimizers() {
            this.ls.optimizers = {};
            this.ls.stepCount = 0;

            const types = ['sgd', 'momentum', 'adagrad', 'rmsprop', 'adam'];
            for (const type of types) {
                const checkbox = document.getElementById(`opt-${type}`);
                if (checkbox && checkbox.checked) {
                    this.ls.optimizers[type] = {
                        optimizer: createOptimizer(type, this.ls.lr),
                        path: [this.ls.startPoint.slice()],
                        active: true,
                        converged: false,
                        diverged: false,
                        pathLength: 0,
                    };
                }
            }
            this.updateMetrics();
        },

        // ==================== LANDSCAPE: DRAWING ====================

        drawLandscape() {
            const ctx = this.ls.ctx;
            const fn = LOSS_FUNCTIONS[this.ls.fnKey];

            // Redraw contour base
            if (this.ls.contourImgData) {
                ctx.putImageData(this.ls.contourImgData, 0, 0);
            }

            // Draw optimizer paths
            for (const [type, data] of Object.entries(this.ls.optimizers)) {
                if (data.path.length < 1) continue;
                const color = OPTIMIZER_INFO[type].color;
                const rgb = OPTIMIZER_INFO[type].colorRgb;

                // Path line with gradient opacity
                if (data.path.length >= 2) {
                    for (let i = 1; i < data.path.length; i++) {
                        const alpha = 0.3 + 0.7 * (i / data.path.length);
                        const from = this.coordToPixel(data.path[i - 1][0], data.path[i - 1][1], fn);
                        const to = this.coordToPixel(data.path[i][0], data.path[i][1], fn);
                        ctx.beginPath();
                        ctx.moveTo(from.x, from.y);
                        ctx.lineTo(to.x, to.y);
                        ctx.strokeStyle = `rgba(${rgb}, ${alpha})`;
                        ctx.lineWidth = 2;
                        ctx.stroke();
                    }
                }

                // Current position dot
                const last = data.path[data.path.length - 1];
                const lastPx = this.coordToPixel(last[0], last[1], fn);

                // Glow
                ctx.beginPath();
                ctx.arc(lastPx.x, lastPx.y, 10, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(${rgb}, 0.15)`;
                ctx.fill();

                // Dot
                ctx.beginPath();
                ctx.arc(lastPx.x, lastPx.y, 5, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
                ctx.strokeStyle = 'rgba(255,255,255,0.8)';
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }

            // Draw start point
            if (this.ls.startPoint) {
                const sp = this.coordToPixel(this.ls.startPoint[0], this.ls.startPoint[1], fn);
                ctx.beginPath();
                ctx.arc(sp.x, sp.y, 7, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 2;
                ctx.stroke();

                // Cross
                ctx.beginPath();
                ctx.moveTo(sp.x - 4, sp.y); ctx.lineTo(sp.x + 4, sp.y);
                ctx.moveTo(sp.x, sp.y - 4); ctx.lineTo(sp.x, sp.y + 4);
                ctx.strokeStyle = '#000';
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }
        },

        // ==================== LANDSCAPE: STEP ====================

        landscapeStep() {
            const fn = LOSS_FUNCTIONS[this.ls.fnKey];
            let anyActive = false;

            for (const [type, data] of Object.entries(this.ls.optimizers)) {
                if (!data.active || data.converged || data.diverged) continue;

                const pos = data.path[data.path.length - 1];
                const grad = fn.grad(pos[0], pos[1]);
                const gradMag = Math.sqrt(grad[0] * grad[0] + grad[1] * grad[1]);

                // Check convergence
                if (gradMag < 1e-6) {
                    data.converged = true;
                    continue;
                }

                // Compute update
                const deltas = data.optimizer.step(grad);
                let nx = pos[0] + deltas[0];
                let ny = pos[1] + deltas[1];

                // Check divergence (NaN or out of bounds)
                if (!isFinite(nx) || !isFinite(ny) ||
                    Math.abs(nx) > 1e6 || Math.abs(ny) > 1e6) {
                    data.diverged = true;
                    continue;
                }

                // Clamp to bounds
                nx = Math.max(fn.bounds.x[0], Math.min(fn.bounds.x[1], nx));
                ny = Math.max(fn.bounds.y[0], Math.min(fn.bounds.y[1], ny));

                // Update path
                const dx = nx - pos[0];
                const dy = ny - pos[1];
                data.pathLength += Math.sqrt(dx * dx + dy * dy);
                data.path.push([nx, ny]);

                anyActive = true;
            }

            this.ls.stepCount++;
            document.getElementById('step-display').textContent = `Step: ${this.ls.stepCount}`;

            if (!anyActive || this.ls.stepCount >= this.ls.maxSteps) {
                this.pauseLandscape();
            }
        },

        stepLandscapeOnce() {
            if (!this.ls.startPoint) return;
            this.landscapeStep();
            this.drawLandscape();
            this.updateMetrics();
        },

        // ==================== LANDSCAPE: ANIMATION ====================

        runLandscape() {
            if (this.ls.running || !this.ls.startPoint) return;
            this.ls.running = true;
            document.getElementById('btn-run').disabled = true;
            document.getElementById('btn-pause').disabled = false;

            const animate = () => {
                if (!this.ls.running) return;

                for (let i = 0; i < this.ls.speed; i++) {
                    this.landscapeStep();
                }
                this.drawLandscape();
                this.updateMetrics();

                if (this.ls.running) {
                    this.ls.animId = requestAnimationFrame(animate);
                }
            };
            this.ls.animId = requestAnimationFrame(animate);
        },

        pauseLandscape() {
            this.ls.running = false;
            if (this.ls.animId) {
                cancelAnimationFrame(this.ls.animId);
                this.ls.animId = null;
            }
            document.getElementById('btn-run').disabled = false;
            document.getElementById('btn-pause').disabled = true;
        },

        resetLandscape() {
            this.pauseLandscape();
            this.ls.startPoint = null;
            this.ls.optimizers = {};
            this.ls.stepCount = 0;
            document.getElementById('btn-run').disabled = true;
            document.getElementById('btn-step').disabled = true;
            document.getElementById('btn-pause').disabled = true;
            document.getElementById('canvas-hint').classList.remove('hidden');
            document.getElementById('step-display').textContent = 'Step: 0';
            document.getElementById('coord-display').textContent = 'Position: —';
            document.getElementById('metrics-table-body').innerHTML = '';
        },

        // ==================== LANDSCAPE: METRICS ====================

        updateMetrics() {
            const tbody = document.getElementById('metrics-table-body');
            const fn = LOSS_FUNCTIONS[this.ls.fnKey];
            let html = '';

            for (const [type, data] of Object.entries(this.ls.optimizers)) {
                const info = OPTIMIZER_INFO[type];
                const last = data.path[data.path.length - 1];
                const fVal = fn.f(last[0], last[1]);
                const grad = fn.grad(last[0], last[1]);
                const gradMag = Math.sqrt(grad[0] * grad[0] + grad[1] * grad[1]);

                let status, statusClass;
                if (data.diverged) { status = 'Diverged'; statusClass = 'status-diverged'; }
                else if (data.converged) { status = 'Converged'; statusClass = 'status-converged'; }
                else if (gradMag < 0.01 && data.path.length > 10) { status = 'Near-Converged'; statusClass = 'status-converged'; }
                else if (this.ls.running) { status = 'Running'; statusClass = 'status-running'; }
                else { status = 'Paused'; statusClass = 'status-stuck'; }

                html += `<tr>
                    <td><span class="opt-name-cell"><span class="opt-name-dot" style="background:${info.color}"></span>${info.name}</span></td>
                    <td>${data.path.length - 1}</td>
                    <td>${fVal < 1e6 ? fVal.toFixed(6) : fVal.toExponential(2)}</td>
                    <td>${data.pathLength.toFixed(3)}</td>
                    <td>${gradMag.toFixed(6)}</td>
                    <td><span class="${statusClass}">${status}</span></td>
                </tr>`;
            }
            tbody.innerHTML = html;
        },

        // ==================== NETWORK: CONTROLS ====================

        initNetworkControls() {
            const $ = id => document.getElementById(id);

            $('network-dataset').addEventListener('change', () => this.resetNetwork());
            $('network-optimizer').addEventListener('change', e => {
                this.nn.optType = e.target.value;
                if (this.nn.running) {
                    this.stopNetworkTraining();
                    this.nn.optimizer = createOptimizer(this.nn.optType, this.nn.lr);
                    this.trainNetwork();
                }
            });

            $('network-lr').addEventListener('input', e => {
                this.nn.lr = Math.pow(10, parseFloat(e.target.value));
                $('net-lr-value').textContent = this.nn.lr.toFixed(4);
            });

            $('network-hidden').addEventListener('change', () => this.resetNetwork());
            $('network-activation').addEventListener('change', e => {
                this.nn.activation = e.target.value;
                this.resetNetwork();
            });

            $('network-epochs-per-frame').addEventListener('input', e => {
                this.nn.epochsPerFrame = parseInt(e.target.value);
                $('epf-value').textContent = this.nn.epochsPerFrame;
            });

            $('btn-train').addEventListener('click', () => this.trainNetwork());
            $('btn-stop-train').addEventListener('click', () => this.stopNetworkTraining());
            $('btn-reset-network').addEventListener('click', () => this.resetNetwork());
        },

        // ==================== NETWORK: INITIALIZATION ====================

        initNetworkPlayground() {
            this.generateDataset();
            this.createNetwork();
            this.renderBoundary();
            this.renderNetworkArch();
            this.renderChart();
            this.updateNetworkStats();
        },

        generateDataset() {
            const dsType = document.getElementById('network-dataset').value;
            this.nn.dataset = Datasets[dsType](300);
        },

        createNetwork() {
            const hiddenStr = document.getElementById('network-hidden').value;
            const hiddenLayers = hiddenStr.split(',').map(Number);
            const layerSizes = [2, ...hiddenLayers, 1];
            this.nn.net = new NeuralNetwork(layerSizes, this.nn.activation);
            this.nn.optimizer = createOptimizer(this.nn.optType, this.nn.lr);
            document.getElementById('stat-params').textContent = this.nn.net.getParamCount();
        },

        // ==================== NETWORK: DECISION BOUNDARY ====================

        renderBoundary() {
            const canvas = document.getElementById('boundary-canvas');
            const ctx = canvas.getContext('2d');
            const w = canvas.width;
            const h = canvas.height;
            const ds = this.nn.dataset;
            if (!ds || !this.nn.net) return;

            const range = ds.range || [-1.8, 1.8];
            const imageData = ctx.createImageData(w, h);
            const data = imageData.data;

            // Render decision regions
            for (let py = 0; py < h; py++) {
                for (let px = 0; px < w; px++) {
                    const x = range[0] + (px / w) * (range[1] - range[0]);
                    const y = range[0] + ((h - 1 - py) / h) * (range[1] - range[0]);
                    const prob = this.nn.net.predictRaw([x, y]);
                    const i4 = (py * w + px) * 4;

                    // Class 0: deep blue, Class 1: orange, blend by probability
                    const r = Math.round(prob * 220 + (1 - prob) * 30);
                    const g = Math.round(prob * 120 + (1 - prob) * 50);
                    const b = Math.round(prob * 50 + (1 - prob) * 180);

                    data[i4] = r;
                    data[i4 + 1] = g;
                    data[i4 + 2] = b;
                    data[i4 + 3] = 160;
                }
            }

            ctx.putImageData(imageData, 0, 0);

            // Draw data points
            for (let i = 0; i < ds.inputs.length; i++) {
                const x = ds.inputs[i][0];
                const y = ds.inputs[i][1];
                const cls = ds.targets[i];

                const px = ((x - range[0]) / (range[1] - range[0])) * w;
                const py = h - ((y - range[0]) / (range[1] - range[0])) * h;

                ctx.beginPath();
                ctx.arc(px, py, 3, 0, Math.PI * 2);
                ctx.fillStyle = cls === 1 ? '#ff8c42' : '#4dabf7';
                ctx.fill();
                ctx.strokeStyle = 'rgba(0,0,0,0.5)';
                ctx.lineWidth = 0.5;
                ctx.stroke();
            }
        },

        // ==================== NETWORK: ARCHITECTURE VISUALIZATION ====================

        renderNetworkArch() {
            const canvas = document.getElementById('network-arch-canvas');
            const ctx = canvas.getContext('2d');
            const w = canvas.width;
            const h = canvas.height;
            ctx.clearRect(0, 0, w, h);

            const net = this.nn.net;
            if (!net) return;

            const layers = net.layerSizes;
            const numLayers = layers.length;
            const padX = 50;
            const padY = 25;
            const layerSpacing = (w - 2 * padX) / (numLayers - 1);

            // Compute node positions
            const nodePos = [];
            for (let l = 0; l < numLayers; l++) {
                const x = padX + l * layerSpacing;
                const n = layers[l];
                const availH = h - 2 * padY;
                const nodeGap = Math.min(30, availH / (n + 1));
                const totalH = (n - 1) * nodeGap;
                const startY = (h - totalH) / 2;
                const positions = [];
                for (let i = 0; i < n; i++) {
                    positions.push({ x, y: startY + i * nodeGap });
                }
                nodePos.push(positions);
            }

            // Draw connections with weight-based coloring
            for (let l = 0; l < net.weights.length; l++) {
                for (let j = 0; j < net.weights[l].length; j++) {
                    for (let k = 0; k < net.weights[l][j].length; k++) {
                        const weight = net.weights[l][j][k];
                        const from = nodePos[l][k];
                        const to = nodePos[l + 1][j];

                        const intensity = Math.min(1, Math.abs(weight) / 2);
                        const alpha = 0.08 + intensity * 0.6;
                        const lineW = 0.3 + intensity * 2.5;

                        ctx.beginPath();
                        ctx.moveTo(from.x, from.y);
                        ctx.lineTo(to.x, to.y);
                        ctx.strokeStyle = weight > 0
                            ? `rgba(59, 130, 246, ${alpha})`
                            : `rgba(239, 68, 68, ${alpha})`;
                        ctx.lineWidth = lineW;
                        ctx.stroke();
                    }
                }
            }

            // Draw nodes
            for (let l = 0; l < numLayers; l++) {
                for (const pos of nodePos[l]) {
                    // Glow
                    ctx.beginPath();
                    ctx.arc(pos.x, pos.y, 12, 0, Math.PI * 2);
                    const gradient = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, 12);
                    gradient.addColorStop(0, 'rgba(34, 211, 238, 0.15)');
                    gradient.addColorStop(1, 'rgba(34, 211, 238, 0)');
                    ctx.fillStyle = gradient;
                    ctx.fill();

                    // Node
                    ctx.beginPath();
                    ctx.arc(pos.x, pos.y, 8, 0, Math.PI * 2);
                    ctx.fillStyle = '#0d0f1a';
                    ctx.fill();
                    ctx.strokeStyle = 'rgba(34, 211, 238, 0.6)';
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                }
            }

            // Layer labels
            const labels = [];
            for (let l = 0; l < numLayers; l++) {
                if (l === 0) labels.push('Input');
                else if (l === numLayers - 1) labels.push('Output');
                else labels.push(`H${l}`);
            }
            ctx.font = '10px Inter, sans-serif';
            ctx.fillStyle = '#64748b';
            ctx.textAlign = 'center';
            for (let l = 0; l < numLayers; l++) {
                ctx.fillText(`${labels[l]} (${layers[l]})`, nodePos[l][0].x, h - 8);
            }
        },

        // ==================== NETWORK: CHART ====================

        renderChart() {
            const canvas = document.getElementById('chart-canvas');
            const ctx = canvas.getContext('2d');
            const w = canvas.width;
            const h = canvas.height;
            ctx.clearRect(0, 0, w, h);

            const padL = 50, padR = 15, padT = 10, padB = 25;
            const plotW = w - padL - padR;
            const plotH = h - padT - padB;

            // Background grid
            ctx.strokeStyle = 'rgba(255,255,255,0.05)';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 4; i++) {
                const y = padT + (i / 4) * plotH;
                ctx.beginPath();
                ctx.moveTo(padL, y);
                ctx.lineTo(w - padR, y);
                ctx.stroke();
            }

            const lossHistory = this.nn.lossHistory;
            const accHistory = this.nn.accHistory;
            if (lossHistory.length < 2) {
                ctx.font = '12px Inter, sans-serif';
                ctx.fillStyle = '#64748b';
                ctx.textAlign = 'center';
                ctx.fillText('Training curves will appear here', w / 2, h / 2);
                return;
            }

            const n = lossHistory.length;
            const maxLoss = Math.max(...lossHistory, 1);

            // Y-axis labels
            ctx.font = '9px Courier New, monospace';
            ctx.fillStyle = '#64748b';
            ctx.textAlign = 'right';
            for (let i = 0; i <= 4; i++) {
                const val = (1 - i / 4);
                ctx.fillText(val.toFixed(1), padL - 5, padT + (i / 4) * plotH + 3);
            }

            // X-axis label
            ctx.textAlign = 'center';
            ctx.fillText(`Epoch ${n}`, w / 2, h - 3);

            // Draw loss curve
            ctx.beginPath();
            for (let i = 0; i < n; i++) {
                const x = padL + (i / (n - 1)) * plotW;
                const y = padT + (1 - Math.min(lossHistory[i] / maxLoss, 1)) * plotH;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.strokeStyle = '#ef4444';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Draw accuracy curve
            ctx.beginPath();
            for (let i = 0; i < n; i++) {
                const x = padL + (i / (n - 1)) * plotW;
                const y = padT + (1 - accHistory[i]) * plotH;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.strokeStyle = '#22d3ee';
            ctx.lineWidth = 2;
            ctx.stroke();
        },

        // ==================== NETWORK: TRAINING ====================

        trainNetwork() {
            if (this.nn.running) return;
            this.nn.running = true;
            document.getElementById('btn-train').disabled = true;
            document.getElementById('btn-stop-train').disabled = false;

            if (!this.nn.optimizer || this.nn.optimizer.type !== this.nn.optType || this.nn.optimizer.lr !== this.nn.lr) {
                this.nn.optimizer = createOptimizer(this.nn.optType, this.nn.lr);
            }

            const animate = () => {
                if (!this.nn.running) return;

                for (let i = 0; i < this.nn.epochsPerFrame; i++) {
                    const result = this.nn.net.trainEpoch(
                        this.nn.dataset.inputs,
                        this.nn.dataset.targets,
                        this.nn.optimizer
                    );
                    this.nn.epoch++;
                    this.nn.lossHistory.push(result.loss);
                    this.nn.accHistory.push(result.accuracy);
                }

                this.renderBoundary();
                this.renderNetworkArch();
                this.renderChart();
                this.updateNetworkStats();

                if (this.nn.running) {
                    this.nn.animId = requestAnimationFrame(animate);
                }
            };
            this.nn.animId = requestAnimationFrame(animate);
        },

        stopNetworkTraining() {
            this.nn.running = false;
            if (this.nn.animId) {
                cancelAnimationFrame(this.nn.animId);
                this.nn.animId = null;
            }
            document.getElementById('btn-train').disabled = false;
            document.getElementById('btn-stop-train').disabled = true;
        },

        resetNetwork() {
            this.stopNetworkTraining();
            this.nn.epoch = 0;
            this.nn.lossHistory = [];
            this.nn.accHistory = [];
            this.generateDataset();
            this.createNetwork();
            this.renderBoundary();
            this.renderNetworkArch();
            this.renderChart();
            this.updateNetworkStats();
        },

        updateNetworkStats() {
            const loss = this.nn.lossHistory.length > 0 ? this.nn.lossHistory[this.nn.lossHistory.length - 1] : 0.693;
            const acc = this.nn.accHistory.length > 0 ? this.nn.accHistory[this.nn.accHistory.length - 1] : 0.5;
            document.getElementById('stat-epoch').textContent = this.nn.epoch;
            document.getElementById('stat-loss').textContent = loss.toFixed(4);
            document.getElementById('stat-accuracy').textContent = (acc * 100).toFixed(1) + '%';
        },

        // ==================== LAB TOUR ====================

        initTour() {
            const PRESETS = {
                valley: {
                    fn: 'rosenbrock',
                    start: [-1.5, 2.0],
                    lr: -2.5,      // 10^-2.5 ≈ 0.003
                    opts: ['sgd', 'adam'],
                    explanation: 'The Rosenbrock "Valley of Death" — a narrow banana-shaped valley. SGD uses a fixed learning rate and oscillates across the valley walls because the gradient direction doesn\'t align with the valley. Adam uses adaptive per-parameter learning rates and momentum to smoothly follow the curve. Notice how SGD takes many more steps and may not converge within the step limit, while Adam reaches (1,1) efficiently.'
                },
                minima: {
                    fn: 'rastrigin',
                    start: [4, 4],
                    lr: -2,        // 0.01
                    opts: ['sgd', 'momentum', 'adam'],
                    explanation: 'The Rastrigin function has many local minima (the "wells" you see). SGD without momentum gets trapped in the nearest local minimum because it can\'t build up enough velocity to escape the well. Momentum accumulates velocity over time, giving the optimizer inertia to push past small barriers. Adam combines momentum with adaptive learning rates, giving it the best chance of approaching the global minimum at (0,0).'
                },
                saddle: {
                    fn: 'saddle',
                    start: [0.5, 0.001],
                    lr: -1.5,      // ~0.03
                    opts: ['sgd', 'momentum'],
                    explanation: 'At the origin, the Saddle Point function has zero gradient — the surface is flat in all directions locally. SGD stalls because ∇f = 0, so -lr * ∇f = 0 — no update! Momentum remembers previous gradient directions and builds velocity, allowing it to "roll" through the saddle point. This is a critical problem in deep learning where saddle points are far more common than local minima in high-dimensional spaces.'
                },
                lr_high: {
                    fn: 'quadratic',
                    start: [3.5, 3.5],
                    lr: -0.3,      // ~0.5
                    opts: ['sgd', 'adam'],
                    explanation: 'With a learning rate of 0.5 on this simple bowl, SGD overshoots dramatically — each step jumps past the minimum and oscillates or diverges. The step size is too large relative to the curvature. Adam\'s adaptive learning rates automatically scale down large gradients, preventing explosive oscillation. This demonstrates why adaptive optimizers are preferred in practice — they\'re much more robust to learning rate choices.'
                },
                adam_vs_sgd: {
                    fn: 'beale',
                    start: [-3, 3],
                    lr: -2,        // 0.01
                    opts: ['sgd', 'adam'],
                    explanation: 'The Beale function has large flat plateaus where gradients are tiny, surrounding a sharp minimum at (3, 0.5). SGD with a fixed learning rate makes tiny steps in flat regions (gradient ≈ 0 → update ≈ 0). Adam\'s second moment (v_t) tracks the variance of gradients — in flat regions where gradients are consistently small, the denominator shrinks, effectively increasing the step size. This "adaptive" behavior lets Adam traverse plateaus much faster.'
                },
                race: {
                    fn: 'himmelblau',
                    start: [-4, 4],
                    lr: -2,        // 0.01
                    opts: ['sgd', 'momentum', 'adagrad', 'rmsprop', 'adam'],
                    explanation: 'All five optimizers start from the same point on Himmelblau\'s function (which has 4 identical minima). Watch how they each find potentially different minima depending on their trajectory. SGD follows the steepest descent. Momentum builds velocity. AdaGrad accumulates all past gradients (may slow down too much). RMSprop uses exponential decay to forget old gradients. Adam combines the best of momentum and RMSprop. Notice the differences in convergence speed and which minimum each finds.'
                }
            };

            document.querySelectorAll('.tour-card').forEach(card => {
                const launchBtn = card.querySelector('.tour-launch');
                if (launchBtn) {
                    launchBtn.addEventListener('click', () => {
                        const presetKey = card.dataset.preset;
                        const preset = PRESETS[presetKey];
                        if (!preset) return;

                        // Switch to landscape tab
                        this.switchTab('landscape');

                        // Configure landscape
                        this.resetLandscape();
                        this.ls.fnKey = preset.fn;
                        document.getElementById('landscape-function').value = preset.fn;
                        const fn = LOSS_FUNCTIONS[preset.fn];
                        document.getElementById('landscape-fn-title').textContent = fn.name + ' Function';
                        document.getElementById('landscape-fn-desc').textContent = fn.description;

                        // Set learning rate
                        document.getElementById('landscape-lr').value = preset.lr;
                        this.ls.lr = Math.pow(10, preset.lr);
                        document.getElementById('lr-value').textContent = this.ls.lr.toFixed(4);

                        // Set optimizer checkboxes
                        ['sgd', 'momentum', 'adagrad', 'rmsprop', 'adam'].forEach(type => {
                            document.getElementById(`opt-${type}`).checked = preset.opts.includes(type);
                        });

                        // Render contour
                        this.renderContour();

                        // Set start point
                        this.ls.startPoint = preset.start.slice();
                        this.setupLandscapeOptimizers();
                        this.drawLandscape();

                        // Show explanation
                        const explCard = document.getElementById('landscape-explanation');
                        document.getElementById('explanation-text').textContent = preset.explanation;
                        explCard.style.display = 'block';

                        // Enable and auto-run
                        document.getElementById('btn-run').disabled = false;
                        document.getElementById('btn-step').disabled = false;
                        document.getElementById('canvas-hint').classList.add('hidden');

                        setTimeout(() => this.runLandscape(), 300);
                    });
                }
            });
        },
    };

    // ==================== BOOT ====================

    document.addEventListener('DOMContentLoaded', () => App.init());

})();
