/**
 * diagnostics-ui.js — Real Model Diagnostics Tab
 * Handles code upload, dataset pasting, API calls, SSE streaming,
 * and real-time chart rendering for side-by-side training comparison.
 */

(function () {
    'use strict';

    const API_BASE = 'http://localhost:8000';

    // ─────────────────────────────────────────────
    //  STATE
    // ─────────────────────────────────────────────

    const DiagState = {
        sessionId: null,
        code: '',
        filename: 'model.py',
        datasetCsv: '',
        parsed: null,
        diagnostics: null,
        suggestion: null,
        // Training state
        training: false,
        eventSource: null,
        epochsA: [],   // { loss, acc }
        epochsB: [],
        totalEpochs: 30,
        currentEpoch: 0,
    };

    // ─────────────────────────────────────────────
    //  DEMO CODE SNIPPETS (same as optisuggest-v2)
    // ─────────────────────────────────────────────

    const DEMOS = {
        sgd_high_lr: {
            filename: 'classifier_sgd.py',
            code: `import torch
import torch.nn as nn
import torch.optim as optim

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

model = Classifier()
optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()`,
            dataset: `x1,x2,x3,x4,y
0.2,0.8,0.5,0.1,0
0.9,0.1,0.3,0.7,1
0.4,0.6,0.8,0.2,0
0.7,0.3,0.1,0.9,1
0.1,0.9,0.6,0.4,0
0.8,0.2,0.4,0.6,1
0.3,0.7,0.9,0.3,0
0.6,0.4,0.2,0.8,1
0.5,0.5,0.7,0.5,0
0.85,0.15,0.35,0.75,1
0.25,0.75,0.55,0.15,0
0.95,0.05,0.25,0.85,1
0.15,0.85,0.65,0.35,0
0.75,0.25,0.15,0.95,1
0.35,0.65,0.85,0.25,0
0.65,0.35,0.45,0.55,1`
        },
        sigmoid_deep: {
            filename: 'deep_sigmoid_net.py',
            code: `import torch.nn as nn
import torch.optim as optim

class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.Sigmoid(),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Linear(512, 256),
            nn.Sigmoid(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x)

model = DeepNet()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()`,
            dataset: `f1,f2,f3,f4,f5,label
0.1,0.9,0.4,0.6,0.2,0
0.8,0.2,0.7,0.3,0.9,1
0.3,0.7,0.5,0.5,0.4,0
0.9,0.1,0.8,0.2,0.7,1
0.2,0.8,0.3,0.7,0.1,0
0.7,0.3,0.6,0.4,0.8,1
0.4,0.6,0.2,0.8,0.3,0
0.6,0.4,0.9,0.1,0.6,1
0.15,0.85,0.35,0.65,0.25,0
0.85,0.15,0.75,0.25,0.85,1
0.25,0.75,0.45,0.55,0.35,0
0.75,0.25,0.65,0.35,0.75,1`
        },
        no_regularization: {
            filename: 'adam_no_reg.py',
            code: `import torch.nn as nn
import torch.optim as optim

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3072, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)

model = ImageClassifier()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()`,
            dataset: `a,b,c,d,target
1.2,3.4,5.6,7.8,0
2.3,4.5,6.7,8.9,1
3.4,5.6,7.8,9.0,0
4.5,6.7,8.9,1.2,1
5.6,7.8,9.0,2.3,0
6.7,8.9,1.2,3.4,1
7.8,9.0,2.3,4.5,0
8.9,1.2,3.4,5.6,1
1.1,3.3,5.5,7.7,0
2.2,4.4,6.6,8.8,1
3.3,5.5,7.7,9.9,0
4.4,6.6,8.8,1.1,1`
        },
        rnn_no_clip: {
            filename: 'lstm_classifier.py',
            code: `import torch
import torch.nn as nn
import torch.optim as optim

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=3, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.embed(x)
        out, (h, c) = self.lstm(x)
        return self.fc(h[-1])

model = SentimentLSTM(10000, 128, 256)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()`,
            dataset: `w1,w2,w3,w4,sentiment
0.5,0.8,0.3,0.1,0
0.9,0.2,0.7,0.6,1
0.1,0.6,0.4,0.8,0
0.7,0.4,0.9,0.3,1
0.3,0.9,0.2,0.5,0
0.8,0.1,0.6,0.7,1
0.2,0.7,0.5,0.9,0
0.6,0.3,0.8,0.4,1
0.4,0.5,0.1,0.2,0
0.95,0.15,0.75,0.65,1
0.15,0.65,0.35,0.85,0
0.75,0.35,0.85,0.25,1`
        }
    };

    // ─────────────────────────────────────────────
    //  PYTHON SYNTAX HIGHLIGHTER
    // ─────────────────────────────────────────────

    function highlightPython(code) {
        const keywords = ['import', 'from', 'class', 'def', 'return', 'for', 'in', 'if', 'else', 'elif', 'super', 'self', 'True', 'False', 'None'];
        let h = code
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        h = h.replace(/(["'])(?:(?=(\\?))\2.)*?\1/g, m => `<span class="diag-str">${m}</span>`);
        h = h.replace(/(#.*$)/gm, m => `<span class="diag-cm">${m}</span>`);
        h = h.replace(/\b(\d+\.?\d*(?:e[+-]?\d+)?)\b/g, m => `<span class="diag-num">${m}</span>`);
        h = h.replace(/\b([A-Z][a-zA-Z0-9]+)\b/g, m => `<span class="diag-cls">${m}</span>`);
        keywords.forEach(kw => {
            h = h.replace(new RegExp(`\\b${kw}\\b`, 'g'), `<span class="diag-kw">${kw}</span>`);
        });
        h = h.replace(/\b([a-z_][a-z0-9_]*)\s*(?=\()/g, m => `<span class="diag-fn">${m}</span>`);
        return h;
    }

    // ─────────────────────────────────────────────
    //  INITIALIZATION
    // ─────────────────────────────────────────────

    window.initDiagnostics = function () {
        // Tab switching (upload vs paste)
        const tabFile = document.getElementById('diag-tab-file');
        const tabPaste = document.getElementById('diag-tab-paste');
        if (tabFile) tabFile.addEventListener('click', () => switchDiagUploadTab('file'));
        if (tabPaste) tabPaste.addEventListener('click', () => switchDiagUploadTab('paste'));

        // Dropzone
        const dz = document.getElementById('diag-dropzone');
        if (dz) {
            dz.addEventListener('click', () => document.getElementById('diag-file-input').click());
            dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag-over'); });
            dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
            dz.addEventListener('drop', e => {
                e.preventDefault();
                dz.classList.remove('drag-over');
                if (e.dataTransfer.files[0]) handleDiagFile(e.dataTransfer.files[0]);
            });
        }

        const fileInput = document.getElementById('diag-file-input');
        if (fileInput) fileInput.addEventListener('change', () => {
            if (fileInput.files[0]) handleDiagFile(fileInput.files[0]);
        });

        // Analyze button
        const analyzeBtn = document.getElementById('diag-analyze-btn');
        if (analyzeBtn) analyzeBtn.addEventListener('click', () => startAnalysis());

        // Train button
        const trainBtn = document.getElementById('diag-train-btn');
        if (trainBtn) trainBtn.addEventListener('click', () => toggleTraining());

        // Reset button
        const resetBtn = document.getElementById('diag-reset-btn');
        if (resetBtn) resetBtn.addEventListener('click', () => resetTraining());

        // Back button
        const backBtn = document.getElementById('diag-back-btn');
        if (backBtn) backBtn.addEventListener('click', () => goBackToUpload());

        // Demo cards
        document.querySelectorAll('.diag-demo-card').forEach(card => {
            card.addEventListener('click', () => {
                const key = card.dataset.demo;
                loadDiagDemo(key);
            });
        });
    };

    // ─────────────────────────────────────────────
    //  UI TAB SWITCHING
    // ─────────────────────────────────────────────

    function switchDiagUploadTab(tab) {
        document.getElementById('diag-tab-file').classList.toggle('active', tab === 'file');
        document.getElementById('diag-tab-paste').classList.toggle('active', tab === 'paste');
        document.getElementById('diag-file-section').style.display = tab === 'file' ? 'block' : 'none';
        document.getElementById('diag-paste-section').style.display = tab === 'paste' ? 'flex' : 'none';
    }

    // ─────────────────────────────────────────────
    //  FILE HANDLING
    // ─────────────────────────────────────────────

    async function handleDiagFile(file) {
        if (!file) return;
        const ext = file.name.split('.').pop().toLowerCase();

        const allowedExtensions = ['py', 'txt', 'ipynb', 'pth', 'pt'];
        if (!allowedExtensions.includes(ext)) {
            showNotification(`Unsupported file: .${ext}`, 'bad');
            return;
        }

        // Try to upload to backend first
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${API_BASE}/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Upload failed');
            }

            const result = await response.json();
            DiagState.code = result.code;
            DiagState.filename = result.filename;
            document.getElementById('diag-code-input').value = DiagState.code;
            switchDiagUploadTab('paste');
            showNotification(`Successfully processed ${file.name} via backend`, 'good');
        } catch (error) {
            console.error('Backend upload failed, falling back to local reading:', error);
            
            // Fallback for text-based files if server is down
            if (['py', 'txt', 'ipynb'].includes(ext)) {
                const reader = new FileReader();
                reader.onload = e => {
                    let code = e.target.result;
                    
                    // Local fallback parsing for .ipynb
                    if (ext === 'ipynb') {
                        try {
                            const nb = JSON.parse(code);
                            const cells = nb.cells || [];
                            const codeCells = cells
                                .filter(c => c.cell_type === 'code')
                                .map(c => Array.isArray(c.source) ? c.source.join('') : c.source || '');
                            code = codeCells.join('\n\n');
                        } catch (err) {
                            // ignore json error, keep raw content
                        }
                    }
                    
                    DiagState.code = code;
                    DiagState.filename = file.name;
                    document.getElementById('diag-code-input').value = DiagState.code;
                    switchDiagUploadTab('paste');
                    showNotification(`Loaded ${file.name} (Local Fallback)`, 'good');
                };
                reader.readAsText(file);
            } else {
                showNotification(`Cannot parse .${ext} file. Is the backend running on port 8000?`, 'bad');
            }
        }
    }

    function loadDiagDemo(key) {
        const demo = DEMOS[key];
        if (!demo) return;
        DiagState.code = demo.code;
        DiagState.filename = demo.filename;
        DiagState.datasetCsv = demo.dataset;

        document.getElementById('diag-code-input').value = demo.code;
        document.getElementById('diag-dataset-input').value = demo.dataset;
        switchDiagUploadTab('paste');
        showNotification(`Loaded demo: ${demo.filename}`, 'good');
    }

    // ─────────────────────────────────────────────
    //  ANALYSIS
    // ─────────────────────────────────────────────

    async function startAnalysis() {
        const code = document.getElementById('diag-code-input').value.trim();
        const dataset = document.getElementById('diag-dataset-input').value.trim();

        if (code.length < 10) {
            showNotification('Please provide more code to analyze.', 'bad');
            return;
        }

        DiagState.code = code;
        DiagState.datasetCsv = dataset;

        // Show loading
        const overlay = document.getElementById('diag-loading');
        overlay.classList.add('active');

        const steps = ['dls1', 'dls2', 'dls3', 'dls4', 'dls5'];
        steps.forEach((id, i) => {
            setTimeout(() => {
                const el = document.getElementById(id);
                if (el) el.classList.add('vis');
            }, i * 300);
            setTimeout(() => {
                const el = document.getElementById(id);
                if (el) el.classList.add('done');
            }, i * 300 + 250);
        });

        try {
            const response = await fetch(`${API_BASE}/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    code: code,
                    dataset_csv: dataset || null,
                    epochs: 30,
                }),
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Analysis failed');
            }

            const result = await response.json();
            DiagState.sessionId = result.session_id;
            DiagState.parsed = result.parsed;
            DiagState.diagnostics = result.diagnostics;
            DiagState.suggestion = result.suggestion;

            // Wait for loading animation to finish
            setTimeout(() => {
                overlay.classList.remove('active');
                steps.forEach(id => {
                    const el = document.getElementById(id);
                    if (el) el.classList.remove('vis', 'done');
                });
                renderDiagResults();
                showResultsView();
            }, 1800);

        } catch (err) {
            overlay.classList.remove('active');
            steps.forEach(id => {
                const el = document.getElementById(id);
                if (el) el.classList.remove('vis', 'done');
            });
            showNotification(`Error: ${err.message}. Is the backend running on port 8000?`, 'bad');
        }
    }

    // ─────────────────────────────────────────────
    //  RENDER RESULTS
    // ─────────────────────────────────────────────

    function renderDiagResults() {
        const { parsed, diagnostics, suggestion } = DiagState;

        // File badge
        document.getElementById('diag-file-badge').textContent = DiagState.filename;

        // Code view with highlighting
        renderDiagCodeView();

        // Score ring
        const score = diagnostics.score;
        const ringFill = document.getElementById('diag-score-ring-fill');
        const circ = 163.4;
        ringFill.style.strokeDashoffset = circ - (score / 100) * circ;
        ringFill.style.stroke = score >= 70 ? 'var(--accent-green)' : score >= 40 ? 'var(--accent-amber)' : 'var(--accent-red)';
        document.getElementById('diag-score-num').textContent = score;
        document.getElementById('diag-score-title').textContent =
            score >= 70 ? 'Healthy model' : score >= 40 ? 'Issues detected' : 'Critical problems';

        const critCount = diagnostics.issues.filter(i => i.severity === 'critical').length;
        const warnCount = diagnostics.issues.filter(i => i.severity === 'warning').length;
        document.getElementById('diag-score-subtitle').textContent =
            `${critCount} critical · ${warnCount} warnings detected`;

        // Findings list
        const fl = document.getElementById('diag-finding-list');
        fl.innerHTML = diagnostics.issues.map(f => `
            <div class="diag-finding-item diag-sev-${f.severity}">
                <div class="diag-finding-icon">${f.icon}</div>
                <div class="diag-finding-body">
                    <div class="diag-finding-title">${f.title}</div>
                    <div class="diag-finding-desc">${f.desc}</div>
                </div>
            </div>
        `).join('');

        // Model card labels
        document.getElementById('diag-bad-card-title').textContent = suggestion.bad_label;
        document.getElementById('diag-good-card-title').textContent = suggestion.good_label;
        document.getElementById('diag-bad-score-chip').textContent = `Score: ${suggestion.bad_score}/100`;
        document.getElementById('diag-good-score-chip').textContent = `Score: ${suggestion.good_score}/100`;

        // Update labels on chart
        document.getElementById('diag-bad-sim-label').textContent = suggestion.bad_label;
        document.getElementById('diag-good-sim-label').textContent = suggestion.good_label;

        // Why section
        document.getElementById('diag-why-title').textContent = suggestion.root_cause;
        document.getElementById('diag-why-body').textContent = suggestion.root_body;

        // Fix steps
        document.getElementById('diag-fix-steps').innerHTML = (suggestion.fixes || []).map((f, i) => `
            <div class="diag-fix-step">
                <div class="diag-fix-step-num">${i + 1}</div>
                <div class="diag-fix-step-text">
                    <strong>${f[0]}</strong>
                    <span>${f[1]}</span>
                </div>
            </div>
        `).join('');

        // Code diff
        document.getElementById('diag-code-diff').innerHTML = (suggestion.diff_lines || []).map(d => {
            if (!d.text) return '<br>';
            const cls = d.type === 'minus' ? 'diag-diff-minus' : d.type === 'plus' ? 'diag-diff-plus' : 'diag-diff-neutral';
            const prefix = d.type === 'minus' ? '− ' : d.type === 'plus' ? '+ ' : '  ';
            return `<div class="diag-diff-line"><span class="${cls}">${prefix}${d.text}</span></div>`;
        }).join('');

        // Enable/disable train button based on dataset
        const trainBtn = document.getElementById('diag-train-btn');
        if (DiagState.datasetCsv && DiagState.datasetCsv.trim().length > 10) {
            trainBtn.disabled = false;
            trainBtn.textContent = '▶ TRAIN & COMPARE';
        } else {
            trainBtn.disabled = true;
            trainBtn.textContent = '▶ Paste dataset first';
        }

        // Clear charts
        clearDiagChart('diag-bad-chart', '#ef4444');
        clearDiagChart('diag-good-chart', '#10b981');
    }

    function renderDiagCodeView() {
        const lines = DiagState.code.split('\n');
        const container = document.getElementById('diag-code-view');
        container.innerHTML = '';

        const parsed = DiagState.parsed;
        const flagPatterns = [
            { re: /optim\.sgd\s*\(|SGD\s*\(/i, cls: 'diag-flagged' },
            { re: /lr\s*=\s*0\.[1-9]/i, cls: 'diag-flagged' },
            { re: /sigmoid/i, cls: 'diag-flagged' },
            { re: /optimizer\.step\(\)\s*$/i, cls: parsed && parsed.is_rnn && !parsed.has_grad_clip ? 'diag-flagged' : '' },
            { re: /dropout/i, cls: 'diag-noted' },
            { re: /adam\s*\(/i, cls: 'diag-noted' },
            { re: /relu/i, cls: 'diag-noted' },
            { re: /batchnorm|batch_norm/i, cls: 'diag-noted' },
        ];

        lines.forEach((line, i) => {
            const div = document.createElement('div');
            div.className = 'diag-code-line';

            let flagClass = '';
            for (const { re, cls } of flagPatterns) {
                if (cls && re.test(line)) { flagClass = cls; break; }
            }
            if (flagClass) div.classList.add(flagClass);

            div.innerHTML = `
                <span class="diag-code-ln">${i + 1}</span>
                <span class="diag-code-txt">${highlightPython(line) || '&nbsp;'}</span>
            `;
            container.appendChild(div);
        });
    }

    // ─────────────────────────────────────────────
    //  VIEW SWITCHING
    // ─────────────────────────────────────────────

    function showResultsView() {
        document.getElementById('diag-upload-view').style.display = 'none';
        document.getElementById('diag-results-view').style.display = 'block';
    }

    function goBackToUpload() {
        stopTraining();
        document.getElementById('diag-results-view').style.display = 'none';
        document.getElementById('diag-upload-view').style.display = 'block';
    }

    // ─────────────────────────────────────────────
    //  TRAINING (SSE)
    // ─────────────────────────────────────────────

    function toggleTraining() {
        if (DiagState.training) {
            stopTraining();
        } else {
            startTraining();
        }
    }

    function startTraining() {
        if (!DiagState.sessionId) {
            showNotification('Run analysis first.', 'bad');
            return;
        }

        DiagState.training = true;
        DiagState.epochsA = [];
        DiagState.epochsB = [];
        DiagState.currentEpoch = 0;

        document.getElementById('diag-train-btn').textContent = '⏸ PAUSE';
        document.getElementById('diag-insight-bar').innerHTML = 'Connecting to training server...';

        const url = `${API_BASE}/train?session_id=${DiagState.sessionId}`;

        // Use fetch + ReadableStream for SSE (more compatible than EventSource for CORS)
        fetch(url)
            .then(response => {
                if (!response.ok) throw new Error('Training request failed');
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                function pump() {
                    return reader.read().then(({ done, value }) => {
                        if (done || !DiagState.training) {
                            onTrainingComplete();
                            return;
                        }

                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split('\n');
                        buffer = lines.pop(); // keep incomplete line

                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                try {
                                    const data = JSON.parse(line.slice(6));
                                    handleTrainingEvent(data);
                                } catch (e) { /* skip invalid JSON */ }
                            }
                        }

                        return pump();
                    });
                }

                return pump();
            })
            .catch(err => {
                showNotification(`Training error: ${err.message}`, 'bad');
                stopTraining();
            });
    }

    function stopTraining() {
        DiagState.training = false;
        document.getElementById('diag-train-btn').textContent = '▶ TRAIN & COMPARE';
    }

    function resetTraining() {
        stopTraining();
        DiagState.epochsA = [];
        DiagState.epochsB = [];
        DiagState.currentEpoch = 0;
        document.getElementById('diag-epoch-counter').textContent = '0';
        document.getElementById('diag-progress-fill').style.width = '0%';
        document.getElementById('diag-epoch-val').textContent = '0 / 30';
        document.getElementById('diag-insight-bar').innerHTML = 'Press <span>▶ TRAIN & COMPARE</span> to start real training.';
        clearDiagChart('diag-bad-chart', '#ef4444');
        clearDiagChart('diag-good-chart', '#10b981');
    }

    function onTrainingComplete() {
        DiagState.training = false;
        document.getElementById('diag-train-btn').textContent = '⟳ RETRAIN';

        // Final insight
        if (DiagState.epochsA.length > 0 && DiagState.epochsB.length > 0) {
            const lastA = DiagState.epochsA[DiagState.epochsA.length - 1];
            const lastB = DiagState.epochsB[DiagState.epochsB.length - 1];
            const lossA = typeof lastA.loss === 'number' ? lastA.loss : Infinity;
            const lossB = typeof lastB.loss === 'number' ? lastB.loss : Infinity;

            if (lossB < lossA) {
                const improve = lossA > 0 ? Math.round((1 - lossB / lossA) * 100) : 0;
                document.getElementById('diag-insight-bar').innerHTML =
                    `✅ Training complete! Suggested model is <span>${improve}% better</span> — ` +
                    `final loss ${lossB.toFixed(4)} vs ${lossA.toFixed(4)}`;
            } else {
                document.getElementById('diag-insight-bar').innerHTML =
                    `Training complete. Your model loss: <span>${lossA.toFixed(4)}</span>, ` +
                    `Suggested: <span>${lossB.toFixed(4)}</span>`;
            }
        }
    }

    function handleTrainingEvent(data) {
        if (data.status === 'error') {
            showNotification(`Training error: ${data.message}`, 'bad');
            stopTraining();
            return;
        }

        if (data.status === 'done') {
            onTrainingComplete();
            return;
        }

        if (data.epoch === 0) {
            DiagState.totalEpochs = data.total_epochs;
            document.getElementById('diag-insight-bar').innerHTML = 'Training started — watching for patterns...';
            return;
        }

        const epoch = data.epoch;
        DiagState.currentEpoch = epoch;

        // Store metrics
        const lossA = data.model_a.train_loss;
        const accA = data.model_a.train_acc;
        const lossB = data.model_b.train_loss;
        const accB = data.model_b.train_acc;

        DiagState.epochsA.push({ loss: lossA, acc: accA });
        DiagState.epochsB.push({ loss: lossB, acc: accB });

        // Update progress
        const pct = (epoch / DiagState.totalEpochs * 100).toFixed(0);
        document.getElementById('diag-epoch-counter').textContent = epoch;
        document.getElementById('diag-progress-fill').style.width = pct + '%';
        document.getElementById('diag-epoch-val').textContent = `${epoch} / ${DiagState.totalEpochs}`;

        // Update charts
        drawDiagChart('diag-bad-chart', DiagState.epochsA, '#ef4444', DiagState.totalEpochs);
        drawDiagChart('diag-good-chart', DiagState.epochsB, '#10b981', DiagState.totalEpochs);

        // Update model card metrics
        updateModelCardMetrics(data);

        // Update insight bar
        updateDiagInsight(epoch, lossA, lossB, accA, accB);
    }

    // ─────────────────────────────────────────────
    //  CHART DRAWING
    // ─────────────────────────────────────────────

    function drawDiagChart(canvasId, data, color, totalEpochs) {
        const c = document.getElementById(canvasId);
        if (!c || data.length < 1) return;
        const W = c.width, H = c.height;
        const ctx = c.getContext('2d');

        ctx.fillStyle = '#0d0f1a';
        ctx.fillRect(0, 0, W, H);

        // Grid
        ctx.strokeStyle = color + '15';
        ctx.lineWidth = 1;
        for (let i = 1; i <= 4; i++) {
            const y = (i / 5) * H;
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
        }

        if (data.length < 2) return;

        // Get max loss for scaling
        const losses = data.map(d => typeof d.loss === 'number' ? d.loss : 0);
        const visMax = Math.max(1.2, ...losses.slice(-Math.min(20, losses.length)));

        // Fill area
        ctx.beginPath();
        data.forEach((d, i) => {
            const x = (i / (totalEpochs - 1)) * W;
            const v = typeof d.loss === 'number' ? d.loss : visMax;
            const y = H - (v / visMax) * (H - 12) - 6;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        const lx = ((data.length - 1) / (totalEpochs - 1)) * W;
        ctx.lineTo(lx, H); ctx.lineTo(0, H); ctx.closePath();
        ctx.fillStyle = color + '18';
        ctx.fill();

        // Line
        ctx.beginPath();
        data.forEach((d, i) => {
            const x = (i / (totalEpochs - 1)) * W;
            const v = typeof d.loss === 'number' ? d.loss : visMax;
            const y = H - (v / visMax) * (H - 12) - 6;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.shadowColor = color;
        ctx.shadowBlur = 5;
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Current point
        const last = data[data.length - 1];
        const lastLoss = typeof last.loss === 'number' ? last.loss : 0;
        const clx = ((data.length - 1) / (totalEpochs - 1)) * W;
        const cly = H - (lastLoss / visMax) * (H - 12) - 6;
        ctx.beginPath(); ctx.arc(clx, cly, 4.5, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.shadowColor = color; ctx.shadowBlur = 10; ctx.fill(); ctx.shadowBlur = 0;

        // Label
        ctx.fillStyle = color;
        ctx.font = 'bold 10px Inter, sans-serif';
        const labelText = `loss: ${lastLoss === 'NaN' ? 'NaN' : lastLoss.toFixed(4)}`;
        ctx.fillText(labelText, clx > W - 90 ? clx - 80 : clx + 8, cly < 18 ? cly + 14 : cly - 5);
    }

    function clearDiagChart(canvasId, color) {
        const c = document.getElementById(canvasId);
        if (!c) return;
        const ctx = c.getContext('2d');
        ctx.fillStyle = '#0d0f1a';
        ctx.fillRect(0, 0, c.width, c.height);
        ctx.fillStyle = color + '50';
        ctx.font = '11px Inter, sans-serif';
        ctx.fillText('Press TRAIN to start', 14, c.height - 10);
    }

    function updateModelCardMetrics(data) {
        // Model A (bad)
        const badMetrics = document.getElementById('diag-bad-metrics');
        const a = data.model_a;
        badMetrics.innerHTML = buildMetricRows(a.train_loss, a.train_acc, a.val_loss, a.val_acc, false);

        // Model B (good)
        const goodMetrics = document.getElementById('diag-good-metrics');
        const b = data.model_b;
        goodMetrics.innerHTML = buildMetricRows(b.train_loss, b.train_acc, b.val_loss, b.val_acc, true);
    }

    function buildMetricRows(trainLoss, trainAcc, valLoss, valAcc, isGood) {
        const fmtLoss = v => typeof v === 'number' ? v.toFixed(4) : 'NaN';
        const fmtAcc = v => typeof v === 'number' ? (v * 100).toFixed(1) + '%' : '—';
        const pctLoss = v => typeof v === 'number' ? Math.max(0, Math.min(100, 100 - v * 30)) : 0;
        const pctAcc = v => typeof v === 'number' ? v * 100 : 0;

        const rows = [
            ['TRAIN LOSS', fmtLoss(trainLoss), pctLoss(trainLoss)],
            ['TRAIN ACC', fmtAcc(trainAcc), pctAcc(trainAcc)],
            ['VAL LOSS', fmtLoss(valLoss), pctLoss(valLoss)],
            ['VAL ACC', fmtAcc(valAcc), pctAcc(valAcc)],
        ];

        return rows.map(([name, val, pct]) => `
            <div class="diag-metric-row">
                <span class="diag-mr-name">${name}</span>
                <div class="diag-mr-bar-track">
                    <div class="diag-mr-bar-fill ${isGood ? 'good' : 'bad'}" style="width:${pct}%"></div>
                </div>
                <span class="diag-mr-val ${isGood ? 'good' : 'bad'}">${val}</span>
            </div>
        `).join('');
    }

    function updateDiagInsight(epoch, lossA, lossB, accA, accB) {
        const bar = document.getElementById('diag-insight-bar');
        const la = typeof lossA === 'number' ? lossA : Infinity;
        const lb = typeof lossB === 'number' ? lossB : Infinity;
        const improve = la > 0 && isFinite(la) ? Math.round((1 - lb / la) * 100) : 0;

        let msg;
        if (epoch < 3) {
            msg = 'Training started — watching for instability patterns...';
        } else if (la > 5 || !isFinite(la)) {
            msg = `⚠️ Your model loss is <span>${la === Infinity ? 'NaN' : la.toFixed(4)}</span> — diverging! Suggested: <span>${lb.toFixed(4)}</span>`;
        } else if (improve > 50) {
            msg = `✅ Suggested model is <span>${improve}% better</span> — loss ${lb.toFixed(4)} vs ${la.toFixed(4)} at epoch ${epoch}`;
        } else if (improve > 15) {
            msg = `📈 Improvement visible — suggested loss <span>${lb.toFixed(4)}</span> vs your model <span>${la.toFixed(4)}</span>`;
        } else {
            msg = `Epoch ${epoch}: Your model <span>${la.toFixed(4)}</span> · Suggested <span>${lb.toFixed(4)}</span>`;
        }
        bar.innerHTML = msg;
    }

    // ─────────────────────────────────────────────
    //  NOTIFICATIONS
    // ─────────────────────────────────────────────

    function showNotification(msg, type) {
        let notif = document.getElementById('diag-notification');
        if (!notif) {
            notif = document.createElement('div');
            notif.id = 'diag-notification';
            notif.className = 'diag-notification';
            document.body.appendChild(notif);
        }
        notif.textContent = msg;
        notif.className = `diag-notification diag-notif-${type} show`;
        setTimeout(() => notif.classList.remove('show'), 3500);
    }

})();
