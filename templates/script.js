let selectedFile = null;
let chartQ = null;
let chartM = null;

const fileInput = document.getElementById('file-input');
const imgPreview = document.getElementById('image-preview');
const previewPlaceholder = document.getElementById('preview-placeholder');
const boxBadge = document.getElementById('box-badge');
const btnClassify = document.getElementById('btn-classify');

if (fileInput) {
    fileInput.addEventListener('change', (e) => {
        if (!e.target.files.length) return;
        selectedFile = e.target.files[0];
        const reader = new FileReader();
        reader.onload = (event) => {
            imgPreview.src = event.target.result;
            imgPreview.style.display = 'block';
            previewPlaceholder.style.display = 'none';
            if (boxBadge) boxBadge.style.display = 'none';
            btnClassify.disabled = false;
        };
        reader.readAsDataURL(selectedFile);
    });
}

async function changeAlgorithm() {
    const algo = document.getElementById('algo-select').value;
    document.getElementById('meta-title').innerText = `Metaheuristic Extraction (${algo})`;
    document.getElementById('meta-badge').innerText = `${algo} Pruning`;
    await loadMetrics(algo);
}

async function classifyScan() {
    if (!selectedFile) return;

    const algo = document.getElementById('algo-select').value;
    const btnText = document.getElementById('btn-text');
    btnText.innerText = "Extracting Features & Running Model...";
    btnClassify.disabled = true;

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('algorithm', algo);

    try {
        const res = await fetch('/api/predict', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.annotated_image) {
            imgPreview.src = data.annotated_image;
        }

        if (boxBadge) {
            if (data.follicles_detected > 0) {
                boxBadge.innerText = `${data.follicles_detected} candidate follicle regions (heuristic)`;
                boxBadge.style.display = 'block';
            } else {
                boxBadge.style.display = 'none';
            }
        }

        const hStatus = document.getElementById('h-live-status');
        hStatus.innerText = data.hybrid_pred.label;
        hStatus.className = 'pred-status ' + (data.hybrid_pred.is_infected ? 'status-pcos' : 'status-normal');
        document.getElementById('h-live-conf').innerText = data.hybrid_pred.confidence + '%';

        const qStatus = document.getElementById('q-live-status');
        qStatus.innerText = data.quantum_pred.label;
        qStatus.className = 'pred-status ' + (data.quantum_pred.is_infected ? 'status-pcos' : 'status-normal');
        document.getElementById('q-live-conf').innerText = data.quantum_pred.confidence + '%';

        const mStatus = document.getElementById('m-live-status');
        mStatus.innerText = data.metaheuristic_pred.label;
        mStatus.className = 'pred-status ' + (data.metaheuristic_pred.is_infected ? 'status-pcos' : 'status-normal');
        document.getElementById('m-live-conf').innerText = data.metaheuristic_pred.confidence + '%';

    } catch (err) {
        console.error("Diagnostic error:", err);
    } finally {
        btnText.innerText = "Execute Dual-Branch Feature Diagnostic";
        btnClassify.disabled = false;
    }
}

async function loadMetrics(algo = "PSO") {
    try {
        console.log(`Fetching metrics for: ${algo}...`);
        const res = await fetch(`/api/metrics/${algo}`);
        if (!res.ok) {
            throw new Error(`HTTP error ${res.status}`);
        }
        const data = await res.json();
        console.log("Loaded empirical metrics:", data);

        if (data.hybrid && data.hybrid.accuracy !== undefined) {
            document.getElementById('h-acc').innerText = (data.hybrid.accuracy * 100).toFixed(1) + '%';
        }

        // 1. Render Quantum Metrics
        const q = data.quantum;
        if (q && q.accuracy !== undefined) {
            document.getElementById('q-acc').innerText = (q.accuracy * 100).toFixed(1) + '%';
            document.getElementById('q-feats').innerText = q.selected_features || 64;
            if (q.confusion_matrix) {
                document.getElementById('q-tn').innerText = q.confusion_matrix[0][0];
                document.getElementById('q-fp').innerText = q.confusion_matrix[0][1];
                document.getElementById('q-fn').innerText = q.confusion_matrix[1][0];
                document.getElementById('q-tp').innerText = q.confusion_matrix[1][1];
            }
        }

        // 2. Render Metaheuristic Metrics
        const m = data.metaheuristic;
        if (m && m.accuracy !== undefined) {
            document.getElementById('m-acc').innerText = (m.accuracy * 100).toFixed(1) + '%';
            document.getElementById('m-feats').innerText = m.selected_features;
            if (m.confusion_matrix) {
                document.getElementById('m-tn').innerText = m.confusion_matrix[0][0];
                document.getElementById('m-fp').innerText = m.confusion_matrix[0][1];
                document.getElementById('m-fn').innerText = m.confusion_matrix[1][0];
                document.getElementById('m-tp').innerText = m.confusion_matrix[1][1];
            }
        }

        renderCharts(q, m);
    } catch (err) {
        console.error("Failed to load metrics from API:", err);
    }
}

function renderCharts(q, m) {
    if (!q || !m) return;
    if (chartQ) chartQ.destroy();
    if (chartM) chartM.destroy();

    const chartConfig = (values, color) => ({
        type: 'bar',
        data: {
            labels: ['Acc', 'Prec', 'Recall', 'F1'],
            datasets: [{
                data: values.map(v => (v * 100).toFixed(1)),
                backgroundColor: color,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { min: 40, max: 100, grid: { color: 'rgba(255,255,255,0.05)' } },
                x: { grid: { display: false } }
            }
        }
    });

    const canvasQ = document.getElementById('chart-quantum');
    const canvasM = document.getElementById('chart-meta');

    if (canvasQ && q.accuracy !== undefined) {
        chartQ = new Chart(canvasQ, chartConfig([q.accuracy, q.precision, q.recall, q.f1], '#a855f7'));
    }
    if (canvasM && m.accuracy !== undefined) {
        chartM = new Chart(canvasM, chartConfig([m.accuracy, m.precision, m.recall, m.f1], '#06b6d4'));
    }
}

// Ensure execution happens after DOM elements are accessible
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => loadMetrics("PSO"));
} else {
    loadMetrics("PSO");
}