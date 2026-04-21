// ─── Config ──────────────────────────────────────────────────────────────────
const API_BASE = 'http://localhost:8000';
let priceChart = null;

// ─── Utility Functions ────────────────────────────────────────────────────────
const formatPrice = (price) => price ? `$${parseFloat(price).toFixed(2)}` : '$--';
const formatPct = (pct) => pct ? `${parseFloat(pct).toFixed(2)}%` : '--%';
const formatNumber = (n) => n ? parseInt(n).toLocaleString() : '--';

function setElement(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

function setClass(id, className) {
    const el = document.getElementById(id);
    if (el) {
        el.className = className;
    }
}

// ─── Fetch Latest Prediction ──────────────────────────────────────────────────
async function loadLatestPrediction() {
    try {
        const res = await fetch(`${API_BASE}/api/prediction/latest`);
        if (!res.ok) throw new Error('No prediction available');
        const data = await res.json();

        // Price
        setElement('predicted-price', formatPrice(data.predicted_price));
        setElement('current-price', formatPrice(data.current_price));
        setElement('prediction-date', data.date);

        // Direction
        const isUp = data.predicted_dir === 'UP';
        setElement('direction-icon', isUp ? '▲' : '▼');
        setElement('direction-label', isUp ? 'UP' : 'DOWN');
        setClass('direction-label', `direction-label ${isUp ? 'up' : 'down'}`);

        // Change
        const changeEl = document.getElementById('price-change');
        if (changeEl) {
            changeEl.textContent = formatPrice(data.price_change);
            changeEl.style.color = isUp
                ? 'var(--accent-green)'
                : 'var(--accent-red)';
        }

        const changePctEl = document.getElementById('price-change-pct');
        if (changePctEl) {
            changePctEl.textContent = formatPct(data.price_change_pct);
            changePctEl.style.color = isUp
                ? 'var(--accent-green)'
                : 'var(--accent-red)';
        }

        // Confidence
        setElement('confidence', formatPct(data.confidence));

        // Last updated
        setElement('last-updated',
            `Last updated: ${new Date(data.created_at).toLocaleString()}`
        );

    } catch (err) {
        console.error('Error loading prediction:', err);
        setElement('predicted-price', 'Unavailable');
        setElement('direction-label', 'Run pipeline first');
    }
}

// ─── Fetch Latest Sentiment ───────────────────────────────────────────────────
async function loadLatestSentiment() {
    try {
        const res = await fetch(`${API_BASE}/api/sentiment/latest`);
        if (!res.ok) throw new Error('No sentiment available');
        const data = await res.json();

        // Label and color
        let labelColor = 'var(--text-primary)';
        if (data.avg_sentiment > 0.05) {
            labelColor = 'var(--accent-green)';
        } else if (data.avg_sentiment < -0.05) {
            labelColor = 'var(--accent-red)';
        }

        setElement('sentiment-emoji', '');
        setElement('sentiment-label', data.sentiment_label
            .replace('Positive', 'Positive')
            .replace('Negative', 'Negative')
            .replace('Neutral', 'Neutral')
        );
        setElement('sentiment-score',
            `Score: ${parseFloat(data.avg_sentiment).toFixed(4)}`
        );

        const labelEl = document.getElementById('sentiment-label');
        if (labelEl) labelEl.style.color = labelColor;

        // Meta
        setElement('tweet-count', formatNumber(data.tweet_count));
        setElement('positive-count', formatNumber(data.positive_count));
        setElement('negative-count', formatNumber(data.negative_count));
        setElement('total-engagement', formatNumber(data.total_engagement));

    } catch (err) {
        console.error('Error loading sentiment:', err);
        setElement('sentiment-emoji', '');
        setElement('sentiment-label', 'Unavailable');
    }
}

// ─── Fetch Model Accuracy ─────────────────────────────────────────────────────
async function loadAccuracy() {
    try {
        const res = await fetch(`${API_BASE}/api/accuracy`);
        if (!res.ok) throw new Error('No accuracy data');
        const data = await res.json();

        setElement('rmse', formatPrice(data.model_rmse));
        setElement('mae', formatPrice(data.model_mae));
        setElement('dir-accuracy', formatPct(data.model_dir_accuracy));
        setElement('live-accuracy', formatPct(data.live_accuracy_pct));
        setElement('total-predictions', formatNumber(data.total_predictions));
        setElement('model-version', data.model_version || 'finbert_v1');

    } catch (err) {
        console.error('Error loading accuracy:', err);
    }
}

// ─── Fetch Prediction History Table ──────────────────────────────────────────
async function loadPredictionHistory() {
    try {
        const res = await fetch(`${API_BASE}/api/prediction/history?limit=15`);
        if (!res.ok) throw new Error('No history available');
        const data = await res.json();

        const tbody = document.getElementById('history-tbody');
        if (!tbody) return;

        if (!data.predictions || data.predictions.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="5" class="loading">
                        No predictions yet — run the pipeline first
                    </td>
                </tr>`;
            return;
        }

        tbody.innerHTML = data.predictions.map(p => `
            <tr>
                <td>${p.date}</td>
                <td>${formatPrice(p.predicted_price)}</td>
                <td>${p.actual_price ? formatPrice(p.actual_price) : '--'}</td>
                <td style="color: ${p.predicted_dir === 'UP'
                    ? 'var(--accent-green)'
                    : 'var(--accent-red)'}">
                    ${p.predicted_dir === 'UP' ? '▲ UP' : '▼ DOWN'}
                </td>
                <td class="${p.was_correct === null
                    ? ''
                    : p.was_correct
                        ? 'correct-yes'
                        : 'correct-no'}">
                    ${p.was_correct === null
                        ? '--'
                        : p.was_correct ? 'Yes' : 'No'}
                </td>
            </tr>
        `).join('');

    } catch (err) {
        console.error('Error loading history:', err);
        const tbody = document.getElementById('history-tbody');
        if (tbody) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="5" class="loading">Failed to load history</td>
                </tr>`;
        }
    }
}

// ─── Price Chart ──────────────────────────────────────────────────────────────
async function loadPriceChart(days = 30) {
    try {
        // Update active button
        document.querySelectorAll('.chart-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        event?.target?.classList.add('active');

        const res = await fetch(`${API_BASE}/api/price/history?days=${days}`);
        if (!res.ok) throw new Error('No price data');
        const data = await res.json();

        const prices = data.prices.reverse();
        const labels = prices.map(p => p.date);
        const closes = prices.map(p => p.close);

        const ctx = document.getElementById('priceChart')?.getContext('2d');
        if (!ctx) return;

        // Destroy existing chart
        if (priceChart) priceChart.destroy();

        priceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label: 'TSLA Close Price',
                    data: closes,
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.05)',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: (ctx) => `$${ctx.parsed.y.toFixed(2)}`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: '#1e293b' },
                        ticks: {
                            color: '#64748b',
                            maxTicksLimit: 8,
                            font: { size: 11 }
                        }
                    },
                    y: {
                        grid: { color: '#1e293b' },
                        ticks: {
                            color: '#64748b',
                            font: { size: 11 },
                            callback: (val) => `$${val}`
                        }
                    }
                }
            }
        });

    } catch (err) {
        console.error('Error loading price chart:', err);
    }
}

// ─── Manual Pipeline Trigger ──────────────────────────────────────────────────
async function triggerPipeline() {
    const btn = document.getElementById('trigger-btn');
    const status = document.getElementById('trigger-status');

    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Running...';
    }
    if (status) status.textContent = '';

    try {
        const res = await fetch(`${API_BASE}/api/prediction/trigger`, {
            method: 'POST'
        });

        if (!res.ok) throw new Error('Pipeline trigger failed');

        if (status) {
            status.textContent = 'Pipeline triggered. Refreshing in 10s...';
            status.style.color = 'var(--accent-green)';
        }

        // Refresh data after 10 seconds
        setTimeout(() => {
            loadAllData();
            if (status) status.textContent = '';
        }, 10000);

    } catch (err) {
        console.error('Error triggering pipeline:', err);
        if (status) {
            status.textContent = 'Failed to trigger pipeline';
            status.style.color = 'var(--accent-red)';
        }
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.textContent = 'Run Pipeline Now';
        }
    }
}

// ─── Load All Data ────────────────────────────────────────────────────────────
async function loadAllData() {
    await Promise.all([
        loadLatestPrediction(),
        loadLatestSentiment(),
        loadAccuracy(),
        loadPredictionHistory(),
        loadPriceChart(30)
    ]);
}

// ─── Auto Refresh every 5 minutes ────────────────────────────────────────────
setInterval(loadAllData, 5 * 60 * 1000);

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', loadAllData);