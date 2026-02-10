import { useState } from 'react';
import ProbabilityChart from './ProbabilityChart';

const API_URL = import.meta.env.VITE_API_URL || '';

const CLASS_NAMES = ['Normal', 'Ball Fault', 'Inner Race Fault', 'Outer Race Fault'];
const HEATMAP_COLORS = [
    'rgba(102,126,234,0.05)', 'rgba(102,126,234,0.15)', 'rgba(102,126,234,0.3)',
    'rgba(102,126,234,0.45)', 'rgba(102,126,234,0.6)', 'rgba(102,126,234,0.75)',
    'rgba(102,126,234,0.9)',
];

function getCellColor(value, maxVal) {
    if (maxVal === 0) return HEATMAP_COLORS[0];
    const ratio = value / maxVal;
    const idx = Math.min(Math.floor(ratio * (HEATMAP_COLORS.length - 1)), HEATMAP_COLORS.length - 1);
    return HEATMAP_COLORS[idx];
}

export default function FullEvaluation({ modelName, datasetName }) {
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const runEval = async () => {
        setLoading(true);
        setError('');
        setResult(null);

        const formData = new FormData();
        formData.append('model_name', modelName);
        formData.append('dataset_name', datasetName);

        try {
            const res = await fetch(`${API_URL}/api/evaluate`, {
                method: 'POST',
                body: formData,
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Evaluation failed');
            }
            const data = await res.json();
            setResult(data);
        } catch (e) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    };

    const maxCmVal = result ? Math.max(...result.confusion_matrix.flat()) : 0;

    return (
        <div className="fade-in">
            <h3 className="section-title">📊 Full Dataset Evaluation</h3>

            <button
                className={`predict-btn ${loading ? 'loading' : ''}`}
                onClick={runEval}
                disabled={!modelName || !datasetName || loading}
                style={{ marginBottom: '1.5rem' }}
            >
                {loading ? (
                    <>
                        <span className="spinner" />
                        Evaluating model on entire dataset…
                    </>
                ) : (
                    '🚀 Run Full Evaluation'
                )}
            </button>

            {error && <div className="error-box">❌ {error}</div>}

            {!result && !loading && !error && (
                <div className="glass-card" style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
                    👆 Click 'Run Full Evaluation' to evaluate the model on the entire dataset.
                </div>
            )}

            {result && (
                <div className="results-section">
                    {/* Overall metrics */}
                    <h4 className="section-title" style={{ fontSize: '1.1rem' }}>📈 Overall Metrics</h4>
                    <div className="metrics-grid">
                        <div className="metric-card glass-card">
                            <div className="metric-label">🎯 Accuracy</div>
                            <div className="metric-value">{(result.accuracy * 100).toFixed(2)}%</div>
                        </div>
                        <div className="metric-card glass-card">
                            <div className="metric-label">📊 Precision</div>
                            <div className="metric-value">{(result.precision * 100).toFixed(2)}%</div>
                        </div>
                        <div className="metric-card glass-card">
                            <div className="metric-label">🔍 Recall</div>
                            <div className="metric-value">{(result.recall * 100).toFixed(2)}%</div>
                        </div>
                        <div className="metric-card glass-card">
                            <div className="metric-label">⚖️ F1-Score</div>
                            <div className="metric-value">{(result.f1_score * 100).toFixed(2)}%</div>
                        </div>
                    </div>

                    {/* Confusion Matrix */}
                    <h4 className="section-title" style={{ fontSize: '1.1rem' }}>🔢 Confusion Matrix</h4>
                    <div className="glass-card" style={{ padding: '1.5rem', marginBottom: '1.5rem', overflowX: 'auto' }}>
                        <table className="window-table" style={{ minWidth: '500px' }}>
                            <thead>
                                <tr>
                                    <th style={{ borderBottom: '2px solid var(--glass-border)' }}>Actual ↓ / Predicted →</th>
                                    {CLASS_NAMES.map(n => (
                                        <th key={n} style={{ textAlign: 'center', borderBottom: '2px solid var(--glass-border)' }}>{n.replace(' Fault', '')}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {result.confusion_matrix.map((row, i) => (
                                    <tr key={i}>
                                        <td style={{ fontWeight: 700, color: 'var(--text-primary)' }}>{CLASS_NAMES[i]}</td>
                                        {row.map((val, j) => (
                                            <td
                                                key={j}
                                                style={{
                                                    textAlign: 'center',
                                                    fontWeight: i === j ? 800 : 400,
                                                    color: i === j ? 'var(--text-primary)' : 'var(--text-secondary)',
                                                    background: getCellColor(val, maxCmVal),
                                                    fontSize: '1rem',
                                                }}
                                            >
                                                {val}
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    {/* Per-class metrics */}
                    <h4 className="section-title" style={{ fontSize: '1.1rem' }}>📋 Per-Class Metrics</h4>
                    <div className="glass-card" style={{ padding: '1.5rem', marginBottom: '1.5rem', overflowX: 'auto' }}>
                        <table className="window-table">
                            <thead>
                                <tr>
                                    <th>Class</th>
                                    <th>Precision</th>
                                    <th>Recall</th>
                                    <th>F1-Score</th>
                                    <th>Support</th>
                                </tr>
                            </thead>
                            <tbody>
                                {result.per_class.map(m => (
                                    <tr key={m.class}>
                                        <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{m.class}</td>
                                        <td>{(m.precision * 100).toFixed(2)}%</td>
                                        <td>{(m.recall * 100).toFixed(2)}%</td>
                                        <td>{(m.f1_score * 100).toFixed(2)}%</td>
                                        <td>{m.support}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    {/* Summary */}
                    <div style={{
                        padding: '1.25rem',
                        background: 'rgba(56, 239, 125, 0.08)',
                        border: '1px solid rgba(56, 239, 125, 0.15)',
                        borderRadius: '10px',
                        color: 'var(--text-primary)',
                    }}>
                        <h4 style={{ color: '#38ef7d', marginBottom: '0.5rem' }}>✅ Evaluation Complete!</h4>
                        <div style={{ color: 'var(--text-secondary)', lineHeight: 1.8 }}>
                            <strong>Model:</strong> {result.model}<br />
                            <strong>Dataset:</strong> {result.dataset}<br />
                            <strong>Total Samples:</strong> {result.total_samples}<br />
                            <strong>Correct Predictions:</strong> {result.correct}<br />
                            <strong>Accuracy:</strong> {(result.accuracy * 100).toFixed(2)}%
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
