import { useState, useEffect } from 'react';
import SignalChart from './SignalChart';
import ProbabilityChart from './ProbabilityChart';

const API_URL = import.meta.env.VITE_API_URL || '';

function getBadgeClass(label) {
    if (label.includes('Normal')) return 'Normal';
    if (label.includes('Ball')) return 'Ball';
    if (label.includes('Inner')) return 'Inner';
    if (label.includes('Outer')) return 'Outer';
    return '';
}

export default function SingleSample({ modelName, datasetName }) {
    const [sampleIndex, setSampleIndex] = useState(0);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [totalSamples, setTotalSamples] = useState(0);
    const [error, setError] = useState('');

    // Auto-predict when model, dataset, or sample changes
    useEffect(() => {
        if (!modelName || !datasetName) return;
        predict(sampleIndex);
    }, [modelName, datasetName]);

    const predict = async (idx) => {
        setLoading(true);
        setError('');
        try {
            const res = await fetch(
                `${API_URL}/api/datasets/${datasetName}/sample?model_name=${encodeURIComponent(modelName)}&sample_index=${idx}`
            );
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Failed to predict');
            }
            const data = await res.json();
            setResult(data);
            setTotalSamples(data.total_samples);
        } catch (e) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    };

    const handleSliderChange = (e) => {
        const idx = parseInt(e.target.value);
        setSampleIndex(idx);
        predict(idx);
    };

    const handleRandom = async () => {
        try {
            const res = await fetch(`${API_URL}/api/datasets/${datasetName}/random_index`);
            const data = await res.json();
            setSampleIndex(data.index);
            predict(data.index);
        } catch (e) {
            setError('Failed to get random sample');
        }
    };

    return (
        <div className="fade-in">
            <h3 className="section-title">🎯 Single Sample Fault Prediction</h3>

            {/* Sample selector */}
            <div className="glass-card" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
                    <div style={{ flex: 1, minWidth: '200px' }}>
                        <label className="model-info-label">Sample Index</label>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                            <input
                                type="range"
                                min={0}
                                max={totalSamples > 0 ? totalSamples - 1 : 0}
                                value={sampleIndex}
                                onChange={handleSliderChange}
                                style={{ flex: 1, accentColor: '#667eea' }}
                            />
                            <span style={{ color: 'var(--text-secondary)', minWidth: '90px', fontWeight: 600, fontSize: '0.95rem' }}>
                                {sampleIndex} / {totalSamples > 0 ? totalSamples - 1 : '?'}
                            </span>
                        </div>
                    </div>
                    <button
                        onClick={handleRandom}
                        className="predict-btn"
                        style={{ width: 'auto', padding: '0.6rem 1.5rem', fontSize: '0.9rem' }}
                    >
                        🎲 Random Sample
                    </button>
                </div>
            </div>

            {loading && (
                <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-secondary)' }}>
                    <span className="spinner" /> Loading prediction...
                </div>
            )}

            {error && <div className="error-box">❌ {error}</div>}

            {result && !loading && (
                <>
                    {/* Result metrics */}
                    <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
                        <div className="metric-card glass-card">
                            <div className="metric-label">🎯 Predicted Fault</div>
                            <div className="metric-value" style={{ fontSize: '1.2rem' }}>
                                <span className={`fault-badge ${getBadgeClass(result.predicted_label)}`} style={{ fontSize: '1rem', padding: '0.3rem 0.8rem' }}>
                                    {result.predicted_label}
                                </span>
                            </div>
                        </div>
                        <div className="metric-card glass-card">
                            <div className="metric-label">📊 Confidence</div>
                            <div className="metric-value">{(result.confidence * 100).toFixed(1)}%</div>
                        </div>
                        <div className="metric-card glass-card">
                            <div className="metric-label">✓ Ground Truth</div>
                            <div className="metric-value" style={{ fontSize: '1.2rem' }}>
                                <span className={`fault-badge ${getBadgeClass(result.true_name)}`} style={{ fontSize: '1rem', padding: '0.3rem 0.8rem' }}>
                                    {result.true_name}
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* Correct/Incorrect */}
                    <div style={{ marginBottom: '1.5rem' }}>
                        {result.is_correct ? (
                            <div style={{ padding: '0.75rem 1.25rem', background: 'rgba(56, 239, 125, 0.1)', border: '1px solid rgba(56, 239, 125, 0.2)', borderRadius: '10px', color: '#38ef7d', fontWeight: 600 }}>
                                ✅ Correct Prediction!
                            </div>
                        ) : (
                            <div className="error-box" style={{ marginBottom: 0 }}>
                                ❌ Misclassified (True: {result.true_name})
                            </div>
                        )}
                    </div>

                    {/* Charts */}
                    <div className="charts-grid">
                        <SignalChart signalData={result.signal_data} />
                        <ProbabilityChart probabilities={result.probabilities} />
                    </div>

                    {/* Detailed probabilities */}
                    <h4 className="section-title" style={{ fontSize: '1.1rem', marginBottom: '1rem' }}>📋 Detailed Class Probabilities</h4>
                    <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
                        {Object.entries(result.probabilities).map(([name, prob], i) => (
                            <div key={name} className="metric-card glass-card">
                                <div className="metric-label">
                                    {result.predicted_class === i ? '🟢' : '⚪'} {name}
                                </div>
                                <div className="metric-value">{(prob * 100).toFixed(2)}%</div>
                            </div>
                        ))}
                    </div>
                </>
            )}
        </div>
    );
}
