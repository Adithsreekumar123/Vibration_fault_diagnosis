import SignalChart from './SignalChart';
import ProbabilityChart from './ProbabilityChart';

function getBadgeClass(label) {
    if (label.includes('Normal')) return 'Normal';
    if (label.includes('Ball')) return 'Ball';
    if (label.includes('Inner')) return 'Inner';
    if (label.includes('Outer')) return 'Outer';
    return '';
}

export default function Results({ result }) {
    if (!result) return null;

    const { overall_prediction, average_probabilities, window_results, signal_data, total_windows, signal_length, model, filename, ground_truth, is_correct } = result;
    const badgeClass = getBadgeClass(overall_prediction.label);

    return (
        <div className="results-section">
            <h3 className="section-title" style={{ marginBottom: '1.5rem' }}>🎯 Prediction Results</h3>

            {/* Hero result card */}
            <div className={`result-hero glass-card ${badgeClass}`}>
                <span className={`result-label-badge ${badgeClass}`}>
                    Predicted Fault
                </span>
                <div className="result-fault-name">{overall_prediction.label}</div>
                <div className="result-confidence">
                    Confidence: <span className="confidence-value">{(overall_prediction.confidence * 100).toFixed(1)}%</span>
                </div>
            </div>

            {/* Ground truth + correct/incorrect */}
            {ground_truth && (
                <>
                    <div className="metrics-grid" style={{ gridTemplateColumns: '1fr 1fr', marginBottom: '1rem' }}>
                        <div className="metric-card glass-card">
                            <div className="metric-label">🎯 Predicted Fault</div>
                            <div className="metric-value" style={{ fontSize: '1.1rem' }}>
                                <span className={`fault-badge ${badgeClass}`} style={{ fontSize: '0.95rem', padding: '0.3rem 0.8rem' }}>
                                    {overall_prediction.label}
                                </span>
                            </div>
                        </div>
                        <div className="metric-card glass-card">
                            <div className="metric-label">✓ Ground Truth</div>
                            <div className="metric-value" style={{ fontSize: '1.1rem' }}>
                                <span className={`fault-badge ${getBadgeClass(ground_truth.label)}`} style={{ fontSize: '0.95rem', padding: '0.3rem 0.8rem' }}>
                                    {ground_truth.label}
                                </span>
                            </div>
                        </div>
                    </div>
                    <div style={{ marginBottom: '1.5rem' }}>
                        {is_correct ? (
                            <div style={{ padding: '0.75rem 1.25rem', background: 'rgba(56, 239, 125, 0.1)', border: '1px solid rgba(56, 239, 125, 0.2)', borderRadius: '10px', color: '#38ef7d', fontWeight: 600 }}>
                                ✅ Correct Prediction!
                            </div>
                        ) : (
                            <div className="error-box" style={{ marginBottom: 0 }}>
                                ❌ Misclassified — Predicted: {overall_prediction.label}, True: {ground_truth.label}
                            </div>
                        )}
                    </div>
                </>
            )}

            {/* Metric cards */}
            <div className="metrics-grid">
                <div className="metric-card glass-card">
                    <div className="metric-label">Model Used</div>
                    <div className="metric-value" style={{ fontSize: '1.1rem' }}>{model}</div>
                </div>
                <div className="metric-card glass-card">
                    <div className="metric-label">Signal Length</div>
                    <div className="metric-value">{signal_length.toLocaleString()}</div>
                </div>
                <div className="metric-card glass-card">
                    <div className="metric-label">Windows Analyzed</div>
                    <div className="metric-value">{total_windows}</div>
                </div>
                <div className="metric-card glass-card">
                    <div className="metric-label">File</div>
                    <div className="metric-value" style={{ fontSize: '0.9rem', wordBreak: 'break-all' }}>{filename}</div>
                </div>
            </div>

            {/* Charts */}
            <div className="charts-grid">
                <SignalChart signalData={signal_data} />
                <ProbabilityChart probabilities={average_probabilities} />
            </div>

            {/* Per-window results table */}
            {window_results && window_results.length > 0 && (
                <div className="window-table-card glass-card">
                    <div className="chart-title">🔍 Per-Window Predictions (first {window_results.length} of {total_windows})</div>
                    <div style={{ overflowX: 'auto' }}>
                        <table className="window-table">
                            <thead>
                                <tr>
                                    <th>Window</th>
                                    <th>Prediction</th>
                                    <th>Confidence</th>
                                    <th>Normal</th>
                                    <th>Ball</th>
                                    <th>Inner Race</th>
                                    <th>Outer Race</th>
                                </tr>
                            </thead>
                            <tbody>
                                {window_results.map((w) => {
                                    const wBadge = getBadgeClass(w.predicted_label);
                                    return (
                                        <tr key={w.window_index}>
                                            <td>#{w.window_index + 1}</td>
                                            <td>
                                                <span className={`fault-badge ${wBadge}`}>{w.predicted_label}</span>
                                            </td>
                                            <td>{(w.confidence * 100).toFixed(1)}%</td>
                                            {Object.values(w.probabilities).map((p, i) => (
                                                <td key={i}>{(p * 100).toFixed(1)}%</td>
                                            ))}
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    );
}
