import { useState, useEffect } from 'react';

const API_URL = import.meta.env.VITE_API_URL || '';

export default function ModelSelector({ selectedModel, onModelChange }) {
    const [models, setModels] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch(`${API_URL}/api/models`)
            .then(res => res.json())
            .then(data => {
                setModels(data.models || []);
                setLoading(false);
                // Default to first available model
                if (!selectedModel && data.models?.length) {
                    const first = data.models.find(m => m.available);
                    if (first) onModelChange(first.name);
                }
            })
            .catch(() => {
                setLoading(false);
                // Fallback model list if backend not reachable
                const fallback = ['CNN', 'CNN-LSTM', 'Transformer', 'Hybrid (CNN-LSTM-Transformer)', 'DANN'];
                setModels(fallback.map(name => ({ name, description: '', type: '', available: true })));
                if (!selectedModel) onModelChange('CNN');
            });
    }, []);

    const currentInfo = models.find(m => m.name === selectedModel);

    return (
        <div>
            <h3 className="section-title">🤖 Select Model</h3>
            <select
                className="model-select"
                value={selectedModel}
                onChange={e => onModelChange(e.target.value)}
                disabled={loading}
            >
                {loading && <option>Loading models…</option>}
                {models.map(m => (
                    <option key={m.name} value={m.name} disabled={!m.available}>
                        {m.name} {!m.available ? '(not trained)' : ''}
                    </option>
                ))}
            </select>

            {currentInfo && (
                <div className="model-info">
                    <div className="model-info-label">Type</div>
                    <div className="model-info-value">{currentInfo.type}</div>
                    {currentInfo.description && (
                        <>
                            <div className="model-info-label" style={{ marginTop: '0.5rem' }}>Description</div>
                            <div className="model-info-value">{currentInfo.description}</div>
                        </>
                    )}
                    {currentInfo.paderborn_accuracy && (
                        <>
                            <div className="model-info-label" style={{ marginTop: '0.5rem' }}>Paderborn Accuracy</div>
                            <div className="model-info-value">{currentInfo.paderborn_accuracy}</div>
                        </>
                    )}
                </div>
            )}
        </div>
    );
}
