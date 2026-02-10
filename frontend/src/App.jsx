import { useState, useEffect } from 'react';
import FileUpload from './components/FileUpload';
import ModelSelector from './components/ModelSelector';
import Results from './components/Results';
import SingleSample from './components/SingleSample';
import FullEvaluation from './components/FullEvaluation';
import './App.css';

const API_URL = import.meta.env.VITE_API_URL || '';

export default function App() {
  // Config state
  const [model, setModel] = useState('');
  const [dataset, setDataset] = useState('Paderborn');
  const [datasets, setDatasets] = useState([]);
  const [deviceInfo, setDeviceInfo] = useState('cpu');

  // Tab state
  const [activeTab, setActiveTab] = useState('single'); // 'single' | 'evaluation' | 'upload'

  // Upload tab state
  const [file, setFile] = useState(null);
  const [uploadResult, setUploadResult] = useState(null);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadError, setUploadError] = useState('');

  // Fetch datasets
  useEffect(() => {
    fetch(`${API_URL}/api/datasets`)
      .then(res => res.json())
      .then(data => {
        setDatasets(data.datasets || []);
        setDeviceInfo(data.device || 'cpu');
        if (data.datasets?.length) {
          const first = data.datasets.find(d => d.available);
          if (first) setDataset(first.name);
        }
      })
      .catch(() => {
        setDatasets([
          { name: 'CWRU', available: true, samples: 0, window_size: 4096 },
          { name: 'Paderborn', available: true, samples: 0, window_size: 4096 },
        ]);
      });
  }, []);

  const handleUploadPredict = async () => {
    if (!file) return;
    setUploadLoading(true);
    setUploadError('');
    setUploadResult(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_name', model);

    try {
      const res = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Prediction failed');
      }
      setUploadResult(await res.json());
    } catch (e) {
      setUploadError(e.message || 'Failed to connect to server.');
    } finally {
      setUploadLoading(false);
    }
  };

  const currentDatasetInfo = datasets.find(d => d.name === dataset);

  const tabs = [
    { id: 'single', label: '🎯 Single Sample Prediction', icon: '🎯' },
    { id: 'evaluation', label: '📊 Full Dataset Evaluation', icon: '📊' },
    { id: 'upload', label: '📁 Upload .mat File', icon: '📁' },
  ];

  return (
    <div className="app-layout">
      {/* ── Sidebar ── */}
      <aside className="sidebar glass-card">
        <div className="sidebar-header">
          <h2>⚙️ Configuration</h2>
        </div>

        {/* Model selection */}
        <div className="sidebar-section">
          <ModelSelector selectedModel={model} onModelChange={setModel} />
        </div>

        {/* Dataset selection */}
        <div className="sidebar-section">
          <h3 className="section-title">📊 Dataset</h3>
          <select
            className="model-select"
            value={dataset}
            onChange={e => setDataset(e.target.value)}
          >
            {datasets.map(d => (
              <option key={d.name} value={d.name} disabled={!d.available}>
                {d.name} {!d.available ? '(not found)' : ''}
              </option>
            ))}
          </select>
        </div>

        {/* Dataset info */}
        {currentDatasetInfo && currentDatasetInfo.available && (
          <div className="sidebar-section">
            <h3 className="section-title" style={{ fontSize: '0.95rem' }}>📋 Dataset Info</h3>
            <div className="sidebar-info-list">
              <div className="sidebar-info-item">
                <span>Samples</span>
                <span>{currentDatasetInfo.samples.toLocaleString()}</span>
              </div>
              <div className="sidebar-info-item">
                <span>Window Size</span>
                <span>{currentDatasetInfo.window_size.toLocaleString()}</span>
              </div>
              <div className="sidebar-info-item">
                <span>Device</span>
                <span style={{ color: deviceInfo === 'cuda' ? '#38ef7d' : 'var(--text-secondary)' }}>
                  {deviceInfo.toUpperCase()}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Training instructions */}
        <div className="sidebar-section">
          <h3 className="section-title" style={{ fontSize: '0.95rem' }}>🚀 Train Models</h3>
          <div className="sidebar-code">
            <code># Train DANN + Few-shot</code>
            <code>python run_dann_fewshot.py</code>
          </div>
          <div className="model-info-value" style={{ marginTop: '0.5rem', fontSize: '0.8rem' }}>
            Edit FEWSHOT_FRACTION for 1%/5%/20%.
          </div>
        </div>
      </aside>

      {/* ── Main content ── */}
      <main className="main-content">
        {/* Header */}
        <header className="app-header">
          <h1>🔧 Vibration Fault Diagnosis System</h1>
          <p>Deep Learning-Based Predictive Maintenance with Domain Adaptation</p>
        </header>

        {/* Tab bar */}
        <div className="tab-bar">
          {tabs.map(tab => (
            <button
              key={tab.id}
              className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        <div className="tab-content">
          {activeTab === 'single' && (
            <SingleSample modelName={model} datasetName={dataset} />
          )}

          {activeTab === 'evaluation' && (
            <FullEvaluation modelName={model} datasetName={dataset} />
          )}

          {activeTab === 'upload' && (
            <div className="fade-in">
              <div className="glass-card" style={{ padding: '1.75rem', marginBottom: '1.5rem' }}>
                <FileUpload file={file} onFileChange={setFile} />
              </div>

              <button
                className={`predict-btn ${uploadLoading ? 'loading' : ''}`}
                onClick={handleUploadPredict}
                disabled={!file || !model || uploadLoading}
                style={{ marginBottom: '1.5rem' }}
              >
                {uploadLoading ? (
                  <>
                    <span className="spinner" />
                    Analyzing Vibration Data…
                  </>
                ) : (
                  '⚡ Predict Fault'
                )}
              </button>

              {uploadError && <div className="error-box">❌ {uploadError}</div>}
              <Results result={uploadResult} />
            </div>
          )}
        </div>

        {/* Footer */}
        <footer className="app-footer">
          <p>🔧 Vibration Fault Diagnosis System · Deep Learning with Domain Adaptation</p>
          <p>Cross-Domain: CWRU → Paderborn · Models: CNN, CNN-LSTM, Transformer, Hybrid, DANN</p>
        </footer>
      </main>
    </div>
  );
}
