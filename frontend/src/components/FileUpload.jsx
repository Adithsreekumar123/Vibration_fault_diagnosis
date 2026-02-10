import { useState, useRef } from 'react';

export default function FileUpload({ file, onFileChange }) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped && dropped.name.endsWith('.mat')) {
      onFileChange(dropped);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => setDragOver(false);

  const handleClick = () => inputRef.current?.click();

  const handleInput = (e) => {
    const selected = e.target.files[0];
    if (selected) onFileChange(selected);
  };

  return (
    <div>
      <h3 className="section-title">📂 Upload Vibration File</h3>
      <div
        className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
        onClick={handleClick}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <span className="upload-icon">📁</span>
        <div className="upload-title">
          {file ? 'Change file' : 'Drop your .mat file here'}
        </div>
        <div className="upload-subtitle">
          or click to browse · Supports CWRU &amp; Paderborn .mat formats
        </div>
        {file && (
          <div className="upload-file-info">
            📄 {file.name} ({(file.size / 1024).toFixed(1)} KB)
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept=".mat"
          onChange={handleInput}
          style={{ display: 'none' }}
        />
      </div>
    </div>
  );
}
