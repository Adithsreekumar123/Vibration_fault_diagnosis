# ui_app.py
"""
Comprehensive Streamlit UI for Vibration Fault Diagnosis

Features:
- Model selection (CNN, CNN-LSTM, Transformer, DANN)
- Dataset selection (CWRU, Paderborn)
- Signal visualization
- Fault prediction with confidence
- All class probabilities
- Confusion matrix visualization
- Accuracy, Precision, Recall, F1-score
- Full classification report
"""

import os
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)

# Import models
from src.models.cnn_classifier import CNNClassifier
from src.models.cnn_lstm_classifier import CNNLSTMClassifier
from src.models.transformer_classifier import TransformerClassifier
from src.train.train_dann import DANNModel

# Page config
st.set_page_config(
    page_title="Vibration Fault Diagnosis",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .fault-normal { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .fault-ball { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .fault-inner { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .fault-outer { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Constants
CLASS_NAMES = ["Normal", "Ball Fault", "Inner Race Fault", "Outer Race Fault"]
CLASS_COLORS = ["#38ef7d", "#f5576c", "#00f2fe", "#fee140"]

MODEL_PATHS = {
    "CNN (CWRU Supervised)": "results/supervised/cnn_cwru_supervised.pt",
    "CNN-LSTM (CWRU)": "results/supervised/cnn_lstm_cwru.pt",
    "Transformer (CWRU)": "results/supervised/transformer_cwru.pt",
    "CNN-DANN (Domain Adapted)": "results/dann/cnn_dann.pt"
}

DATA_PATHS = {
    "CWRU (Lab Data)": "data/processed/cwru_windows.npz",
    "Paderborn (Real-World Data)": "data/processed/paderborn_windows.npz"
}


@st.cache_resource
def load_model(model_name, model_path):
    """Load model with caching"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(model_path):
        return None, device
    
    if "DANN" in model_name:
        model = DANNModel(num_classes=4).to(device)
    elif "CNN-LSTM" in model_name:
        model = CNNLSTMClassifier(num_classes=4).to(device)
    elif "Transformer" in model_name:
        model = TransformerClassifier(num_classes=4).to(device)
    else:
        model = CNNClassifier(num_classes=4).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, device


@st.cache_data
def load_data(data_path):
    """Load dataset with caching"""
    if not os.path.exists(data_path):
        return None, None
    data = np.load(data_path)
    return data["X"], data["y"]


def predict_single(model, signal, device, is_dann=False):
    """Predict fault for a single signal"""
    x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if is_dann:
            logits, _, _ = model(x)
        else:
            logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    pred_class = np.argmax(probs)
    confidence = probs[pred_class]
    
    return pred_class, confidence, probs


def evaluate_dataset(model, X, y, device, is_dann=False):
    """Evaluate model on entire dataset"""
    y_true = []
    y_pred = []
    all_probs = []
    
    batch_size = 256
    n_samples = len(X)
    
    for i in range(0, n_samples, batch_size):
        batch_X = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]
        
        x = torch.tensor(batch_X, dtype=torch.float32).unsqueeze(1).to(device)
        
        with torch.no_grad():
            if is_dann:
                logits, _, _ = model(x)
            else:
                logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        preds = np.argmax(probs, axis=1)
        
        y_true.extend(batch_y.tolist())
        y_pred.extend(preds.tolist())
        all_probs.extend(probs.tolist())
    
    return np.array(y_true), np.array(y_pred), np.array(all_probs)


def plot_signal(signal, title="Vibration Signal"):
    """Plot vibration signal"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=signal,
        mode='lines',
        name='Signal',
        line=dict(color='#1E88E5', width=1)
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis_title="Sample",
        yaxis_title="Amplitude",
        template="plotly_white",
        height=300,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig


def plot_probabilities(probs, class_names):
    """Plot class probabilities"""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=class_names,
        y=probs * 100,
        marker_color=CLASS_COLORS,
        text=[f"{p:.1f}%" for p in probs * 100],
        textposition='auto'
    ))
    fig.update_layout(
        title=dict(text="Class Probabilities", x=0.5, font=dict(size=16)),
        xaxis_title="Fault Type",
        yaxis_title="Probability (%)",
        template="plotly_white",
        height=300,
        yaxis=dict(range=[0, 100]),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig


def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix heatmap"""
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=class_names,
        y=class_names,
        color_continuous_scale="Blues",
        text_auto=True
    )
    fig.update_layout(
        title=dict(text="Confusion Matrix", x=0.5, font=dict(size=16)),
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🔧 Vibration Fault Diagnosis System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deep Learning-Based Predictive Maintenance with Domain Adaptation</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Model selection
    st.sidebar.subheader("Model Selection")
    model_name = st.sidebar.selectbox(
        "Select Model",
        list(MODEL_PATHS.keys()),
        index=3  # Default to DANN
    )
    model_path = MODEL_PATHS[model_name]
    
    # Dataset selection
    st.sidebar.subheader("Dataset Selection")
    dataset_name = st.sidebar.selectbox(
        "Select Dataset",
        list(DATA_PATHS.keys()),
        index=1  # Default to Paderborn
    )
    data_path = DATA_PATHS[dataset_name]
    
    # Load model
    model, device = load_model(model_name, model_path)
    is_dann = "DANN" in model_name
    
    if model is None:
        st.error(f"❌ Model not found: {model_path}")
        st.info("Please train the model first using the appropriate run script.")
        return
    
    # Load data
    X, y = load_data(data_path)
    
    if X is None:
        st.error(f"❌ Dataset not found: {data_path}")
        st.info("Please run preprocessing first: `python run_preprocess.py`")
        return
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Dataset Info")
    st.sidebar.text(f"Samples: {len(X)}")
    st.sidebar.text(f"Window Size: {X.shape[1]}")
    st.sidebar.text(f"Device: {device}")
    
    # Main content tabs
    tab1, tab2 = st.tabs(["🎯 Single Sample Prediction", "📊 Full Dataset Evaluation"])
    
    # ==================== TAB 1: Single Sample ====================
    with tab1:
        st.subheader("Single Sample Fault Prediction")
        
        # Sample selector
        col1, col2 = st.columns([2, 1])
        with col1:
            sample_idx = st.slider(
                "Select Sample Index",
                0, len(X) - 1, 0,
                help="Choose a sample from the dataset"
            )
        with col2:
            random_btn = st.button("🎲 Random Sample")
            if random_btn:
                sample_idx = np.random.randint(0, len(X))
                st.experimental_rerun()
        
        # Get sample
        signal = X[sample_idx]
        true_label = y[sample_idx]
        
        # Predict
        pred_class, confidence, probs = predict_single(model, signal, device, is_dann)
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🎯 Predicted Fault", CLASS_NAMES[pred_class])
        with col2:
            st.metric("📊 Confidence", f"{confidence:.1%}")
        with col3:
            is_correct = pred_class == true_label
            st.metric("✓ Ground Truth", CLASS_NAMES[true_label])
        
        # Status indicator
        if is_correct:
            st.success("✅ Correct Prediction!")
        else:
            st.error(f"❌ Misclassified (True: {CLASS_NAMES[true_label]})")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(plot_signal(signal, "Vibration Signal"), use_container_width=True)
        
        with col2:
            st.plotly_chart(plot_probabilities(probs, CLASS_NAMES), use_container_width=True)
        
        # Detailed probabilities
        st.subheader("📋 Detailed Class Probabilities")
        prob_cols = st.columns(4)
        for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
            with prob_cols[i]:
                color = "🟢" if i == pred_class else "⚪"
                st.metric(f"{color} {name}", f"{prob:.2%}")
    
    # ==================== TAB 2: Full Evaluation ====================
    with tab2:
        st.subheader("Full Dataset Evaluation")
        
        # Evaluate button
        if st.button("🚀 Run Full Evaluation", type="primary"):
            with st.spinner("Evaluating model on entire dataset..."):
                y_true, y_pred, all_probs = evaluate_dataset(model, X, y, device, is_dann)
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            cm = confusion_matrix(y_true, y_pred)
            
            # Display main metrics
            st.markdown("### 📈 Overall Metrics")
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                st.metric("🎯 Accuracy", f"{accuracy:.2%}")
            with metric_cols[1]:
                st.metric("📊 Precision", f"{precision:.2%}")
            with metric_cols[2]:
                st.metric("🔍 Recall", f"{recall:.2%}")
            with metric_cols[3]:
                st.metric("⚖️ F1-Score", f"{f1:.2%}")
            
            # Confusion matrix
            st.markdown("### 🔢 Confusion Matrix")
            st.plotly_chart(plot_confusion_matrix(cm, CLASS_NAMES), use_container_width=True)
            
            # Per-class metrics
            st.markdown("### 📋 Per-Class Metrics")
            report = classification_report(
                y_true, y_pred,
                target_names=CLASS_NAMES,
                output_dict=True
            )
            
            # Create metrics table
            class_metrics = []
            for name in CLASS_NAMES:
                metrics = report[name]
                class_metrics.append({
                    "Class": name,
                    "Precision": f"{metrics['precision']:.2%}",
                    "Recall": f"{metrics['recall']:.2%}",
                    "F1-Score": f"{metrics['f1-score']:.2%}",
                    "Support": int(metrics['support'])
                })
            
            st.table(class_metrics)
            
            # Raw confusion matrix values
            st.markdown("### 📊 Confusion Matrix Values")
            cm_df = {
                "": CLASS_NAMES,
                CLASS_NAMES[0]: cm[:, 0].tolist(),
                CLASS_NAMES[1]: cm[:, 1].tolist(),
                CLASS_NAMES[2]: cm[:, 2].tolist(),
                CLASS_NAMES[3]: cm[:, 3].tolist()
            }
            st.dataframe(cm_df)
            
            # Summary
            st.markdown("---")
            st.success(f"""
            ### ✅ Evaluation Complete!
            
            **Model:** {model_name}  
            **Dataset:** {dataset_name}  
            **Total Samples:** {len(y_true)}  
            **Correct Predictions:** {np.sum(y_true == y_pred)}  
            **Accuracy:** {accuracy:.2%}
            """)
        else:
            st.info("👆 Click 'Run Full Evaluation' to evaluate the model on the entire dataset.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔧 Vibration Fault Diagnosis System | Deep Learning with Domain Adaptation</p>
        <p>Cross-Domain: CWRU → Paderborn | Models: CNN, CNN-LSTM, Transformer, DANN</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
