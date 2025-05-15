import streamlit as st
from streamlit_mic_recorder import mic_recorder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Set page config for a polished look
st.set_page_config(page_title="Urdu Deepfake Audio Detection", page_icon="🎤", layout="wide")

# Custom CSS for an amazing UI
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { background-color: #4CAF50; color: white; }
    .stSelectbox { background-color: #ffffff; border-radius: 5px; }
    .prediction-box { background-color: #e8f5e9; padding: 10px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# Model Definitions
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze(1)

class PerceptronModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.linear(x).squeeze(1)

class LinearSVM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.linear(x).squeeze(1)

class CNNFrontEnd(nn.Module):
    def __init__(self, in_channels=13):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.MaxPool1d(2),
        )
    def forward(self, x):
        b, fm = x.size()
        t = fm // 13
        x = x.view(b, 13, t)
        x = self.conv(x)
        return x.flatten(1)

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
    def forward(self, x):
        return F.relu(self.net(x) + x)

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction), nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim), nn.Sigmoid()
        )
    def forward(self, x):
        b, f = x.size()
        y = self.pool(x.view(b, f, 1)).view(b, f)
        w = self.fc(y)
        return x * w

class ImprovedDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.front = CNNFrontEnd(in_channels=13)
        dim_after_cnn = 64 * ((200 // 2) // 2)
        self.initial_fc = nn.Linear(dim_after_cnn, 128)
        self.res_blocks = nn.Sequential(ResidualBlock(128), ResidualBlock(128))
        self.attn = ChannelAttention(128)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        x = self.front(x)
        x = F.relu(self.initial_fc(x))
        x = self.res_blocks(x)
        x = self.attn(x)
        return self.classifier(x).squeeze(1)

# Feature Extraction
def extract_mfcc(audio_data, sr=16000, n_mfcc=13, max_len=200):
    if isinstance(audio_data, str):
        y, sr = librosa.load(audio_data, sr=sr)
    else:
        y = audio_data
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.flatten()

# Training Function
def train_model(model, criterion, optimizer, train_loader, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device).float()
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb) if callable(criterion) else criterion(out, yb)
            loss.backward()
            optimizer.step()

# Load or Train Models with Caching
@st.cache_resource
def load_or_train_models():
    device = 'cpu'  # Use CPU for simplicity in Streamlit
    input_dim = 13 * 200
    model_files = {
        'LogReg': 'logreg_model.pth',
        'Perceptron': 'perceptron_model.pth',
        'SVM': 'svm_model.pth',
        'DNN': 'dnn_model.pth'
    }
    models = {}
    settings = {
        'LogReg': (nn.BCELoss(), optim.Adam),
        'Perceptron': (lambda o, l: torch.clamp(1 - (l.float() * 2 - 1) * o, min=0).mean(), optim.SGD),
        'SVM': (lambda o, l: torch.clamp(1 - (l.float() * 2 - 1) * o, min=0).mean(), optim.SGD),
        'DNN': (nn.BCELoss(), optim.Adam)
    }

    # Load dataset if training is needed
    if not all(os.path.exists(f) for f in model_files.values()):
        ds = load_dataset("CSALT/deepfake_detection_dataset_urdu", split="train")
        ds = ds.map(lambda ex: {"label": 0 if "Bonafide" in ex["audio"]["path"] else 1})
        ds = ds.map(lambda ex: {"features": extract_mfcc(ex["audio"]["array"])}, remove_columns=["audio"])
        X = np.stack(ds["features"])
        y = np.array(ds["label"])
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    for name, file in model_files.items():
        if name == 'DNN':
            model = ImprovedDNN(input_dim)
        elif name == 'LogReg':
            model = LogisticRegressionModel(input_dim)
        elif name == 'Perceptron':
            model = PerceptronModel(input_dim)
        elif name == 'SVM':
            model = LinearSVM(input_dim)
        
        model.to(device)
        if os.path.exists(file):
            model.load_state_dict(torch.load(file, map_location=device))
        else:
            crit, opt_class = settings[name]
            optimizer = opt_class(model.parameters(), lr=1e-3, weight_decay=1e-4)
            train_model(model, crit, optimizer, train_loader, device)
            torch.save(model.state_dict(), file)
            st.success(f"Trained and saved {name} model!")
        model.eval()
        models[name] = model
    return models

models = load_or_train_models()

# Streamlit UI
st.title("🎙️ Urdu Deepfake Audio Detection")
st.markdown("Detect if an audio is **Original (Bonafide)** or **Fake (Deepfake)** with confidence scores!")

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Select Prediction Model")
    model_choice = st.selectbox("Choose a model", list(models.keys()), help="Select the model for prediction")

with col2:
    st.subheader("Audio Input Method")
    audio_option = st.radio("How would you like to provide audio?", ("Upload File", "Record Live"))

# Audio Input
if audio_option == "Upload File":
    uploaded_file = st.file_uploader("Upload an Audio File", type=["wav", "mp3"], help="Supported formats: WAV, MP3")
    if uploaded_file:
        audio_bytes = uploaded_file.read()
        audio_path = "temp_upload.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
    else:
        audio_path = None
else:
    st.write("Press the button below to start recording.")
    audio = mic_recorder(start_prompt="🎤 Start", stop_prompt="⏹️ Stop", key="recorder")
    if audio:
        audio_path = "temp_record.wav"
        with open(audio_path, "wb") as f:
            f.write(audio['bytes'])
    else:
        audio_path = None

# Prediction
if audio_path and model_choice in models:
    with st.spinner("Analyzing audio..."):
        features = extract_mfcc(audio_path)
        features_tensor = torch.from_numpy(features).float().unsqueeze(0).to('cpu')
        model = models[model_choice]
        with torch.no_grad():
            output = model(features_tensor)
            prob = output.item() if model_choice in ['LogReg', 'DNN'] else torch.sigmoid(output).item()
            prediction = "Fake" if prob > 0.5 else "Original"
            confidence = prob if prediction == "Fake" else 1 - prob
    
    st.markdown(f"<div class='prediction-box'><h3>Prediction: {prediction}</h3><p>Confidence: {confidence:.2f}</p></div>", unsafe_allow_html=True)
    st.audio(audio_path)
    if os.path.exists(audio_path):
        os.remove(audio_path)
else:
    if not audio_path:
        st.info("Please upload an audio file or record audio to get a prediction.")
    if model_choice not in models:
        st.error("Selected model is unavailable. Please try another.")

st.markdown("---")
st.write("Built with ❤️ using Streamlit and PyTorch by Anas Altaf")
