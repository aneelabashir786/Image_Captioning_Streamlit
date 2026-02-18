"""
Streamlit App for Image Captioning
-----------------------------------
- Downloads model & vocab from Hugging Face
- Greedy and Beam Search (k=3)
- Robust error handling
"""

import pickle
import re
from io import BytesIO

import streamlit as st
import requests
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ----------------------------------------------------------------------
#    YOUR HUGGING FACE LINKS – REPLACE THESE!
# ----------------------------------------------------------------------

MODEL_URL = "https://huggingface.co/aneelaBashir22f3414/captioning/resolve/main/best_model.pth"
VOCAB_URL = "https://huggingface.co/aneelaBashir22f3414/captioning/resolve/main/vocab.pkl"


# ----------------------------------------------------------------------

# ------------------------------
# 1. Model Definitions (same as training)
# ------------------------------
class Encoder(nn.Module):
    def __init__(self, feature_dim=2048, hidden_size=512, dropout=0.5):
        super().__init__()
        self.fc = nn.Linear(feature_dim, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        out = self.fc(features)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output, hidden

    def init_hidden(self, encoder_output):
        batch_size = encoder_output.size(0)
        num_layers = 2
        hidden_size = encoder_output.size(1)
        h0 = torch.zeros(num_layers, batch_size, hidden_size, device=encoder_output.device)
        h0[0] = encoder_output
        c0 = torch.zeros_like(h0)
        return (h0, c0)

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, feature_dim=2048, embed_size=256, hidden_size=512):
        super().__init__()
        self.encoder = Encoder(feature_dim, hidden_size)
        self.decoder = Decoder(vocab_size, embed_size, hidden_size)

    def forward(self, features, captions):
        enc_out = self.encoder(features)
        hidden = self.decoder.init_hidden(enc_out)
        outputs, _ = self.decoder(captions, hidden)
        return outputs

# ------------------------------
# 2. Download Helpers (cached)
# ------------------------------
@st.cache_resource
def download_vocab(url):
    response = requests.get(url)
    response.raise_for_status()
    vocab_data = pickle.loads(response.content)
    return vocab_data['word2idx'], vocab_data['idx2word']

@st.cache_resource
def download_model(url, vocab_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageCaptioningModel(vocab_size=vocab_size).to(device)
    response = requests.get(url)
    response.raise_for_status()
    buffer = BytesIO(response.content)
    state_dict = torch.load(buffer, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device

@st.cache_resource
def load_feature_extractor():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet = resnet.to(device)
    resnet.eval()
    return resnet, device

# ------------------------------
# 3. Image preprocessing
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_features(image, resnet, device):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet(img_tensor)
        features = features.view(1, -1)
    return features

# ------------------------------
# 4. Robust Inference Functions
# ------------------------------
def greedy_search(model, feature_tensor, word2idx, idx2word, max_len=30):
    model.eval()
    device = next(model.parameters()).device
    feature_tensor = feature_tensor.to(device)

    with torch.no_grad():
        enc_out = model.encoder(feature_tensor)
        hidden = model.decoder.init_hidden(enc_out)

        start_token = word2idx.get('<start>', None)
        end_token = word2idx.get('<end>', None)
        if start_token is None or end_token is None:
            raise ValueError("Missing <start> or <end> token in vocabulary!")

        curr_token = torch.tensor([[start_token]], device=device)
        caption_tokens = []

        for _ in range(max_len):
            embedded = model.decoder.embedding(curr_token)
            lstm_out, hidden = model.decoder.lstm(embedded, hidden)
            logits = model.decoder.fc(lstm_out)
            next_token = logits.argmax(dim=-1).item()

            if next_token == end_token:
                break
            # Safely get word
            word = idx2word.get(next_token, '<UNK>')
            caption_tokens.append(word)
            curr_token = torch.tensor([[next_token]], device=device)

    return ' '.join(caption_tokens)

def beam_search(model, feature_tensor, word2idx, idx2word, beam_size=3, max_len=30):
    model.eval()
    device = next(model.parameters()).device
    feature_tensor = feature_tensor.to(device)

    with torch.no_grad():
        enc_out = model.encoder(feature_tensor)
        hidden = model.decoder.init_hidden(enc_out)

        start_token = word2idx.get('<start>')
        end_token = word2idx.get('<end>')
        if start_token is None or end_token is None:
            raise ValueError("Missing <start> or <end> token in vocabulary!")

        beams = [([start_token], 0.0, hidden)]

        for _ in range(max_len):
            new_beams = []
            for seq, score, h in beams:
                if seq[-1] == end_token:
                    new_beams.append((seq, score, h))
                    continue
                curr_token = torch.tensor([[seq[-1]]], device=device)
                embedded = model.decoder.embedding(curr_token)
                lstm_out, new_h = model.decoder.lstm(embedded, h)
                logits = model.decoder.fc(lstm_out)
                log_probs = torch.log_softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(log_probs, beam_size, dim=-1)

                for i in range(beam_size):
                    token = topk_indices[0, 0, i].item()
                    token_prob = topk_probs[0, 0, i].item()
                    new_seq = seq + [token]
                    new_score = score + token_prob
                    new_beams.append((new_seq, new_score, new_h))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            if all(b[0][-1] == end_token for b in beams):
                break

        best_seq = beams[0][0]

    words = []
    for idx in best_seq[1:]:
        if idx == end_token:
            break
        words.append(idx2word.get(idx, '<UNK>'))
    return ' '.join(words)

# ------------------------------
# 5. Streamlit UI
# ------------------------------
st.set_page_config(page_title="Image Captioning", layout="centered")
st.title("Neural Storyteller – Image Captioning with Seq2Seq")
st.markdown("Upload an image and choose a decoding strategy.")

# --- Load resources ---
with st.spinner(" Downloading vocabulary..."):
    try:
        word2idx, idx2word = download_vocab(VOCAB_URL)
        vocab_size = len(word2idx)
        st.success("✅ Vocabulary loaded")
    except Exception as e:
        st.error(f"Failed to load vocabulary: {e}")
        st.stop()

with st.spinner(" Downloading model weights..."):
    try:
        model, device = download_model(MODEL_URL, vocab_size)
        st.success(f"✅ Model loaded on {device}")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

with st.spinner(" Loading ResNet50..."):
    resnet, resnet_device = load_feature_extractor()
    st.success("✅ ResNet50 ready")

# --- Image upload ---
st.subheader("1. Select an image")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info(" Please upload an image to continue.")
    st.stop()

image = Image.open(uploaded_file).convert('RGB')
st.image(image, caption="Uploaded Image", use_column_width=True)

# --- Decoding method ---
st.subheader("2. Choose decoding method")
method = st.radio(
    "Select search strategy:",
    ("Greedy Search", "Beam Search (k=3)"),
    horizontal=True
)

# --- Generate button ---
if st.button("Generate Caption", type="primary"):
    with st.spinner("Extracting features and generating..."):
        try:
            features = extract_features(image, resnet, resnet_device)
            # Features are on resnet_device, model might be on different device
            # Move to model's device
            features = features.to(device)

            if method == "Beam Search (k=3)":
                caption = beam_search(model, features, word2idx, idx2word, beam_size=3)
            else:
                caption = greedy_search(model, features, word2idx, idx2word)

            st.success("Caption generated!")
            st.markdown(f"**{method}**")
            if caption.strip():
                st.markdown(f"<h2 style='color: #2e86ab;'>{caption}</h2>", unsafe_allow_html=True)
            else:
                st.warning(" Model generated an empty caption. Try another image.")

        except Exception as e:
            st.error(f"Caption generation failed: {str(e)}")
else:
    st.info("Click the button to generate a caption.")

st.markdown("---")
# st.caption(f"Model: `{MODEL_URL}`  |  Vocab: `{VOCAB_URL}`")

