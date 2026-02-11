import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pickle
import requests
import os

# --------------------------------------------------
# 🔽 DOWNLOAD MODEL & VOCAB FROM HUGGINGFACE
# --------------------------------------------------

MODEL_URL = "https://huggingface.co/aneelaBashir22f3414/Image_Captioning/blob/main/best_model.pth"
VOCAB_URL = "https://huggingface.co/aneelaBashir22f3414/Image_Captioning/blob/main/vocab.pkl"

def download_file(url, filename):
    if not os.path.exists(filename):
        r = requests.get(url)
        with open(filename, "wb") as f:
            f.write(r.content)

download_file(MODEL_URL, "best_model.pth")
download_file(VOCAB_URL, "vocab.pkl")

# --------------------------------------------------
# 🔤 LOAD VOCAB
# --------------------------------------------------

with open("vocab.pkl", "rb") as f:
    vocab_data = pickle.load(f)

word2idx = vocab_data["word2idx"]
idx2word = vocab_data["idx2word"]

START_TOKEN = "<start>"
END_TOKEN = "<end>"
PAD_TOKEN = "<pad>"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# 🧠 MODEL ARCHITECTURE (MATCHES TRAINING)
# --------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, feature_dim=2048, hidden_size=512, dropout=0.5):
        super().__init__()
        self.fc = nn.Linear(feature_dim, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


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
        output, hidden = self.lstm(embedded, hidden)
        output = self.dropout(output)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, encoder_output):
        batch_size = encoder_output.size(0)
        num_layers = 2
        hidden_size = encoder_output.size(1)

        h0 = torch.zeros(num_layers, batch_size, hidden_size).to(encoder_output.device)
        h0[0] = encoder_output
        c0 = torch.zeros_like(h0)

        return (h0, c0)


class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(vocab_size)

    def forward(self, features, captions):
        enc_out = self.encoder(features)
        hidden = self.decoder.init_hidden(enc_out)
        outputs, _ = self.decoder(captions, hidden)
        return outputs


# --------------------------------------------------
# 📦 LOAD TRAINED MODEL
# --------------------------------------------------

model = ImageCaptioningModel(len(word2idx)).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# --------------------------------------------------
# 🖼️ RESNET FEATURE EXTRACTOR
# --------------------------------------------------

resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet = resnet.to(device)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# --------------------------------------------------
# ✨ GREEDY SEARCH
# --------------------------------------------------

def generate_caption(image):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feature = resnet(image).view(1,-1)
        enc_out = model.encoder(feature)
        hidden = model.decoder.init_hidden(enc_out)

        curr_token = torch.tensor([[word2idx[START_TOKEN]]]).to(device)
        caption = []

        for _ in range(30):
            output, hidden = model.decoder(curr_token, hidden)
            next_token = output.argmax(-1).item()

            word = idx2word[next_token]
            if word == END_TOKEN:
                break

            caption.append(word)
            curr_token = torch.tensor([[next_token]]).to(device)

    return " ".join(caption)


# --------------------------------------------------
# 🌐 STREAMLIT UI
# --------------------------------------------------

st.title("🖼️ Neural Storyteller - Image Captioning")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            caption = generate_caption(image)
        st.success("Caption: " + caption)
