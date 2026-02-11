import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pickle
import numpy as np

# -----------------------------
# Load Vocabulary
# -----------------------------
with open("vocab.pkl", "rb") as f:
    vocab_data = pickle.load(f)

word2idx = vocab_data["word2idx"]
idx2word = vocab_data["idx2word"]

START_TOKEN = "<start>"
END_TOKEN = "<end>"
PAD_TOKEN = "<pad>"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Model Definition
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, feature_dim=2048, hidden_size=512):
        super().__init__()
        self.fc = nn.Linear(feature_dim, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, encoder_output):
        h0 = encoder_output.unsqueeze(0)
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


# -----------------------------
# Load Model
# -----------------------------
model = ImageCaptioningModel(len(word2idx)).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# -----------------------------
# ResNet Feature Extractor
# -----------------------------
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

# -----------------------------
# Greedy Search
# -----------------------------
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

            if idx2word[next_token] == END_TOKEN:
                break

            caption.append(idx2word[next_token])
            curr_token = torch.tensor([[next_token]]).to(device)

    return " ".join(caption)


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🖼️ Neural Storyteller - Image Captioning")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        caption = generate_caption(image)
        st.success("Caption: " + caption)
