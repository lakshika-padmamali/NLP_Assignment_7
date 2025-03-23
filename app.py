import streamlit as st
import torch
from transformers import DistilBertForSequenceClassification, AutoTokenizer
import loralib as lora
import random

# ✅ Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "lora_distilbert_cpu.pt"

# Load base model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
for layer in model.distilbert.transformer.layer:
    d_in = layer.attention.q_lin.in_features
    d_out = layer.attention.q_lin.out_features
    layer.attention.q_lin = lora.Linear(d_in, d_out, r=8)
    layer.attention.v_lin = lora.Linear(d_in, d_out, r=8)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
model.to(device)
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# ✅ Set page config and yellow background with black text
st.set_page_config(page_title="Toxic Comment Classifier", layout="centered")

st.markdown("""
<style>
    body, .stApp {
        background-color: #fff6cc;
        color: black;
    }
    .black-text {
        color: black;
        font-weight: bold;
        font-size: 18px;
    }
    .confidence {
        color: black;
        font-size: 16px;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# ✅ Title
st.markdown('<h1 class="black-text">🛡️ Toxic Comment Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="black-text">Enter a comment below to check if it\'s toxic or non-toxic using a fine-tuned DistilBERT model with LoRA.</p>', unsafe_allow_html=True)

# ✅ Prediction Function
def classify_toxicity(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1).squeeze()
    confidence = float(torch.max(probs))
    prediction = torch.argmax(probs, dim=-1).item()
    return prediction, confidence

# ✅ Emojis and Explanation Text
toxic_emojis = ["💢", "😡", "🔥", "🚫", "🤬"]
non_toxic_emojis = ["😇", "🌸", "😊", "✅", "💚"]
explanations = {
    "toxic": "This comment may contain offensive language, threats, or hate speech.",
    "non-toxic": "This appears to be a respectful or neutral comment."
}

# ✅ Input
st.markdown('<p class="black-text">💬 Type a comment:</p>', unsafe_allow_html=True)
user_input = st.text_area("", height=100)

# ✅ Prediction Button
if st.button("🚀 Classify Comment"):
    if user_input.strip():
        label, confidence = classify_toxicity(user_input)
        emoji = random.choice(toxic_emojis if label == 1 else non_toxic_emojis)

        if label == 1:
            st.markdown(f'<p class="black-text">Prediction:<br>🔥 Toxic {emoji}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="black-text">🧠 {explanations["toxic"]}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="black-text">Prediction:<br>✅ Non-Toxic {emoji}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="black-text">🧠 {explanations["non-toxic"]}</p>', unsafe_allow_html=True)

        st.markdown(f'<p class="confidence">Confidence: {confidence*100:.2f}%</p>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a valid comment.")


