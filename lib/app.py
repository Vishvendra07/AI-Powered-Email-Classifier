import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from bs4 import BeautifulSoup
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load vectorizer and model
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = load_model("cnn_email_classifier.h5")

# Labels and explanations
label_map = {
    0: "Ham ✅",
    1: "Spam ⚠️",
    2: "Phishing 🚨"
}

explanation = {
    0: (
        "This message appears to be a regular, non-malicious email. "
        "It lacks spam-like characteristics such as suspicious links, exaggerated language, or urgency. "
        "Typical signs of legitimate communication are present — such as neutral tone and clear intent."
    ),
    1: (
        "The email displays typical spam indicators such as promotional language, urgency (e.g., 'act now'), "
        "and possibly questionable links or offers. While it may not be overtly dangerous, it resembles unsolicited marketing or bulk messages."
    ),
    2: (
        "Phishing traits detected. This message attempts to impersonate a trusted service or institution, "
        "often requesting login credentials, payment information, or verification. It may contain fake links, scare tactics, "
        "or impersonation techniques commonly found in phishing attempts."
    )
}

STOPWORDS = set(stopwords.words("english"))

def preprocess_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower().split()
    return " ".join([word for word in text if word not in STOPWORDS])

# Streamlit UI
st.set_page_config(page_title="AI-Powered Email Classifier", page_icon="📧")
st.title("📧 AI-Powered Email Classifier")
email_input = st.text_area("Paste your email content here:", height=250)

if st.button("Classify"):
    if not email_input.strip():
        st.warning("Please enter some email content.")
    else:
        clean_text = preprocess_text(email_input)
        vec = vectorizer.transform([clean_text])
        cnn_input = vec.toarray().reshape(1, vec.shape[1], 1)

        probs = model.predict(cnn_input)[0]
        predicted_label = np.argmax(probs)

        st.subheader("📊 Prediction Probabilities:")
        for i, p in enumerate(probs):
            st.write(f"{label_map[i]}: {p*100:.2f}%")

        st.success(f"✅ **Final Prediction:** {label_map[predicted_label]}")
        st.info(f"💡 **Why?** {explanation[predicted_label]}")

        st.caption("📝 Note: Phishing and spam can overlap — more phishing samples improve detection.")
