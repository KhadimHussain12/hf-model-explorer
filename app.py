import os
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

st.set_page_config(page_title="HF Model Explorer", page_icon="🤗", layout="wide")

@st.cache_resource
def get_hf_client():
    hf_token = os.getenv("HF_TOKEN")
    return InferenceClient(token=hf_token)

client = get_hf_client()

st.title("🤗 Hugging Face Model Explorer (Colab Demo)")

tab_sentiment, tab_summarize, tab_vision = st.tabs([
    "💬 Sentiment Analysis", "📝 Text Summarization", "🖼️ Object Recognition"
])

with tab_sentiment:
    st.header("Sentiment Analysis")
    user_text = st.text_area("Enter text:", "I am having an amazing time building AI apps with Streamlit!", key="sent_in")
    if st.button("Analyze Sentiment", type="primary"):
        with st.spinner("Processing..."):
            try:
                res = client.text_classification(text=user_text, model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
                col1, col2 = st.columns(2)
                col1.metric("Label", res[0].label)
                col2.metric("Score", f"{res[0].score * 100:.2f}%")
            except Exception as e:
                st.error(f"Error: {e}")

with tab_summarize:
    st.header("Text Summarization")
    article = st.text_area("Enter article (at least 30 words):", height=150, key="sum_in")
    if st.button("Summarize", type="primary"):
        with st.spinner("Summarizing..."):
            try:
                res = client.summarization(text=article, model="facebook/bart-large-cnn")
                st.write(res.summary_text)
            except Exception as e:
                st.error(f"Error: {e}")

with tab_vision:
    st.header("Object Recognition")
    img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if img_file and st.button("Classify Image", type="primary"):
        with st.spinner("Classifying..."):
            try:
                img_bytes = img_file.read()
                preds = client.image_classification(image=img_bytes, model="google/vit-base-patch16-224")
                for p in preds[:5]:
                    st.write(f"**{p.label.title()}**: {p.score * 100:.1f}%")
                    st.progress(float(p.score))
            except Exception as e:
                st.error(f"Error: {e}")
