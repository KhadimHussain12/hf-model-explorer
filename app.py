import os
import io
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from PIL import Image

# Load environment variables (for local dev)
load_dotenv()

# Streamlit Page Setup
st.set_page_config(
    page_title="Hugging Face Model Explorer",
    page_icon="🤗",
    layout="wide"
)

# Initialize HF Inference Client
@st.cache_resource
def get_hf_client():
    hf_token = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN", None)
    return InferenceClient(token=hf_token)

client = get_hf_client()

# Header Section
st.title("🤗 Hugging Face Model Explorer")
st.markdown("""
Explore open-source machine learning models hosted on the Hugging Face Hub using direct inference endpoints.
""")

# Setup Navigation Tabs
tab_sentiment, tab_summarize, tab_vision = st.tabs([
    "💬 Sentiment Analysis",
    "📝 Text Summarization", 
    "🖼️ Object Recognition"
])

# ------------------------------------------------------------------
# Tab 1: Sentiment Analysis
# ------------------------------------------------------------------
with tab_sentiment:
    st.header("Sentiment Analysis")
    st.caption("Model: `distilbert/distilbert-base-uncased-finetuned-sst-2-english`")
    
    user_text = st.text_area(
        "Enter text to analyze sentiment:",
        placeholder="I absolutely love building web apps with Streamlit and open source AI!",
        key="sentiment_input"
    )
    
    if st.button("Analyze Sentiment", type="primary", key="btn_sentiment"):
        if not user_text.strip():
            st.warning("Please enter text to analyze.")
        else:
            with st.spinner("Querying Hugging Face Inference API..."):
                try:
                    response = client.text_classification(
                        text=user_text,
                        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
                    )
                    
                    st.success("Analysis Complete!")
                    
                    top_label = response[0].label
                    top_score = response[0].score
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Predicted Sentiment", top_label)
                    col2.metric("Confidence Score", f"{top_score * 100:.2f}%")
                    
                    st.write("##### Classification Details")
                    scores_dict = {item.label: item.score for item in response}
                    st.bar_chart(scores_dict)

                except Exception as e:
                    st.error(f"Inference Error: {str(e)}")

# ------------------------------------------------------------------
# Tab 2: Text Summarization
# ------------------------------------------------------------------
with tab_summarize:
    st.header("Text Summarization")
    st.caption("Model: `facebook/bart-large-cnn`")
    
    article_text = st.text_area(
        "Paste article or long text block:",
        height=200,
        placeholder="Paste a multi-paragraph article here to test summarization...",
        key="summary_input"
    )
    
    if st.button("Generate Summary", type="primary", key="btn_summarize"):
        if len(article_text.split()) < 30:
            st.warning("Please enter a longer text sample (at least 30 words) for meaningful summarization.")
        else:
            with st.spinner("Summarizing text..."):
                try:
                    summary_result = client.summarization(
                        text=article_text,
                        model="facebook/bart-large-cnn"
                    )
                    
                    summary_text = summary_result.summary_text
                    
                    st.subheader("Summary Result")
                    st.write(summary_text)
                    
                    orig_words = len(article_text.split())
                    sum_words = len(summary_text.split())
                    st.info(f"Reduced from **{orig_words} words** down to **{sum_words} words** ({(1 - sum_words/orig_words)*100:.1f}% reduction).")

                except Exception as e:
                    st.error(f"Inference Error: {str(e)}")

# ------------------------------------------------------------------
# Tab 3: Object Recognition (Vision) - FIXED FOR PIL BYTES
# ------------------------------------------------------------------
with tab_vision:
    st.header("Object Recognition / Image Classification")
    st.caption("Model: `google/vit-base-patch16-224`")
    
    uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col_img, col_preds = st.columns([1, 1])
        
        # Open image with PIL
        image = Image.open(uploaded_file)
        
        with col_img:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
        with col_preds:
            if st.button("Classify Image", type="primary", key="btn_vision"):
                with st.spinner("Analyzing image features..."):
                    try:
                        # Convert PIL image to raw bytes with explicit PNG/JPEG format
                        buf = io.BytesIO()
                        img_format = image.format if image.format else "JPEG"
                        image.save(buf, format=img_format)
                        img_bytes = buf.getvalue()
                        
                        predictions = client.image_classification(
                            image=img_bytes,
                            model="google/vit-base-patch16-224"
                        )
                        
                        st.subheader("Top Predictions")
                        for pred in predictions[:5]:
                            st.write(f"**{pred.label.title()}**: {pred.score * 100:.1f}%")
                            st.progress(float(pred.score))
                            
                    except Exception as e:
                        st.error(f"Inference Error: {str(e)}")
