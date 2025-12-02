import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer
import json
import pandas as pd
import matplotlib.pyplot as plt
import re

# Page config
st.set_page_config(
    page_title="Comment Classifier",
    page_icon="💬",
    layout="centered"
)

# Title
st.title("💬 Comment Classifier")
st.markdown("Classify comments into 8 categories with AI")


# ===== SIMPLE TEXT CLEANING =====
def clean_text(text):
    """Simple text cleaning"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters
    return text.strip()

model_path = "./model_files"
# ===== LOAD MODEL =====
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        # Load model and tokenizer

        # model = DistilBertForSequenceClassification.from_pretrained("./model_files")
        # tokenizer = DistilBertTokenizer.from_pretrained("./model_files")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        with open(f"{model_path}/config.json", "r") as f:
            config = json.load(f)
        # Load label mappings (from your file)
        if "id2label" in config:
            id2label = config["id2label"]
            # Convert string keys to int if needed
            id2label = {int(k): v for k, v in id2label.items()}
        elif "label2id" in config:
            label2id = config["label2id"]
            # Create id2label from label2id
            id2label = {v: k for k, v in label2id.items()}
        else:
            # Fallback: create generic labels
            num_labels = config.get("num_labels", 8)
            id2label = {i: f"Class_{i}" for i in range(num_labels)}

        try:
            with open(f"{model_path}/response_templates.json", "r") as f:
                responses = json.load(f)
        except:
            responses = {}

        return model, tokenizer, id2label, responses

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None


# ===== PREDICTION FUNCTION =====
def predict(comment, model, tokenizer, id2label):
    """Make prediction for a single comment"""
    # Clean the text
    cleaned = clean_text(comment)

    # Tokenize
    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    # Move to same device as model
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    return id2label[pred_idx], confidence


# ===== MAIN APP =====
def main():
    # Load model
    with st.spinner("Loading AI model..."):
        model, tokenizer, id2label, responses = load_model()

    if model is None:
        st.error("Could not load model. Please check model files.")
        return

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        mode = st.radio("Mode:", ["Single Comment", "Batch Analysis"])

        st.divider()
        st.markdown("**Model Info:**")
        st.caption("Accuracy: 85-87%")
        st.caption("Classes: 8")
        st.caption("Powered by DistilBERT")

    # SINGLE COMMENT MODE
    if mode == "Single Comment":
        st.subheader("Analyze Single Comment")

        # Text input
        comment = st.text_area(
            "Enter comment:",
            height=100,
            placeholder="Example: 'Amazing work! Loved it!'"
        )

        col1, col2 = st.columns(2)
        with col1:
            analyze_btn = st.button("🔍 Analyze", type="primary")
        with col2:
            if st.button("🔄 Example"):
                comment = "This video made me emotional... it reminded me of my childhood."

        if analyze_btn and comment:
            with st.spinner("Analyzing..."):
                # Predict
                category, confidence = predict(comment, model, tokenizer, id2label)

                # Display results
                st.success(f"**Category:** {category}")

                # Show response template
                if category in responses:
                    st.markdown("**Suggested Response:**")
                    st.write(responses[category])

        elif not comment and analyze_btn:
            st.warning("Please enter a comment")

    # BATCH MODE
    else:
        st.subheader("Batch Analysis")

        # Upload or paste
        option = st.radio("Input:", ["Upload CSV", "Paste Text"])

        comments = []

        if option == "Upload CSV":
            uploaded = st.file_uploader("Choose CSV file", type=['csv'])
            if uploaded:
                df = pd.read_csv(uploaded)
                if 'comment' in df.columns:
                    comments = df['comment'].tolist()
                elif 'text' in df.columns:
                    comments = df['text'].tolist()
                else:
                    st.error("File must have 'comment' or 'text' column")

        else:  # Paste Text
            text_input = st.text_area(
                "Enter comments (one per line):",
                height=150,
                placeholder="Comment 1\nComment 2\nComment 3"
            )
            if text_input:
                comments = [line.strip() for line in text_input.split('\n') if line.strip()]

        # Analyze button
        if comments and st.button("📊 Analyze All", type="primary"):
            results = []
            progress_bar = st.progress(0)

            for i, comment in enumerate(comments):
                category, confidence = predict(comment, model, tokenizer, id2label)
                results.append({
                    "Comment": comment[:50] + "..." if len(comment) > 50 else comment,
                    "Category": category
                })
                progress_bar.progress((i + 1) / len(comments))

            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)

            # Download button
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="results.csv",
                mime="text/csv"
            )

            # Simple chart
            st.subheader("Distribution")
            category_counts = results_df['Category'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
            st.pyplot(fig)


# Run the app
if __name__ == "__main__":
    main()