import os
import json
import re
import time
import emoji
import langdetect
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from threading import Thread
from collections import Counter
from wordcloud import WordCloud
from streamlit_autorefresh import st_autorefresh
from sklearn.metrics import accuracy_score, precision_score, f1_score

from config import load_config
from utils import extract_video_id
import consumer
import producer

# --- Streamlit Configuration ---
os.environ["STREAMLIT_WATCHED_MODULES"] = "none"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
st.set_page_config(layout="wide")
st.title("🚀 YouTube Sentiment Analysis Dashboard")

# --- Session State Initialization ---
for key in ["analysis_started", "video_id", "model_choice"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "analysis_started" else False

# --- Input Section ---
video_url = st.text_input("🔗 Enter YouTube Video URL")
model_choice = st.radio("🧠 Choose sentiment analysis model(s):", 
                        ["TextBlob", "BERT", "Azure", "TextBlob and BERT", "All"])

# --- Start Button Logic ---
if st.button("Start Analysis"):
    video_id = extract_video_id(video_url)
    if not video_id:
        st.warning("❌ Please enter a valid YouTube video URL")
    else:
        if os.path.exists("sentiment_output.json"):
            os.remove("sentiment_output.json")

        st.session_state.update({
            "analysis_started": True,
            "video_id": video_url,
            "model_choice": model_choice
        })

        with st.spinner("⏳ Starting sentiment analysis..."):
            Thread(target=producer.main, args=(video_url,), daemon=True).start()
            Thread(target=consumer.main, args=(model_choice, None), daemon=True).start()

        st.success("✅ Sentiment analysis is running in the background. Dashboard will auto-update.")

# --- Dashboard Visualizations ---
if st.session_state.analysis_started and os.path.exists("sentiment_output.json"):

    # 🔁 Auto-refresh every 10 seconds
    st_autorefresh(interval=10_000, limit=None, key="sentiment_autorefresh")

    try:
        df = pd.read_json("sentiment_output.json", lines=True)

        if df.empty or "comment" not in df.columns:
            st.info("⏳ Waiting for comments to be processed...")
            st.stop()

        # Show latest comment
        st.markdown(f"**🆕 Latest Comment Processed:** _{df['comment'].iloc[-1]}_")

        # Normalize BERT sentiment labels
        if "bert_sentiment_label" in df.columns:
            df["bert_sentiment_label"] = df["bert_sentiment_label"].replace({
                "1 star": "negative", "2 stars": "negative", "3 stars": "neutral",
                "4 stars": "positive", "5 stars": "positive"
            })

        # Select Sentiment Column
        sentiment_cols = [col for col in df.columns if col.endswith("_sentiment_label")]
        selected_col = st.selectbox("📊 Choose sentiment column to visualize:", sentiment_cols)

        # Pie Chart
        sentiment_counts = df[selected_col].value_counts()
        fig_pie = px.pie(
            names=sentiment_counts.index,
            values=sentiment_counts.values,
            title=f"Sentiment Distribution - {selected_col.capitalize()}"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Total Comments Processed
        st.markdown(f"### 📝 Comments processed so far: *{len(df)}*")

        # 📈 Model Performance
        if "textblob_sentiment_label" in df.columns:
            st.markdown("## 📈 Model Performance Metrics (vs TextBlob)")

            ref = df["textblob_sentiment_label"]

            for col in sentiment_cols:
                if col == "textblob_sentiment_label":
                    continue
                pred = df[col]
                acc = accuracy_score(ref, pred)
                prec = precision_score(ref, pred, average="weighted", zero_division=0)
                f1 = f1_score(ref, pred, average="weighted", zero_division=0)

                st.metric(label=f"{col} Accuracy", value=f"{acc:.2f}")
                st.metric(label=f"{col} Precision", value=f"{prec:.2f}")
                st.metric(label=f"{col} F1 Score", value=f"{f1:.2f}")

        # ☁️ Word Cloud
        def clean_text(text):
            text = emoji.replace_emoji(text, replace='')
            return re.sub(r'[^\w\s]', '', text).lower()

        all_words = ' '.join(df['comment'].astype(str).apply(clean_text)).split()
        word_freq = Counter(all_words)

        wordcloud = WordCloud(
            width=300,
            height=150,
            background_color='white',
            margin=1
        ).generate_from_frequencies(word_freq)

        fig_wc, ax = plt.subplots(figsize=(3, 1.5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        plt.tight_layout(pad=0)
        st.markdown("### ☁️ Word Cloud of Comments")
        st.pyplot(fig_wc, clear_figure=True)

        # 🌐 Language Detection
        def detect_language_safe(comment):
            try:
                return langdetect.detect(comment)
            except:
                return 'unknown'

        lang_counts = df['comment'].astype(str).apply(detect_language_safe).value_counts()
        fig_lang = px.bar(
            x=lang_counts.index,
            y=lang_counts.values,
            labels={'x': "Language", 'y': "Count"},
            title="🌐 Language Distribution"
        )
        st.plotly_chart(fig_lang, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
