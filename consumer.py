import json
import time
import re
import emoji
from confluent_kafka import Consumer
from config import load_config
from textblob import TextBlob
from transformers import pipeline
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# Load configuration
config = load_config()
kafka_conf = config["kafka"]
azure_conf = config["azure"]
bert_conf = config["bert"]

# Initialize BERT sentiment model
bert_pipeline = pipeline("sentiment-analysis", model=bert_conf["model_name"])

# Initialize Azure sentiment client
azure_client = TextAnalyticsClient(
    endpoint=azure_conf["endpoint"],
    credential=AzureKeyCredential(azure_conf["key"])
)

def map_bert_stars_to_label(label):
    mapping = {
        "1 star": "negative", "2 stars": "negative", "3 stars": "neutral",
        "4 stars": "positive", "5 stars": "positive",
        "negative": "negative", "neutral": "neutral", "positive": "positive",
        "label_0": "negative", "label_1": "neutral", "label_2": "positive",
        "LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive",
    }
    return mapping.get(label.strip().lower(), "neutral")

def is_valid_comment(comment):
    if not comment or len(comment.strip()) < 3:
        return False
    cleaned = emoji.replace_emoji(comment, replace='')
    cleaned = re.sub(r'\W+', '', cleaned)
    return len(cleaned) > 2

def analyze_sentiment(text, method):
    try:
        if method == "TextBlob":
            polarity = TextBlob(text).sentiment.polarity
            return "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral"
        elif method == "BERT":
            result = bert_pipeline(text)[0]
            return map_bert_stars_to_label(result["label"])
        elif method == "Azure":
            response = azure_client.analyze_sentiment([text])[0]
            return response.sentiment.lower()
    except Exception as e:
        print(f"[Error] {method} sentiment failed: {e}")
        return "neutral"

def create_kafka_consumer():
    return Consumer({
        'bootstrap.servers': kafka_conf["bootstrap_servers"],
        'security.protocol': kafka_conf["security_protocol"],
        'sasl.mechanism': kafka_conf["sasl_mechanism"],
        'sasl.username': kafka_conf["sasl_username"],
        'sasl.password': kafka_conf["sasl_password"],
        'ssl.ca.location': kafka_conf["ssl_ca_location"],
        'group.id': 'youtube-sentiment-group',
        'auto.offset.reset': 'earliest'
    })

def stream_sentiments(model_choice="All"):
    """
    Yields sentiment analysis results for valid comments from Kafka.
    Each result is a dictionary:
    {
        "comment": "...",
        "textblob_sentiment_label": "...",
        "bert_sentiment_label": "...",
        "azure_sentiment_label": "..."
    }
    """
    consumer = create_kafka_consumer()
    consumer.subscribe([kafka_conf["topic"]])

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None or msg.error():
                continue

            try:
                data = json.loads(msg.value().decode("utf-8"))
                comment = data.get("comment", "").strip()
                if not is_valid_comment(comment):
                    continue

                result = {"comment": comment}

                if model_choice in ["TextBlob", "TextBlob and BERT", "All"]:
                    result["textblob_sentiment_label"] = analyze_sentiment(comment, "TextBlob")
                if model_choice in ["BERT", "TextBlob and BERT", "All"]:
                    result["bert_sentiment_label"] = analyze_sentiment(comment, "BERT")
                if model_choice in ["Azure", "All"]:
                    result["azure_sentiment_label"] = analyze_sentiment(comment, "Azure")

                yield result

            except Exception as e:
                print(f"[Error] Failed to process message: {e}")

    finally:
        consumer.close()

def main(model_choice="All", max_comments=None):  # Allow unlimited comments if None
    with open("sentiment_output.json", "a", encoding="utf-8") as f:
        for i, sentiment in enumerate(stream_sentiments(model_choice)):
            if max_comments and i >= max_comments:
                break
            print(sentiment)
            f.write(json.dumps(sentiment, ensure_ascii=False) + "\n")
            f.flush()  

