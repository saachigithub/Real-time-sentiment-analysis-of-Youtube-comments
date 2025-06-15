import json
import time
import logging
import re
import html
import emoji
from confluent_kafka import Producer
from config import load_config
from utils import extract_video_id
from googleapiclient.discovery import build
import pytchat

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("producer")

# Load config
config = load_config()
api_key = config["youtube"]["api_key"]

# Kafka config
kafka_config = {
    'bootstrap.servers': config['kafka']['bootstrap_servers'],
    'security.protocol': config['kafka']['security_protocol'],
    'sasl.mechanisms': config['kafka']['sasl_mechanism'],
    'sasl.username': config['kafka']['sasl_username'],
    'sasl.password': config['kafka']['sasl_password'],
    'ssl.ca.location': config['kafka']['ssl_ca_location']
}

producer = Producer(kafka_config)
topic = config['kafka']['topic']

_keep_running = True  # Graceful shutdown flag

def stop_producing():
    global _keep_running
    _keep_running = False
    logger.info("Stop signal received for producer.")

def clean_text(text):
    text = html.unescape(text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[\r\n]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def is_valid_comment(text):
    cleaned = clean_text(text)
    return len(cleaned) >= 3

def delivery_report(err, msg):
    if err:
        logger.error(f"Delivery failed: {err}")
    else:
        logger.info(f"Delivered to {msg.topic()} [{msg.partition()}]")

def is_live_video(video_id):
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        response = youtube.videos().list(
            part="snippet,liveStreamingDetails",
            id=video_id
        ).execute()
        items = response.get("items", [])
        return bool(items and "liveStreamingDetails" in items[0])
    except Exception as e:
        logger.error(f"Failed to check if video is live: {e}")
        return False

def produce_live_comments(video_id):
    global _keep_running
    _keep_running = True
    logger.info(f"Starting live comment streaming for video ID: {video_id}")

    try:
        chat = pytchat.create(video_id=video_id, interruptable=False)

        while chat.is_alive() and _keep_running:
            for c in chat.get().sync_items():
                cleaned = clean_text(c.message)
                if is_valid_comment(cleaned):
                    message = json.dumps({"comment": cleaned}, ensure_ascii=False)
                    producer.produce(topic, value=message.encode("utf-8"), callback=delivery_report)
                    producer.poll(0)
                    logger.info(f"Produced comment: {cleaned[:60]}")
                if not _keep_running:
                    break
            time.sleep(0.5)

        chat.terminate()
        logger.info("Live chat ended or stopped.")
    except Exception as e:
        logger.error(f"Error in producing live comments: {e}")
    finally:
        producer.flush()
        logger.info("Producer flushed and stopped.")

def produce_regular_comments(video_id, max_comments=1000):
    logger.info(f"Fetching non-live video comments for video ID: {video_id}")
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        next_page_token = None
        total_comments = 0

        while True:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText"
            ).execute()

            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                cleaned = clean_text(comment)
                if is_valid_comment(cleaned):
                    message = json.dumps({"comment": cleaned}, ensure_ascii=False)
                    producer.produce(topic, value=message.encode("utf-8"), callback=delivery_report)
                    producer.poll(0)
                    logger.info(f"Produced comment: {cleaned[:60]}")
                    total_comments += 1

            next_page_token = response.get("nextPageToken")
            if not next_page_token or total_comments >= max_comments:
                break
            time.sleep(0.3)

    except Exception as e:
        logger.error(f"Error in producing regular video comments: {e}")
    finally:
        producer.flush()
        logger.info("Producer flushed and stopped.")

def main(video_url: str):
    video_id = extract_video_id(video_url)
    if not video_id:
        logger.error("Invalid YouTube video URL")
        return

    if is_live_video(video_id):
        logger.info("Detected live video")
        produce_live_comments(video_id)
    else:
        logger.info("Detected non-live video")
        produce_regular_comments(video_id)

    logger.info("Finished producing comments")
