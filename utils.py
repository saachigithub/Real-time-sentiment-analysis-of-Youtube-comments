# utils.py
from urllib.parse import urlparse, parse_qs

def extract_video_id(url):
    query = urlparse(url)
    if query.hostname in ["www.youtube.com", "youtube.com"]:
        return parse_qs(query.query).get("v", [None])[0]
    elif query.hostname == "youtu.be":
        return query.path[1:]
    return None
