import os
import torch
import clip
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from icrawler.builtin import BingImageCrawler
from icrawler import ImageDownloader

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Prompts: building good, watermark and non-building bad
text_prompts = clip.tokenize([
    "a photo of a building exterior",
    "a photo with a watermark",
    "a random object or person"
]).to(device)

# CLIP score thresholds
THRESHOLDS = {
    "building": 0.25,
    "watermark": 0.25,
    "non_building": 0.35
}

# Block known stock image domains
banned_sources = ['alamy', 'shutterstock', 'getty', 'istock', 'dreamstime']


# Downloader with CLIP filtering
class CLIPFilteredDownloader(ImageDownloader):
    def download(self, task, default_ext, timeout=5, max_retry=3, **kwargs):
        try:
            url = task["file_url"]
            if any(bad in url.lower() for bad in banned_sources):
                return  # skip based on URL

            response = requests.get(url, timeout=timeout)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            image_tensor = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features = model.encode_text(text_prompts)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ text_features.T).squeeze(0).tolist()

            score_building, score_watermark, score_nonbuilding = similarity

            if (
                score_building >= THRESHOLDS["building"]
                and score_watermark < THRESHOLDS["watermark"]
                and score_nonbuilding < THRESHOLDS["non_building"]
            ):
                return super().download(task, default_ext, timeout, max_retry, **kwargs)

        except Exception:
            return


# Custom Bing Crawler
class CLIPFilteredBingCrawler(BingImageCrawler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, downloader_cls=CLIPFilteredDownloader, **kwargs)


# Scrape target countries
countries = {
    "Japan": [
        "Japanese traditional houses",
        "Tokyo modern buildings",
        "Kyoto street architecture",
        "Japanese residential apartments",
        "urban houses in Japan",
        "Japanese building exteriors",
        "suburban neighborhoods in Japan",
        "Japanese architecture styles",
        "city buildings in Osaka",
        "apartment complexes in Tokyo"
    ],
    "Mexico": [
        "Mexican colonial buildings",
        "Mexico City residential buildings",
        "colorful houses in Mexico",
        "Mexican street architecture",
        "Mexican town homes",
        "traditional houses in Oaxaca",
        "Mexican modern apartments",
        "buildings in Guadalajara",
        "urban streets in Mexico",
        "Mexican neighborhood houses"
    ],
    "France": [
        "Paris apartment buildings",
        "French countryside homes",
        "Haussmann architecture France",
        "residential buildings in France",
        "old buildings in Lyon",
        "urban buildings in Marseille",
        "village houses in Provence",
        "modern architecture in France",
        "French street view",
        "French building exteriors"
    ],
    "Italy": [
        "Italian stone houses",
        "Florence residential buildings",
        "Venetian apartment exteriors",
        "Rome historic buildings",
        "Tuscany countryside homes",
        "Italian city buildings",
        "Naples urban buildings",
        "residential streets in Italy",
        "Italian apartment blocks",
        "Italian architecture styles"
    ],
    "Greece": [
        "Greek village houses",
        "Santorini buildings",
        "Athens apartment buildings",
        "traditional houses in Greece",
        "urban buildings in Greece",
        "Greek island architecture",
        "neoclassical buildings in Athens",
        "whitewashed buildings in Greece",
        "Greek residential streets",
        "Greek architecture styles"
    ]
}


max_images = 10000

for country, prompts in countries.items():
    save_dir = os.path.join("filtered_images", country.replace(" ", "_"))
    os.makedirs(save_dir, exist_ok=True)

    for keyword in prompts:
        # Count how many images already exist
        existing_images = [
            f for f in os.listdir(save_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        file_idx_offset = len(existing_images)

        print(f"Scraping & filtering: {country} -> {keyword} (starting from {file_idx_offset})")

        crawler = CLIPFilteredBingCrawler(storage={"root_dir": save_dir})
        crawler.crawl(
            keyword=keyword,
            max_num=min(1000, max_images - file_idx_offset),  # Avoid exceeding limit
            filters=None,
            file_idx_offset=file_idx_offset
        )

        # Stop if we've hit the image cap
        if len(existing_images) >= max_images:
            print(f" {country} reached {max_images} images. Skipping remaining prompts.")
            break
