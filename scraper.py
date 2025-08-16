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
        "houses near Japanese rice fields",
        "Kyoto traditional machiya houses",
        "Gassho-style farmhouses in Japan",
        "Japanese coastal houses",
        "residential Osaka neighborhoods",
        "apartment balconies in Tokyo",
        "Japanese townhouses",
        "homes along narrow alleys in Japan",
        "small shops and residences in Japan",
        "mountain villages in Japan",
        "Shinto shrine buildings in Japan",
        "Japanese homes with gardens",
        "townhouses in Kobe Japan",
        "traditional houses with sliding doors Japan",
        "modern glass houses in Japan",
        "historic Japanese streets",
        "Japanese residential areas with power lines",
        "wooden mountain lodges in Japan",
        "homes near Japanese temples",
        "Japanese houses by the sea"
    ],
    "Mexico": [
        "colonial plazas with surrounding buildings Mexico",
        "mountain village homes in Mexico",
        "houses in Oaxaca Mexico",
        "colorful hillside houses in Guanajuato",
        "Mexican ranch-style homes",
        "traditional Yucatan homes",
        "homes along cobblestone streets Mexico",
        "residential areas in Cancun",
        "old hacienda buildings Mexico",
        "Mexican beach houses",
        "homes with flat roofs in Mexico",
        "whitewashed houses in Mexico",
        "apartments in Mexico City",
        "street views in Zacatecas",
        "painted houses in Mexican small towns",
        "residential neighborhoods in Baja Sur",
        "houses with wrought iron balconies Mexico",
        "homes near Mexican markets",
        "low-rise apartments in Mexico",
        "colorful street facades in Mexico"
    ],
    "France": [
        "urban buildings in Lille",
        "historic buildings in Strasbourg",
        "residential homes in Cannes",
        "coastal homes in Normandy",
        "French provincial houses",
        "modern housing in Lyon",
        "residential streets in Toulouse",
        "French homes in Alsace",
        "French apartments with balconies",
        "sunset over French buildings",
        "residential areas in Marseille",
        "stone houses in rural France",
        "village rooftops in Provence",
        "Parisian apartment facades",
        "half-timbered houses in France",
        "medieval town buildings in France",
        "homes along the French Riviera",
        "French cottages in the countryside",
        "row houses in French towns",
        "houses near lavender fields in Provence",
    ],
    "Italy": [
        "houses in Puglia Italy",
        "medieval Italian hill towns",
        "Italian coastal villas",
        "townhouses in Florence",
        "stone farmhouses in Italy",
        "colorful seaside buildings in Italy",
        "residential streets in Genoa",
        "historic piazza buildings Italy",
        "old city walls and houses Italy",
        "residential areas in Turin",
        "apartments with shutters in Italy",
        "mountain villages in northern Italy",
        "houses in Capri Italy",
        "Italian streets with laundry hanging",
        "residential neighborhoods in Rome",
        "Italian rustic cottages",
        "colorful fishing village homes Italy",
        "houses on Italian lake shores",
        "vineyard farmhouses in Italy",
        "old stone steps and homes Italy"
    ],
    "Greece": [
        "apartment buildings in Greek cities",
        "Greek seaside houses",
        "fishing village homes in Greece",
        "historic homes in Rhodes",
        "residential streets in Naxos",
        "Greek village squares and houses",
        "coastal villas in Greece",
        "whitewashed houses with blue doors Greece",
        "residential areas in Chania Crete",
        "houses near Greek olive groves",
        "Greek hillside homes",
        "stone cottages in Greek countryside",
        "harbor houses in Greek islands",
        "modern homes in Athens suburbs",
        "houses with bougainvillea in Greece",
        "narrow stepped streets with houses Greece",
        "old windmills and nearby homes Greece",
        "island homes overlooking the Aegean",
        "Greek houses with domed roofs",
        "residential neighborhoods in Greek coastal towns"
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
            if f.lower().endswith((".jpg"))
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
