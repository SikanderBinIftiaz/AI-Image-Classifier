"""Central configuration: folders, file types, app name."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"          # thumbnails shown in History
DATASET_DIR = DATA_DIR / "datasets"        # extracted ZIP datasets
SCREENSHOT_DIR = DATA_DIR / "screenshots"  # saved from the live camera script
MODELS_DIR = BASE_DIR / "saved_models"     # custom models you train
DB_PATH = DATA_DIR / "history.db"          # SQLite database (prediction history)

for _folder in (DATA_DIR, UPLOAD_DIR, DATASET_DIR, SCREENSHOT_DIR, MODELS_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

APP_NAME = "VisionAI"
APP_TAGLINE = "Image classification studio"
ALLOWED_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]
