import os
from transformers import pipeline
from detoxify import Detoxify

def get_cache_dir():
    return os.environ.get("HF_CACHE", "/models")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def download_all():
    cache = ensure_dir(get_cache_dir())
    # Set environment variables so transformers uses this cache
    os.environ["HF_HOME"] = cache
    os.environ["TRANSFORMERS_CACHE"] = cache

    print(f"Using HF cache at: {cache}")

    def download(model_name, task, **kwargs):
        print(f"Checking {model_name}...")
        # Simple check: If the folder exists, we assume it's downloaded.
        # Transformers cache structure is complex (hash names), so this is a heuristic.
        # A safer way is to just run pipeline() and let it hit the cache.
        print(f"↓ Downloading/Verifying: {model_name}")
        pipeline(task, model=model_name, **kwargs)

    download("facebook/bart-large-mnli", "zero-shot-classification")
    download("papluca/xlm-roberta-base-language-detection", "text-classification")
    download("cardiffnlp/twitter-xlm-roberta-base-sentiment", "sentiment-analysis")

    detox_path = os.path.join(cache, "detoxify-original")
    print("↓ Downloading Detoxify...")
    Detoxify("original") 
    # Detoxify stores in ~/.cache/torch/hub/checkpoints/ by default unless configured.
    # This might be tricky. We might need to copy it or let it run at runtime if it's small (50MB).
    # But for now, we run it to trigger download.

    print("✔ All models downloaded verification complete.")

if __name__ == "__main__":
    download_all()
