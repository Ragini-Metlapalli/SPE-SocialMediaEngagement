import os
import subprocess

def install_deps():
    print("Installing heavy dependencies inside initContainer...")
    subprocess.run([
        "pip", "install", "--no-cache-dir",
        "torch==2.1.0+cpu",
        "transformers",
        "sentencepiece",
        "detoxify",
        "protobuf",
        "-f", "https://download.pytorch.org/whl/cpu"
    ], check=True)

def download_models():
    from transformers import pipeline
    from detoxify import Detoxify

    cache = os.environ.get("HF_CACHE", "/models")
    os.makedirs(cache, exist_ok=True)

    os.environ["TRANSFORMERS_CACHE"] = cache
    os.environ["HF_HOME"] = cache

    print(f"Downloading models into: {cache}")

    pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
    pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

    Detoxify("original")

if __name__ == "__main__":
    install_deps()
    download_models()

