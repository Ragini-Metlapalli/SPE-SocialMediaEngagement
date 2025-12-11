import os
from transformers import pipeline
from detoxify import Detoxify

def download_all():
    print(" Downloading Topic Classifier (facebook/bart-large-mnli)...")
    pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    print(" Downloading Language Detector (papluca/xlm-roberta-base-language-detection)...")
    pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")

    print(" Downloading Sentiment Analyzer (cardiffnlp/twitter-xlm-roberta-base-sentiment)...")
    pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

    print(" Downloading Detoxify (original)...")
    Detoxify("original")

    print(" All heavy models downloaded successfully!")

if __name__ == "__main__":
    download_all()
