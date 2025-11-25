# Configuration file
# config.py
import os
from pathlib import Path

class Config:
    # Database settings
    DATABASE_PATH = Path("data/database.db")
    CACHE_DIR = Path("cache")
    
    # Model settings
    MODELS = {
        "sentence_embeddings": "sentence-transformers/all-MiniLM-L6-v2",
        "abstractive": "facebook/bart-large-cnn",
        "spacy_model": "en_core_web_sm"
    }
    
    # Summarization parameters
    DEFAULT_SUMMARY_LENGTH = 3
    MAX_SUMMARY_LENGTH = 10
    MIN_SENTENCE_LENGTH = 10
    MAX_SENTENCE_LENGTH = 5000
    
    # TF-IDF parameters
    TFIDF_MAX_FEATURES = 10000
    TFIDF_MIN_DF = 1
    TFIDF_MAX_DF = 0.95
    
    # TextRank parameters
    TEXTRANK_WINDOW_SIZE = 2
    TEXTRANK_ITERATIONS = 50
    TEXTRANK_THRESHOLD = 0.0001
    
    # Scoring weights
    IMPORTANCE_WEIGHTS = {
        "tfidf": 0.3,
        "position": 0.2,
        "length": 0.1,
        "entity": 0.15,
        "keyword": 0.1,
        "semantic": 0.1,
        "textrank": 0.05
    }
    
    # Personalization
    USER_PROFILE_FEATURES = [
        "domain_expertise", "reading_speed", "detail_preference",
        "summary_length_preference", "technical_level"
    ]
    
    # Create necessary directories
    def __init__(self):
        self.DATABASE_PATH.parent.mkdir(exist_ok=True)
        self.CACHE_DIR.mkdir(exist_ok=True)
