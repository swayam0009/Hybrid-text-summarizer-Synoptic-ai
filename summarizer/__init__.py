# summarizer/__init__.py
"""
Hybrid Text Summarization System

A comprehensive text summarization system that combines extractive and abstractive
techniques with personalization capabilities.

This package provides:
- Text preprocessing and content analysis
- Extractive summarization using multiple algorithms
- Abstractive summarization with transformer models
- Personalization based on user profiles
- Summary optimization and quality assessment
- Decision explanations and transparency
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__description__ = "Hybrid Text Summarization System with Personalization"

# Import main classes
from .hybrid_summarizer import HybridSummarizer
from .preprocessor import TextPreprocessor
from .importance import ContentImportanceAssessment
from .extractive import ExtractiveSummarizer
from .abstractive import AbstractiveSummarizer
from .optimizer import SummaryOptimizer
from .personalizer import PersonalizationEngine
from .explainer import DecisionExplainer

# Export all classes
__all__ = [
    'HybridSummarizer',
    'TextPreprocessor',
    'ContentImportanceAssessment',
    'ExtractiveSummarizer',
    'AbstractiveSummarizer',
    'SummaryOptimizer',
    'PersonalizationEngine',
    'DecisionExplainer'
]

# Package-level constants
SUPPORTED_SUMMARY_TYPES = ['extractive', 'abstractive', 'hybrid']
SUPPORTED_LANGUAGES = ['en']  # Currently only English
DEFAULT_SUMMARY_LENGTH = 3
MAX_SUMMARY_LENGTH = 10
MIN_SUMMARY_LENGTH = 1

# Model configurations
DEFAULT_MODELS = {
    'sentence_embeddings': 'sentence-transformers/all-MiniLM-L6-v2',
    'abstractive': 'facebook/bart-large-cnn',
    'spacy_model': 'en_core_web_sm'
}

# Importance scoring weights
DEFAULT_IMPORTANCE_WEIGHTS = {
    'tfidf': 0.3,
    'position': 0.2,
    'length': 0.1,
    'entity': 0.15,
    'keyword': 0.1,
    'semantic': 0.1,
    'textrank': 0.05
}

# Personalization features
USER_PROFILE_FEATURES = [
    'domain_expertise',
    'reading_speed',
    'detail_preference',
    'summary_length_preference',
    'technical_level'
]

# Quality thresholds
QUALITY_THRESHOLDS = {
    'min_coherence_score': 0.3,
    'min_readability_score': 30,
    'max_grammar_issues': 5,
    'min_compression_ratio': 0.1,
    'max_compression_ratio': 0.9
}

def get_version():
    """Get the current version of the package"""
    return __version__

def get_supported_features():
    """Get list of supported features"""
    return {
        'summary_types': SUPPORTED_SUMMARY_TYPES,
        'languages': SUPPORTED_LANGUAGES,
        'models': list(DEFAULT_MODELS.keys()),
        'personalization_features': USER_PROFILE_FEATURES,
        'explanation_capabilities': [
            'sentence_selection',
            'importance_scoring',
            'personalization_factors',
            'confidence_scores',
            'visualizations'
        ]
    }

def get_model_info():
    """Get information about the models used"""
    return {
        'sentence_transformer': DEFAULT_MODELS['sentence_embeddings'],
        'abstractive_model': DEFAULT_MODELS['abstractive'],
        'preprocessing_model': DEFAULT_MODELS['spacy_model'],
        'supported_languages': SUPPORTED_LANGUAGES,
        'max_input_length': 1000000,  # characters
        'recommended_input_length': 50000  # characters
    }

# Package initialization
def initialize_package():
    """Initialize the package with necessary downloads"""
    try:
        import nltk
        import spacy
        
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        
        # Check if spaCy model is available
        try:
            spacy.load(DEFAULT_MODELS['spacy_model'])
        except OSError:
            print(f"Warning: spaCy model '{DEFAULT_MODELS['spacy_model']}' not found.")
            print("Please install it using: python -m spacy download en_core_web_sm")
        
        return True
    except ImportError as e:
        print(f"Error initializing package: {e}")
        return False

# Auto-initialize when package is imported
try:
    initialize_package()
except Exception as e:
    print(f"Warning: Package initialization failed: {e}")
