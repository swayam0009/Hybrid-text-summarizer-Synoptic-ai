# Preprocessing module
# summarizer/preprocessor.py
import re
import nltk
import spacy
from typing import List, Dict, Any
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string

class TextPreprocessor:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        
        # Initialize tools
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Downloading {spacy_model}...")
            from spacy.cli import download
            download(spacy_model)
            self.nlp = spacy.load(spacy_model)
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Contractions dictionary
        self.contractions = {
            "aren't": "are not", "can't": "cannot", "couldn't": "could not",
            "didn't": "did not", "doesn't": "does not", "don't": "do not",
            "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
            "he'd": "he would", "he'll": "he will", "he's": "he is",
            "i'd": "i would", "i'll": "i will", "i'm": "i am",
            "i've": "i have", "isn't": "is not", "it'd": "it would",
            "it'll": "it will", "it's": "it is", "let's": "let us",
            "shouldn't": "should not", "that's": "that is",
            "there's": "there is", "they'd": "they would",
            "they'll": "they will", "they're": "they are",
            "they've": "they have", "we'd": "we would",
            "we're": "we are", "we've": "we have", "weren't": "were not",
            "what's": "what is", "where's": "where is",
            "who's": "who is", "won't": "will not", "wouldn't": "would not",
            "you'd": "you would", "you'll": "you will",
            "you're": "you are", "you've": "you have"
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        return text.strip()
    
    def expand_contractions(self, text: str) -> str:
        """Expand contractions in text"""
        for contraction, expansion in self.contractions.items():
            text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
        return text
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """Split text into sentences with validation"""
        if not text or not text.strip():
            return []
        try:
            sentences = sent_tokenize(text.strip())
            # Filter out empty sentences
            return [sent.strip() for sent in sentences if sent.strip()]
        except Exception as e:
            print(f"Error in sentence tokenization: {e}")
            # Fallback: split by periods
            sentences = text.split('.')
            return [sent.strip() for sent in sentences if sent.strip()]
    
    def tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words with validation"""
        if not text or not text.strip():
            return []
        try:
            return word_tokenize(text.strip())
        except Exception as e:
            print(f"Error in word tokenization: {e}")
            # Fallback: split by spaces
            return text.split()
    
    def remove_stopwords(self, words: List[str]) -> List[str]:
        """Remove stop words from word list"""
        return [word for word in words if word.lower() not in self.stop_words]
    
    def stem_words(self, words: List[str]) -> List[str]:
        """Apply stemming to words"""
        return [self.stemmer.stem(word) for word in words]
    
    def lemmatize_words(self, words: List[str]) -> List[str]:
        """Apply lemmatization to words"""
        return [self.lemmatizer.lemmatize(word) for word in words]
    
    def pos_tagging(self, words: List[str]) -> List[tuple]:
        """Apply part-of-speech tagging"""
        return nltk.pos_tag(words)
    
    def extract_named_entities(self, text: str) -> List[Dict]:
        """Extract named entities using spaCy"""
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        return entities
    
    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """Complete preprocessing pipeline"""
        # Clean and normalize
        cleaned_text = self.clean_text(text)
        expanded_text = self.expand_contractions(cleaned_text)
        
        # Sentence segmentation
        sentences = self.tokenize_sentences(expanded_text)
        
        # Process each sentence
        processed_sentences = []
        for sentence in sentences:
            # Word tokenization
            words = self.tokenize_words(sentence)
            
            # Remove punctuation
            words = [word for word in words if word not in string.punctuation]
            
            # Convert to lowercase
            words = [word.lower() for word in words]
            
            # Remove stop words
            words_no_stop = self.remove_stopwords(words)
            
            # Lemmatization
            lemmatized_words = self.lemmatize_words(words_no_stop)
            
            # POS tagging
            pos_tags = self.pos_tagging(words)
            
            # Named entities
            entities = self.extract_named_entities(sentence)
            
            processed_sentences.append({
                'original': sentence,
                'words': words,
                'words_no_stop': words_no_stop,
                'lemmatized': lemmatized_words,
                'pos_tags': pos_tags,
                'entities': entities
            })
        
        return {
            'original_text': text,
            'cleaned_text': expanded_text,
            'sentences': processed_sentences,
            'total_sentences': len(sentences),
            'total_words': sum(len(sent['words']) for sent in processed_sentences)
        }
