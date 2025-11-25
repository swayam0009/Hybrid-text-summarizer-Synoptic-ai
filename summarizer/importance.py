# Content importance assessment
# summarizer/importance.py
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import math

class ContentImportanceAssessment:
    def __init__(self, sentence_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.sentence_transformer = SentenceTransformer(sentence_model)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=1,
            max_df=0.95,
            stop_words='english'
        )
    
    def calculate_tfidf_scores(self, sentences: List[str]) -> np.ndarray:
        """Calculate TF-IDF scores for sentences"""
        if not sentences or len(sentences) == 0:
            return np.array([])
        # Filter out empty sentences
        valid_sentences = [s for s in sentences if s.strip()]
        if not valid_sentences:
            return np.array([])
        try:
            # Fit TF-IDF vectorizer
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(valid_sentences)
            # Calculate sentence scores as sum of TF-IDF values
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            # Normalize scores
            if sentence_scores.max() > 0:
                sentence_scores = sentence_scores / sentence_scores.max()
            return sentence_scores
        except Exception as e:
            print(f"Error in TF-IDF calculation: {e}")
            return np.ones(len(valid_sentences))  # Return uniform scores as fallback
    
    def calculate_position_weights(self, num_sentences: int) -> np.ndarray:
        """Calculate position-based weights (higher for intro/conclusion)"""
        weights = np.ones(num_sentences)
        
        if num_sentences <= 3:
            return weights
        
        # Higher weights for first and last sentences
        intro_end = max(1, num_sentences // 10)  # First 10%
        conclusion_start = num_sentences - max(1, num_sentences // 10)  # Last 10%
        
        # Boost introduction sentences
        weights[:intro_end] *= 1.5
        
        # Boost conclusion sentences
        weights[conclusion_start:] *= 1.3
        
        return weights
    
    def calculate_length_scores(self, sentences: List[str]) -> np.ndarray:
        """Calculate length-based scores (penalize very short/long sentences)"""
        lengths = np.array([len(sent.split()) for sent in sentences])
        
        if len(lengths) == 0:
            return np.array([])
        
        # Optimal length range (10-30 words)
        optimal_min, optimal_max = 10, 30
        
        scores = np.ones(len(sentences))
        
        for i, length in enumerate(lengths):
            if length < optimal_min:
                scores[i] = length / optimal_min
            elif length > optimal_max:
                scores[i] = optimal_max / length
            else:
                scores[i] = 1.0
        
        return scores
    
    def calculate_entity_density(self, processed_sentences: List[Dict]) -> np.ndarray:
        """Calculate named entity density scores"""
        scores = []
        
        for sent_data in processed_sentences:
            entities = sent_data.get('entities', [])
            words = sent_data.get('words', [])
            
            if len(words) == 0:
                scores.append(0.0)
            else:
                entity_density = len(entities) / len(words)
                scores.append(entity_density)
        
        scores = np.array(scores)
        
        # Normalize scores
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return scores
    
    def calculate_keyword_frequency(self, processed_sentences: List[Dict], 
                                   top_k: int = 10) -> np.ndarray:
        """Calculate keyword frequency scores"""
        # Extract all words
        all_words = []
        for sent_data in processed_sentences:
            all_words.extend(sent_data.get('lemmatized', []))
        
        # Count word frequencies
        word_freq = {}
        for word in all_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_keywords = [word for word, freq in top_keywords]
        
        # Calculate sentence scores based on keyword presence
        scores = []
        for sent_data in processed_sentences:
            lemmatized_words = sent_data.get('lemmatized', [])
            keyword_count = sum(1 for word in lemmatized_words if word in top_keywords)
            
            if len(lemmatized_words) == 0:
                scores.append(0.0)
            else:
                scores.append(keyword_count / len(lemmatized_words))
        
        scores = np.array(scores)
        
        # Normalize scores
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return scores
    
    def calculate_semantic_similarity(self, sentences: List[str]) -> np.ndarray:
        """Calculate semantic similarity to document centroid"""
        if len(sentences) == 0:
            return np.array([])
        
        # Get sentence embeddings
        sentence_embeddings = self.sentence_transformer.encode(sentences)
        
        # Calculate document centroid
        document_centroid = np.mean(sentence_embeddings, axis=0)
        
        # Calculate cosine similarity to centroid
        similarities = cosine_similarity(sentence_embeddings, [document_centroid])
        scores = similarities.flatten()
        
        return scores
    
    def calculate_textrank_scores(self, sentences: List[str], 
                                 window_size: int = 2) -> np.ndarray:
        """Calculate TextRank scores for sentences"""
        if len(sentences) <= 1:
            return np.ones(len(sentences))
        
        # Get sentence embeddings
        sentence_embeddings = self.sentence_transformer.encode(sentences)
        
        # Create similarity matrix
        similarity_matrix = cosine_similarity(sentence_embeddings)
        
        # Create graph
        graph = nx.from_numpy_array(similarity_matrix)
        
        # Calculate PageRank scores
        try:
            pagerank_scores = nx.pagerank(graph, max_iter=50, tol=1e-4)
            scores = np.array([pagerank_scores[i] for i in range(len(sentences))])
        except:
            # Fallback to uniform scores if PageRank fails
            scores = np.ones(len(sentences))
        
        # Normalize scores
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return scores
    
    def calculate_unified_importance_scores(self, preprocessed_data: Dict[str, Any],
                                          weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Calculate unified importance scores combining all methods"""
        sentences = [sent['original'] for sent in preprocessed_data['sentences']]
        processed_sentences = preprocessed_data['sentences']
        
        if len(sentences) == 0:
            return {'scores': np.array([]), 'components': {}}
        
        # Default weights
        if weights is None:
            weights = {
                'tfidf': 0.3,
                'position': 0.2,
                'length': 0.1,
                'entity': 0.15,
                'keyword': 0.1,
                'semantic': 0.1,
                'textrank': 0.05
            }
        
        # Calculate individual scores
        tfidf_scores = self.calculate_tfidf_scores(sentences)
        position_weights = self.calculate_position_weights(len(sentences))
        length_scores = self.calculate_length_scores(sentences)
        entity_scores = self.calculate_entity_density(processed_sentences)
        keyword_scores = self.calculate_keyword_frequency(processed_sentences)
        semantic_scores = self.calculate_semantic_similarity(sentences)
        textrank_scores = self.calculate_textrank_scores(sentences)
        
        # Combine scores
        unified_scores = (
            weights['tfidf'] * tfidf_scores +
            weights['position'] * position_weights +
            weights['length'] * length_scores +
            weights['entity'] * entity_scores +
            weights['keyword'] * keyword_scores +
            weights['semantic'] * semantic_scores +
            weights['textrank'] * textrank_scores
        )
        
        # Normalize final scores
        if unified_scores.max() > 0:
            unified_scores = unified_scores / unified_scores.max()
        
        return {
            'scores': unified_scores,
            'components': {
                'tfidf': tfidf_scores,
                'position': position_weights,
                'length': length_scores,
                'entity': entity_scores,
                'keyword': keyword_scores,
                'semantic': semantic_scores,
                'textrank': textrank_scores
            }
        }
