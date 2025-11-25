# Extractive summarization
# summarizer/extractive.py
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import heapq

class ExtractiveSummarizer:
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
    
    def greedy_selection(self, sentences: List[str], 
                        importance_scores: np.ndarray,
                        target_length: int) -> List[int]:
        """Greedy sentence selection based on importance scores"""
        if len(sentences) == 0 or len(importance_scores) == 0:
            return []
        
        # Sort sentences by importance score
        sentence_indices = list(range(len(sentences)))
        sorted_indices = sorted(sentence_indices, 
                              key=lambda i: importance_scores[i], 
                              reverse=True)
        
        # Select top sentences
        selected_indices = sorted_indices[:target_length]
        
        # Sort selected indices to maintain chronological order
        selected_indices.sort()
        
        return selected_indices
    
    def mmr_selection(self, sentences: List[str], 
                     importance_scores: np.ndarray,
                     sentence_embeddings: np.ndarray,
                     target_length: int,
                     lambda_param: float = 0.7) -> List[int]:
        """Maximal Marginal Relevance (MMR) selection"""
        if len(sentences) == 0 or len(importance_scores) == 0:
            return []
        
        selected_indices = []
        remaining_indices = list(range(len(sentences)))
        
        # Select first sentence with highest importance
        if remaining_indices:
            first_idx = max(remaining_indices, key=lambda i: importance_scores[i])
            selected_indices.append(first_idx)
            remaining_indices.remove(first_idx)
        
        # Select remaining sentences using MMR
        while len(selected_indices) < target_length and remaining_indices:
            mmr_scores = []
            
            for idx in remaining_indices:
                # Relevance score (importance)
                relevance = importance_scores[idx]
                
                # Redundancy score (max similarity to selected sentences)
                if selected_indices:
                    similarities = cosine_similarity(
                        [sentence_embeddings[idx]], 
                        sentence_embeddings[selected_indices]
                    ).flatten()
                    redundancy = np.max(similarities)
                else:
                    redundancy = 0
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
                mmr_scores.append((mmr_score, idx))
            
            # Select sentence with highest MMR score
            if mmr_scores:
                _, best_idx = max(mmr_scores, key=lambda x: x[0])
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        # Sort selected indices to maintain chronological order
        selected_indices.sort()
        
        return selected_indices
    
    def remove_redundancy(self, sentences: List[str], 
                         selected_indices: List[int],
                         sentence_embeddings: np.ndarray) -> List[int]:
        """Remove redundant sentences based on similarity threshold"""
        if len(selected_indices) <= 1:
            return selected_indices
        
        # Calculate pairwise similarities
        selected_embeddings = sentence_embeddings[selected_indices]
        similarity_matrix = cosine_similarity(selected_embeddings)
        
        # Remove redundant sentences
        final_indices = []
        for i, idx in enumerate(selected_indices):
            is_redundant = False
            
            for j, prev_idx in enumerate(final_indices):
                if similarity_matrix[i][selected_indices.index(prev_idx)] > self.similarity_threshold:
                    is_redundant = True
                    break
            
            if not is_redundant:
                final_indices.append(idx)
        
        return final_indices
    
    def preserve_key_quotes(self, sentences: List[str], 
                          selected_indices: List[int]) -> List[int]:
        """Preserve sentences with quotes or important facts"""
        import re
        
        quote_patterns = [
            r'"[^"]*"',  # Direct quotes
            r"'[^']*'",  # Single quotes
            r'\b(said|says|stated|according to|reported)\b',  # Attribution
        ]
        
        preserved_indices = []
        
        for i, sentence in enumerate(sentences):
            # Check for quotes or attributions
            has_quote = any(re.search(pattern, sentence, re.IGNORECASE) 
                          for pattern in quote_patterns)
            
            # Check for numerical facts
            has_numbers = bool(re.search(r'\b\d+(\.\d+)?%?\b', sentence))
            
            # Check for proper nouns (potential important entities)
            has_proper_nouns = bool(re.search(r'\b[A-Z][a-z]+\b', sentence))
            
            if has_quote or (has_numbers and has_proper_nouns):
                preserved_indices.append(i)
        
        # Combine with selected indices
        final_indices = list(set(selected_indices + preserved_indices))
        final_indices.sort()
        
        return final_indices
    
    def extractive_summarize(self, preprocessed_data: Dict[str, Any],
                           importance_data: Dict[str, Any],
                           sentence_embeddings: np.ndarray,
                           target_length: int = 3,
                           method: str = "mmr") -> Dict[str, Any]:
        """Generate extractive summary"""
        sentences = [sent['original'] for sent in preprocessed_data['sentences']]
        importance_scores = importance_data['scores']
        
        if len(sentences) == 0:
            return {
                'summary': '',
                'selected_sentences': [],
                'selected_indices': [],
                'method': method
            }
        
        # Ensure target length doesn't exceed available sentences
        target_length = min(target_length, len(sentences))
        
        # Select sentences based on method
        if method == "greedy":
            selected_indices = self.greedy_selection(sentences, importance_scores, target_length)
        elif method == "mmr":
            selected_indices = self.mmr_selection(sentences, importance_scores, 
                                                sentence_embeddings, target_length)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Remove redundancy
        selected_indices = self.remove_redundancy(sentences, selected_indices, sentence_embeddings)
        
        # Preserve key quotes and facts
        selected_indices = self.preserve_key_quotes(sentences, selected_indices)
        
        # Create summary
        selected_sentences = [sentences[i] for i in selected_indices]
        summary = ' '.join(selected_sentences)
        
        return {
            'summary': summary,
            'selected_sentences': selected_sentences,
            'selected_indices': selected_indices,
            'method': method,
            'importance_scores': importance_scores[selected_indices] if len(selected_indices) > 0 else []
        }
