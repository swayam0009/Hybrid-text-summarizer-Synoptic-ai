# Summary optimization
# summarizer/optimizer.py
import re
import textstat
from typing import Dict, Any, List, Tuple
from rouge_score import rouge_scorer
import numpy as np
from collections import Counter

class SummaryOptimizer:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], 
                                                   use_stemmer=True)
        
        # Common grammar patterns for checking
        self.grammar_patterns = [
            (r'\b(a|an)\s+(?=[aeiouAEIOU])', 'article_error'),
            (r'\b(a|an)\s+(?=[^aeiouAEIOU])', 'article_error'),
            (r'\s+', 'whitespace'),
            (r'([.!?])\s*([a-z])', 'capitalization'),
        ]
    
    def control_length(self, summary: str, target_length: int, 
                      unit: str = "sentences") -> str:
        """Control summary length based on target"""
        if unit == "sentences":
            sentences = summary.split('. ')
            if len(sentences) > target_length:
                return '. '.join(sentences[:target_length]) + '.'
            return summary
        
        elif unit == "words":
            words = summary.split()
            if len(words) > target_length:
                truncated = ' '.join(words[:target_length])
                # Try to end at sentence boundary
                last_punct = max(truncated.rfind('.'), truncated.rfind('!'), 
                               truncated.rfind('?'))
                if last_punct > len(truncated) * 0.8:  # If punctuation is near end
                    return truncated[:last_punct + 1]
                return truncated + '...'
            return summary
        
        elif unit == "characters":
            if len(summary) > target_length:
                truncated = summary[:target_length]
                last_space = truncated.rfind(' ')
                if last_space > target_length * 0.8:
                    return truncated[:last_space] + '...'
                return truncated + '...'
            return summary
        
        return summary
    
    def assess_coherence(self, summary: str) -> float:
        """Assess coherence of summary"""
        sentences = summary.split('. ')
        
        if len(sentences) <= 1:
            return 1.0
        
        # Check for transition words
        transition_words = [
            'however', 'therefore', 'furthermore', 'moreover', 'additionally',
            'consequently', 'meanwhile', 'subsequently', 'nevertheless',
            'thus', 'hence', 'accordingly', 'besides', 'likewise'
        ]
        
        transition_count = 0
        for sentence in sentences:
            if any(word in sentence.lower() for word in transition_words):
                transition_count += 1
        
        # Check for pronoun references
        pronouns = ['he', 'she', 'it', 'they', 'this', 'that', 'these', 'those']
        pronoun_count = 0
        
        for sentence in sentences[1:]:  # Skip first sentence
            if any(pronoun in sentence.lower().split() for pronoun in pronouns):
                pronoun_count += 1
        
        # Calculate coherence score
        coherence_score = 0.0
        
        # Transition score (max 0.5)
        if len(sentences) > 1:
            transition_score = min(0.5, transition_count / (len(sentences) - 1))
            coherence_score += transition_score
        
        # Pronoun reference score (max 0.3)
        if len(sentences) > 1:
            pronoun_score = min(0.3, pronoun_count / (len(sentences) - 1))
            coherence_score += pronoun_score
        
        # Length consistency score (max 0.2)
        sentence_lengths = [len(sent.split()) for sent in sentences]
        if sentence_lengths:
            length_std = np.std(sentence_lengths)
            length_mean = np.mean(sentence_lengths)
            if length_mean > 0:
                length_consistency = max(0, 0.2 - (length_std / length_mean))
                coherence_score += length_consistency
        
        return min(1.0, coherence_score)
    
    def detect_redundancy(self, summary: str, threshold: float = 0.7) -> List[str]:
        """Detect redundant sentences in summary"""
        sentences = summary.split('. ')
        redundant_sentences = []
        
        for i, sent1 in enumerate(sentences):
            for j, sent2 in enumerate(sentences[i+1:], i+1):
                # Simple word overlap check
                words1 = set(sent1.lower().split())
                words2 = set(sent2.lower().split())
                
                if len(words1) == 0 or len(words2) == 0:
                    continue
                
                overlap = len(words1.intersection(words2))
                similarity = overlap / min(len(words1), len(words2))
                
                if similarity > threshold:
                    redundant_sentences.append(sent2)
        
        return redundant_sentences
    
    def remove_redundancy(self, summary: str, threshold: float = 0.7) -> str:
        """Remove redundant sentences from summary"""
        sentences = summary.split('. ')
        redundant_sentences = set(self.detect_redundancy(summary, threshold))
        
        filtered_sentences = [sent for sent in sentences if sent not in redundant_sentences]
        
        if not filtered_sentences:
            return summary  # Return original if all sentences are redundant
        
        return '. '.join(filtered_sentences)
    
    def assess_readability(self, summary: str) -> Dict[str, float]:
        """Assess readability using multiple metrics"""
        if not summary.strip():
            return {
                'flesch_kincaid': 0.0,
                'flesch_reading_ease': 0.0,
                'smog_index': 0.0,
                'automated_readability_index': 0.0
            }
        
        try:
            readability_scores = {
                'flesch_kincaid': textstat.flesch_kincaid_grade(summary),
                'flesch_reading_ease': textstat.flesch_reading_ease(summary),
                'smog_index': textstat.smog_index(summary),
                'automated_readability_index': textstat.automated_readability_index(summary)
            }
        except:
            readability_scores = {
                'flesch_kincaid': 0.0,
                'flesch_reading_ease': 0.0,
                'smog_index': 0.0,
                'automated_readability_index': 0.0
            }
        
        return readability_scores
    
    def check_grammar(self, summary: str) -> List[Dict[str, Any]]:
        """Basic grammar checking"""
        issues = []
        
        # Check for common grammar issues
        for pattern, issue_type in self.grammar_patterns:
            matches = re.finditer(pattern, summary)
            for match in matches:
                issues.append({
                    'type': issue_type,
                    'position': match.start(),
                    'text': match.group(),
                    'suggestion': self._get_grammar_suggestion(match.group(), issue_type)
                })
        
        return issues
    
    def _get_grammar_suggestion(self, text: str, issue_type: str) -> str:
        """Get grammar correction suggestion"""
        if issue_type == 'whitespace':
            return ' '
        elif issue_type == 'capitalization':
            return text.upper()
        elif issue_type == 'article_error':
            if text.lower().startswith('a ') and text[2] in 'aeiou':
                return 'an' + text[1:]
            elif text.lower().startswith('an ') and text[3] not in 'aeiou':
                return 'a' + text[2:]
        
        return text
    
    def fix_grammar(self, summary: str) -> str:
        """Apply basic grammar fixes"""
        # Fix whitespace
        summary = re.sub(r'\s+', ' ', summary)
        
        # Fix capitalization after punctuation
        summary = re.sub(r'([.!?])\s*([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), summary)
        
        # Fix article errors (basic)
        summary = re.sub(r'\ba\s+(?=[aeiouAEIOU])', 'an ', summary)
        summary = re.sub(r'\ban\s+(?=[^aeiouAEIOU])', 'a ', summary)
        
        return summary.strip()
    
    def calculate_rouge_scores(self, summary: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores against reference"""
        if not summary.strip() or not reference.strip():
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        scores = self.rouge_scorer.score(reference, summary)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def optimize_summary(self, summary: str, 
                        target_length: int = 150,
                        reference_summary: str = None,
                        optimization_goals: List[str] = None) -> Dict[str, Any]:
        """Comprehensive summary optimization"""
        if optimization_goals is None:
            optimization_goals = ['length', 'coherence', 'redundancy', 'readability', 'grammar']
        
        optimized_summary = summary
        optimization_log = []
        
        # Length optimization
        if 'length' in optimization_goals:
            original_length = len(optimized_summary.split())
            optimized_summary = self.control_length(optimized_summary, target_length, 'words')
            optimization_log.append(f"Length: {original_length} → {len(optimized_summary.split())} words")
        
        # Redundancy removal
        if 'redundancy' in optimization_goals:
            redundant_sentences = self.detect_redundancy(optimized_summary)
            optimized_summary = self.remove_redundancy(optimized_summary)
            optimization_log.append(f"Removed {len(redundant_sentences)} redundant sentences")
        
        # Grammar fixes
        if 'grammar' in optimization_goals:
            grammar_issues = self.check_grammar(optimized_summary)
            optimized_summary = self.fix_grammar(optimized_summary)
            optimization_log.append(f"Fixed {len(grammar_issues)} grammar issues")
        
        # Calculate metrics
        coherence_score = self.assess_coherence(optimized_summary) if 'coherence' in optimization_goals else 0.0
        readability_scores = self.assess_readability(optimized_summary) if 'readability' in optimization_goals else {}
        rouge_scores = self.calculate_rouge_scores(optimized_summary, reference_summary) if reference_summary else {}
        
        return {
            'optimized_summary': optimized_summary,
            'original_summary': summary,
            'coherence_score': coherence_score,
            'readability_scores': readability_scores,
            'rouge_scores': rouge_scores,
            'optimization_log': optimization_log,
            'word_count': len(optimized_summary.split()),
            'sentence_count': len(optimized_summary.split('. '))
        }
