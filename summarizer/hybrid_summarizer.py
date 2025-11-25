# Main summarizer class
# summarizer/hybrid_summarizer.py
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from .preprocessor import TextPreprocessor
from .importance import ContentImportanceAssessment
from .extractive import ExtractiveSummarizer
from .abstractive import AbstractiveSummarizer
from .optimizer import SummaryOptimizer
from .personalizer import PersonalizationEngine
from .explainer import DecisionExplainer

class HybridSummarizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.preprocessor = TextPreprocessor(config.get('spacy_model', 'en_core_web_sm'))
        self.importance_assessor = ContentImportanceAssessment(
            config.get('sentence_model', 'sentence-transformers/all-MiniLM-L6-v2')
        )
        self.extractive_summarizer = ExtractiveSummarizer()
        self.abstractive_summarizer = AbstractiveSummarizer(
            config.get('abstractive_model', 'facebook/bart-large-cnn')
        )
        self.optimizer = SummaryOptimizer()
        self.personalizer = PersonalizationEngine()
        self.explainer = DecisionExplainer()
        
        # Initialize sentence transformer for embeddings
        self.sentence_transformer = SentenceTransformer(
            config.get('sentence_model', 'sentence-transformers/all-MiniLM-L6-v2')
        )
    
    def summarize(self, text: str, 
                  user_id: Optional[int] = None,
                  summary_type: str = "hybrid",
                  target_length: int = 3,
                  available_time: Optional[int] = None,
                  personalize: bool = True,
                  explain_decisions: bool = False,
                  **kwargs) -> Dict[str, Any]:
        """
        Main summarization method that orchestrates the entire pipeline
        
        Args:
            text: Input text to summarize
            user_id: User ID for personalization
            summary_type: Type of summary ('extractive', 'abstractive', 'hybrid')
            target_length: Target number of sentences
            available_time: Available reading time in minutes
            personalize: Whether to apply personalization
            explain_decisions: Whether to generate explanations
            
        Returns:
            Dictionary containing summary and metadata
        """
        try:
            # Input validation
            if not text or not text.strip():
                return {
                    'summary': '',
                    'summary_type': summary_type,
                    'error': 'Empty text input provided'
                }
            if len(text.strip()) < 10:
                return {
                    'summary': text.strip(),
                    'summary_type': summary_type,
                    'error': 'Text too short for summarization'
                }
            # Step 1: Preprocessing
            preprocessed_data = self.preprocessor.preprocess_text(text)
            if not preprocessed_data.get('sentences') or len(preprocessed_data['sentences']) == 0:
                return {
                    'summary': '',
                    'summary_type': summary_type,
                    'error': 'No valid sentences found in input text'
                }
            # ...existing code...
            # Step 2: Content Importance Assessment
            importance_data = self.importance_assessor.calculate_unified_importance_scores(
                preprocessed_data, self.config.get('importance_weights')
            )
            # Step 3: Get sentence embeddings
            sentence_texts = [sent['original'] for sent in preprocessed_data['sentences']]
            sentence_embeddings = self.sentence_transformer.encode(sentence_texts)
            # Step 4: Personalization (if enabled and user provided)
            base_parameters = {
                'summary_length': target_length,
                'summary_type': summary_type,
                'refinement_style': 'balanced'
            }
            if personalize and user_id:
                personalized_params = self.personalizer.personalize_summary_parameters(
                    user_id, text, base_parameters, available_time
                )
                target_length = personalized_params['summary_length']
                summary_type = personalized_params.get('summary_type', summary_type)
            else:
                personalized_params = base_parameters
            # Step 5: Generate summary based on type
            if summary_type == "extractive":
                summary_result = self._generate_extractive_summary(
                    preprocessed_data, importance_data, sentence_embeddings, target_length
                )
            elif summary_type == "abstractive":
                # For pure abstractive, first extract more sentences then refine
                extractive_result = self._generate_extractive_summary(
                    preprocessed_data, importance_data, sentence_embeddings, target_length * 2
                )
                summary_result = self._generate_abstractive_summary(extractive_result, target_length)
            else:  # hybrid
                summary_result = self._generate_hybrid_summary(
                    preprocessed_data, importance_data, sentence_embeddings, 
                    target_length, personalized_params
                )
            # Step 6: Summary Optimization
            try:
                optimized_result = self.optimizer.optimize_summary(
                    summary_result['summary'],
                    target_length * 30,  # Approximate words per sentence
                    optimization_goals=['length', 'coherence', 'redundancy', 'grammar']
                )
                # Step 7: Update summary with optimized version
                summary_result['summary'] = optimized_result['optimized_summary']
                summary_result['optimization_log'] = optimized_result['optimization_log']
                summary_result['coherence_score'] = optimized_result['coherence_score']
                summary_result['readability_scores'] = optimized_result['readability_scores']
            except Exception as e:
                print(f"Error in post-processing: {e}")
                # Use the summary as-is without optimization
                pass
            # Step 8: Generate explanations (if requested)
            explanations = {}
            if explain_decisions:
                explanation_data = {
                    'selected_sentences': summary_result.get('selected_sentences', []),
                    'selected_indices': summary_result.get('selected_indices', []),
                    'importance_scores': importance_data['scores'],
                    'importance_components': importance_data['components'],
                    'all_sentences': sentence_texts,
                    'personalization_data': personalized_params,
                    'method': summary_type
                }
                explanations = self.explainer.create_comprehensive_explanation(explanation_data)
            # Step 9: Compile final result
            final_result = {
                'summary': summary_result['summary'],
                'summary_type': summary_type,
                'original_length': len(text.split()),
                'summary_length': len(summary_result['summary'].split()),
                'compression_ratio': 1 - (len(summary_result['summary'].split()) / len(text.split())),
                'selected_sentences': summary_result.get('selected_sentences', []),
                'selected_indices': summary_result.get('selected_indices', []),
                'importance_scores': importance_data['scores'].tolist(),
                'importance_components': {k: v.tolist() for k, v in importance_data['components'].items()},
                'personalization_applied': personalize and user_id is not None,
                'personalization_factors': personalized_params.get('personalization_factors', {}),
                'coherence_score': summary_result.get('coherence_score', 0.0),
                'readability_scores': summary_result.get('readability_scores', {}),
                'optimization_log': summary_result.get('optimization_log', []),
                'processing_metadata': {
                    'total_sentences': len(sentence_texts),
                    'sentences_selected': len(summary_result.get('selected_sentences', [])),
                    'method_used': summary_type,
                    'target_length': target_length
                }
            }
            if explain_decisions:
                final_result['explanations'] = explanations
            return final_result
            
        except Exception as e:
            return {
                'summary': '',
                'summary_type': summary_type,
                'error': f'Summarization failed: {str(e)}'
            }
    
    def _generate_extractive_summary(self, preprocessed_data: Dict[str, Any],
                                   importance_data: Dict[str, Any],
                                   sentence_embeddings: np.ndarray,
                                   target_length: int) -> Dict[str, Any]:
        """Generate extractive summary"""
        return self.extractive_summarizer.extractive_summarize(
            preprocessed_data, importance_data, sentence_embeddings, target_length, method="mmr"
        )
    
    def _generate_abstractive_summary(self, extractive_result: Dict[str, Any],
                                    target_length: int) -> Dict[str, Any]:
        """Generate abstractive summary from extractive result"""
        refined_result = self.abstractive_summarizer.refine_extractive_summary(
            extractive_result, target_length * 25  # Approximate target word count
        )
        
        return {
            'summary': refined_result['refined_summary'],
            'selected_sentences': extractive_result.get('selected_sentences', []),
            'selected_indices': extractive_result.get('selected_indices', []),
            'refinement_applied': True,
            'original_extractive': refined_result['original_extractive']
        }
    
    def _generate_hybrid_summary(self, preprocessed_data: Dict[str, Any],
                               importance_data: Dict[str, Any],
                               sentence_embeddings: np.ndarray,
                               target_length: int,
                               personalized_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate hybrid summary combining extractive and abstractive approaches"""
        try:
            print("🔄 Starting hybrid summarization...")
            # Step 1: Extract key sentences
            extraction_length = max(target_length, min(target_length * 2, 8))
            max_available_sentences = len(preprocessed_data['sentences'])
            extraction_length = min(extraction_length, max_available_sentences)
            print(f"🔄 Extracting {extraction_length} sentences from {max_available_sentences} available")
            if extraction_length == 0:
                return {
                    'summary': '',
                    'selected_sentences': [],
                    'selected_indices': [],
                    'error': 'No sentences available for extraction'
                }
            extractive_result = self.extractive_summarizer.extractive_summarize(
                preprocessed_data, importance_data, sentence_embeddings, extraction_length, method="mmr"
            )
            print(f"🔄 Extractive result: {extractive_result.get('summary', '')[:100]}...")
            # Validate extractive result
            if not extractive_result or not extractive_result.get('summary'):
                return {
                    'summary': '',
                    'selected_sentences': [],
                    'selected_indices': [],
                    'error': 'Extractive summarization failed'
                }
            # Step 2: Apply abstractive refinement
            refinement_style = personalized_params.get('refinement_style', 'balanced')
            target_word_count = target_length * 25
            print(f"🔄 Applying abstractive refinement with style: {refinement_style}")
            refined_result = self.abstractive_summarizer.refine_extractive_summary(
                extractive_result, 
                target_word_count, 
                refinement_style
            )
            print(f"🔄 Refined result type: {type(refined_result)}")
            print(f"🔄 Refined result keys: {list(refined_result.keys()) if isinstance(refined_result, dict) else 'Not a dict'}")
            # Validate refined result with more careful checking
            if not refined_result:
                print("❌ Refined result is None or empty")
                refined_summary = extractive_result.get('summary', '')
            elif not isinstance(refined_result, dict):
                print("❌ Refined result is not a dictionary")
                refined_summary = extractive_result.get('summary', '')
            else:
                refined_summary = refined_result.get('refined_summary', '')
                if not refined_summary:
                    print("❌ No refined_summary in result")
                    refined_summary = extractive_result.get('summary', '')
            print(f"🔄 Final refined summary: {refined_summary[:100] if refined_summary else 'EMPTY'}...")
            # Safe extraction of other values
            selected_sentences = extractive_result.get('selected_sentences', [])
            selected_indices = extractive_result.get('selected_indices', [])
            extractive_summary = extractive_result.get('summary', '')
            print(f"🔄 Building final result...")
            # Build result dictionary safely
            result = {
                'summary': refined_summary if refined_summary else extractive_summary,
                'selected_sentences': selected_sentences,
                'selected_indices': selected_indices,
                'extractive_summary': extractive_summary,
                'refinement_applied': bool(refined_summary and refined_summary != extractive_summary),
                'refinement_style': refinement_style,
                'hybrid_approach': True
            }
            print(f"🔄 Result built successfully")
            return result
        except Exception as e:
            print(f"❌ Error in hybrid summarization: {e}")
            import traceback
            traceback.print_exc()
            return {
                'summary': '',
                'selected_sentences': [],
                'selected_indices': [],
                'error': f'Hybrid summarization failed: {str(e)}'
            }
    
    def batch_summarize(self, texts: List[str], 
                       user_id: Optional[int] = None,
                       **kwargs) -> List[Dict[str, Any]]:
        """Summarize multiple texts in batch"""
        results = []
        
        for i, text in enumerate(texts):
            try:
                result = self.summarize(text, user_id=user_id, **kwargs)
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'batch_index': i,
                    'summary': '',
                    'error': f'Batch summarization failed for text {i}: {str(e)}'
                })
        
        return results
    
    def evaluate_summary_quality(self, summary: str, reference: str = None) -> Dict[str, Any]:
        """Evaluate summary quality using various metrics"""
        # Basic quality metrics
        readability_scores = self.optimizer.assess_readability(summary)
        coherence_score = self.optimizer.assess_coherence(summary)
        
        # Grammar check
        grammar_issues = self.optimizer.check_grammar(summary)
        
        # ROUGE scores (if reference provided)
        rouge_scores = {}
        if reference:
            rouge_scores = self.optimizer.calculate_rouge_scores(summary, reference)
        
        return {
            'readability_scores': readability_scores,
            'coherence_score': coherence_score,
            'grammar_issues': len(grammar_issues),
            'rouge_scores': rouge_scores,
            'word_count': len(summary.split()),
            'sentence_count': len(summary.split('. ')),
            'overall_quality': self._calculate_overall_quality(
                readability_scores, coherence_score, len(grammar_issues)
            )
        }
    
    def _calculate_overall_quality(self, readability_scores: Dict[str, float],
                                 coherence_score: float,
                                 grammar_issues: int) -> float:
        """Calculate overall quality score"""
        # Normalize readability scores
        flesch_score = max(0, min(100, readability_scores.get('flesch_reading_ease', 50))) / 100
        
        # Penalize grammar issues
        grammar_penalty = max(0, 1 - (grammar_issues * 0.1))
        
        # Combine scores
        overall_quality = (flesch_score * 0.4 + coherence_score * 0.4 + grammar_penalty * 0.2)
        
        return overall_quality
    
    def get_supported_formats(self) -> List[str]:
        """Return list of supported input formats"""
        return ['text', 'txt', 'pdf', 'docx', 'html', 'url']
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return information about loaded models"""
        return {
            'sentence_model': self.sentence_transformer.get_sentence_embedding_dimension(),
            'abstractive_model': self.abstractive_summarizer.model_name,
            'preprocessing_model': 'spacy + nltk',
            'supported_languages': ['en'],  # Currently only English
            'max_input_length': 10000000,  # characters
            'recommended_input_length': 500000  # characters
        }
