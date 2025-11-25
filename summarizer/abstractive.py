# Abstractive refinement
# summarizer/abstractive.py
import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, 
    T5Tokenizer, T5ForConditionalGeneration,
    BartTokenizer, BartForConditionalGeneration
)
from typing import Dict, Any, List, Optional
import re

class AbstractiveSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer based on model type
        if "bart" in model_name.lower():
            self.tokenizer = BartTokenizer.from_pretrained(model_name)
            self.model = BartForConditionalGeneration.from_pretrained(model_name)
            self.max_model_input_length = 1024
        elif "t5" in model_name.lower():
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.max_model_input_length = 512  # generally T5 variants have 512 or 1024 depending on model
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.max_model_input_length = 1024  # fallback default
        
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess_for_abstractive(self, extractive_summary: str, max_length: Optional[int] = None) -> str:
        """
        Prepare input for abstractive model - truncate at max token length
        """
        if not extractive_summary or not extractive_summary.strip():
            return ""
        text = re.sub(r'\s+', ' ', extractive_summary).strip()
        
        # Use max_length param or default model max length
        max_len = max_length or self.max_model_input_length
        
        # Add prefix for T5
        if "t5" in self.model_name.lower():
            text = f"summarize: {text}"
            
        try:
            tokens = self.tokenizer.encode(text, truncation=True, max_length=max_len)
            truncated = self.tokenizer.decode(tokens, skip_special_tokens=True)
            return truncated
        except Exception as e:
            print(f"Tokenization error: {e}")
            return text
    
    def generate_abstractive_summary(self, input_text: str,
                                     max_length: int = 180,
                                     min_length: int = 50,
                                     num_beams: int = 4,
                                     length_penalty: float = 1.0,
                                     no_repeat_ngram_size: int = 3,
                                     early_stopping: bool = True,
                                     do_sample: bool = True,
                                     top_k: int = 50,
                                     top_p: float = 0.9) -> str:
        """
        Generate abstractive summary with sampling to get more diverse outputs
        """
        try:
            processed_input = self.preprocess_for_abstractive(input_text, max_length=self.max_model_input_length)
            
            inputs = self.tokenizer(
                processed_input,
                max_length=self.max_model_input_length,
                truncation=True,
                padding='longest',
                return_tensors='pt'
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                summary_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    early_stopping=early_stopping,
                    do_sample=do_sample,
                    top_k=top_k,
                    top_p=top_p
                )
                
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summary = self.post_process_summary(summary)
            return summary
        
        except Exception as e:
            print(f"Abstractive summarization failed: {e}")
            return input_text
    
    def post_process_summary(self, summary: str) -> str:
        """Post-process generated summary"""
        # Remove T5 prefix if present
        if summary.startswith("summarize:"):
            summary = summary[10:].strip()
        
        # Clean up formatting
        summary = re.sub(r'\s+', ' ', summary).strip()
        
        # Ensure proper capitalization
        sentences = summary.split('. ')
        capitalized_sentences = []
        
        for sentence in sentences:
            if sentence:
                sentence = sentence.strip()
                if sentence:
                    sentence = sentence[0].upper() + sentence[1:]
                    capitalized_sentences.append(sentence)
        
        summary = '. '.join(capitalized_sentences)
        
        # Ensure summary ends with proper punctuation
        if summary and not summary.endswith(('.', '!', '?')):
            summary += '.'
        
        return summary
    
    def refine_extractive_summary(self, extractive_data: Dict[str, Any],
                                 target_length: int = 150,
                                 refinement_style: str = "concise") -> Dict[str, Any]:
        """Refine extractive summary using abstractive techniques"""
        try:
            print(f"🔄 refine_extractive_summary called with target_length={target_length}, style={refinement_style}")
            # Validate input
            if not extractive_data or not isinstance(extractive_data, dict):
                print("❌ Invalid extractive_data")
                return {
                    'refined_summary': '',
                    'original_extractive': '',
                    'refinement_method': 'abstractive',
                    'refinement_style': refinement_style,
                    'error': 'Invalid extractive data'
                }
            extractive_summary = extractive_data.get('summary', '')
            print(f"🔄 Input extractive summary: {extractive_summary[:100] if extractive_summary else 'EMPTY'}...")
            if not extractive_summary or not extractive_summary.strip():
                print("❌ Empty extractive summary")
                return {
                    'refined_summary': '',
                    'original_extractive': '',
                    'refinement_method': 'abstractive',
                    'refinement_style': refinement_style,
                    'error': 'Empty extractive summary'
                }
            # Adjust generation parameters based on style
            if refinement_style == "concise":
                max_length = min(target_length, 100)
                min_length = max(20, target_length // 3)
                length_penalty = 2.5
            elif refinement_style == "detailed":
                max_length = min(target_length, 300)
                min_length = max(50, target_length // 2)
                length_penalty = 1.5
            else:  # balanced
                max_length = min(target_length, 200)
                min_length = max(30, target_length // 2)
                length_penalty = 2.0
            print(f"🔄 Calling generate_abstractive_summary with max_length={max_length}")
            refined_summary = self.generate_abstractive_summary(
                extractive_summary,
                max_length=max_length,
                min_length=min_length,
                length_penalty=length_penalty
            )
            print(f"🔄 Generated refined summary: {refined_summary[:100] if refined_summary else 'EMPTY'}...")
            # Validate refined summary
            if not refined_summary or not refined_summary.strip():
                refined_summary = extractive_summary
            result = {
                'refined_summary': refined_summary,
                'original_extractive': extractive_summary,
                'refinement_method': 'abstractive',
                'refinement_style': refinement_style,
                'selected_sentences': extractive_data.get('selected_sentences', []),
                'selected_indices': extractive_data.get('selected_indices', [])
            }
            print(f"🔄 Returning result with keys: {list(result.keys())}")
            return result
        except Exception as e:
            print(f"❌ Error in refine_extractive_summary: {e}")
            import traceback
            traceback.print_exc()
            return {
                'refined_summary': extractive_data.get('summary', '') if extractive_data else '',
                'original_extractive': extractive_data.get('summary', '') if extractive_data else '',
                'refinement_method': 'abstractive',
                'refinement_style': refinement_style,
                'error': f'Refinement failed: {str(e)}',
                'selected_sentences': extractive_data.get('selected_sentences', []) if extractive_data else [],
                'selected_indices': extractive_data.get('selected_indices', []) if extractive_data else []
            }
    
    def paraphrase_sentences(self, sentences: List[str]) -> List[str]:
        """Paraphrase individual sentences for better coherence"""
        paraphrased = []
        
        for sentence in sentences:
            if len(sentence.strip()) > 10:  # Only paraphrase meaningful sentences
                try:
                    paraphrased_sentence = self.generate_abstractive_summary(
                        sentence,
                        max_length=100,
                        min_length=10,
                        num_beams=2,
                        length_penalty=1.0
                    )
                    paraphrased.append(paraphrased_sentence)
                except:
                    paraphrased.append(sentence)  # Keep original if paraphrasing fails
            else:
                paraphrased.append(sentence)
        
        return paraphrased
    
    def ensure_coherence(self, summary: str) -> str:
        """Ensure coherence in the summary"""
        # Split into sentences
        sentences = summary.split('. ')
        
        if len(sentences) <= 1:
            return summary
        
        # Add transition words where appropriate
        transitions = [
            "Additionally", "Furthermore", "Moreover", "However", 
            "Therefore", "Consequently", "Meanwhile", "Subsequently"
        ]
        
        coherent_sentences = [sentences[0]]  # Keep first sentence as is
        
        for i, sentence in enumerate(sentences[1:], 1):
            if sentence.strip():
                # Randomly add transitions (but not too frequently)
                if i % 3 == 0 and len(sentence.split()) > 5:
                    transition = transitions[i % len(transitions)]
                    if not sentence.startswith(tuple(transitions)):
                        sentence = f"{transition.lower()}, {sentence.lower()}"
                
                coherent_sentences.append(sentence)
        
        return '. '.join(coherent_sentences)
