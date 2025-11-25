# debug_hybrid.py
import traceback
from summarizer.hybrid_summarizer import HybridSummarizer
from config import Config

def debug_hybrid_error():
    """Debug the exact location of the string index error"""
    print("🔍 Debugging Hybrid Summarization Error...")
    
    # Initialize
    config = Config()
    config_dict = {
        'spacy_model': config.MODELS['spacy_model'],
        'sentence_model': config.MODELS['sentence_embeddings'],
        'abstractive_model': config.MODELS['abstractive'],
        'importance_weights': config.IMPORTANCE_WEIGHTS
    }
    
    summarizer = HybridSummarizer(config_dict)
    
    # Test text
    test_text = "This is a test sentence for summarization. This is another sentence that provides more context. This is the third sentence to make it longer. Finally, this is the fourth sentence to complete the test."
    
    try:
        # Step 1: Test preprocessing
        print("🔄 Step 1: Testing preprocessing...")
        preprocessed_data = summarizer.preprocessor.preprocess_text(test_text)
        print(f"✅ Preprocessing successful: {len(preprocessed_data['sentences'])} sentences")
        
        # Step 2: Test importance assessment
        print("🔄 Step 2: Testing importance assessment...")
        importance_data = summarizer.importance_assessor.calculate_unified_importance_scores(preprocessed_data)
        print(f"✅ Importance assessment successful: {len(importance_data['scores'])} scores")
        
        # Step 3: Test sentence embeddings
        print("🔄 Step 3: Testing sentence embeddings...")
        sentence_texts = [sent['original'] for sent in preprocessed_data['sentences']]
        sentence_embeddings = summarizer.sentence_transformer.encode(sentence_texts)
        print(f"✅ Sentence embeddings successful: {sentence_embeddings.shape}")
        
        # Step 4: Test extractive summarization
        print("🔄 Step 4: Testing extractive summarization...")
        extractive_result = summarizer.extractive_summarizer.extractive_summarize(
            preprocessed_data, importance_data, sentence_embeddings, 3, method="mmr"
        )
        print(f"✅ Extractive summarization successful")
        print(f"📝 Extractive summary: {extractive_result['summary']}")
        
        # Step 5: Test abstractive refinement (this is likely where it fails)
        print("🔄 Step 5: Testing abstractive refinement...")
        print(f"Input to abstractive: '{extractive_result['summary']}'")
        print(f"Input length: {len(extractive_result['summary'])} characters")
        
        # Test the abstractive refinement step by step
        refined_result = summarizer.abstractive_summarizer.refine_extractive_summary(
            extractive_result, 150, 'balanced'
        )
        print(f"✅ Abstractive refinement successful")
        print(f"📝 Refined summary: {refined_result['refined_summary']}")
        
    except Exception as e:
        print(f"❌ Error at step: {e}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_hybrid_error()
