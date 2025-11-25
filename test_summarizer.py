
# test_summarizer.py
import sys
import os
from summarizer.preprocessor import TextPreprocessor
from summarizer.importance import ContentImportanceAssessment
from summarizer.extractive import ExtractiveSummarizer
from sentence_transformers import SentenceTransformer

def test_components():
    """Test individual components step by step"""
    print("🧪 Testing Hybrid Summarizer Components...")
    
    try:
        # Test 1: Preprocessor
        print("\n--- Test 1: Text Preprocessor ---")
        preprocessor = TextPreprocessor()
        print("✅ Preprocessor initialized")
        
        test_text = "This is a test sentence for summarization. This is another sentence that provides more context. This is the third sentence to make it longer."
        
        if not test_text.strip():
            print("❌ Empty text provided")
            return
            
        result = preprocessor.preprocess_text(test_text)
        print(f"✅ Preprocessing successful: {len(result['sentences'])} sentences found")
        
        # Test 2: Importance Assessment
        print("\n--- Test 2: Importance Assessment ---")
        importance_assessor = ContentImportanceAssessment()
        print("✅ Importance assessor initialized")
        
        importance_data = importance_assessor.calculate_unified_importance_scores(result)
        print(f"✅ Importance scores calculated: {len(importance_data['scores'])} scores")
        
        # Test 3: Extractive Summarization
        print("\n--- Test 3: Extractive Summarization ---")
        extractive_summarizer = ExtractiveSummarizer()
        print("✅ Extractive summarizer initialized")
        
        # Get sentence embeddings
        sentence_texts = [sent['original'] for sent in result['sentences']]
        sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        sentence_embeddings = sentence_transformer.encode(sentence_texts)
        
        # Generate extractive summary
        extractive_result = extractive_summarizer.extractive_summarize(
            result, importance_data, sentence_embeddings, target_length=2
        )
        
        print(f"✅ Extractive summary generated:")
        print(f"📝 Summary: {extractive_result['summary']}")
        print(f"📊 Selected {len(extractive_result['selected_sentences'])} sentences")
        
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

def test_simple_cases():
    """Test with different text inputs"""
    print("\n🔍 Testing Different Input Cases...")
    
    test_cases = [
        {
            'name': 'Empty string',
            'text': '',
        },
        {
            'name': 'Single sentence',
            'text': 'This is just one sentence.',
        },
        {
            'name': 'Normal text',
            'text': 'Artificial intelligence is transforming our world. Machine learning algorithms can process vast amounts of data. Natural language processing helps computers understand human language. These technologies are revolutionizing many industries.',
        }
    ]
    
    preprocessor = TextPreprocessor()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Case {i}: {test_case['name']} ---")
        print(f"Input length: {len(test_case['text'])} characters")
        
        try:
            if not test_case['text'].strip():
                print("❌ Empty input - skipping")
                continue
                
            result = preprocessor.preprocess_text(test_case['text'])
            
            if not result['sentences']:
                print("❌ No sentences found")
                continue
                
            print(f"✅ Found {len(result['sentences'])} sentences")
            
            # Show first sentence details
            first_sentence = result['sentences'][0]
            print(f"📝 First sentence: {first_sentence['original']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_components()
    test_simple_cases()
