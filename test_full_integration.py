# test_full_integration.py
import sys
import os

def test_hybrid_summarizer():
    """Test the complete HybridSummarizer class"""
    print("🧪 Testing Complete Hybrid Summarizer...")
    
    try:
        # Import the main class
        from summarizer.hybrid_summarizer import HybridSummarizer
        from config import Config
        
        # Initialize config
        config = Config()
        config_dict = {
            'spacy_model': config.MODELS['spacy_model'],
            'sentence_model': config.MODELS['sentence_embeddings'],
            'abstractive_model': config.MODELS['abstractive'],
            'importance_weights': config.IMPORTANCE_WEIGHTS
        }
        
        print("✅ Config loaded successfully")
        
        # Initialize summarizer (this might be where the error occurs)
        print("🔄 Initializing HybridSummarizer...")
        summarizer = HybridSummarizer(config_dict)
        print("✅ HybridSummarizer initialized successfully")
        
        # Test with the same text that worked in components
        test_text = "This is a test sentence for summarization. This is another sentence that provides more context. This is the third sentence to make it longer. Finally, this is the fourth sentence to complete the test."
        
        print(f"🔄 Testing with text ({len(test_text)} characters)...")
        
        # Try extractive first (simpler)
        result = summarizer.summarize(
            text=test_text,
            summary_type='extractive',
            target_length=2,
            personalize=False,
            explain_decisions=False
        )
        
        if 'error' in result:
            print(f"❌ Error in extractive summarization: {result['error']}")
        else:
            print(f"✅ Extractive summarization successful!")
            print(f"📝 Summary: {result['summary']}")
            print(f"📊 Original: {result.get('original_length', 0)} words")
            print(f"📊 Summary: {result.get('summary_length', 0)} words")
        
        # Try hybrid approach
        print("\n🔄 Testing hybrid approach...")
        result2 = summarizer.summarize(
            text=test_text,
            summary_type='hybrid',
            target_length=2,
            personalize=False,
            explain_decisions=False
        )
        
        if 'error' in result2:
            print(f"❌ Error in hybrid summarization: {result2['error']}")
        else:
            print(f"✅ Hybrid summarization successful!")
            print(f"📝 Summary: {result2['summary']}")
        
    except Exception as e:
        print(f"❌ Error in full integration: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

def test_streamlit_integration():
    """Test the specific pattern used in Streamlit app"""
    print("\n🖥️  Testing Streamlit Integration Pattern...")
    
    try:
        # Simulate what happens in your Streamlit app
        from summarizer.hybrid_summarizer import HybridSummarizer
        from config import Config
        
        # Initialize exactly as in your app
        config = Config()
        config_dict = {
            'spacy_model': config.MODELS['spacy_model'],
            'sentence_model': config.MODELS['sentence_embeddings'],
            'abstractive_model': config.MODELS['abstractive'],
            'importance_weights': config.IMPORTANCE_WEIGHTS
        }
        
        summarizer = HybridSummarizer(config_dict)
        
        # Test with various scenarios that might occur in Streamlit
        test_scenarios = [
            {
                'name': 'Normal user input',
                'text': 'Artificial intelligence is revolutionizing many industries. Machine learning algorithms can process vast amounts of data quickly. Natural language processing helps computers understand human language. These technologies are transforming how we work and live.',
                'params': {
                    'user_id': None,
                    'summary_type': 'hybrid',
                    'target_length': 2,
                    'available_time': 5,
                    'personalize': False,
                    'explain_decisions': False
                }
            },
            {
                'name': 'With personalization',
                'text': 'Technology companies are investing heavily in AI research. Machine learning models require large datasets for training. Deep learning has shown remarkable results in image recognition. Natural language processing continues to improve.',
                'params': {
                    'user_id': 1,
                    'summary_type': 'hybrid',
                    'target_length': 3,
                    'available_time': 10,
                    'personalize': True,
                    'explain_decisions': True
                }
            }
        ]
        
        for scenario in test_scenarios:
            print(f"\n--- Testing: {scenario['name']} ---")
            
            try:
                result = summarizer.summarize(
                    text=scenario['text'],
                    **scenario['params']
                )
                
                if 'error' in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    print(f"✅ Success! Summary: {result['summary'][:100]}...")
                    
            except Exception as e:
                print(f"❌ Exception in {scenario['name']}: {str(e)}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"❌ Error in Streamlit integration test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hybrid_summarizer()
    test_streamlit_integration()
