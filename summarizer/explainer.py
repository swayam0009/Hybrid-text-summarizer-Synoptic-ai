# Decision explanations
# summarizer/explainer.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from typing import Dict, Any, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

class DecisionExplainer:
    def __init__(self):
        self.explanation_templates = {
            'sentence_selection': "This sentence was selected because it scored {score:.2f} in importance, "
                                "ranking {rank} out of {total} sentences.",
            'importance_factors': "The importance score is based on: TF-IDF ({tfidf:.2f}), "
                                "Position ({position:.2f}), Entities ({entities:.2f}), "
                                "Semantic relevance ({semantic:.2f})",
            'personalization': "This summary was personalized based on your {domain} expertise, "
                             "preference for {detail_level} detail, and {time_constraint} time constraints.",
            'method_choice': "The {method} method was chosen because it provides {reasoning}."
        }
    
    def explain_sentence_selection(self, selected_sentences: List[str],
                                 selected_indices: List[int],
                                 importance_scores: np.ndarray,
                                 all_sentences: List[str]) -> List[Dict[str, Any]]:
        """Explain why specific sentences were selected"""
        explanations = []
        
        for i, (sentence, idx) in enumerate(zip(selected_sentences, selected_indices)):
            score = importance_scores[idx] if idx < len(importance_scores) else 0.0
            rank = np.sum(importance_scores > score) + 1
            
            explanation = {
                'sentence': sentence,
                'index': idx,
                'importance_score': score,
                'rank': int(rank),
                'total_sentences': len(all_sentences),
                'explanation': self.explanation_templates['sentence_selection'].format(
                    score=score, rank=rank, total=len(all_sentences)
                )
            }
            explanations.append(explanation)
        
        return explanations
    
    def explain_importance_scoring(self, importance_components: Dict[str, np.ndarray],
                                 selected_indices: List[int]) -> List[Dict[str, Any]]:
        """Explain how importance scores were calculated"""
        explanations = []
        
        for idx in selected_indices:
            components = {}
            for component_name, scores in importance_components.items():
                if idx < len(scores):
                    components[component_name] = scores[idx]
                else:
                    components[component_name] = 0.0
            
            explanation = {
                'sentence_index': idx,
                'components': components,
                'explanation': self.explanation_templates['importance_factors'].format(
                    tfidf=components.get('tfidf', 0.0),
                    position=components.get('position', 0.0),
                    entities=components.get('entity', 0.0),
                    semantic=components.get('semantic', 0.0)
                )
            }
            explanations.append(explanation)
        
        return explanations
    
    def explain_personalization_factors(self, personalization_data: Dict[str, Any]) -> Dict[str, Any]:
        """Explain how personalization affected the summary"""
        factors = personalization_data.get('personalization_factors', {})
        
        explanation = {
            'content_based': factors.get('content_based', {}),
            'collaborative': factors.get('collaborative', {}),
            'time_constraint': factors.get('time_constraint', 1.0),
            'length_adjustment': {
                'original': factors.get('original_length', 3),
                'personalized': factors.get('personalized_length', 3)
            }
        }
        
        # Generate text explanation
        content_prefs = factors.get('content_based', {})
        domain = content_prefs.get('domain_expertise', 'general')
        detail_level = 'high' if content_prefs.get('detail_preference', 0.5) > 0.7 else 'moderate'
        time_factor = factors.get('time_constraint', 1.0)
        
        if time_factor < 0.8:
            time_constraint = 'limited'
        elif time_factor > 1.2:
            time_constraint = 'flexible'
        else:
            time_constraint = 'moderate'
        
        explanation['text_explanation'] = self.explanation_templates['personalization'].format(
            domain=domain, detail_level=detail_level, time_constraint=time_constraint
        )
        
        return explanation
    
    def explain_method_choice(self, method: str, method_reasoning: str) -> Dict[str, Any]:
        """Explain why a specific summarization method was chosen"""
        method_explanations = {
            'extractive': 'direct extraction of key sentences while preserving original content',
            'abstractive': 'paraphrasing and synthesis for better readability',
            'hybrid': 'combination of extraction and abstraction for optimal results',
            'mmr': 'balance between relevance and diversity in sentence selection',
            'greedy': 'selection of highest-scoring sentences for maximum relevance'
        }
        
        return {
            'method': method,
            'reasoning': method_explanations.get(method, method_reasoning),
            'explanation': self.explanation_templates['method_choice'].format(
                method=method, reasoning=method_explanations.get(method, method_reasoning)
            )
        }
    
    def generate_importance_visualization(self, importance_components: Dict[str, np.ndarray],
                                        selected_indices: List[int],
                                        sentence_texts: List[str]) -> go.Figure:
        """Generate visualization of importance scores"""
        # Prepare data for visualization
        component_names = list(importance_components.keys())
        selected_data = []
        
        for idx in selected_indices:
            sentence_data = {
                'sentence_index': idx,
                'sentence_text': sentence_texts[idx][:50] + '...' if len(sentence_texts[idx]) > 50 else sentence_texts[idx]
            }
            
            for component in component_names:
                if idx < len(importance_components[component]):
                    sentence_data[component] = importance_components[component][idx]
                else:
                    sentence_data[component] = 0.0
            
            selected_data.append(sentence_data)
        
        # Create DataFrame
        df = pd.DataFrame(selected_data)
        
        # Create stacked bar chart
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3[:len(component_names)]
        
        for i, component in enumerate(component_names):
            fig.add_trace(go.Bar(
                name=component.title(),
                x=df['sentence_text'],
                y=df[component],
                marker_color=colors[i]
            ))
        
        fig.update_layout(
            title='Importance Score Components for Selected Sentences',
            xaxis_title='Selected Sentences',
            yaxis_title='Score',
            barmode='stack',
            height=500
        )
        
        return fig
    
    def generate_word_cloud(self, text: str, max_words: int = 100) -> WordCloud:
        """Generate word cloud for key terms"""
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=max_words,
            colormap='viridis'
        ).generate(text)
        
        return wordcloud
    
    def create_summary_comparison_chart(self, original_length: int,
                                      summary_length: int,
                                      compression_ratio: float) -> go.Figure:
        """Create visualization comparing original and summary lengths"""
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['Original Document', 'Summary'],
            y=[original_length, summary_length],
            marker_color=['lightblue', 'darkblue'],
            text=[f'{original_length} words', f'{summary_length} words'],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f'Document Compression (Ratio: {compression_ratio:.1%})',
            yaxis_title='Word Count',
            height=400
        )
        
        return fig
    
    def generate_confidence_scores(self, importance_scores: np.ndarray,
                                 selected_indices: List[int]) -> Dict[str, float]:
        """Generate confidence scores for summary quality"""
        if len(importance_scores) == 0 or len(selected_indices) == 0:
            return {'overall_confidence': 0.0, 'selection_confidence': 0.0, 'coverage_confidence': 0.0}
        
        # Selection confidence: how well the selected sentences score
        selected_scores = importance_scores[selected_indices]
        selection_confidence = np.mean(selected_scores)
        
        # Coverage confidence: how well the summary covers the document
        total_importance = np.sum(importance_scores)
        covered_importance = np.sum(selected_scores)
        coverage_confidence = covered_importance / total_importance if total_importance > 0 else 0.0
        
        # Overall confidence: combination of selection and coverage
        overall_confidence = (selection_confidence + coverage_confidence) / 2
        
        return {
            'overall_confidence': overall_confidence,
            'selection_confidence': selection_confidence,
            'coverage_confidence': coverage_confidence
        }
    
    def create_attention_heatmap(self, attention_weights: np.ndarray,
                               sentence_texts: List[str]) -> go.Figure:
        """Create attention heatmap for sentence relationships"""
        if len(attention_weights) == 0 or len(sentence_texts) == 0:
            return go.Figure()
        
        # Truncate sentence texts for display
        display_texts = [text[:30] + '...' if len(text) > 30 else text for text in sentence_texts]
        
        fig = go.Figure(data=go.Heatmap(
            z=attention_weights,
            x=display_texts,
            y=display_texts,
            colorscale='Viridis',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Sentence Attention Weights',
            xaxis_title='Target Sentences',
            yaxis_title='Source Sentences',
            height=600
        )
        
        return fig
    
    def create_topic_distribution_chart(self, topic_scores: Dict[str, float]) -> go.Figure:
        """Create chart showing topic distribution in the summary"""
        if not topic_scores:
            return go.Figure()
        
        topics = list(topic_scores.keys())
        scores = list(topic_scores.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=topics,
            values=scores,
            hole=0.3
        )])
        
        fig.update_layout(
            title='Topic Distribution in Summary',
            height=400
        )
        
        return fig
    
    def create_sentiment_analysis_chart(self, sentiment_scores: Dict[str, float]) -> go.Figure:
        """Create chart showing sentiment analysis of the summary"""
        if not sentiment_scores:
            return go.Figure()
        
        sentiments = list(sentiment_scores.keys())
        scores = list(sentiment_scores.values())
        
        colors = ['green' if s > 0.1 else 'red' if s < -0.1 else 'gray' for s in scores]
        
        fig = go.Figure(data=[go.Bar(
            x=sentiments,
            y=scores,
            marker_color=colors
        )])
        
        fig.update_layout(
            title='Sentiment Analysis of Summary',
            xaxis_title='Sentiment',
            yaxis_title='Score',
            height=400
        )
        
        return fig
    
    def create_readability_radar_chart(self, readability_scores: Dict[str, float]) -> go.Figure:
        """Create radar chart for readability metrics"""
        if not readability_scores:
            return go.Figure()
        
        # Normalize scores to 0-100 scale
        normalized_scores = {}
        for metric, score in readability_scores.items():
            if metric == 'flesch_reading_ease':
                normalized_scores[metric] = max(0, min(100, score))
            elif metric == 'flesch_kincaid':
                normalized_scores[metric] = max(0, min(100, (20 - score) * 5))  # Invert and scale
            else:
                normalized_scores[metric] = max(0, min(100, (20 - score) * 5))  # Invert and scale
        
        categories = list(normalized_scores.keys())
        values = list(normalized_scores.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Readability Scores'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title='Readability Metrics Radar Chart',
            height=500
        )
        
        return fig
    
    def create_comprehensive_explanation(self, summary_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive explanation of summarization decisions"""
        # Extract data
        selected_sentences = summary_data.get('selected_sentences', [])
        selected_indices = summary_data.get('selected_indices', [])
        importance_scores = summary_data.get('importance_scores', np.array([]))
        importance_components = summary_data.get('importance_components', {})
        all_sentences = summary_data.get('all_sentences', [])
        personalization_data = summary_data.get('personalization_data', {})
        method = summary_data.get('method', 'hybrid')
        
        # Generate explanations
        sentence_explanations = self.explain_sentence_selection(
            selected_sentences, selected_indices, importance_scores, all_sentences
        )
        
        importance_explanations = self.explain_importance_scoring(
            importance_components, selected_indices
        )
        
        personalization_explanation = self.explain_personalization_factors(personalization_data)
        
        method_explanation = self.explain_method_choice(method, 'optimal balance of relevance and readability')
        
        confidence_scores = self.generate_confidence_scores(importance_scores, selected_indices)
        
        # Create visualizations
        visualizations = {}
        
        if importance_components and selected_indices and all_sentences:
            visualizations['importance_chart'] = self.generate_importance_visualization(
                importance_components, selected_indices, all_sentences
            )
        
        # Generate summary statistics
        original_length = len(' '.join(all_sentences).split()) if all_sentences else 0
        summary_length = len(' '.join(selected_sentences).split()) if selected_sentences else 0
        compression_ratio = (original_length - summary_length) / original_length if original_length > 0 else 0
        
        visualizations['comparison_chart'] = self.create_summary_comparison_chart(
            original_length, summary_length, compression_ratio
        )
        
        # Add readability radar chart if readability scores are available
        readability_scores = summary_data.get('readability_scores', {})
        if readability_scores:
            visualizations['readability_radar'] = self.create_readability_radar_chart(readability_scores)
        
        # Add topic distribution if available
        topic_scores = summary_data.get('topic_scores', {})
        if topic_scores:
            visualizations['topic_distribution'] = self.create_topic_distribution_chart(topic_scores)
        
        # Add sentiment analysis if available
        sentiment_scores = summary_data.get('sentiment_scores', {})
        if sentiment_scores:
            visualizations['sentiment_analysis'] = self.create_sentiment_analysis_chart(sentiment_scores)
        
        return {
            'sentence_explanations': sentence_explanations,
            'importance_explanations': importance_explanations,
            'personalization_explanation': personalization_explanation,
            'method_explanation': method_explanation,
            'confidence_scores': confidence_scores,
            'visualizations': visualizations,
            'summary_statistics': {
                'original_length': original_length,
                'summary_length': summary_length,
                'compression_ratio': compression_ratio,
                'sentences_selected': len(selected_sentences),
                'total_sentences': len(all_sentences)
            }
        }
    
    def generate_explanation_report(self, summary_data: Dict[str, Any]) -> str:
        """Generate a text-based explanation report"""
        comprehensive_explanation = self.create_comprehensive_explanation(summary_data)
        
        report = []
        report.append("# Summarization Decision Report\n")
        
        # Summary statistics
        stats = comprehensive_explanation['summary_statistics']
        report.append(f"## Summary Statistics")
        report.append(f"- Original document: {stats['original_length']} words")
        report.append(f"- Generated summary: {stats['summary_length']} words")
        report.append(f"- Compression ratio: {stats['compression_ratio']:.1%}")
        report.append(f"- Sentences selected: {stats['sentences_selected']} out of {stats['total_sentences']}")
        report.append("")
        
        # Method explanation
        method_exp = comprehensive_explanation['method_explanation']
        report.append(f"## Method Used: {method_exp['method'].title()}")
        report.append(f"Reasoning: {method_exp['reasoning']}")
        report.append("")
        
        # Confidence scores
        confidence = comprehensive_explanation['confidence_scores']
        report.append(f"## Confidence Scores")
        report.append(f"- Overall confidence: {confidence['overall_confidence']:.1%}")
        report.append(f"- Selection confidence: {confidence['selection_confidence']:.1%}")
        report.append(f"- Coverage confidence: {confidence['coverage_confidence']:.1%}")
        report.append("")
        
        # Personalization
        pers_exp = comprehensive_explanation['personalization_explanation']
        if pers_exp.get('text_explanation'):
            report.append(f"## Personalization")
            report.append(pers_exp['text_explanation'])
            report.append("")
        
        # Selected sentences
        sentence_exps = comprehensive_explanation['sentence_explanations']
        if sentence_exps:
            report.append(f"## Selected Sentences")
            for i, exp in enumerate(sentence_exps, 1):
                report.append(f"### Sentence {i}")
                report.append(f"**Text:** {exp['sentence']}")
                report.append(f"**Score:** {exp['importance_score']:.3f}")
                report.append(f"**Rank:** {exp['rank']} out of {exp['total_sentences']}")
                report.append("")
        
        return "\n".join(report)
    
    def export_explanations(self, summary_data: Dict[str, Any], format: str = 'json') -> str:
        """Export explanations in various formats"""
        comprehensive_explanation = self.create_comprehensive_explanation(summary_data)
        
        if format == 'json':
            import json
            # Convert numpy arrays to lists for JSON serialization
            serializable_data = self._make_json_serializable(comprehensive_explanation)
            return json.dumps(serializable_data, indent=2)
        
        elif format == 'markdown':
            return self.generate_explanation_report(summary_data)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
