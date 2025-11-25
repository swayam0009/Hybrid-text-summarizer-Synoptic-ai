# Personalization engine
# summarizer/personalizer.py
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime, timedelta

class PersonalizationEngine:
    def __init__(self):
        self.user_profiles = {}
        self.content_vectors = {}
        self.interaction_history = {}
        
        # Domain expertise mapping
        self.domain_keywords = {
            'technology': ['software', 'hardware', 'computer', 'internet', 'digital', 'AI', 'machine learning'],
            'science': ['research', 'study', 'experiment', 'hypothesis', 'theory', 'analysis'],
            'business': ['company', 'market', 'profit', 'revenue', 'strategy', 'management'],
            'healthcare': ['patient', 'treatment', 'medical', 'health', 'disease', 'therapy'],
            'finance': ['investment', 'money', 'bank', 'stock', 'financial', 'economy'],
            'sports': ['game', 'team', 'player', 'score', 'championship', 'tournament'],
            'politics': ['government', 'policy', 'election', 'political', 'vote', 'candidate']
        }
    
    def create_user_profile(self, user_id: int, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new user profile"""
        default_profile = {
            'user_id': user_id,
            'domain_expertise': profile_data.get('domain_expertise', 'general'),
            'reading_speed': profile_data.get('reading_speed', 200),  # words per minute
            'detail_preference': profile_data.get('detail_preference', 0.5),  # 0 = concise, 1 = detailed
            'summary_length_preference': profile_data.get('summary_length_preference', 3),
            'technical_level': profile_data.get('technical_level', 0.5),  # 0 = beginner, 1 = expert
            'preferred_topics': profile_data.get('preferred_topics', []),
            'reading_history': [],
            'feedback_history': [],
            'interaction_patterns': {
                'avg_reading_time': 0,
                'preferred_summary_lengths': [],
                'topic_preferences': {},
                'style_preferences': {}
            },
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
        
        self.user_profiles[user_id] = default_profile
        return default_profile
    
    def update_user_profile(self, user_id: int, interaction_data: Dict[str, Any]):
        """Update user profile based on interaction data"""
        if user_id not in self.user_profiles:
            return
        
        profile = self.user_profiles[user_id]
        
        # Update reading history
        if 'document_topic' in interaction_data:
            profile['reading_history'].append({
                'topic': interaction_data['document_topic'],
                'timestamp': datetime.now().isoformat(),
                'reading_time': interaction_data.get('reading_time', 0),
                'summary_length': interaction_data.get('summary_length', 0)
            })
        
        # Update feedback history
        if 'feedback_rating' in interaction_data:
            profile['feedback_history'].append({
                'rating': interaction_data['feedback_rating'],
                'summary_type': interaction_data.get('summary_type', 'hybrid'),
                'timestamp': datetime.now().isoformat()
            })
        
        # Update interaction patterns
        self._update_interaction_patterns(user_id, interaction_data)
        
        profile['last_updated'] = datetime.now().isoformat()
    
    def _update_interaction_patterns(self, user_id: int, interaction_data: Dict[str, Any]):
        """Update user interaction patterns"""
        profile = self.user_profiles[user_id]
        patterns = profile['interaction_patterns']
        
        # Update average reading time
        if 'reading_time' in interaction_data:
            reading_times = [h.get('reading_time', 0) for h in profile['reading_history']]
            reading_times.append(interaction_data['reading_time'])
            patterns['avg_reading_time'] = np.mean(reading_times)
        
        # Update preferred summary lengths
        if 'summary_length' in interaction_data and 'feedback_rating' in interaction_data:
            if interaction_data['feedback_rating'] >= 4:  # Good feedback
                patterns['preferred_summary_lengths'].append(interaction_data['summary_length'])
        
        # Update topic preferences
        if 'document_topic' in interaction_data and 'feedback_rating' in interaction_data:
            topic = interaction_data['document_topic']
            if topic not in patterns['topic_preferences']:
                patterns['topic_preferences'][topic] = []
            patterns['topic_preferences'][topic].append(interaction_data['feedback_rating'])
    
    def detect_domain_expertise(self, document_content: str) -> str:
        """Detect document domain based on content"""
        content_lower = document_content.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            domain_scores[domain] = score
        
        if not domain_scores or max(domain_scores.values()) == 0:
            return 'general'
        
        return max(domain_scores, key=domain_scores.get)
    
    def calculate_time_constraint_factor(self, user_id: int, available_time: Optional[int] = None) -> float:
        """Calculate how time constraints affect summary preferences"""
        if user_id not in self.user_profiles:
            return 1.0
        
        profile = self.user_profiles[user_id]
        reading_speed = profile['reading_speed']  # words per minute
        
        if available_time is None:
            return 1.0
        
        # Calculate optimal summary length based on available time
        optimal_word_count = available_time * reading_speed
        
        # Normalize to a factor between 0.5 and 2.0
        if optimal_word_count < 50:
            return 0.5  # Very short summary
        elif optimal_word_count > 300:
            return 2.0  # Can handle longer summary
        else:
            return 0.5 + (optimal_word_count - 50) / 250 * 1.5
    
    def get_content_based_preferences(self, user_id: int, document_content: str) -> Dict[str, Any]:
        """Get content-based personalization preferences"""
        if user_id not in self.user_profiles:
            return {}
        
        profile = self.user_profiles[user_id]
        document_domain = self.detect_domain_expertise(document_content)
        
        preferences = {
            'technical_level': profile['technical_level'],
            'detail_preference': profile['detail_preference'],
            'domain_match': document_domain == profile['domain_expertise'],
            'summary_length_preference': profile['summary_length_preference']
        }
        
        # Adjust preferences based on domain expertise
        if preferences['domain_match']:
            preferences['technical_level'] = min(1.0, preferences['technical_level'] + 0.2)
            preferences['detail_preference'] = min(1.0, preferences['detail_preference'] + 0.1)
        
        return preferences
    
    def get_collaborative_filtering_recommendations(self, user_id: int, 
                                                  document_features: Dict[str, Any]) -> Dict[str, float]:
        """Get recommendations based on similar users"""
        if user_id not in self.user_profiles:
            return {}
        
        current_profile = self.user_profiles[user_id]
        similar_users = []
        
        # Find similar users based on profile similarity
        for other_id, other_profile in self.user_profiles.items():
            if other_id == user_id:
                continue
            
            similarity = self._calculate_profile_similarity(current_profile, other_profile)
            if similarity > 0.3:  # Threshold for similarity
                similar_users.append((other_id, similarity))
        
        # Sort by similarity
        similar_users.sort(key=lambda x: x[1], reverse=True)
        
        # Get recommendations from top 5 similar users
        recommendations = {
            'summary_length': current_profile['summary_length_preference'],
            'detail_level': current_profile['detail_preference'],
            'technical_level': current_profile['technical_level']
        }
        
        if similar_users:
            top_similar = similar_users[:5]
            
            # Average preferences from similar users
            similar_lengths = []
            similar_details = []
            similar_technical = []
            
            for similar_id, similarity in top_similar:
                similar_profile = self.user_profiles[similar_id]
                similar_lengths.append(similar_profile['summary_length_preference'])
                similar_details.append(similar_profile['detail_preference'])
                similar_technical.append(similar_profile['technical_level'])
            
            # Weighted average
            recommendations['summary_length'] = int(np.average(similar_lengths, weights=[s[1] for s in top_similar]))
            recommendations['detail_level'] = np.average(similar_details, weights=[s[1] for s in top_similar])
            recommendations['technical_level'] = np.average(similar_technical, weights=[s[1] for s in top_similar])
        
        return recommendations
    
    def _calculate_profile_similarity(self, profile1: Dict[str, Any], profile2: Dict[str, Any]) -> float:
        """Calculate similarity between two user profiles"""
        # Compare key attributes
        domain_match = 1.0 if profile1['domain_expertise'] == profile2['domain_expertise'] else 0.0
        
        reading_speed_sim = 1.0 - abs(profile1['reading_speed'] - profile2['reading_speed']) / 1000
        reading_speed_sim = max(0.0, reading_speed_sim)
        
        detail_sim = 1.0 - abs(profile1['detail_preference'] - profile2['detail_preference'])
        technical_sim = 1.0 - abs(profile1['technical_level'] - profile2['technical_level'])
        
        # Weighted similarity
        similarity = (0.3 * domain_match + 0.2 * reading_speed_sim + 
                     0.25 * detail_sim + 0.25 * technical_sim)
        
        return similarity
    
    def personalize_summary_parameters(self, user_id: int, 
                                     document_content: str,
                                     base_parameters: Dict[str, Any],
                                     available_time: Optional[int] = None) -> Dict[str, Any]:
        """Personalize summary parameters based on user profile"""
        if user_id not in self.user_profiles:
            return base_parameters
        
        # Get personalization factors
        content_prefs = self.get_content_based_preferences(user_id, document_content)
        collab_recs = self.get_collaborative_filtering_recommendations(user_id, {})
        time_factor = self.calculate_time_constraint_factor(user_id, available_time)
        
        # Create personalized parameters
        personalized_params = base_parameters.copy()
        
        # Adjust summary length
        base_length = base_parameters.get('summary_length', 3)
        detail_factor = content_prefs.get('detail_preference', 0.5)
        collaborative_length = collab_recs.get('summary_length', base_length)
        
        personalized_length = int(base_length * (1 + detail_factor * 0.5) * time_factor)
        personalized_length = max(1, min(personalized_length, 10))  # Bounds checking
        
        personalized_params['summary_length'] = personalized_length
        
        # Adjust technical level
        personalized_params['technical_level'] = content_prefs.get('technical_level', 0.5)
        
        # Adjust abstractive refinement style
        detail_pref = content_prefs.get('detail_preference', 0.5)
        if detail_pref < 0.3:
            personalized_params['refinement_style'] = 'concise'
        elif detail_pref > 0.7:
            personalized_params['refinement_style'] = 'detailed'
        else:
            personalized_params['refinement_style'] = 'balanced'
        
        # Store personalization factors for explanation
        personalized_params['personalization_factors'] = {
            'content_based': content_prefs,
            'collaborative': collab_recs,
            'time_constraint': time_factor,
            'original_length': base_length,
            'personalized_length': personalized_length
        }
        
        return personalized_params
    
    def get_user_summary_history(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's summarization history"""
        if user_id not in self.user_profiles:
            return []
        
        profile = self.user_profiles[user_id]
        history = profile.get('reading_history', [])
        
        # Sort by timestamp (most recent first)
        history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return history[:limit]
    
    def analyze_user_preferences(self, user_id: int) -> Dict[str, Any]:
        """Analyze user preferences from interaction history"""
        if user_id not in self.user_profiles:
            return {}
        
        profile = self.user_profiles[user_id]
        
        # Analyze feedback patterns
        feedback_history = profile.get('feedback_history', [])
        if feedback_history:
            avg_rating = np.mean([f['rating'] for f in feedback_history])
            preferred_types = {}
            
            for feedback in feedback_history:
                summary_type = feedback.get('summary_type', 'hybrid')
                if summary_type not in preferred_types:
                    preferred_types[summary_type] = []
                preferred_types[summary_type].append(feedback['rating'])
            
            # Calculate average rating per type
            for summary_type in preferred_types:
                preferred_types[summary_type] = np.mean(preferred_types[summary_type])
        else:
            avg_rating = 0.0
            preferred_types = {}
        
        # Analyze reading patterns
        reading_history = profile.get('reading_history', [])
        topic_distribution = {}
        
        for reading in reading_history:
            topic = reading.get('topic', 'general')
            topic_distribution[topic] = topic_distribution.get(topic, 0) + 1
        
        return {
            'average_feedback_rating': avg_rating,
            'preferred_summary_types': preferred_types,
            'topic_distribution': topic_distribution,
            'total_interactions': len(reading_history),
            'profile_completeness': self._calculate_profile_completeness(profile)
        }
    
    def _calculate_profile_completeness(self, profile: Dict[str, Any]) -> float:
        """Calculate how complete a user profile is"""
        required_fields = ['domain_expertise', 'reading_speed', 'detail_preference', 
                          'summary_length_preference', 'technical_level']
        
        completeness = 0.0
        for field in required_fields:
            if field in profile and profile[field] is not None:
                completeness += 0.2
        
        # Bonus for interaction history
        if profile.get('reading_history'):
            completeness += 0.1
        if profile.get('feedback_history'):
            completeness += 0.1
        
        return min(1.0, completeness)
