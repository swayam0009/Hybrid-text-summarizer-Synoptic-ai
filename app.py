# Main Streamlit app
# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# Import custom modules
from config import Config
from database import DatabaseManager
from summarizer.hybrid_summarizer import HybridSummarizer
from utils.file_handler import FileHandler
from utils.auth import AuthManager

# Page configuration
st.set_page_config(
    page_title="Hybrid Text Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize configuration
config = Config()
config_dict = {
    'spacy_model': config.MODELS['spacy_model'],
    'sentence_model': config.MODELS['sentence_embeddings'],
    'abstractive_model': config.MODELS['abstractive'],
    'importance_weights': config.IMPORTANCE_WEIGHTS
}

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize all components (cached for performance)"""
    db_manager = DatabaseManager(config.DATABASE_PATH)
    auth_manager = AuthManager(str(config.DATABASE_PATH))
    file_handler = FileHandler()
    summarizer = HybridSummarizer(config_dict)
    
    return db_manager, auth_manager, file_handler, summarizer

# Custom CSS
def load_custom_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .summary-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .explanation-box {
        background: #fef7e0;
        border: 1px solid #f6cc02;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    load_custom_css()
    
    # Initialize components
    db_manager, auth_manager, file_handler, summarizer = initialize_components()
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    
    # Main header
    st.markdown('<h1 class="main-header">🤖 Hybrid Text Summarizer</h1>', unsafe_allow_html=True)
    
    # Authentication check
    if not st.session_state.authenticated:
        show_auth_page(auth_manager, db_manager)
    else:
        show_main_application(db_manager, auth_manager, file_handler, summarizer)

def show_auth_page(auth_manager, db_manager):
    """Show authentication page"""
    st.markdown("### 🔐 Authentication Required")
    
    # Create tabs for login and registration
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        show_login_form(auth_manager)
    
    with tab2:
        show_registration_form(auth_manager, db_manager)

def show_login_form(auth_manager):
    """Show login form"""
    with st.form("login_form"):
        st.subheader("Login to Your Account")
        
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.form_submit_button("Login", use_container_width=True):
            if username and password:
                # Add debug info
                st.write(f"Attempting login for user: {username}")
                result = auth_manager.authenticate_user(username, password)
                st.write(f"Login result: {result}")  # Debug output
                
                if result['success']:
                    st.session_state.authenticated = True
                    st.session_state.user_id = result['user_id']
                    st.session_state.username = result['username']
                    st.session_state.session_token = result['session_token']
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error(f"Login failed: {result['error']}")
            else:
                st.warning("Please enter both username and password")

def show_registration_form(auth_manager, db_manager):
    """Show registration form"""
    with st.form("registration_form"):
        st.subheader("Create New Account")
        
        username = st.text_input("Username", help="3-20 characters, alphanumeric and underscore only")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        # Show password requirements
        with st.expander("Password Requirements"):
            requirements = auth_manager.get_password_requirements()
            st.write(f"• Minimum {requirements['min_length']} characters")
            st.write("• At least one uppercase letter")
            st.write("• At least one lowercase letter")
            st.write("• At least one digit")
            st.write("• At least one special character")
        
        if st.form_submit_button("Register", use_container_width=True):
            if username and email and password and confirm_password:
                if password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    result = auth_manager.create_user_account(username, email, password)
                    
                    if result['success']:
                        # Create default user profile
                        profile_data = {
                            'domain_expertise': 'general',
                            'reading_speed': 200,
                            'detail_preference': 0.5,
                            'summary_length_preference': 3,
                            'technical_level': 0.5
                        }
                        db_manager.create_user_profile(result['user_id'], profile_data)
                        
                        st.success("Account created successfully! Please login.")
                    else:
                        st.error(result['error'])
            else:
                st.warning("Please fill in all fields")

def show_main_application(db_manager, auth_manager, file_handler, summarizer):
    """Show main application interface"""
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👋 Welcome, {st.session_state.username}!")
        
        if st.button("Logout", use_container_width=True):
            auth_manager.logout_user(st.session_state.get('session_token', ''))
            st.session_state.authenticated = False
            st.session_state.user_id = None
            st.session_state.username = None
            st.rerun()
        
        st.markdown("---")
        
        # Navigation
        page = st.selectbox(
            "Navigate to:",
            ["📝 Summarize Text", "👤 User Profile", "📊 Analytics", "🔧 Settings", "ℹ️ Help"]
        )
    
    # Main content based on selected page
    if page == "📝 Summarize Text":
        show_summarization_page(db_manager, file_handler, summarizer)
    elif page == "👤 User Profile":
        show_profile_page(db_manager)
    elif page == "📊 Analytics":
        show_analytics_page(db_manager)
    elif page == "🔧 Settings":
        show_settings_page(db_manager, auth_manager)
    elif page == "ℹ️ Help":
        show_help_page()

def show_summarization_page(db_manager, file_handler, summarizer):
    """Show text summarization page"""
    st.markdown("## 📝 Text Summarization")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["📝 Direct Text Input", "📁 File Upload", "🌐 URL Input"],
        horizontal=True
    )
    
    text_content = ""
    document_title = ""
    
    if input_method == "📝 Direct Text Input":
        text_content = st.text_area(
            "Enter your text here:",
            height=200,
            placeholder="Paste your text content here..."
        )
        document_title = st.text_input("Document Title (optional)")
    
    elif input_method == "📁 File Upload":
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=['txt', 'pdf', 'docx', 'html'],
            help="Supported formats: TXT, PDF, DOCX, HTML"
        )
        
        if uploaded_file is not None:
            with st.spinner("Processing file..."):
                result = file_handler.process_file(uploaded_file, uploaded_file.type)
                
                if result['success']:
                    text_content = result['text']
                    document_title = result['metadata'].get('filename', 'Uploaded Document')
                    
                    # Show file info
                    st.success(f"File processed successfully!")
                    with st.expander("File Information"):
                        st.write(f"**Filename:** {document_title}")
                        st.write(f"**File Size:** {result['metadata'].get('file_size', 'N/A')} bytes")
                        st.write(f"**Content Length:** {len(text_content)} characters")
                else:
                    st.error(result['error'])
    
    elif input_method == "🌐 URL Input":
        url = st.text_input("Enter URL:", placeholder="https://example.com/article")
        
        if url and st.button("Fetch Content"):
            with st.spinner("Fetching content from URL..."):
                result = file_handler.process_file(url)
                
                if result['success']:
                    text_content = result['text']
                    document_title = f"Web Content from {url}"
                    st.success("Content fetched successfully!")
                else:
                    st.error(result['error'])
    
    # Summarization options
    if text_content:
        st.markdown("### ⚙️ Summarization Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            summary_type = st.selectbox(
                "Summary Type:",
                ["hybrid", "extractive", "abstractive"],
                help="Hybrid combines both extractive and abstractive methods"
            )
        
        with col2:
            target_length = st.slider(
                "Target Length (sentences):",
                min_value=1,
                max_value=10,
                value=3
            )
        
        with col3:
            available_time = st.number_input(
                "Available reading time (minutes):",
                min_value=1,
                max_value=60,
                value=5,
                help="This will influence summary length based on your reading speed"
            )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                personalize = st.checkbox("Enable Personalization", value=True)
                explain_decisions = st.checkbox("Show Explanations", value=False)
            
            with col2:
                if personalize:
                    st.info("Personalization will adapt the summary based on your profile")
                if explain_decisions:
                    st.info("Explanations will show why certain sentences were selected")
        
        # Generate summary
        if st.button("🚀 Generate Summary", use_container_width=True):
            if not text_content or not text_content.strip():
                st.error("Please enter some text to summarize")
            else:
                # Add debug info
                st.write(f"**Debug Info:**")
                st.write(f"- Text length: {len(text_content)} characters")
                st.write(f"- Text preview: {text_content[:100]}...")
                with st.spinner("Generating summary..."):
                    try:
                        result = summarizer.summarize(
                            text=text_content,
                            user_id=st.session_state.user_id if personalize else None,
                            summary_type=summary_type,
                            target_length=target_length,
                            available_time=available_time,
                            personalize=personalize,
                            explain_decisions=explain_decisions
                        )
                        # Debug the result
                        st.write(f"**Result keys:** {list(result.keys())}")
                        if 'error' in result and summary_type == 'hybrid':
                            # Fallback to extractive if hybrid fails
                            st.warning("Hybrid summarization failed, falling back to extractive method...")
                            result = summarizer.summarize(
                                text=text_content,
                                user_id=st.session_state.user_id if personalize else None,
                                summary_type='extractive',
                                target_length=target_length,
                                available_time=available_time,
                                personalize=personalize,
                                explain_decisions=explain_decisions
                            )
                        if 'error' in result:
                            st.error(f"Error: {result['error']}")
                        else:
                            show_summary_results(result, db_manager, explain_decisions)
                    except Exception as e:
                        st.error(f"Summarization failed: {str(e)}")
                        st.write(f"**Error details:** {type(e).__name__}: {str(e)}")

def show_summary_results(result, db_manager, explain_decisions):
    """Display summarization results"""
    st.markdown("## 📋 Summary Results")
    
    # Summary text
    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
    st.markdown("### 📝 Generated Summary")
    st.write(result['summary'])
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Original Length", f"{result['original_length']} words")
    
    with col2:
        st.metric("Summary Length", f"{result['summary_length']} words")
    
    with col3:
        compression_ratio = result['compression_ratio'] * 100
        st.metric("Compression", f"{compression_ratio:.1f}%")
    
    with col4:
        coherence_score = result.get('coherence_score', 0) * 100
        st.metric("Coherence Score", f"{coherence_score:.1f}%")
    
    # Additional information
    with st.expander("📊 Detailed Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Summary Type:**", result['summary_type'].title())
            st.write("**Total Sentences:**", result['processing_metadata']['total_sentences'])
            st.write("**Selected Sentences:**", result['processing_metadata']['sentences_selected'])
        
        with col2:
            if result['personalization_applied']:
                st.write("**Personalization:** Applied")
                factors = result['personalization_factors']
                if factors:
                    st.write("**Personalization Factors:**")
                    for key, value in factors.items():
                        if isinstance(value, dict):
                            st.write(f"  - {key}: {value}")
            else:
                st.write("**Personalization:** Not applied")
    
    # Readability scores
    readability_scores = result.get('readability_scores', {})
    if readability_scores:
        with st.expander("📈 Readability Scores"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Flesch Reading Ease", f"{readability_scores.get('flesch_reading_ease', 0):.1f}")
                st.metric("Flesch-Kincaid Grade", f"{readability_scores.get('flesch_kincaid', 0):.1f}")
            
            with col2:
                st.metric("SMOG Index", f"{readability_scores.get('smog_index', 0):.1f}")
                st.metric("ARI", f"{readability_scores.get('automated_readability_index', 0):.1f}")
    
    # Explanations
    if explain_decisions and 'explanations' in result:
        show_explanations(result['explanations'])
    
    # User feedback
    st.markdown("### 💬 Feedback")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        feedback_text = st.text_area("How was this summary? (optional)", height=100)
    
    with col2:
        rating = st.selectbox("Rating", [1, 2, 3, 4, 5], index=4)
        
        if st.button("Submit Feedback"):
            # Save feedback to database
            feedback_data = {
                'rating': rating,
                'feedback_text': feedback_text,
                'improvement_suggestions': ''
            }
            
            # This would need the summary_id from the database
            st.success("Thank you for your feedback!")

def show_explanations(explanations):
    """Display explanations for summarization decisions"""
    st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
    st.markdown("### 🔍 Decision Explanations")
    
    # Sentence explanations
    if 'sentence_explanations' in explanations:
        st.markdown("#### Selected Sentences")
        for i, exp in enumerate(explanations['sentence_explanations']):
            with st.expander(f"Sentence {i+1}: {exp['sentence'][:50]}..."):
                st.write(f"**Importance Score:** {exp['importance_score']:.3f}")
                st.write(f"**Rank:** {exp['rank']} out of {exp['total_sentences']}")
                st.write(f"**Explanation:** {exp['explanation']}")
    
    # Personalization explanation
    if 'personalization_explanation' in explanations:
        st.markdown("#### Personalization Factors")
        pers_exp = explanations['personalization_explanation']
        st.write(pers_exp.get('text_explanation', 'No personalization applied'))
    
    # Visualizations
    if 'visualizations' in explanations:
        viz = explanations['visualizations']
        
        if viz.get('importance_chart'):
            st.markdown("#### Importance Score Breakdown")
            st.plotly_chart(viz['importance_chart'], use_container_width=True)
        
        if viz.get('comparison_chart'):
            st.markdown("#### Length Comparison")
            st.plotly_chart(viz['comparison_chart'], use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def save_to_history(db_manager, result, document_title, text_content):
    """Save summarization to user history"""
    try:
        summary_data = {
            'document_title': document_title,
            'document_content': text_content[:1000],  # Truncate for storage
            'summary': result['summary'],
            'summary_type': result['summary_type'],
            'summary_length': result['summary_length'],
            'importance_scores': result.get('importance_scores', []),
            'personalization_factors': result.get('personalization_factors', {})
        }
        
        db_manager.save_summarization_history(st.session_state.user_id, summary_data)
        
    except Exception as e:
        st.error(f"Failed to save to history: {str(e)}")

def show_profile_page(db_manager):
    """Show user profile page"""
    st.markdown("## 👤 User Profile")
    
    # Get user profile
    profile = db_manager.get_user_profile(st.session_state.user_id)
    
    if not profile:
        st.warning("No profile found. Creating default profile...")
        profile_data = {
            'domain_expertise': 'general',
            'reading_speed': 200,
            'detail_preference': 0.5,
            'summary_length_preference': 3,
            'technical_level': 0.5
        }
        db_manager.create_user_profile(st.session_state.user_id, profile_data)
        st.rerun()
    
    # Profile editing form
    with st.form("profile_form"):
        st.subheader("📝 Edit Profile")
        
        col1, col2 = st.columns(2)
        
        with col1:
            domain_expertise = st.selectbox(
                "Domain Expertise:",
                ["general", "technology", "science", "business", "healthcare", "finance", "sports", "politics"],
                index=["general", "technology", "science", "business", "healthcare", "finance", "sports", "politics"].index(profile.get('domain_expertise', 'general'))
            )
            
            reading_speed = st.slider(
                "Reading Speed (words per minute):",
                min_value=100,
                max_value=500,
                value=profile.get('reading_speed', 200),
                help="Average reading speed for adults is 200-250 WPM"
            )
            
            detail_preference = st.slider(
                "Detail Preference:",
                min_value=0.0,
                max_value=1.0,
                value=profile.get('detail_preference', 0.5),
                help="0 = Prefer concise summaries, 1 = Prefer detailed summaries"
            )
        
        with col2:
            summary_length_preference = st.slider(
                "Preferred Summary Length (sentences):",
                min_value=1,
                max_value=10,
                value=profile.get('summary_length_preference', 3)
            )
            
            technical_level = st.slider(
                "Technical Level:",
                min_value=0.0,
                max_value=1.0,
                value=profile.get('technical_level', 0.5),
                help="0 = Beginner, 1 = Expert"
            )
        
        if st.form_submit_button("Update Profile", use_container_width=True):
            # Update profile logic would go here
            st.success("Profile updated successfully!")
    
    # Profile analytics
    st.markdown("### 📊 Profile Analytics")
    
    # This would show user's summarization history, preferences, etc.
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Summaries", "25")  # This would come from database
    
    with col2:
        st.metric("Average Rating", "4.2")  # This would come from database
    
    with col3:
        st.metric("Days Active", "30")  # This would come from database

def show_analytics_page(db_manager):
    """Show analytics page"""
    st.markdown("## 📊 Analytics Dashboard")
    
    # This would contain comprehensive analytics
    st.info("📈 Analytics dashboard coming soon!")
    
    # Placeholder for actual analytics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Summary Statistics")
        # Create sample chart
        data = {
            'Summary Type': ['Extractive', 'Abstractive', 'Hybrid'],
            'Count': [15, 8, 12]
        }
        fig = px.bar(data, x='Summary Type', y='Count', title='Summary Types Used')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Reading Patterns")
        # Create sample chart
        data = {
            'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            'Summaries': [3, 5, 2, 4, 6, 1, 2]
        }
        fig = px.line(data, x='Day', y='Summaries', title='Weekly Summary Activity')
        st.plotly_chart(fig, use_container_width=True)

def show_settings_page(db_manager, auth_manager):
    """Show settings page"""
    st.markdown("## 🔧 Settings")
    
    # Password change
    st.markdown("### 🔒 Change Password")
    with st.form("password_form"):
        old_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")
        
        if st.form_submit_button("Change Password"):
            if new_password != confirm_password:
                st.error("New passwords do not match")
            else:
                result = auth_manager.change_password(
                    st.session_state.user_id, old_password, new_password
                )
                if result['success']:
                    st.success(result['message'])
                else:
                    st.error(result['error'])
    
    st.markdown("---")
    
    # Data management
    st.markdown("### 📁 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export My Data", use_container_width=True):
            st.info("Data export functionality coming soon!")
    
    with col2:
        if st.button("Delete My Account", use_container_width=True):
            st.warning("Account deletion functionality coming soon!")

def show_help_page():
    """Show help page"""
    st.markdown("## ℹ️ Help & Documentation")
    
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Input Your Text**: Choose from direct text input, file upload, or URL
    2. **Select Options**: Choose summary type, length, and personalization settings
    3. **Generate Summary**: Click the generate button to create your summary
    4. **Review Results**: View your summary with metrics and explanations
    
    ### 📝 Summary Types
    
    - **Extractive**: Selects the most important sentences from the original text
    - **Abstractive**: Creates new sentences by paraphrasing and synthesizing content
    - **Hybrid**: Combines both approaches for optimal results
    
    ### 🎯 Personalization
    
    The system personalizes summaries based on:
    - Your domain expertise
    - Reading speed and preferences
    - Historical feedback
    - Available time constraints
    
    ### 📊 Quality Metrics
    
    - **Compression Ratio**: How much the text was reduced
    - **Coherence Score**: How well the summary flows
    - **Readability Scores**: How easy the summary is to read
    
    ### 💡 Tips for Better Summaries
    
    - Provide clear, well-structured input text
    - Set appropriate length targets for your needs
    - Use personalization for better results
    - Provide feedback to improve future summaries
    
    ### 🔧 Technical Details
    
    - **Models Used**: BART, T5, BERT, spaCy
    - **Supported Formats**: TXT, PDF, DOCX, HTML, URLs
    - **Maximum File Size**: 10MB
    - **Processing Time**: 10-60 seconds depending on text length
    """)

if __name__ == "__main__":
    main()
