import streamlit as st
import os
from dotenv import load_dotenv
import yaml
from rag_system import RAGSystem, UserContext
from typing import Dict, Any
import time
import tempfile

# Load environment variables
load_dotenv()

# Page configuration with saffron theme
st.set_page_config(
    page_title="🙏 JAI GURU DEV AI Chatbot",
    page_icon="🙏",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "JAI GURU DEV AI - Wisdom from Sri Sri Ravi Shankar's Teachings"
    }
)

# Custom CSS for saffron theme with improved dropdown visibility
st.markdown("""
<style>
    .main {
        background-color: #FFF8DC;
    }
    
    .stApp {
        background: linear-gradient(135deg, #FFF8DC 0%, #FFEFD5 100%);
    }
    
    .css-1d391kg {
        background-color: #FF8C00;
    }
    
    h1, h2, h3 {
        color: #8B4513 !important;
        font-family: 'serif';
    }
    
    .stSelectbox label, .stTextInput label, .stTextArea label {
        color: #8B4513 !important;
        font-weight: bold;
        font-size: 16px;
    }
    
    /* Improved dropdown visibility */
    .stSelectbox > div > div {
        background-color: white !important;
        border: 2px solid #FF8C00 !important;
        border-radius: 8px !important;
        color: #333333 !important;
    }
    
    .stSelectbox > div > div > div {
        color: #333333 !important;
        font-weight: 500 !important;
    }
    
    /* Dropdown options styling */
    .stSelectbox [role="listbox"] {
        background-color: white !important;
        border: 2px solid #FF8C00 !important;
        border-radius: 8px !important;
    }
    
    .stSelectbox [role="option"] {
        background-color: white !important;
        color: #333333 !important;
        font-weight: 500 !important;
        padding: 8px 12px !important;
    }
    
    .stSelectbox [role="option"]:hover {
        background-color: #FFE4B5 !important;
        color: #8B4513 !important;
    }
    
    .stSelectbox [aria-selected="true"] {
        background-color: #FF8C00 !important;
        color: white !important;
    }
    
    /* Text area and input styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: white !important;
        border: 2px solid #FF8C00 !important;
        border-radius: 8px !important;
        color: #333333 !important;
    }
    
    .user-message {
        background-color: #FFE4B5;
        padding: 10px;
        border-radius: 10px;
        border-left: 4px solid #FF8C00;
        margin: 10px 0;
    }
    
    .bot-message {
        background-color: #FFF8DC;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #DAA520;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .context-card {
        background-color: #FFFACD;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #DDD;
        margin: 10px 0;
    }
    
    .source-card {
        background-color: #F5F5DC;
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid #CD853F;
        margin: 5px 0;
        font-size: 0.9em;
    }
    
    .welcome-header {
        text-align: center;
        background: linear-gradient(45deg, #FF8C00, #DAA520);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin-bottom: 20px;
    }
    
    .context-header {
        text-align: center;
        background: linear-gradient(45deg, #FF8C00, #DAA520);
        padding: 15px;
        border-radius: 12px;
        color: white;
        margin-bottom: 15px;
    }
    
    .stButton button {
        background-color: #FF8C00;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stButton button:hover {
        background-color: #DAA520;
        color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .sidebar .stSelectbox {
        background-color: #FFF8DC;
    }
    
    /* Form button styling */
    .stFormSubmitButton button {
        background-color: #FF8C00 !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-size: 16px !important;
        font-weight: bold !important;
    }
    
    .stFormSubmitButton button:hover {
        background-color: #DAA520 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

class ChatbotUI:
    def __init__(self):
        self.config_path = "config.yaml"
        self.knowledge_base_path = "Knowledge_Base"
        
        # Use the same path logic as RAGSystem for consistency
        self.db_path = self._get_writable_db_path()
        
        # Initialize session state
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = None
        if 'user_context' not in st.session_state:
            st.session_state.user_context = UserContext()
        if 'context_gathered' not in st.session_state:
            st.session_state.context_gathered = False
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False
        if 'db_force_recreate' not in st.session_state:
            st.session_state.db_force_recreate = False
    
    def _get_writable_db_path(self) -> str:
        """Get the same writable directory path as RAGSystem"""
        # Try different writable locations in order of preference
        possible_paths = [
            # Railway typically provides /tmp as writable
            "/tmp/chroma_db",
            # System temp directory
            os.path.join(tempfile.gettempdir(), "chroma_db"),
            # Current directory as fallback
            "./chroma_db"
        ]
        
        for path in possible_paths:
            try:
                # Test if we can create directory and write to it
                os.makedirs(path, exist_ok=True)
                test_file = os.path.join(path, "write_test.tmp")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                return path
            except Exception:
                continue
        
        # If all fail, use current directory and hope for the best
        return "./chroma_db"
    
    def initialize_system(self):
        """Initialize the RAG system"""
        try:
            with st.spinner("🙏 Initializing JAI GURU DEV AI... Please wait while I connect to the divine wisdom..."):
                st.session_state.rag_system = RAGSystem(
                    config_path=self.config_path,
                    knowledge_base_path=self.knowledge_base_path
                )
                st.session_state.system_initialized = True
                st.success("✅ System initialized! Ready to share wisdom from Gurudev's teachings.")
        except Exception as e:
            st.error(f"❌ Error initializing system: {str(e)}")
            st.error("Please check your API keys and configuration.")
            return False
        return True
    
    def render_sidebar(self):
        """Render sidebar with configuration options"""
        with st.sidebar:
            st.markdown("## ⚙️ Configuration")
            
            # Model provider selection
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            current_provider = config['model_provider']['default']
            provider = st.selectbox(
                "🤖 Select AI Model Provider:",
                options=["openai", "groq"],
                index=0 if current_provider == "openai" else 1,
                help="Choose between OpenAI GPT models or Groq Llama models"
            )
            
            # Update config if changed
            if provider != current_provider:
                config['model_provider']['default'] = provider
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f)
                st.session_state.system_initialized = False  # Reinitialize system
                st.rerun()
            
            st.markdown("---")
            st.markdown("## 🔌 API Connection Tests")
            
            # Test OpenAI Connection
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🧪 Test OpenAI", use_container_width=True):
                    with st.spinner("Testing OpenAI connection..."):
                        from rag_system import test_openai_connection
                        result = test_openai_connection()
                        if result["success"]:
                            st.success(f"✅ {result['message']}")
                        else:
                            st.error(f"❌ {result['error']}")
            
            with col2:
                if st.button("🧪 Test Groq", use_container_width=True):
                    with st.spinner("Testing Groq connection..."):
                        from rag_system import test_groq_connection
                        result = test_groq_connection()
                        if result["success"]:
                            st.success(f"✅ {result['message']}")
                        else:
                            st.error(f"❌ {result['error']}")
            
            # Test Embeddings
            if st.button("🔍 Test Embeddings", use_container_width=True):
                with st.spinner("Testing embeddings connection..."):
                    from rag_system import test_embeddings_connection
                    result = test_embeddings_connection()
                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                    else:
                        st.error(f"❌ {result['error']}")
            
            # Quick test all
            if st.button("⚡ Test All APIs", use_container_width=True):
                st.write("**Testing all API connections...**")
                
                # Test OpenAI
                from rag_system import test_openai_connection, test_groq_connection, test_embeddings_connection
                
                openai_result = test_openai_connection()
                if openai_result["success"]:
                    st.success(f"OpenAI: ✅ Connected")
                else:
                    st.error(f"OpenAI: ❌ {openai_result['error']}")
                
                # Test Groq
                groq_result = test_groq_connection()
                if groq_result["success"]:
                    st.success(f"Groq: ✅ Connected")
                else:
                    st.error(f"Groq: ❌ {groq_result['error']}")
                
                # Test Embeddings
                embed_result = test_embeddings_connection()
                if embed_result["success"]:
                    st.success(f"Embeddings: ✅ Connected")
                else:
                    st.error(f"Embeddings: ❌ {embed_result['error']}")
                
                # Summary
                total_success = sum([openai_result["success"], groq_result["success"], embed_result["success"]])
                if total_success == 3:
                    st.balloons()
                    st.success("🎉 All APIs are working perfectly!")
                elif total_success >= 1:
                    st.warning(f"⚠️ {total_success}/3 APIs are working. Check your API keys.")
                else:
                    st.error("❌ No APIs are working. Please check your .env file and API keys.")
            
            st.markdown("---")
            st.markdown("## 📚 Knowledge Base Stats")
            if st.session_state.rag_system:
                num_teachings = len(st.session_state.rag_system.teachings)
                st.metric("Total Teachings", num_teachings)
            
            st.markdown("---")
            st.markdown("## 🙏 About")
            st.markdown("""
            **JAI GURU DEV AI** brings you wisdom from Sri Sri Ravi Shankar's teachings.
            
            This chatbot uses advanced AI to find the most relevant spiritual guidance for your questions and life situations.
            """)
            
            if st.button("🔄 Reset Conversation"):
                st.session_state.context_gathered = False
                st.session_state.chat_history = []
                st.session_state.user_context = UserContext()
                st.rerun()
    
    def gather_user_context(self):
        """Gather initial context from user"""
        # Only show context-specific header, not the main header
        st.markdown("""
        <div class="context-header">
            <h2>🙏 Welcome to JAI GURU DEV AI</h2>
            <p>Let me understand your situation better so I can share the most relevant wisdom from Gurudev's teachings.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("context_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                life_aspect = st.selectbox(
                    "🌟 What aspect of life are you seeking guidance about?",
                    options=["", "Relationships", "Spiritual Practice", "Career & Work", "Emotional Well-being", 
                            "Health & Healing", "Family Issues", "Personal Growth", "Life Purpose", "Other"],
                    help="This helps me find the most relevant teachings"
                )
                
                guidance_type = st.selectbox(
                    "🎯 What type of guidance do you seek?",
                    options=["", "General Wisdom", "Specific Situation Help", "Daily Practice Guidance", 
                            "Philosophical Understanding", "Practical Solutions"],
                    help="Choose the kind of guidance that would be most helpful"
                )
            
            with col2:
                emotional_state = st.selectbox(
                    "😌 How are you feeling right now?",
                    options=["", "Peaceful", "Confused", "Anxious", "Sad", "Angry", "Joyful", 
                            "Stressed", "Lonely", "Grateful", "Seeking", "Other"],
                    help="Understanding your emotional state helps me provide appropriate guidance"
                )
                
                specific_situation = st.text_area(
                    "💭 Briefly describe your current situation (optional):",
                    placeholder="Share any specific details about what you're going through...",
                    height=100,
                    help="The more context you provide, the better I can help you"
                )
            
            submitted = st.form_submit_button("🙏 Begin Spiritual Guidance", use_container_width=True)
            
            if submitted and (life_aspect or emotional_state or guidance_type):
                st.session_state.user_context = UserContext(
                    life_aspect=life_aspect,
                    emotional_state=emotional_state,
                    guidance_type=guidance_type,
                    specific_situation=specific_situation
                )
                st.session_state.context_gathered = True
                st.rerun()
    
    def display_user_context(self):
        """Display gathered user context"""
        if not st.session_state.context_gathered:
            return
            
        context = st.session_state.user_context
        context_parts = []
        
        if context.life_aspect:
            context_parts.append(f"**Life Aspect:** {context.life_aspect}")
        if context.emotional_state:
            context_parts.append(f"**Emotional State:** {context.emotional_state}")
        if context.guidance_type:
            context_parts.append(f"**Guidance Type:** {context.guidance_type}")
        if context.specific_situation:
            context_parts.append(f"**Situation:** {context.specific_situation}")
        
        if context_parts:
            st.markdown(f"""
            <div class="context-card">
                <h4>🎯 Your Context:</h4>
                {' • '.join(context_parts)}
            </div>
            """, unsafe_allow_html=True)
    
    def display_chat_history(self):
        """Display chat history"""
        for i, (question, response, sources) in enumerate(st.session_state.chat_history):
            # User message
            st.markdown(f"""
            <div class="user-message">
                <strong>🙋 You:</strong> {question}
            </div>
            """, unsafe_allow_html=True)
            
            # Bot response
            st.markdown(f"""
            <div class="bot-message">
                <strong>🙏 JAI GURU DEV AI:</strong><br>
                {response}
            </div>
            """, unsafe_allow_html=True)
            
            # Sources
            if sources:
                with st.expander(f"📚 Source Teachings ({len(sources)} teachings referenced)"):
                    for source in sources:
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>Teaching #{source['teaching_number']}: {source['title']}</strong><br>
                            <em>Date: {source['date']} | Topics: {source['topics']}</em>
                        </div>
                        """, unsafe_allow_html=True)
    
    def handle_user_query(self):
        """Handle user query input"""
        user_query = st.text_input(
            "💬 Ask your question:",
            placeholder="How can I find peace in difficult times?",
            key="user_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🙏 Ask Gurudev", use_container_width=True):
                if user_query:
                    self.process_query(user_query)
                else:
                    st.warning("Please enter your question first.")
        
        with col2:
            if st.button("🎲 Get Random Wisdom", use_container_width=True):
                random_queries = [
                    "What is the essence of happiness?",
                    "How do I find inner peace?",
                    "What is the purpose of life?",
                    "How do I deal with stress?",
                    "What is true love?",
                    "How do I overcome fear?",
                    "What is the secret to success?",
                    "How do I find my life purpose?"
                ]
                import random
                random_query = random.choice(random_queries)
                self.process_query(random_query)
    
    def process_query(self, query: str):
        """Process user query and get response"""
        if not st.session_state.rag_system:
            st.error("System not initialized. Please wait for initialization to complete.")
            return
        
        with st.spinner("🙏 Consulting Gurudev's teachings..."):
            try:
                response_data = st.session_state.rag_system.get_response(
                    query, 
                    st.session_state.user_context
                )
                
                if response_data['success']:
                    # Add to chat history
                    st.session_state.chat_history.append((
                        query, 
                        response_data['answer'], 
                        response_data['sources']
                    ))
                    st.rerun()
                else:
                    st.error(f"Error getting response: {response_data.get('error', 'Unknown error')}")
            
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
    
    def run(self):
        """Main application runner"""
        # Render sidebar
        self.render_sidebar()
        
        # Show main header only when not in context gathering mode
        if st.session_state.context_gathered or st.session_state.system_initialized:
            st.markdown("""
            <div class="welcome-header">
                <h1>🙏 JAI GURU DEV AI</h1>
                <p>Divine Wisdom from Sri Sri Ravi Shankar's Teachings</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Initialize system if not done
        if not st.session_state.system_initialized:
            if not self.initialize_system():
                return
        
        # Gather user context if not done
        if not st.session_state.context_gathered:
            self.gather_user_context()
            return
        
        # Display user context
        self.display_user_context()
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("## 💬 Our Conversation")
            self.display_chat_history()
        
        # Query input section
        st.markdown("---")
        st.markdown("## 🙏 Ask for Guidance")
        self.handle_user_query()
        
        # Footer
        st.markdown("""
        ---
        <div style="text-align: center; color: #8B4513; font-style: italic;">
            🙏 "In the depth of silence is the source of love" - Sri Sri Ravi Shankar 🙏
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main function"""
    chatbot = ChatbotUI()
    chatbot.run()

if __name__ == "__main__":
    main()
