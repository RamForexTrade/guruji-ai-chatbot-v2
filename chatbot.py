import streamlit as st
import os
from rag_system import RAGSystem, UserContext

class ChatbotApp:
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
        import tempfile
        
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

if __name__ == "__main__":
    app = ChatbotApp()
