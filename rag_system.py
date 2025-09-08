    def __init__(self, config_path: str, knowledge_base_path: str):
        self.config_path = config_path
        self.knowledge_base_path = knowledge_base_path
        self.config = self.load_config()
        self.teachings = []
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.llm = None
        
        # Use a writable directory for database - prioritize temp directories for Railway compatibility
        import tempfile
        base_dir = os.environ.get('TMPDIR') or os.environ.get('TMP') or tempfile.gettempdir() or '/tmp'
        self.db_path = os.path.join(base_dir, "chroma_db")
        
        # Fallback to current directory if temp dir is not writable
        try:
            os.makedirs(self.db_path, exist_ok=True)
            # Test write permissions
            test_file = os.path.join(self.db_path, "write_test.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"📂 Using database directory: {self.db_path}")
        except Exception as e:
            print(f"⚠️ Cannot use temp directory {self.db_path}: {e}")
            # Fallback to current directory
            self.db_path = "./chroma_db"
            print(f"📂 Falling back to: {self.db_path}")
        
        self.metadata_file = os.path.join(self.db_path, "db_metadata.json")
        
        # Initialize components step by step with error handling
        try:
            print("🔧 Setting up LLM...")
            self.setup_llm()
            print("🔧 Setting up embeddings...")
            self.setup_embeddings()
            print("🔧 Loading and processing documents...")
            self.load_and_process_documents()
            print("🔧 Setting up retrieval chain...")
            self.setup_retrieval_chain()
            print("✅ RAG System initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing RAG system: {e}")
            raise