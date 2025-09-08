import os
import yaml
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import BaseRetriever
import numpy as np
from document_processor import DocumentProcessor, Teaching
import hashlib
import json
import tempfile
import shutil

@dataclass
class UserContext:
    """Store user context from preliminary questions"""
    life_aspect: str = ""
    emotional_state: str = ""
    guidance_type: str = ""
    specific_situation: str = ""

class CustomRetriever(BaseRetriever):
    """Custom retriever that combines vector search with metadata filtering"""
    
    def __init__(self, vectorstore, teachings: List[Teaching], top_k: int = 5, **kwargs):
        # Initialize parent class properly
        super().__init__(**kwargs)
        # Set instance attributes
        self.vectorstore = vectorstore
        self.teachings = teachings
        self.top_k = top_k
        self.processor = DocumentProcessor("")
    
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        """Get relevant documents using vector search and metadata filtering"""
        try:
            # Get vector search results
            vector_results = self.vectorstore.similarity_search(query, k=self.top_k * 2)
            
            # Extract teaching numbers from results
            teaching_numbers = []
            for doc in vector_results:
                if 'number' in doc.metadata:
                    teaching_numbers.append(doc.metadata['number'])
            
            # Find corresponding teachings
            relevant_teachings = [t for t in self.teachings if t.number in teaching_numbers]
            
            # Convert back to documents with enhanced metadata
            enhanced_docs = []
            for teaching in relevant_teachings[:self.top_k]:
                doc = Document(
                    page_content=teaching.content,
                    metadata={
                        'number': teaching.number,
                        'title': teaching.title,
                        'date': teaching.date,
                        'location': teaching.location,
                        'topics': ', '.join(teaching.topics),
                        'keywords': ', '.join(teaching.keywords),
                        'problem_categories': ', '.join(teaching.problem_categories),
                        'emotional_states': ', '.join(teaching.emotional_states),
                        'life_situations': ', '.join(teaching.life_situations)
                    }
                )
                enhanced_docs.append(doc)
            
            return enhanced_docs
            
        except Exception as e:
            print(f"Error in CustomRetriever._get_relevant_documents: {e}")
            # Fallback to basic vector search
            return self.vectorstore.similarity_search(query, k=self.top_k)

class RAGSystem:
    """RAG System for JAI GURU DEV AI Chatbot"""
    
    def __init__(self, config_path: str, knowledge_base_path: str):
        self.config_path = config_path
        self.knowledge_base_path = knowledge_base_path
        self.config = self.load_config()
        self.teachings = []
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.llm = None
        
        # Use a writable directory for database - Railway compatibility
        self.db_path = self._get_writable_db_path()
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
    
    def _detect_readonly_filesystem(self, error_msg: str) -> bool:
        """Detect if the error is related to read-only filesystem"""
        readonly_indicators = [
            "readonly", "read-only", "read only",
            "permission denied", "permissions",
            "1032", "attempt to write a readonly database",
            "database is locked", "disk i/o error",
            "unable to open database"
        ]
        
        error_lower = error_msg.lower()
        return any(indicator in error_lower for indicator in readonly_indicators)
    
    def _get_writable_db_path(self) -> str:
        """Get a writable directory path for ChromaDB - Railway compatible"""
        # Check for environment variable override first
        env_db_path = os.getenv('CHROMA_DB_PATH')
        if env_db_path:
            try:
                os.makedirs(env_db_path, exist_ok=True)
                test_file = os.path.join(env_db_path, "write_test.tmp")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                print(f"📂 Using environment-specified database directory: {env_db_path}")
                return env_db_path
            except Exception as e:
                print(f"⚠️ Cannot use environment-specified path {env_db_path}: {e}")
        
        # Try different writable locations in order of preference
        possible_paths = [
            # Railway typically provides /tmp as writable
            "/tmp/chroma_db",
            # Alternative tmp paths for different environments
            "/var/tmp/chroma_db",
            # System temp directory
            os.path.join(tempfile.gettempdir(), "chroma_db"),
            # User home directory if available
            os.path.expanduser("~/chroma_db") if os.path.expanduser("~") != "~" else None,
            # App directory fallback
            os.path.join(os.getcwd(), "tmp", "chroma_db"),
            # Current directory as last resort
            "./chroma_db"
        ]
        
        # Filter out None values
        possible_paths = [p for p in possible_paths if p is not None]
        
        for path in possible_paths:
            try:
                # Test if we can create directory and write to it
                os.makedirs(path, exist_ok=True)
                test_file = os.path.join(path, "write_test.tmp")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                print(f"📂 Using writable database directory: {path}")
                return path
            except Exception as e:
                print(f"⚠️ Cannot use {path}: {e}")
                continue
        
        # If all fail, use current directory and hope for the best
        fallback_path = "./chroma_db"
        print(f"📂 Using fallback directory: {fallback_path}")
        print("⚠️ Warning: This may not be writable in Railway deployment")
        return fallback_path
    
    def _get_chroma_settings(self) -> Settings:
        """Get ChromaDB settings optimized for production deployment"""
        # Check if we're in a production environment
        is_production = (
            os.getenv('ENVIRONMENT') == 'production' or 
            os.getenv('RAILWAY_ENVIRONMENT') is not None or
            os.getenv('PORT') is not None  # Common in Railway deployments
        )
        
        if is_production:
            print("🚀 Configuring ChromaDB for production environment")
            
            # Use settings that work better in constrained environments
            settings = Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.db_path,
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        else:
            print("🔧 Configuring ChromaDB for development environment")
            settings = Settings(
                persist_directory=self.db_path,
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        
        return settings
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def setup_llm(self):
        """Setup Language Model based on configuration with minimal fallback for deprecated models"""
        provider = self.config['model_provider']['default']
        
        if provider == "openai":
            model_config = self.config['model_provider']['openai']
            self.llm = ChatOpenAI(
                model=model_config['model'],
                temperature=model_config['temperature'],
                max_tokens=model_config['max_tokens']
            )
        elif provider == "groq":
            model_config = self.config['model_provider']['groq']
            
            # Try primary model first
            try:
                self.llm = ChatGroq(
                    model=model_config['model'],
                    temperature=model_config['temperature'],
                    max_tokens=model_config['max_tokens']
                )
                print(f"✅ Connected to {model_config['model']}")
                
            except Exception as e:
                # Only try fallback if explicitly specified in config
                fallback_model = model_config.get('fallback_model')
                if fallback_model:
                    print(f"⚠️ Primary model failed, using fallback: {fallback_model}")
                    self.llm = ChatGroq(
                        model=fallback_model,
                        temperature=model_config['temperature'],
                        max_tokens=model_config['max_tokens']
                    )
                    print(f"✅ Connected to fallback: {fallback_model}")
                else:
                    # Preserve original behavior - raise the error
                    raise e
        else:
            raise ValueError(f"Unsupported model provider: {provider}")
    
    def setup_embeddings(self):
        """Setup embeddings model"""
        self.embeddings = OpenAIEmbeddings(
            model=self.config['embeddings']['model']
        )
    
    def _get_knowledge_base_hash(self) -> str:
        """Generate hash of knowledge base files to detect changes"""
        hasher = hashlib.md5()
        
        # Get all .md files in knowledge base
        md_files = []
        if os.path.exists(self.knowledge_base_path):
            for root, dirs, files in os.walk(self.knowledge_base_path):
                for file in files:
                    if file.endswith('.md'):
                        md_files.append(os.path.join(root, file))
        
        # Sort files for consistent hash
        md_files.sort()
        
        # Hash file contents
        for file_path in md_files:
            try:
                with open(file_path, 'rb') as f:
                    hasher.update(f.read())
                # Also include file modification time
                hasher.update(str(os.path.getmtime(file_path)).encode())
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
        
        return hasher.hexdigest()
    
    def _save_db_metadata(self, num_documents: int, knowledge_hash: str):
        """Save database metadata"""
        metadata = {
            'num_documents': num_documents,
            'knowledge_hash': knowledge_hash,
            'created_at': os.path.getctime(self.db_path) if os.path.exists(self.db_path) else None,
            'embeddings_model': self.config['embeddings']['model']
        }
        
        try:
            os.makedirs(self.db_path, exist_ok=True)
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save metadata: {e}")
    
    def _load_db_metadata(self) -> Dict[str, Any]:
        """Load database metadata"""
        if not os.path.exists(self.metadata_file):
            return {}
        
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")
            return {}
    
    def _should_recreate_db(self) -> bool:
        """Check if database should be recreated"""
        # Check if database directory exists
        if not os.path.exists(self.db_path):
            print("📂 ChromaDB directory doesn't exist - will create new database")
            return True
        
        # Check if database files exist
        db_files = ['chroma.sqlite3']
        for file in db_files:
            if not os.path.exists(os.path.join(self.db_path, file)):
                print(f"📂 Missing database file {file} - will recreate database")
                return True
        
        # Load metadata
        metadata = self._load_db_metadata()
        if not metadata:
            print("📂 No metadata found - will recreate database")
            return True
        
        # Check if knowledge base has changed
        current_hash = self._get_knowledge_base_hash()
        if metadata.get('knowledge_hash') != current_hash:
            print("📂 Knowledge base has changed - will recreate database")
            return True
        
        # Check if embeddings model has changed
        current_model = self.config['embeddings']['model']
        if metadata.get('embeddings_model') != current_model:
            print("📂 Embeddings model has changed - will recreate database")
            return True
        
        print("✅ Existing ChromaDB is up to date - will reuse it")
        return False
    
    def _create_vectorstore_with_settings(self, documents: List[Document]) -> Chroma:
        """Create vector store with proper settings and comprehensive error handling"""
        
        # Try with optimized settings first
        try:
            print("🔄 Attempting to create ChromaDB with optimized settings...")
            chroma_settings = self._get_chroma_settings()
            
            # Create ChromaDB client with settings
            client = chromadb.Client(chroma_settings)
            
            # Create vector store
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.db_path,
                client=client
            )
            
            print("✅ Successfully created ChromaDB with optimized settings")
            return vectorstore
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error creating vector store with optimized settings: {error_msg}")
            
            # Check if this is a permissions/readonly error
            if self._detect_readonly_filesystem(error_msg):
                print("🔍 Detected filesystem permissions issue - trying alternatives...")
                
                # Try to clean up any partial creation
                try:
                    if os.path.exists(self.db_path):
                        shutil.rmtree(self.db_path)
                        print("🧹 Cleaned up partial database creation")
                except Exception as cleanup_error:
                    print(f"⚠️ Could not clean up: {cleanup_error}")
                
                # Try alternative writable path
                alternative_paths = [
                    "/tmp/chroma_alt",
                    os.path.join(tempfile.gettempdir(), "chroma_alt"),
                    f"./chroma_alt_{os.getpid()}"
                ]
                
                for alt_path in alternative_paths:
                    try:
                        print(f"🔄 Trying alternative path: {alt_path}")
                        os.makedirs(alt_path, exist_ok=True)
                        
                        # Test write permissions
                        test_file = os.path.join(alt_path, "test_write.tmp")
                        with open(test_file, 'w') as f:
                            f.write("test")
                        os.remove(test_file)
                        
                        # Update db_path and try again
                        self.db_path = alt_path
                        self.metadata_file = os.path.join(self.db_path, "db_metadata.json")
                        
                        vectorstore = Chroma.from_documents(
                            documents=documents,
                            embedding=self.embeddings,
                            persist_directory=self.db_path
                        )
                        
                        print(f"✅ Successfully created ChromaDB at alternative path: {alt_path}")
                        return vectorstore
                        
                    except Exception as alt_error:
                        print(f"❌ Alternative path {alt_path} failed: {alt_error}")
                        continue
                
                # If all persistent options fail, try in-memory
                print("🔄 All persistent storage options failed, trying in-memory database...")
                try:
                    vectorstore = Chroma.from_documents(
                        documents=documents,
                        embedding=self.embeddings
                        # No persist_directory = in-memory only
                    )
                    print("✅ Created in-memory ChromaDB successfully")
                    print("⚠️ WARNING: Database will not persist between restarts")
                    return vectorstore
                    
                except Exception as memory_error:
                    print(f"❌ Even in-memory database failed: {memory_error}")
                    raise Exception(f"All ChromaDB creation methods failed. Last error: {memory_error}")
            
            else:
                # Try basic creation without custom settings
                print("🔄 Trying basic ChromaDB creation without custom settings...")
                try:
                    vectorstore = Chroma.from_documents(
                        documents=documents,
                        embedding=self.embeddings,
                        persist_directory=self.db_path
                    )
                    print("✅ Successfully created basic ChromaDB")
                    return vectorstore
                    
                except Exception as basic_error:
                    print(f"❌ Basic creation also failed: {basic_error}")
                    # Fall back to in-memory as last resort
                    print("🔄 Falling back to in-memory database...")
                    vectorstore = Chroma.from_documents(
                        documents=documents,
                        embedding=self.embeddings
                    )
                    print("✅ Created in-memory ChromaDB as final fallback")
                    print("⚠️ WARNING: Database will not persist between restarts")
                    return vectorstore
    
    def load_and_process_documents(self):
        """Load and process all teachings from markdown files"""
        processor = DocumentProcessor(self.knowledge_base_path)
        self.teachings = processor.load_all_teachings()
        
        if not self.teachings:
            raise ValueError("No teachings found in knowledge base")
        
        # Check if we should recreate the database
        should_recreate = self._should_recreate_db()
        
        if should_recreate:
            print("🔄 Creating new ChromaDB...")
            
            # Convert teachings to LangChain documents
            documents = []
            for teaching in self.teachings:
                doc = Document(
                    page_content=teaching.content,
                    metadata={
                        'number': teaching.number,
                        'title': teaching.title,
                        'date': teaching.date,
                        'location': teaching.location,
                        'topics': ', '.join(teaching.topics),
                        'keywords': ', '.join(teaching.keywords),
                        'problem_categories': ', '.join(teaching.problem_categories),
                        'emotional_states': ', '.join(teaching.emotional_states),
                        'life_situations': ', '.join(teaching.life_situations),
                        'full_text': teaching.get_full_text()
                    }
                )
                documents.append(doc)
            
            # Create new vector store with enhanced error handling
            self.vectorstore = self._create_vectorstore_with_settings(documents)
            print(f"✅ Created ChromaDB with {len(documents)} teachings")
            
            # Save metadata if possible
            try:
                knowledge_hash = self._get_knowledge_base_hash()
                self._save_db_metadata(len(documents), knowledge_hash)
            except Exception as e:
                print(f"⚠️ Could not save metadata: {e}")
                
        else:
            print("♻️ Loading existing ChromaDB...")
            
            # Load existing vector store
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.db_path,
                    embedding_function=self.embeddings
                )
                
                # Verify the database has expected number of documents
                metadata = self._load_db_metadata()
                expected_docs = metadata.get('num_documents', 0)
                
                # Try to get collection info
                try:
                    collection = self.vectorstore._collection
                    actual_count = collection.count()
                    print(f"✅ Loaded existing ChromaDB with {actual_count} documents")
                    
                    if actual_count != len(self.teachings):
                        print(f"⚠️ Warning: Database has {actual_count} documents but found {len(self.teachings)} teachings in knowledge base")
                    
                except Exception as e:
                    print(f"⚠️ Could not verify document count: {e}")
                    print("✅ Loaded existing ChromaDB successfully")
                    
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Error loading existing vector store: {error_msg}")
                
                # Check if it's a permissions issue
                if self._detect_readonly_filesystem(error_msg):
                    print("🔍 Detected permissions issue with existing database")
                    print("🔄 Will recreate database in writable location...")
                else:
                    print("🔄 Will try to recreate database...")
                
                # Fallback to creating new database
                documents = []
                for teaching in self.teachings:
                    doc = Document(
                        page_content=teaching.content,
                        metadata={
                            'number': teaching.number,
                            'title': teaching.title,
                            'date': teaching.date,
                            'location': teaching.location,
                            'topics': ', '.join(teaching.topics),
                            'keywords': ', '.join(teaching.keywords),
                            'problem_categories': ', '.join(teaching.problem_categories),
                            'emotional_states': ', '.join(teaching.emotional_states),
                            'life_situations': ', '.join(teaching.life_situations),
                            'full_text': teaching.get_full_text()
                        }
                    )
                    documents.append(doc)
                
                self.vectorstore = self._create_vectorstore_with_settings(documents)
                print(f"✅ Recreated ChromaDB with {len(documents)} teachings")
                
                # Save metadata if possible
                try:
                    knowledge_hash = self._get_knowledge_base_hash()
                    self._save_db_metadata(len(documents), knowledge_hash)
                except Exception as e:
                    print(f"⚠️ Could not save metadata: {e}")
    
    def setup_retrieval_chain(self):
        """Setup the retrieval QA chain"""
        
        # Custom prompt template
        template = """
        You are "JAI GURU DEV AI", a compassionate spiritual guide based on Sri Sri Ravi Shankar's teachings. 
        Your purpose is to provide wisdom, guidance, and spiritual insights to help users navigate life's challenges.

        Context: Based on the user's questions and situation, here are relevant teachings:

        {context}

        Human Question: {question}

        Guidelines for your response:
        1. Speak with compassion, wisdom, and gentleness
        2. Draw insights from the provided teachings but explain them in a way that's relevant to the user's situation
        3. If multiple teachings are relevant, synthesize the wisdom
        4. Always maintain the spiritual and philosophical tone of the original teachings
        5. End with a practical suggestion or reflection question when appropriate
        6. Use "Jai Guru Dev" as a blessing at the end when appropriate

        Response:
        """
        
        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Setup custom retriever with error handling
        try:
            print("🔧 Creating custom retriever...")
            self.retriever = CustomRetriever(
                vectorstore=self.vectorstore,
                teachings=self.teachings,
                top_k=self.config['rag']['top_k_results']
            )
            print("✅ Custom retriever created successfully")
        except Exception as e:
            print(f"❌ Error creating custom retriever: {e}")
            print("🔧 Falling back to basic vector store retriever...")
            # Fallback to basic retriever
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.config['rag']['top_k_results']}
            )
        
        # Setup QA chain
        try:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            print("✅ QA chain setup complete")
        except Exception as e:
            print(f"❌ Error setting up QA chain: {e}")
            raise
    
    def search_by_context(self, user_context: UserContext) -> List[Teaching]:
        """Search teachings based on user context"""
        processor = DocumentProcessor("")
        
        # Extract search terms from user context
        topics = [user_context.life_aspect] if user_context.life_aspect else []
        emotions = [user_context.emotional_state] if user_context.emotional_state else []
        situations = [user_context.specific_situation] if user_context.specific_situation else []
        
        # Use metadata search
        context_teachings = processor.search_teachings_by_metadata(
            teachings=self.teachings,
            query_topics=topics,
            query_emotions=emotions,
            query_situations=situations
        )
        
        return context_teachings[:5]  # Return top 5 contextually relevant teachings
    
    def get_response(self, question: str, user_context: UserContext = None) -> Dict[str, Any]:
        """Get response from RAG system"""
        try:
            # Enhance query with user context
            enhanced_query = question
            if user_context:
                context_parts = []
                if user_context.life_aspect:
                    context_parts.append(f"Life aspect: {user_context.life_aspect}")
                if user_context.emotional_state:
                    context_parts.append(f"Emotional state: {user_context.emotional_state}")
                if user_context.guidance_type:
                    context_parts.append(f"Seeking: {user_context.guidance_type}")
                if user_context.specific_situation:
                    context_parts.append(f"Situation: {user_context.specific_situation}")
                
                if context_parts:
                    enhanced_query = f"{question}\n\nUser Context: {'; '.join(context_parts)}"
            
            # Get response from QA chain
            result = self.qa_chain({"query": enhanced_query})
            
            # Extract source information
            sources = []
            for doc in result.get('source_documents', []):
                source_info = {
                    'teaching_number': doc.metadata.get('number', 'Unknown'),
                    'title': doc.metadata.get('title', 'Unknown'),
                    'topics': doc.metadata.get('topics', ''),
                    'date': doc.metadata.get('date', 'Not specified')
                }
                sources.append(source_info)
            
            return {
                'answer': result['result'],
                'sources': sources,
                'success': True
            }
            
        except Exception as e:
            return {
                'answer': f"I apologize, but I encountered an error while processing your question. Please try rephrasing your question or contact support. Error: {str(e)}",
                'sources': [],
                'success': False,
                'error': str(e)
            }
    
    def get_initial_questions(self) -> List[str]:
        """Get initial questions for user context gathering"""
        return self.config['chatbot']['initial_questions']

def test_openai_connection():
    """Test OpenAI API connection"""
    try:
        import openai
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return {"success": False, "error": "OpenAI API key not found in environment"}
        
        # Load config to get the model
        with open("config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        model_config = config['model_provider']['openai']
        
        # Test with a simple completion
        client = ChatOpenAI(
            model=model_config['model'], 
            max_tokens=10, 
            temperature=0
        )
        response = client.invoke("Say 'Hello'")
        
        return {"success": True, "message": "OpenAI connection successful!", "response": response.content}
        
    except Exception as e:
        return {"success": False, "error": f"OpenAI connection failed: {str(e)}"}

def test_groq_connection():
    """Test Groq API connection"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            return {"success": False, "error": "Groq API key not found in environment"}
        
        # Load config to get the model
        with open("config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        model_config = config['model_provider']['groq']
        
        # Test with primary model first
        try:
            client = ChatGroq(
                model=model_config['model'], 
                max_tokens=10, 
                temperature=0
            )
            response = client.invoke("Say 'Hello'")
            return {"success": True, "message": f"Groq connection successful with {model_config['model']}!", "response": response.content}
            
        except Exception as primary_error:
            # Try fallback model if available
            fallback_model = model_config.get('fallback_model')
            if fallback_model:
                try:
                    client = ChatGroq(
                        model=fallback_model, 
                        max_tokens=10, 
                        temperature=0
                    )
                    response = client.invoke("Say 'Hello'")
                    return {"success": True, "message": f"Groq connection successful with fallback model {fallback_model}!", "response": response.content}
                    
                except Exception as fallback_error:
                    return {"success": False, "error": f"Both primary ({model_config['model']}) and fallback ({fallback_model}) models failed. Primary error: {primary_error}. Fallback error: {fallback_error}"}
            else:
                return {"success": False, "error": f"Primary model {model_config['model']} failed: {primary_error}"}
        
    except Exception as e:
        return {"success": False, "error": f"Groq connection failed: {str(e)}"}

def test_embeddings_connection():
    """Test OpenAI Embeddings connection"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return {"success": False, "error": "OpenAI API key not found for embeddings"}
        
        # Load config to get the embeddings model
        with open("config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Test embeddings
        embeddings = OpenAIEmbeddings(model=config['embeddings']['model'])
        test_embedding = embeddings.embed_query("Hello world")
        
        return {"success": True, "message": f"Embeddings connection successful! (dimension: {len(test_embedding)})"}
        
    except Exception as e:
        return {"success": False, "error": f"Embeddings connection failed: {str(e)}"}