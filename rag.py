import os
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

# FastAPI imports
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain imports - Updated for latest versions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import (
    PyPDFLoader, 
    DirectoryLoader,
    TextLoader,
    UnstructuredEmailLoader
)

# Additional imports
import tempfile
import shutil
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# SET YOUR DOCUMENT PATH HERE
DOCUMENT_PATH = "policy.pdf"  # ← CHANGE THIS TO YOUR ACTUAL PATH

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: List[str] = []

class DocumentUploadResponse(BaseModel):
    message: str
    processed_files: List[str]

class RAGChatbot:
    def __init__(self):
        self.setup_environment()
        self.embeddings = None
        self.vector_store = None
        self.qa_chain = None
        self.sessions = {}  # Store conversation memories
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
    def setup_environment(self):
        """Setup environment variables and API keys"""
        # Check for required environment variables
        required_vars = ['GOOGLE_API_KEY']
        missing_vars = []
        
        for var in required_vars:
            api_key = os.getenv(var)
            if not api_key:
                missing_vars.append(var)
                logger.error(f"Missing environment variable: {var}")
            else:
                # Show API key status (first 10 chars for debugging)
                key_preview = api_key[:10] + "..." if len(api_key) > 10 else api_key
                logger.info(f"API Key found: {var} = {key_preview} (length: {len(api_key)})")
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            logger.info("Please set the following environment variables:")
            logger.info("export GOOGLE_API_KEY='your_google_api_key_here'")
            logger.info("Or create a .env file with: GOOGLE_API_KEY=your_key_here")
            raise ValueError(f"Environment variables {missing_vars} are required")
        
        # Use FAISS as vector store (simpler and doesn't require external service)
        self.use_pinecone = False
        logger.info("Using FAISS as vector store")
            
    def load_documents_from_path(self, file_path: str) -> List[Document]:
        """Load documents from a given file or directory path"""
        path = Path(file_path)
        documents = []
        
        if not path.exists():
            raise FileNotFoundError(f"Path {file_path} does not exist")
            
        if path.is_file():
            documents.extend(self._load_single_file(path))
        elif path.is_dir():
            documents.extend(self._load_directory(path))
        else:
            raise ValueError(f"Invalid path: {file_path}")
            
        logger.info(f"Loaded {len(documents)} documents from {file_path}")
        return documents
    
    def _load_single_file(self, file_path: Path) -> List[Document]:
        """Load a single file based on its extension"""
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.pdf':
                logger.info(f"Loading PDF file: {file_path}")
                loader = PyPDFLoader(str(file_path))
                documents = loader.load()
            elif extension in ['.txt', '.md']:
                loader = TextLoader(str(file_path), encoding='utf-8')
            elif extension == '.eml':
                loader = UnstructuredEmailLoader(str(file_path))
            else:
                # Try to load as text for other formats
                loader = TextLoader(str(file_path), encoding='utf-8')
                
            return loader.load()
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise
    
    def _load_directory(self, dir_path: Path) -> List[Document]:
        """Load all supported files from a directory"""
        documents = []
        
        # Define file patterns for different loaders
        patterns = {
            "*.pdf": PyPDFLoader,
            "*.txt": TextLoader,
            "*.md": TextLoader,
            "*.eml": UnstructuredEmailLoader,
        }
        
        for pattern, loader_class in patterns.items():
            try:
                if loader_class == TextLoader:
                    # Add encoding for text loaders
                    loader = DirectoryLoader(
                        str(dir_path), 
                        glob=pattern, 
                        loader_cls=loader_class,
                        loader_kwargs={'encoding': 'utf-8'},
                        show_progress=True
                    )
                else:
                    loader = DirectoryLoader(
                        str(dir_path), 
                        glob=pattern, 
                        loader_cls=loader_class,
                        show_progress=True
                    )
                documents.extend(loader.load())
            except Exception as e:
                logger.error(f"Error loading files with pattern {pattern}: {str(e)}")
                
        return documents
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        if not documents:
            return []
            
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(chunks)} chunks")
        return chunks
    
    def setup_embeddings(self):
        """Initialize Google Generative AI embeddings"""
        try:
            api_key = os.getenv('GOOGLE_API_KEY')
            logger.info(f"Setting up embeddings with API key: {api_key[:10] if api_key else 'None'}...")
            
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key
            )
            logger.info("Embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            logger.error("This might be due to an invalid or missing API key")
            raise
    
    def create_vector_store(self, documents: List[Document]):
        """Create vector store from documents"""
        if not documents:
            raise ValueError("No documents provided for vector store creation")
            
        if not self.embeddings:
            self.setup_embeddings()
            
        try:
            # Use FAISS for simplicity
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            # Save FAISS index locally
            self.vector_store.save_local("faiss_index")
            logger.info("FAISS vector store created and saved successfully")
                
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def load_existing_vector_store(self):
        """Load existing vector store if available"""
        try:
            if not self.embeddings:
                self.setup_embeddings()
            if Path("faiss_index").exists():
                self.vector_store = FAISS.load_local(
                    "faiss_index", 
                    self.embeddings,
                    allow_dangerous_deserialization=True  # Required for newer versions
                )
                logger.info("Loaded existing FAISS vector store")
                return True
            else:
                logger.info("No existing FAISS index found")
                return False
        except Exception as e:
            logger.error(f"Error loading FAISS vector store: {str(e)}")
            return False
    
    def setup_qa_chain(self):
        """Setup the conversational retrieval chain"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
            
        try:
            # Initialize the LLM
            api_key = os.getenv('GOOGLE_API_KEY')
            logger.info(f"Setting up LLM with API key: {api_key[:10] if api_key else 'None'}...")
            
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",  # Updated model name
                google_api_key=os.getenv('GOOGLE_API_KEY'),
                temperature=0.7
            )
            
            # Create retriever
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # Create the conversational chain
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
                verbose=True
            )
            
            logger.info("QA chain setup successfully")
            
        except Exception as e:
            logger.error(f"Error setting up QA chain: {str(e)}")
            raise
    
    def get_or_create_memory(self, session_id: str) -> ConversationBufferMemory:
        """Get or create conversation memory for a session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        return self.sessions[session_id]
    
    async def chat(self, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a chat message and return response"""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized")
            
        if not session_id:
            session_id = str(uuid.uuid4())
            
        try:
            memory = self.get_or_create_memory(session_id)
            
            # Get response from the chain
            result = await asyncio.to_thread(
                self.qa_chain,
                {
                    "question": message,
                    "chat_history": memory.chat_memory.messages
                }
            )
            
            # Update memory
            memory.chat_memory.add_user_message(message)
            memory.chat_memory.add_ai_message(result["answer"])
            
            # Extract source information
            sources = []
            if "source_documents" in result:
                sources = [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
            
            return {
                "response": result["answer"],
                "session_id": session_id,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            raise
    
    def test_api_key(self):
        """Test if the API key is valid by making a simple request"""
        try:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                logger.error("No API key found")
                return False
                
            logger.info("Testing API key with Google Generative AI...")
            
            # Test embeddings
            test_embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key
            )
            test_result = test_embeddings.embed_query("test")
            logger.info(f"Embeddings test successful: {len(test_result)} dimensions")
            
            # Test LLM
            test_llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=api_key,
                temperature=0.1
            )
            test_response = test_llm.invoke("Say 'API key is working'")
            logger.info(f"LLM test successful: {test_response.content}")
            
            return True
            
        except Exception as e:
            logger.error(f"API key test failed: {str(e)}")
            return False
    
    def initialize_from_path(self, file_path: str):
        """Initialize the chatbot with documents from a path"""
        try:
            logger.info(f"Initializing chatbot with documents from: {file_path}")
            
            # Try to load existing vector store first
            if not self.load_existing_vector_store():
                logger.info("No existing vector store found, creating new one...")
                # Load and process documents
                documents = self.load_documents_from_path(file_path)
                if not documents:
                    raise ValueError("No documents found in the specified path")
                
                processed_docs = self.process_documents(documents)
                self.create_vector_store(processed_docs)
            
            # Setup QA chain
            self.setup_qa_chain()
            logger.info("RAG Chatbot initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing chatbot: {str(e)}")
            raise

# Initialize FastAPI app
app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global chatbot instance
chatbot = RAGChatbot()

@app.on_event("startup")
async def startup_event():
    """Initialize chatbot on startup"""
    logger.info(f"Attempting to initialize with document: {DOCUMENT_PATH}")
    logger.info(f"File exists: {os.path.exists(DOCUMENT_PATH)}")
    
    # Test API key first
    logger.info("Testing API key...")
    api_key_test = chatbot.test_api_key()
    if not api_key_test:
        logger.error("API key test failed! Please check your GOOGLE_API_KEY")
        return
    
    try:
        # Auto-initialize with the document path if it exists
        if DOCUMENT_PATH and os.path.exists(DOCUMENT_PATH):
            chatbot.initialize_from_path(DOCUMENT_PATH)
            logger.info(f"Chatbot auto-initialized with documents from {DOCUMENT_PATH}")
        else:
            logger.warning(f"Document path {DOCUMENT_PATH} not found. Use /initialize endpoint to set document path")
    except Exception as e:
        logger.error(f"Auto-initialization failed: {str(e)}")
        logger.info("Use /initialize endpoint to manually set document path")

@app.post("/initialize", response_model=Dict[str, str])
async def initialize_chatbot(file_path: str = Form(...)):
    """Initialize the chatbot with documents from a specified path"""
    try:
        chatbot.initialize_from_path(file_path)
        return {"message": f"Chatbot initialized successfully with documents from {file_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat with the RAG bot"""
    try:
        if not chatbot.qa_chain:
            raise HTTPException(status_code=400, detail="Chatbot not initialized. Please call /initialize first.")
        
        result = await chatbot.chat(request.message, request.session_id)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process new documents"""
    try:
        processed_files = []
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Save uploaded files
            for file in files:
                temp_path = Path(temp_dir) / file.filename
                with open(temp_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                processed_files.append(file.filename)
            
            # Load and process documents
            documents = chatbot.load_documents_from_path(temp_dir)
            if documents:
                processed_docs = chatbot.process_documents(documents)
                
                # Add to existing vector store or create new one
                if chatbot.vector_store:
                    new_vector_store = FAISS.from_documents(processed_docs, chatbot.embeddings)
                    chatbot.vector_store.merge_from(new_vector_store)
                    chatbot.vector_store.save_local("faiss_index")
                else:
                    chatbot.create_vector_store(processed_docs)
                    chatbot.setup_qa_chain()
        
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return DocumentUploadResponse(
            message=f"Successfully processed {len(processed_files)} files",
            processed_files=processed_files
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/test-api-key")
async def test_api_key():
    """Test if the API key is working"""
    try:
        success = chatbot.test_api_key()
        return {
            "api_key_test": "success" if success else "failed",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "api_key_test": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "initialized": chatbot.qa_chain is not None,
        "vector_store_type": "FAISS",
        "document_path": DOCUMENT_PATH,
        "document_exists": os.path.exists(DOCUMENT_PATH) if DOCUMENT_PATH else False
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    if session_id in chatbot.sessions:
        del chatbot.sessions[session_id]
        return {"message": f"Session {session_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='RAG Chatbot API')
    parser.add_argument('--docs-path', type=str, help='Path to documents directory')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run on')
    parser.add_argument('--port', type=int, default=8000, help='Port to run on')
    args = parser.parse_args()
    
    # Override document path if provided via command line
    if args.docs_path:
        DOCUMENT_PATH = args.docs_path
    
    print(f"Starting RAG Chatbot API...")
    print(f"Document path: {DOCUMENT_PATH}")
    print(f"Using FAISS as vector store")
    print(f"Make sure to set your GOOGLE_API_KEY environment variable")
    
    uvicorn.run(app, host=args.host, port=args.port)