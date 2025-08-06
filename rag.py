import os
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
import requests
import tempfile
import uuid

# FastAPI imports
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# LangChain imports - Updated for latest versions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredEmailLoader
)

# PDF processing with pdfplumber
import pdfplumber

# Additional imports
import shutil
import pickle
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Settings class for environment variables
class Settings(BaseSettings):
    openai_api_key: str = ""
    pinecone_api_key: str = ""
    api_bearer_token: str = "test_token_123"
    
    class Config:
        env_file = ".env"

settings = Settings()

# Security
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify bearer token"""
    if credentials.credentials != settings.api_bearer_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Pydantic models for the new API format
class HackRXRequest(BaseModel):
    documents: str  # URL to the document
    questions: List[str]  # List of questions to answer

class HackRXResponse(BaseModel):
    answers: List[str]  # List of answers corresponding to the questions

# Original Pydantic models (keeping for backward compatibility)
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
        self.cache_dir = "vector_cache"  # Directory to store cached embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def setup_environment(self):
        """Setup environment variables and API keys"""
        # Check for required environment variables
        required_vars = ['OPENAI_API_KEY']
        missing_vars = []
        
        for var in required_vars:
            api_key = getattr(settings, var.lower(), None) or os.getenv(var)
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
            logger.info("export OPENAI_API_KEY='your_openai_api_key_here'")
            logger.info("Or create a .env file with: OPENAI_API_KEY=your_key_here")
            raise ValueError(f"Environment variables {missing_vars} are required")
        
        # Use FAISS as vector store (simpler and doesn't require external service)
        self.use_pinecone = False
        logger.info("Using FAISS as vector store")

    def download_document_from_url(self, url: str) -> str:
        """Download document from URL and return local path"""
        try:
            logger.info(f"Downloading document from: {url}")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            # Create temporary file
            temp_dir = tempfile.mkdtemp()
            file_extension = ".pdf"  # Assuming PDF for now
            if url.lower().endswith('.txt'):
                file_extension = ".txt"
            elif url.lower().endswith('.md'):
                file_extension = ".md"
            
            temp_file_path = os.path.join(temp_dir, f"document{file_extension}")
            
            with open(temp_file_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Document downloaded successfully to: {temp_file_path}")
            return temp_file_path
            
        except Exception as e:
            logger.error(f"Error downloading document: {str(e)}")
            raise
            
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
                logger.info(f"Loading PDF file with pdfplumber: {file_path}")
                documents = []
                with pdfplumber.open(str(file_path)) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        text = page.extract_text()
                        if text and text.strip():
                            # Create document with metadata
                            doc = Document(
                                page_content=text,
                                metadata={
                                    "source": str(file_path),
                                    "page": page_num + 1,
                                    "total_pages": len(pdf.pages)
                                }
                            )
                            documents.append(doc)
                            logger.info(f"Extracted text from page {page_num + 1}")
                
                logger.info(f"Successfully loaded {len(documents)} pages from PDF")
                return documents
                
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
        
        # Load PDF files with pdfplumber
        pdf_files = list(dir_path.glob("*.pdf"))
        for pdf_file in pdf_files:
            try:
                documents.extend(self._load_single_file(pdf_file))
            except Exception as e:
                logger.error(f"Error loading PDF file {pdf_file}: {str(e)}")
        
        # Load other file types with DirectoryLoader
        patterns = {
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
        """Initialize OpenAI embeddings"""
        try:
            api_key = settings.openai_api_key or os.getenv('OPENAI_API_KEY')
            logger.info(f"Setting up embeddings with API key: {api_key[:10] if api_key else 'None'}...")
            
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=api_key
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
            logger.info("FAISS vector store created successfully")
                
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def generate_cache_key(self, document_url: str) -> str:
        """Generate a unique cache key for a document URL"""
        return hashlib.md5(document_url.encode()).hexdigest()
    
    def get_cache_path(self, cache_key: str) -> str:
        """Get the cache file path for a given cache key"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def save_vector_store_to_cache(self, cache_key: str):
        """Save the current vector store to cache"""
        if not self.vector_store:
            logger.warning("No vector store to save")
            return
            
        try:
            cache_path = self.get_cache_path(cache_key)
            # Save FAISS index
            self.vector_store.save_local(cache_path)
            logger.info(f"Vector store saved to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Error saving vector store to cache: {str(e)}")
    
    def load_vector_store_from_cache(self, cache_key: str) -> bool:
        """Load vector store from cache if it exists"""
        try:
            cache_path = self.get_cache_path(cache_key)
            if not os.path.exists(cache_path):
                logger.info(f"Cache not found for key: {cache_key}")
                return False
                
            if not self.embeddings:
                self.setup_embeddings()
                
            # Load FAISS index
            self.vector_store = FAISS.load_local(cache_path, self.embeddings)
            logger.info(f"Vector store loaded from cache: {cache_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store from cache: {str(e)}")
            return False
    
    def clear_cache(self):
        """Clear all cached vector stores"""
        try:
            if os.path.exists(self.cache_dir):
                for file in os.listdir(self.cache_dir):
                    if file.endswith('.pkl'):
                        os.remove(os.path.join(self.cache_dir, file))
                logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached files"""
        try:
            cache_files = []
            total_size = 0
            if os.path.exists(self.cache_dir):
                for file in os.listdir(self.cache_dir):
                    if file.endswith('.pkl'):
                        file_path = os.path.join(self.cache_dir, file)
                        file_size = os.path.getsize(file_path)
                        cache_files.append({
                            "file": file,
                            "size_bytes": file_size,
                            "size_mb": round(file_size / (1024 * 1024), 2)
                        })
                        total_size += file_size
            
            return {
                "cache_directory": self.cache_dir,
                "total_files": len(cache_files),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "files": cache_files
            }
        except Exception as e:
            logger.error(f"Error getting cache info: {str(e)}")
            return {"error": str(e)}
    
    def setup_qa_chain(self):
        """Setup the conversational retrieval chain"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
            
        try:
            # Initialize the LLM
            api_key = settings.openai_api_key or os.getenv('OPENAI_API_KEY')
            logger.info(f"Setting up LLM with API key: {api_key[:10] if api_key else 'None'}...")
            
            llm = ChatOpenAI(
                model="gpt-4o-mini",  # Using OpenAI's latest model
                openai_api_key=api_key,
                temperature=0.3  # Lower temperature for more consistent answers
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

    async def answer_question(self, question: str) -> str:
        """Answer a single question without session management"""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized")
            
        try:
            # Get response from the chain without memory
            result = await asyncio.to_thread(
                self.qa_chain,
                {
                    "question": question,
                    "chat_history": []  # Empty chat history for standalone questions
                }
            )
            
            return result["answer"]
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            raise
    
    def test_api_key(self):
        """Test if the API key is valid by making a simple request"""
        try:
            api_key = settings.openai_api_key or os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.error("No API key found")
                return False
                
            logger.info("Testing API key with OpenAI...")
            
            # Test embeddings
            test_embeddings = OpenAIEmbeddings(
                openai_api_key=api_key
            )
            test_result = test_embeddings.embed_query("test")
            logger.info(f"Embeddings test successful: {len(test_result)} dimensions")
            
            # Test LLM
            test_llm = ChatOpenAI(
                model="gpt-4o-mini",
                openai_api_key=api_key,
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

    async def initialize_from_url(self, document_url: str):
        """Initialize the chatbot with documents from a URL"""
        temp_file_path = None
        try:
            logger.info(f"Initializing chatbot with document from URL: {document_url}")
            
            # Generate cache key for this document URL
            cache_key = self.generate_cache_key(document_url)
            logger.info(f"Cache key: {cache_key}")
            
            # Try to load from cache first
            if self.load_vector_store_from_cache(cache_key):
                logger.info("Successfully loaded vector store from cache")
                self.setup_qa_chain()
                logger.info("RAG Chatbot initialized successfully from cache")
                return
            
            logger.info("Cache miss - processing document from scratch")
            
            # Download document
            temp_file_path = self.download_document_from_url(document_url)
            
            # Load and process documents
            documents = self.load_documents_from_path(temp_file_path)
            if not documents:
                raise ValueError("No documents found in the downloaded file")
            
            processed_docs = self.process_documents(documents)
            self.create_vector_store(processed_docs)
            
            # Save to cache for future use
            self.save_vector_store_to_cache(cache_key)
            
            # Setup QA chain
            self.setup_qa_chain()
            logger.info("RAG Chatbot initialized successfully from URL and cached")
            
        except Exception as e:
            logger.error(f"Error initializing chatbot from URL: {str(e)}")
            raise
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    # Also remove the temporary directory
                    temp_dir = os.path.dirname(temp_file_path)
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file: {e}")

# Initialize FastAPI app
app = FastAPI(title="HackRX RAG Chatbot API", version="1.0.0")

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
    """Initialize on startup"""
    logger.info("HackRX RAG Chatbot API starting up...")
    
    # Test API key first
    logger.info("Testing API key...")
    api_key_test = chatbot.test_api_key()
    if not api_key_test:
        logger.error("API key test failed! Please check your OPENAI_API_KEY")
    else:
        logger.info("API key test successful!")

@app.post("/api/v1/hackrx/run", response_model=HackRXResponse)
async def hackrx_run(request: HackRXRequest, token: str = Depends(verify_token)):
    """Main HackRX endpoint - process document URL and answer multiple questions"""
    try:
        logger.info(f"Processing HackRX request with {len(request.questions)} questions")
        logger.info(f"Document URL: {request.documents}")
        
        # Initialize chatbot with the document from URL
        await chatbot.initialize_from_url(request.documents)
        
        # Process all questions
        answers = []
        for i, question in enumerate(request.questions):
            logger.info(f"Processing question {i+1}/{len(request.questions)}: {question}")
            try:
                answer = await chatbot.answer_question(question)
                answers.append(answer)
                logger.info(f"Answer {i+1}: {answer[:100]}...")  # Log first 100 chars
            except Exception as e:
                logger.error(f"Error answering question {i+1}: {str(e)}")
                answers.append(f"Error processing question: {str(e)}")
        
        logger.info(f"Completed processing all {len(request.questions)} questions")
        return HackRXResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"HackRX processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# Keep original endpoints for backward compatibility
@app.post("/initialize", response_model=Dict[str, str])
async def initialize_chatbot(file_path: str = Form(...), token: str = Depends(verify_token)):
    """Initialize the chatbot with documents from a specified path"""
    try:
        chatbot.initialize_from_path(file_path)
        return {"message": f"Chatbot initialized successfully with documents from {file_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, token: str = Depends(verify_token)):
    """Chat with the RAG bot"""
    try:
        if not chatbot.qa_chain:
            raise HTTPException(status_code=400, detail="Chatbot not initialized. Please call /initialize first.")
        
        result = await chatbot.chat(request.message, request.session_id)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_documents(files: List[UploadFile] = File(...), token: str = Depends(verify_token)):
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
        "vector_store_type": "FAISS"
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, token: str = Depends(verify_token)):
    """Delete a chat session"""
    if session_id in chatbot.sessions:
        del chatbot.sessions[session_id]
        return {"message": f"Session {session_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/cache/info")
async def get_cache_info(token: str = Depends(verify_token)):
    """Get information about cached vector stores"""
    return chatbot.get_cache_info()

@app.delete("/cache/clear")
async def clear_cache(token: str = Depends(verify_token)):
    """Clear all cached vector stores"""
    chatbot.clear_cache()
    return {"message": "Cache cleared successfully"}

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='HackRX RAG Chatbot API')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run on')
    parser.add_argument('--port', type=int, default=8001, help='Port to run on')
    args = parser.parse_args()
    
    print(f"Starting HackRX RAG Chatbot API...")
    print(f"Using FAISS as vector store")
    print(f"Make sure to set your OPENAI_API_KEY environment variable")
    print(f"Main endpoint: POST /hackrx/run")
    
    uvicorn.run(app, host=args.host, port=args.port)