import requests
import json

def chat_with_rag_bot():
    """Simple command-line chat interface for the RAG bot"""
    
    base_url = "http://localhost:8000"
    session_id = None
    
    # Get document source from server
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            document_source = data.get('document_path', 'Unknown document')
            print("RAG Chatbot is ready! Type 'quit' to exit.")
            print(f"I have knowledge about your document: {document_source}")
        else:
            print("RAG Chatbot is ready! Type 'quit' to exit.")
            print("I have knowledge about your document source")
    except Exception as e:
        print("RAG Chatbot is ready! Type 'quit' to exit.")
        print("I have knowledge about your document source")
    
    print("-" * 50)
    
    while True:
        # Get user input
        user_message = input("\nYou: ").strip()
        
        # Check for exit command
        if user_message.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_message:
            continue
            
        try:
            # Prepare the request
            payload = {"message": user_message}
            if session_id:
                payload["session_id"] = session_id
            
            # Send request to the RAG bot
            response = requests.post(
                f"{base_url}/chat",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Update session ID for conversation continuity
                session_id = data["session_id"]
                
                # Display the response
                print(f"\nBot: {data['response']}")
                
                # Show sources if available
                if data.get('sources'):
                    print(f"\nSources: {', '.join(data['sources'])}")
                    
            else:
                print(f"Error: {response.status_code}")
                print(f"Details: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"Connection error: {e}")
            print("Make sure your RAG server is running on http://localhost:8000")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")

def test_connection():
    """Test if the RAG bot server is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("Server is running!")
            print(f"Status: {data['status']}")
            print(f"Initialized: {data['initialized']}")
            print(f"Vector Store: {data.get('vector_store_type', 'Unknown')}")
            return True
    except Exception as e:
        print("Server is not running or not accessible")
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Checking server connection...")
    
    if test_connection():
        print("\n" + "="*50)
        chat_with_rag_bot()
    else:
        print("\nPlease make sure your RAG server is running:")
        print("   python rag.py")