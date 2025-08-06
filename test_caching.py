import requests
import json
import time
from datetime import datetime

# Test cases from the user
test_cases = [
    {
        "documents": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
        "questions": [
            "When will my root canal claim of Rs 25,000 be settled?",
            "I have done an IVF for Rs 56,000. Is it covered?",
            "I did a cataract treatment of Rs 100,000. Will you settle the full Rs 100,000?",
            "Give me a list of documents to be uploaded for hospitalization for heart surgery."
        ]
    },
    {
        "documents": "https://hackrx.blob.core.windows.net/assets/Super_Splendor_(Feb_2023).pdf?sv=2023-01-03&st=2025-07-21T08%3A10%3A00Z&se=2025-09-22T08%3A10%3A00Z&sr=b&sp=r&sig=vhHrl63YtrEOCsAy%2BpVKr20b3ZUo5HMz1lF9%2BJh6LQ0%3D",
        "questions": [
            "What is the ideal spark plug gap recommeded",
            "Does this comes in tubeless tyre version",
            "Is it compulsoury to have a disc brake"
        ]
    }
]

def test_caching():
    """Test the caching functionality"""
    base_url = "http://localhost:8001"
    headers = {
        "Authorization": "Bearer test_token_123",
        "Content-Type": "application/json"
    }
    
    print("=== Testing Caching Functionality ===\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['documents'].split('/')[-1].split('?')[0]}")
        print("-" * 60)
        
        # First request (should download and cache)
        print(f"First request at {datetime.now().strftime('%H:%M:%S')}")
        start_time = time.time()
        
        try:
            response1 = requests.post(
                f"{base_url}/api/v1/hackrx/run",
                headers=headers,
                json=test_case,
                timeout=120
            )
            
            if response1.status_code == 200:
                first_time = time.time() - start_time
                print(f"✅ First request completed in {first_time:.2f} seconds")
                print(f"Response: {len(response1.json()['answers'])} answers received")
            else:
                print(f"❌ First request failed: {response1.status_code} - {response1.text}")
                continue
                
        except Exception as e:
            print(f"❌ First request error: {str(e)}")
            continue
        
        # Wait a moment
        time.sleep(2)
        
        # Second request (should use cache)
        print(f"Second request at {datetime.now().strftime('%H:%M:%S')}")
        start_time = time.time()
        
        try:
            response2 = requests.post(
                f"{base_url}/api/v1/hackrx/run",
                headers=headers,
                json=test_case,
                timeout=120
            )
            
            if response2.status_code == 200:
                second_time = time.time() - start_time
                print(f"✅ Second request completed in {second_time:.2f} seconds")
                print(f"Response: {len(response2.json()['answers'])} answers received")
                
                # Calculate improvement
                if first_time > 0:
                    improvement = ((first_time - second_time) / first_time) * 100
                    print(f"🚀 Speed improvement: {improvement:.1f}% faster")
                    
                    if second_time < first_time:
                        print("✅ Caching is working - second request was faster!")
                    else:
                        print("⚠️  Second request was not faster - check caching logic")
                else:
                    print("⚠️  Could not calculate improvement")
                    
            else:
                print(f"❌ Second request failed: {response2.status_code} - {response2.text}")
                
        except Exception as e:
            print(f"❌ Second request error: {str(e)}")
        
        print("\n" + "="*60 + "\n")
    
    # Test cache info endpoint
    print("=== Cache Information ===")
    try:
        cache_response = requests.get(f"{base_url}/cache/info", headers=headers)
        if cache_response.status_code == 200:
            cache_info = cache_response.json()
            print(f"Cache directory: {cache_info.get('cache_directory', 'N/A')}")
            print(f"Total cached files: {cache_info.get('total_files', 0)}")
            print(f"Total cache size: {cache_info.get('total_size_mb', 0):.2f} MB")
            
            if cache_info.get('files'):
                print("Cached files:")
                for file_info in cache_info['files']:
                    print(f"  - {file_info['file']}: {file_info['size_mb']} MB")
        else:
            print(f"❌ Cache info failed: {cache_response.status_code}")
    except Exception as e:
        print(f"❌ Cache info error: {str(e)}")

if __name__ == "__main__":
    test_caching() 