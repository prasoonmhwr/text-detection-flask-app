import requests
import json
import time

class FlaskAPIClient:
    def __init__(self, base_url="http://localhost:5001"):
        self.base_url = base_url.rstrip('/')
        
    def health_check(self):
        """Check if the API is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predict(self, text):
        """Make a single prediction"""
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json={"text": text},
                headers={"Content-Type": "application/json"}
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def batch_predict(self, texts):
        """Make batch predictions"""
        try:
            response = requests.post(
                f"{self.base_url}/batch-predict",
                json={"texts": texts},
                headers={"Content-Type": "application/json"}
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_model_info(self):
        """Get model information"""
        try:
            response = requests.get(f"{self.base_url}/model-info")
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def test_api():
    """Test the API endpoints"""
    client = FlaskAPIClient()
    
    print("=== Health Check ===")
    health = client.health_check()
    print(json.dumps(health, indent=2))
    
    print("\n=== Model Info ===")
    model_info = client.get_model_info()
    print(json.dumps(model_info, indent=2))
    
    print("\n=== Single Prediction ===")
    prediction = client.predict(f"The dusty lanes of Jaipur shimmered under the afternoon sun. Eight-year-old Leela clutched her worn doll, its fabric faded but its smile unwavering. Her grandmother, Dadi, sat weaving intricate patterns on a small loom, the rhythmic click-clack a familiar lullaby.Dadi, Leela whispered, will the monsoon ever come? Dadi smiled, her eyes crinkling at the corners. The earth whispers, child. Soon, the clouds will gather, and the peacocks will dance.Just then, a faint rumble echoed from the west. Leela's eyes widened. A cool breeze, carrying the scent of distant rain, stirred the dust. Hope, as vibrant as Dadi's threads, bloomed in Leela's heart.")
    print(json.dumps(prediction, indent=2))
    
    print("\n=== Batch Prediction ===")
    texts = [
        "This is amazing!",
        "I hate this product.",
        "This is a sample text to check if it was written by a human or AI",
        "I want to start my youtube channel what is the process for it. thsis sia as fjfs fjskkf sksksks asda sd asd asd asd as da sda sda sda sd asdasdasd asd asd asd  Also provide the resources required for it."
    ]
    batch_result = client.batch_predict(texts)
    print(json.dumps(batch_result, indent=2))
    
    print("\n=== Performance Test ===")
    start_time = time.time()
    for i in range(10):
        client.predict(f"Test message number {i}")
    end_time = time.time()
    print(f"10 predictions took {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    test_api()