import os

# API server port
API_PORT = int(os.getenv("API_PORT", 8000))

# Path to the trained model file
MODEL_PATH = os.getenv("MODEL_PATH", "model/image_classifier.pth")
