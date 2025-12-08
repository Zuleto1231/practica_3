import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")
GROQ_MODEL = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")

# GAN Configuration
GAN_MODEL_PATH = "gan_training/gen_epoch_30.pth"
Z_DIM = 100
IMAGE_SIZE = 64
DEVICE = "cuda" if os.getenv("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"
