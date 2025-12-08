import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

# Model Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# GAN Configuration
GAN_MODEL_PATH = "formato_de_modelos/generator_epoch_100.keras"
Z_DIM = 128  # Dimensión del espacio latente
IMAGE_SIZE = 64
DEVICE = "cpu"  # No se usa con Keras/TensorFlow
