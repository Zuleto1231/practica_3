import torch
from torchvision.utils import save_image
import io
import sys
import os

# Importar las clases del notebook
# Nota: Primero debes convertir el notebook a .py o importar directamente
try:
    # Intentar importar desde un módulo Python si existe
    from main import Generator, Discriminator
except ImportError:
    # Si no existe, usar nbimport o definir aquí las clases del notebook
    # Por ahora, importamos torch.nn para usar las clases definidas en el notebook
    import torch.nn as nn
    
    # Estas son las mismas clases del notebook main.ipynb
    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),

                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),

                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),

                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),

                nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
                nn.Tanh()
            )

        def forward(self, x):
            return self.model(x)

class GANImageGenerator:
    """Wrapper para usar el generador del notebook"""
    def __init__(self, model_path, z_dim=100, device="cpu"):
        self.device = device
        self.z_dim = z_dim
        
        # Usar la clase Generator del notebook
        self.generator = Generator().to(device)
        
        # Cargar pesos si existe el modelo
        if model_path and os.path.exists(model_path):
            self.generator.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✓ Modelo GAN cargado desde: {model_path}")
        else:
            print(f"⚠️  Modelo no encontrado en {model_path}. Usando generador sin entrenar.")
        
        self.generator.eval()

    def generate_image(self, output_path=None, seed=None):
        """Genera una imagen usando la GAN del notebook"""
        if seed is not None:
            torch.manual_seed(seed)
        
        with torch.no_grad():
            noise = torch.randn(1, self.z_dim, 1, 1, device=self.device)
            fake_image = self.generator(noise)
            fake_image = (fake_image + 1) / 2  # Desnormalizar
            
            if output_path:
                save_image(fake_image, output_path)
                return output_path
            else:
                # Retornar como bytes
                buffer = io.BytesIO()
                save_image(fake_image, buffer, format='PNG')
                buffer.seek(0)
                return buffer.getvalue()
