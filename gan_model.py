import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import keras
from PIL import Image


class GANImageGenerator:
    """Wrapper para usar el generador Keras"""

    def __init__(self, model_path, z_dim=100, device="cpu"):
        self.z_dim = z_dim

        # Cargar modelo Keras
        if model_path and os.path.exists(model_path):
            self.generator = keras.models.load_model(model_path)
            print(f"✓ Modelo GAN cargado desde: {model_path}")
        else:
            print(
                f"⚠️  Modelo no encontrado en {model_path}. Usando generador sin entrenar."
            )
            self.generator = None

    def generate_image(self, output_path=None, seed=None):
        """Genera una imagen usando la GAN Keras"""
        if self.generator is None:
            return "Error: No hay modelo cargado"

        if seed is not None:
            np.random.seed(seed)
            keras.utils.set_random_seed(seed)

        # Generar ruido latente
        noise = np.random.normal(0, 1, (1, self.z_dim))

        # Generar imagen
        generated_image = self.generator.predict(noise, verbose=0)

        # Desnormalizar de [-1, 1] a [0, 255]
        generated_image = (generated_image[0] + 1) * 127.5
        generated_image = np.clip(generated_image, 0, 255).astype(np.uint8)

        # Convertir a PIL Image
        img = Image.fromarray(generated_image)

        if output_path:
            img.save(output_path)
            return output_path
        else:
            # Retornar como bytes
            import io

            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)
            return buffer.getvalue()
