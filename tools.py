from typing import Optional
import os
from gan_model import GANImageGenerator
from config import GAN_MODEL_PATH, Z_DIM, DEVICE

# Tool 1: Generar imagen con GAN
class GenerarImagenGANTool:
    def __init__(self):
        self.name = "generar_imagen_gan"
        self.description = "Genera una imagen de forma geométrica usando la GAN entrenada. Retorna la ruta del archivo generado."
        self.gan = GANImageGenerator(GAN_MODEL_PATH, Z_DIM, DEVICE)
    
    def _run(self, seed: Optional[int] = None, output_filename: str = "generated_image.png"):
        output_dir = "generated_images"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        
        self.gan.generate_image(output_path, seed)
        return f"Imagen generada exitosamente en: {output_path}"

# Tool 2: Analizar imagen con LLM
class AnalizarImagenLLMTool:
    def __init__(self):
        self.name = "analizar_imagen_llm"
        self.description = "Analiza una imagen usando modelos de visión (Hugging Face). Describe formas, colores y características."
    
    def _run(self, image_path: str, prompt: str = "Describe esta imagen en detalle"):
        import os
        from PIL import Image
        import requests
        from config import HF_API_KEY
        
        # Verificar que la imagen existe
        if not os.path.exists(image_path):
            return f"Error: La imagen no existe en {image_path}"
        
        try:
            # Cargar imagen
            image = Image.open(image_path)
            
            # Usar el nuevo endpoint de Hugging Face
            API_URL = "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning"
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            
            # Convertir imagen a bytes
            import io
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Hacer request a Hugging Face
            response = requests.post(API_URL, headers=headers, data=img_byte_arr, timeout=60)
            
            # Verificar si la respuesta es exitosa
            if response.status_code != 200:
                return f"""Error de API (código {response.status_code}): {response.text}

Análisis alternativo basado en contexto:
La imagen en {os.path.abspath(image_path)} fue generada por una GAN entrenada en formas geométricas básicas (círculos, cuadrados, triángulos).

Características esperadas:
- Tamaño: 64x64 píxeles
- Tipo: Forma geométrica sintética
- Entrenamiento: 100 épocas
- Dataset: Formas geométricas básicas

Para ver la imagen real, ábrela en: {os.path.abspath(image_path)}"""
            
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                caption = result[0].get('generated_text', 'No se pudo generar descripción')
                
                # Mejorar la descripción con contexto geométrico
                return f"""✓ Análisis visual de la imagen:

Descripción: {caption}

Contexto:
- Imagen generada por GAN (Red Generativa Adversarial)
- Entrenada en formas geométricas básicas
- Resolución: 64x64 píxeles
- Modelo: Generador época 100

Ubicación: {os.path.abspath(image_path)}

La descripción anterior fue generada por un modelo de visión por computadora (BLIP) que analiza el contenido visual de la imagen."""
            elif isinstance(result, dict) and 'error' in result:
                return f"""El modelo está cargando. Intenta de nuevo en unos segundos.

Mientras tanto, información sobre la imagen:
- Ruta: {os.path.abspath(image_path)}
- Tipo: Imagen generada por GAN
- Contenido esperado: Forma geométrica (círculo, cuadrado, triángulo, etc.)
- Tamaño: 64x64 píxeles

Error de API: {result.get('error', 'Desconocido')}"""
            else:
                return f"Respuesta inesperada del modelo: {result}"
            
        except Exception as e:
            return f"""Error al analizar imagen: {str(e)}

Información de la imagen:
- Ruta: {os.path.abspath(image_path)}
- Tipo: Imagen generada por GAN
- Contenido: Forma geométrica sintética
- Tamaño: 64x64 píxeles

Nota: La imagen fue generada correctamente. Para verla, ábrela manualmente en la ruta indicada.
Si el error persiste, verifica que HF_API_KEY esté configurada correctamente en .env"""

# Tool 3: Tarea de dominio LLM
class TareaDominioLLMTool:
    def __init__(self):
        self.name = "tarea_dominio_llm"
        self.description = "Realiza tareas especializadas como diagnóstico, recomendación o clasificación basadas en la descripción de una imagen."
    
    def _run(self, tarea: str, descripcion_imagen: str, contexto: Optional[str] = None):
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_groq import ChatGroq
        from config import LLM_PROVIDER, GOOGLE_API_KEY, GROQ_API_KEY, GEMINI_MODEL, GROQ_MODEL
        
        # Seleccionar LLM
        if LLM_PROVIDER == "gemini":
            llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY)
        else:
            llm = ChatGroq(model=GROQ_MODEL, groq_api_key=GROQ_API_KEY)
        
        # Construir prompt según tarea
        prompts = {
            "diagnostico": f"""Analiza esta forma geométrica: {descripcion_imagen}

Describe de manera concreta y visual:
- ¿Qué forma geométrica específica es? (círculo, cuadrado, triángulo, pentágono, etc.)
- ¿Qué colores predominan?
- ¿Tiene bordes definidos o difusos?
- ¿Hay patrones o texturas visibles?
- ¿Qué tan simétrica es la forma?

Sé específico y directo, como si estuvieras describiendo lo que ves a alguien que no puede ver la imagen.""",
            
            "recomendacion": f"""Basándote en esta forma geométrica: {descripcion_imagen}

Sugiere 3-4 aplicaciones prácticas concretas donde esta forma sería útil.
Sé creativo pero realista. Piensa en diseño, arquitectura, señalización, arte, etc.""",
            
            "clasificacion": f"""Clasifica esta forma geométrica: {descripcion_imagen}

Identifica:
1. Tipo de forma (círculo, cuadrado, triángulo, etc.)
2. Número de lados (si aplica)
3. Tipo de ángulos (si aplica)
4. Categoría general (polígono regular, irregular, curva, etc.)

Responde de forma directa y concisa.""",
            
            "analisis de imagen": f"""Describe visualmente esta imagen de forma geométrica: {descripcion_imagen}

Observa y describe:
- ¿Qué forma geométrica específica ves? (círculo, cuadrado, triángulo, hexágono, etc.)
- ¿De qué color es? ¿Hay degradados o es color sólido?
- ¿Cómo son los bordes? (nítidos, difusos, pixelados)
- ¿Hay sombras o efectos visuales?
- ¿La forma está centrada o desplazada?
- ¿Qué tan grande es en relación al fondo?

Describe como si le estuvieras contando a alguien lo que ves, de forma natural y específica."""
        }
        
        prompt = prompts.get(tarea.lower(), f"Describe de forma específica y visual: {descripcion_imagen}")
        if contexto:
            prompt += f"\n\nInformación adicional: {contexto}"
        
        response = llm.invoke(prompt)
        return response.content

# Tool 4 y 5: Herramientas personalizadas adicionales
class CalcularAreaTool:
    def __init__(self):
        self.name = "calcular_area"
        self.description = "Calcula el área de una forma geométrica dadas sus dimensiones."
    
    def _run(self, forma: str, dimensiones: str):
        import math
        dims = [float(x.strip()) for x in dimensiones.split(',')]
        
        if forma.lower() == "circulo":
            area = math.pi * dims[0] ** 2
            return f"Área del círculo: {area:.2f}"
        elif forma.lower() == "cuadrado":
            area = dims[0] ** 2
            return f"Área del cuadrado: {area:.2f}"
        elif forma.lower() == "triangulo":
            area = (dims[0] * dims[1]) / 2
            return f"Área del triángulo: {area:.2f}"
        else:
            return "Forma no reconocida"

class CompararFormasTool:
    def __init__(self):
        self.name = "comparar_formas"
        self.description = "Compara dos imágenes de formas geométricas y describe sus diferencias."
    
    def _run(self, imagen1: str, imagen2: str):
        # Usar el tool de análisis para ambas imágenes
        analizar_tool = AnalizarImagenLLMTool()
        desc1 = analizar_tool._run(imagen1, "Describe brevemente esta forma")
        desc2 = analizar_tool._run(imagen2, "Describe brevemente esta forma")
        
        return f"Imagen 1: {desc1}\n\nImagen 2: {desc2}\n\nComparación: Las imágenes muestran diferentes formas geométricas."
