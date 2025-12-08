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
        self.description = "Analiza una imagen usando un modelo de visión (Gemini). Describe formas, colores y características."
    
    def _run(self, image_path: str, prompt: str = "Describe esta imagen en detalle"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        from config import GOOGLE_API_KEY
        import base64
        
        # Leer imagen
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro-vision",
            google_api_key=GOOGLE_API_KEY
        )
        
        response = llm.invoke([
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:image/png;base64,{image_data}"}
        ])
        
        return response.content

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
            "diagnostico": f"Basándote en esta descripción de imagen: '{descripcion_imagen}', proporciona un diagnóstico detallado sobre qué forma geométrica es y sus características.",
            "recomendacion": f"Basándote en esta descripción: '{descripcion_imagen}', recomienda aplicaciones prácticas o usos de esta forma geométrica.",
            "clasificacion": f"Clasifica la forma descrita en: '{descripcion_imagen}'. Indica si es círculo, cuadrado, triángulo, etc."
        }
        
        prompt = prompts.get(tarea.lower(), f"Analiza: {descripcion_imagen}")
        if contexto:
            prompt += f"\nContexto adicional: {contexto}"
        
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
