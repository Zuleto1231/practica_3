from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from tools import (
    GenerarImagenGANTool,
    AnalizarImagenLLMTool,
    TareaDominioLLMTool,
    CalcularAreaTool,
    CompararFormasTool,
)
from config import (
    LLM_PROVIDER,
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    GEMINI_MODEL,
    GROQ_MODEL,
)
import json
import re


class GeometricShapesAgent:
    def __init__(self):
        # Inicializar LLM
        if LLM_PROVIDER == "gemini":
            self.llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.7
            )
        else:
            self.llm = ChatGroq(
                model=GROQ_MODEL, groq_api_key=GROQ_API_KEY, temperature=0.7
            )

        # Inicializar herramientas
        self.tools = {
            "generar_imagen_gan": GenerarImagenGANTool(),
            "analizar_imagen_llm": AnalizarImagenLLMTool(),
            "tarea_dominio_llm": TareaDominioLLMTool(),
            "calcular_area": CalcularAreaTool(),
            "comparar_formas": CompararFormasTool(),
        }

        # Historial de conversación
        self.history = []

        # System prompt
        self.system_prompt = """Eres un asistente experto en formas geométricas con acceso a las siguientes herramientas:

1. generar_imagen_gan: Genera una imagen de forma geométrica usando GAN
   - Parámetros: seed (opcional), output_filename (opcional)
   
2. analizar_imagen_llm: Analiza una imagen usando visión por computadora
   - Parámetros: image_path (requerido), prompt (opcional)
   
3. tarea_dominio_llm: Realiza tareas especializadas (diagnóstico, recomendación, clasificación)
   - Parámetros: tarea (requerido), descripcion_imagen (requerido), contexto (opcional)
   
4. calcular_area: Calcula el área de una forma geométrica
   - Parámetros: forma (requerido), dimensiones (requerido)
   
5. comparar_formas: Compara dos imágenes de formas
   - Parámetros: imagen1 (requerido), imagen2 (requerido)

Para usar una herramienta, responde en este formato:
TOOL: nombre_herramienta
PARAMS: {"param1": "valor1", "param2": "valor2"}

Si no necesitas usar herramientas, responde directamente al usuario."""

    def run(self, query: str, max_iterations=5):
        """Ejecuta una consulta en el agente"""
        try:
            print(f"\n🤖 Procesando: {query}\n")

            # Agregar consulta al historial
            self.history.append(HumanMessage(content=query))
            
            # Mantener solo los últimos 6 mensajes para evitar exceder límites
            if len(self.history) > 6:
                self.history = self.history[-6:]

            for iteration in range(max_iterations):
                # Construir mensajes
                messages = [SystemMessage(content=self.system_prompt)] + self.history

                # Obtener respuesta del LLM
                response = self.llm.invoke(messages)
                response_text = response.content

                print(f"[Iteración {iteration + 1}] Respuesta LLM:\n{response_text}\n")

                # Verificar si el LLM quiere usar una herramienta
                if "TOOL:" in response_text and "PARAMS:" in response_text:
                    tool_match = re.search(r"TOOL:\s*(\w+)", response_text)
                    params_match = re.search(
                        r"PARAMS:\s*(\{.*?\})", response_text, re.DOTALL
                    )

                    if tool_match and params_match:
                        tool_name = tool_match.group(1)
                        try:
                            params_str = params_match.group(1)
                            # Reemplazar barras invertidas por barras normales
                            params_str = params_str.replace("\\", "/")
                            params = json.loads(params_str)
                        except Exception as e:
                            print(f"   Error parseando params: {e}")
                            params = {}

                        if tool_name in self.tools:
                            print(f"🔧 Ejecutando herramienta: {tool_name}")
                            print(f"   Parámetros: {params}\n")

                            # Ejecutar herramienta
                            try:
                                tool_result = self.tools[tool_name]._run(**params)
                                print(f"✓ Resultado: {tool_result}\n")
                            except TypeError as e:
                                # Si falla con kwargs, intentar sin parámetros
                                print(f"   Reintentando sin parámetros...\n")
                                tool_result = self.tools[tool_name]._run()
                                print(f"✓ Resultado: {tool_result}\n")

                            # Agregar resultado al historial
                            self.history.append(AIMessage(content=response_text))
                            self.history.append(
                                HumanMessage(
                                    content=f"Resultado de {tool_name}: {tool_result}"
                                )
                            )
                            continue

                # Si no hay herramientas o es la respuesta final
                self.history.append(AIMessage(content=response_text))
                return response_text

            return "Se alcanzó el límite de iteraciones."

        except Exception as e:
            error_msg = f"Error al procesar la consulta: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

    def reset_memory(self):
        """Reinicia la memoria del agente"""
        self.history = []
        print("✓ Memoria reiniciada")
