# Proyecto 3: Agente AI con GAN y LangChain

Sistema de agente inteligente que integra GANs para generación de imágenes y LLMs (Gemini/Groq) para análisis y tareas especializadas, todo coordinado con LangChain.

## Características

- **Generación de imágenes**: GAN entrenada para crear formas geométricas
- **Análisis con LLM**: Interpretación de imágenes usando Gemini Vision
- **Tareas especializadas**: Diagnóstico, recomendación y clasificación
- **Herramientas personalizadas**: Cálculo de áreas y comparación de formas
- **Agente conversacional**: Memoria y flujo de trabajo inteligente

## Estructura del Proyecto

```
├── main.ipynb              # Notebook de entrenamiento GAN
├── main.py                 # Punto de entrada del agente
├── agent.py                # Configuración del agente LangChain
├── tools.py                # Definición de herramientas
├── gan_model.py            # Modelo GAN
├── config.py               # Configuración
├── requirements.txt        # Dependencias
└── .env                    # Variables de entorno (crear desde .env.example)
```

## Instalación

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Configurar API keys:
```bash
cp .env.example .env
# Editar .env con tus credenciales
```

3. Entrenar la GAN (si no tienes modelo):
```bash
jupyter notebook main.ipynb
# Ejecutar todas las celdas
```

## Uso

### Modo Interactivo

```bash
python main.py
```

### Ejemplos de Consultas

- "Genera una imagen de forma geométrica"
- "Analiza la imagen en generated_images/generated_image.png"
- "Dame un diagnóstico de la forma"
- "Calcula el área de un círculo con radio 5"
- "Compara dos imágenes"

## Herramientas Disponibles

1. **generar_imagen_gan**: Genera imágenes con la GAN
2. **analizar_imagen_llm**: Analiza imágenes con Gemini Vision
3. **tarea_dominio_llm**: Tareas especializadas (diagnóstico, recomendación, clasificación)
4. **calcular_area**: Calcula áreas de formas geométricas
5. **comparar_formas**: Compara dos imágenes

## Configuración

Edita `config.py` o `.env` para:
- Cambiar entre Gemini y Groq
- Ajustar modelos
- Configurar rutas de modelos GAN

## Requisitos

- Python 3.8+
- PyTorch con soporte CUDA (opcional)
- API keys de Google (Gemini) o Groq
