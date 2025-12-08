import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

from agent import GeometricShapesAgent
import os

def main():
    print("=== Agente de Formas Geométricas con LangChain ===\n")
    
    # Verificar que existan las API keys
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GROQ_API_KEY"):
        print("⚠️  Advertencia: No se encontraron API keys.")
        print("Por favor, crea un archivo .env con tus credenciales.")
        print("Puedes usar .env.example como referencia.\n")
        return
    
    # Inicializar agente
    print("Inicializando agente...")
    agent = GeometricShapesAgent()
    print("✓ Agente inicializado correctamente\n")
    
    # Ejemplos de uso
    ejemplos = [
        "Genera una imagen de forma geométrica",
        "Analiza la imagen generada",
        "Dame un diagnóstico de la forma en la imagen",
        "Calcula el área de un círculo con radio 5",
    ]
    
    print("Ejemplos de consultas que puedes hacer:")
    for i, ejemplo in enumerate(ejemplos, 1):
        print(f"{i}. {ejemplo}")
    print()
    
    # Loop interactivo
    while True:
        try:
            query = input("Tu consulta (o 'salir' para terminar): ").strip()
            
            if query.lower() in ['salir', 'exit', 'quit']:
                print("¡Hasta luego!")
                break
            
            if not query:
                continue
            
            print("\n🤖 Procesando...\n")
            response = agent.run(query)
            print(f"\n✓ Respuesta:\n{response}\n")
            print("-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n¡Hasta luego!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")

if __name__ == "__main__":
    main()
