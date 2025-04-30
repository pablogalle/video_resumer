import subprocess

prompt = "Resume el siguiente texto: Este es un ejemplo de prueba para saber si Ollama funciona."
modelo = "mistral"

try:
    result = subprocess.run(
        ["ollama", "run", modelo],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
    )
    print("\n✅ SALIDA DEL LLM:")
    print(result.stdout.decode("utf-8"))
except FileNotFoundError:
    print("❌ Error: No se encontró el ejecutable de 'ollama'.")
except subprocess.CalledProcessError as e:
    print("❌ Error al ejecutar ollama:")
    print(e.stderr.decode("utf-8"))
