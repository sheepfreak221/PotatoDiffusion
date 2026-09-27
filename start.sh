#!/usr/bin/env bash
# PotatoDiffusion starten
# Prüft die benötigte Ordnerstruktur und legt fehlende Verzeichnisse an.
# Prüft außerdem, ob die benötigten Binärdateien vorhanden sind.

cd "$(dirname "$0")"

# --- Verzeichnisse, die vom WebUI benötigt werden ---
DIRS=(
    "bin"
    "models"
    "models/checkpoints"
    "models/checkpoints/SD1.5"
    "models/checkpoints/SDXL"
    "models/diffusion-models"
    "models/diffusion-models/FLUX"
    "models/textencoder"
    "models/vae"
    "models/upscale"
    "outputs"
    "upscale"
    "upscale/uploads"
)

for dir in "${DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "Erstelle Verzeichnis: $dir"
        mkdir -p "$dir"
    fi
done

# --- Prüfen, ob die Binärdateien vorhanden sind ---
MISSING=0

if [ ! -x "bin/sd-cli" ]; then
    echo "FEHLER: bin/sd-cli nicht gefunden oder nicht ausführbar."
    MISSING=1
fi

if [ ! -x "bin/realesrgan-ncnn-vulkan" ]; then
    echo "FEHLER: bin/realesrgan-ncnn-vulkan nicht gefunden oder nicht ausführbar."
    MISSING=1
fi

if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "Bitte lade die fehlenden Binärdateien herunter und lege sie in bin/ ab:"
    echo "  sd-cli:                 https://github.com/leejet/stable-diffusion.cpp"
    echo "  realesrgan-ncnn-vulkan: https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan"
    echo ""
    echo "Stelle außerdem sicher, dass sie ausführbar sind (chmod +x)."
    exit 1
fi

# --- Starten ---
exec python3 app.py "$@"