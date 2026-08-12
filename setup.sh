#!/bin/bash

# ==============================================================================
# CODESPACE AUTOMATION & INITIALIZATION SCRIPT (/dev/sdc1 & DOCLING OPTIMIZED)
# ==============================================================================

# --- CONFIGURATION ---
CHOSEN_MODEL="qwen2.5:3b"
EMBEDDING_MODEL="nomic-embed-text"
APP_PORT=8000
OLLAMA_PORT=11434
NGROK_TOKEN="3GU9mQ5R7dvFUKv7w5R91XzlvT2_4ohtgzS9CkWdYSN8CAxii"
SDC1_MOUNT="/mnt/sdc1"

# --- COLOR UTILITIES ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=== Initializing System & Validating /dev/sdc1 Storage ===${NC}"

# ==============================================================================
# 1. MOUNT AND CONFIGURE /dev/sdc1 STORAGE PARTITION
# ==============================================================================
if [ -e "/dev/sdc1" ]; then
    sudo mkdir -p "$SDC1_MOUNT"
    if ! mountpoint -q "$SDC1_MOUNT"; then
        echo -e "${YELLOW}Mounting /dev/sdc1 to $SDC1_MOUNT...${NC}"
        sudo mount /dev/sdc1 "$SDC1_MOUNT" || {
            echo -e "${YELLOW}Formatting /dev/sdc1 as ext4 and mounting...${NC}"
            sudo mkfs.ext4 -F /dev/sdc1
            sudo mount /dev/sdc1 "$SDC1_MOUNT"
        }
    fi
    sudo chmod 777 "$SDC1_MOUNT"
    STORAGE_BASE="$SDC1_MOUNT"
    echo -e "${GREEN}✓ Successfully mounted and configured /dev/sdc1${NC}"
else
    echo -e "${RED}⚠️ /dev/sdc1 partition not found. Falling back to workspace directory storage.${NC}"
    STORAGE_BASE="./sdc1_fallback"
    mkdir -p "$STORAGE_BASE"
fi

# Define cache directories on /dev/sdc1
TMP_OLLAMA_DIR="$STORAGE_BASE/ollama_cache"
TMP_HF_DIR="$STORAGE_BASE/huggingface_cache"
TMP_PIP_DIR="$STORAGE_BASE/pip_cache"

mkdir -p "$TMP_OLLAMA_DIR" "$TMP_HF_DIR" "$TMP_PIP_DIR"

export OLLAMA_MODELS="$TMP_OLLAMA_DIR"
export HF_HOME="$TMP_HF_DIR"
export PIP_CACHE_DIR="$TMP_PIP_DIR"

# Persist environment variables in ~/.bashrc
grep -qxF "export OLLAMA_MODELS=\"$TMP_OLLAMA_DIR\"" ~/.bashrc || echo "export OLLAMA_MODELS=\"$TMP_OLLAMA_DIR\"" >> ~/.bashrc
grep -qxF "export HF_HOME=\"$TMP_HF_DIR\"" ~/.bashrc || echo "export HF_HOME=\"$TMP_HF_DIR\"" >> ~/.bashrc
grep -qxF "export PIP_CACHE_DIR=\"$TMP_PIP_DIR\"" ~/.bashrc || echo "export PIP_CACHE_DIR=\"$TMP_PIP_DIR\"" >> ~/.bashrc

echo -e "  Ollama Models Directory: $OLLAMA_MODELS"
echo -e "  HuggingFace/Docling Cache: $HF_HOME"
echo -e "  Pip Package Cache: $PIP_CACHE_DIR"

# ==============================================================================
# 2. CHECK & INSTALL OLLAMA
# ==============================================================================
echo -e "${YELLOW}[2/5] Checking Ollama installation...${NC}"
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

if systemctl is-active --quiet ollama; then
    sudo systemctl stop ollama
fi
sudo pkill -9 -f ollama 2>/dev/null

echo "Starting Ollama server daemon..."
OLLAMA_MODELS="$TMP_OLLAMA_DIR" ollama serve > /tmp/ollama_server.log 2>&1 &

echo -n "Waiting for Ollama API server..."
while ! curl -s "http://localhost:$OLLAMA_PORT/api/tags" > /dev/null; do
    sleep 1
    echo -n "."
done
echo -e "\n${GREEN}✓ Ollama daemon is running.${NC}"

# ==============================================================================
# 3. DOWNLOAD REQUIRED OLLAMA MODELS
# ==============================================================================
echo -e "${YELLOW}[3/5] Verifying and pulling models into /dev/sdc1 storage...${NC}"
ollama pull "$CHOSEN_MODEL"
ollama pull "$EMBEDDING_MODEL"
echo -e "${GREEN}✓ Models successfully cached on /dev/sdc1.${NC}"

# ==============================================================================
# 4. INSTALL PYTHON REQUIREMENTS & APP MODULES
# ==============================================================================
echo -e "${YELLOW}[4/5] Checking and installing Python dependencies for app_telegram_hybrid_bm25.py...${NC}"

# Ensure pip uses cache on /dev/sdc1
pip install --cache-dir "$TMP_PIP_DIR" --upgrade pip

# Install required modules used across the RAG application and Docling parser
pip install --cache-dir "$TMP_PIP_DIR" \
    fastapi uvicorn pydantic requests beautifulsoup4 pdfplumber pypdf \
    langchain langchain-core langchain-community langchain-qdrant qdrant-client \
    langchain-ollama python-dotenv python-telegram-bot docling pandas nltk

echo -e "${GREEN}✓ All application modules installed successfully.${NC}"

# ==============================================================================
# 5. CONFIGURE NGROK & AUTHENTICATION TOKEN
# ==============================================================================
echo -e "${YELLOW}[5/5] Configuring ngrok and Authtoken...${NC}"

if ! command -v ngrok &> /dev/null; then
    echo "Installing ngrok..."
    curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
    echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
    sudo apt-get update && sudo apt-get install ngrok -y
fi

echo "Configuring ngrok Authtoken..."
ngrok config add-authtoken "$NGROK_TOKEN"

echo "Launching ngrok public tunnel on port $APP_PORT..."
ngrok http $APP_PORT --log=stdout > /tmp/ngrok_agent.log 2>&1 &
sleep 2

PUBLIC_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o '"public_url":"[^"]*' | grep -o '[^"]*$')
if [ ! -z "$PUBLIC_URL" ]; then
    echo -e "${GREEN}✓ ngrok tunnel active at: ${BLUE}$PUBLIC_URL${NC}"
else
    echo -e "${RED}⚠️ Could not fetch public ngrok URL. Check /tmp/ngrok_agent.log${NC}"
fi

echo -e "\n${GREEN}=== Initialization Complete. All assets stored on /dev/sdc1 partition. ===${NC}"
echo -e "Run your application using: ${BLUE}python app_telegram_hybrid_bm25.py${NC}"