#!/usr/bin/env bash
# Configura el entorno de desarrollo local:
# - Instala mkcert (si no está)
# - Genera certificados TLS confiados por el sistema para localhost
# - Los certs quedan en certs/ y son montados por Traefik en docker-compose.yml
set -euo pipefail

CERT_DIR="$(cd "$(dirname "$0")/.." && pwd)/certs"

# ── Instalar mkcert ────────────────────────────────────────────────────────────
if ! command -v mkcert &>/dev/null; then
    echo "▶ Instalando mkcert..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if ! command -v brew &>/dev/null; then
            echo "Error: Homebrew no encontrado. Instalá Homebrew primero: https://brew.sh"
            exit 1
        fi
        brew install mkcert nss
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &>/dev/null; then
            sudo apt-get update -qq
            sudo apt-get install -y -qq libnss3-tools
        fi
        ARCH=$(uname -m)
        [[ "$ARCH" == "x86_64" ]] && ARCH="amd64"
        [[ "$ARCH" == "aarch64" ]] && ARCH="arm64"
        LATEST=$(curl -fsSL https://api.github.com/repos/FiloSottile/mkcert/releases/latest \
            | grep '"tag_name"' | cut -d'"' -f4)
        curl -fsSL "https://github.com/FiloSottile/mkcert/releases/download/${LATEST}/mkcert-${LATEST}-linux-${ARCH}" \
            -o /tmp/mkcert
        sudo install -m 755 /tmp/mkcert /usr/local/bin/mkcert
    else
        echo "SO no soportado. Instalá mkcert manualmente: https://github.com/FiloSottile/mkcert"
        exit 1
    fi
else
    echo "✓ mkcert ya instalado ($(mkcert --version))"
fi

# ── Instalar CA local en el sistema ───────────────────────────────────────────
echo "▶ Instalando CA local en el sistema (puede pedir contraseña)..."
mkcert -install

# ── Generar certificados ───────────────────────────────────────────────────────
mkdir -p "$CERT_DIR"
mkcert \
    -key-file  "$CERT_DIR/key.pem" \
    -cert-file "$CERT_DIR/cert.pem" \
    localhost 127.0.0.1 ::1

echo ""
echo "✓ Certificados generados en certs/"
echo ""
echo "Próximo paso:"
echo "  docker compose up -d"
echo "  → https://localhost/api/"
echo "  → https://localhost/ui/"
echo "  → https://localhost/mlflow/"
