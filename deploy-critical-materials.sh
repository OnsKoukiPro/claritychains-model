#!/bin/bash

echo "🚀 Deploying with Docker Compose"

# Check Docker
if ! docker version > /dev/null 2>&1; then
    echo "❌ Docker is not running"
    exit 1
fi

# Remove version from docker-compose.yml to fix warning
sed -i '/^version:/d' docker-compose.yml

# Create directories and config
mkdir -p data/raw data/processed logs cache config

# Create .env file
cat > .env << 'EOF'
DEBUG=true
DATA_DIR=./data
CONFIG_DIR=./config
WORLD_BANK_FREE=true
COMTRADE_FREE=true
YAHOO_FINANCE_FREE=true
EOF

# Create config if missing
if [ ! -f "config/config.yaml" ]; then
    cat > config/config.yaml << 'EOF'
paths:
  data_dir: "./data"
  raw_data: "./data/raw"
  processed_data: "./data/processed"

materials:
  lithium:
    comtrade_codes: ["283691"]
  cobalt:
    comtrade_codes: ["260500", "810520"]
  nickel:
    comtrade_codes: ["750100", "750210"]
  copper:
    comtrade_codes: ["740311", "740319"]
  rare_earths:
    comtrade_codes: ["280530"]
EOF
fi

# Deploy with compose
echo "📦 Building and starting services..."
docker-compose up --build -d

echo "✅ Deployment complete!"
echo "📊 Dashboard: http://localhost:8501"
echo "📓 Jupyter: http://localhost:8888 (if enabled)"

sleep 10

# Test
curl -s http://localhost:8501/healthz && echo "✅ App is healthy" || echo "⚠️ Check logs"