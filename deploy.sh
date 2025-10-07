#!/bin/bash

# Critical Materials AI Platform - Fixed Docker Deployment

echo "🚀 Deploying Critical Materials AI Platform..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install docker-compose."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    cat > .env << 'EOF'
# Application Settings
DEBUG=true
LOG_LEVEL=INFO
DATA_DIR=./data
CONFIG_DIR=./config

# API Keys (add your actual keys when available)
WORLD_BANK_API_KEY=your_world_bank_api_key_here
UN_COMTRADE_API_KEY=your_comtrade_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here

# Data Update Settings
UPDATE_FREQUENCY_DAYS=7
ENABLE_AUTO_UPDATE=false

# Model Settings
FORECAST_HORIZON_MONTHS=6
HHI_TARGET=0.25
CONFIDENCE_LEVEL=0.9
EOF
    echo "⚠️  Please edit .env file with your API keys for full functionality"
fi

# Initialize data directories
echo "📊 Initializing data directories..."
mkdir -p data/{raw,processed,templates}
mkdir -p logs cache

# Generate sample data
echo "📊 Generating sample data..."
python scripts/generate-sample-data.py

# Build and start services
echo "🐳 Building and starting Docker containers..."
docker-compose -f /docker-compose.yml up --build -d

echo "✅ Deployment complete!"
echo "📊 Streamlit Dashboard: http://localhost:8501"
echo "📓 Jupyter Lab: http://localhost:8888"
echo "📁 Data directory: ./data"
echo "📋 Logs: ./logs"

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 15

# Check service status
echo "🔍 Checking service status..."
docker-compose -f docker/docker-compose.yml ps

# Test the application
echo "🧪 Testing application..."
curl -f http://localhost:8501/health && echo "✅ Application is healthy" || echo "❌ Application health check failed"