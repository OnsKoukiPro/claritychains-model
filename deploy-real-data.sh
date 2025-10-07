#!/bin/bash

echo "🚀 Deploying Critical Materials AI Platform with Real Data"

# Check Docker
if ! docker version > /dev/null 2>&1; then
    echo "❌ Docker is not running"
    exit 1
fi

# Clean up
echo "🧹 Cleaning up..."
docker stop claritychain-app 2>/dev/null || true
docker rm claritychain-app 2>/dev/null || true

# Create directories
echo "📁 Creating directories..."
mkdir -p data/raw data/processed logs cache

# Build with real data dependencies
echo "📦 Building Docker image with real data support..."
docker build -f docker/Dockerfile -t claritychain-app .

# Run container
echo "🐳 Starting application with real data..."
docker run -d \
    --name claritychain-app \
    -p 8501:8501 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/config:/app/config \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/cache:/app/cache \
    claritychain-app

echo "✅ Deployment complete!"
echo "📊 Access: http://localhost:8501"
echo ""
echo "🔑 Next steps:"
echo "   - Configure API keys in the app"
echo "   - Enable 'Use Real API Data' in the sidebar"
echo "   - Click 'Refresh Data from APIs'"
echo ""
echo "⏳ Waiting for app to start..."
sleep 10

# Test
curl -s http://localhost:8501/healthz && echo "✅ App is healthy" || echo "⚠️  Check app status"