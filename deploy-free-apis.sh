#!/bin/bash

echo "🚀 Deploying with Docker Compose - Enhanced Structure"

# Check Docker
if ! docker version > /dev/null 2>&1; then
    echo "❌ Docker is not running"
    exit 1
fi

# Remove version from docker-compose.yml to fix warning
sed -i '/^version:/d' docker-compose.yml

# Create new directory structure
echo "📁 Creating project structure..."
mkdir -p app/{pages,components,utils}
mkdir -p src/{analytics,utils}
mkdir -p tests/{test_models,test_data_pipeline,test_analytics}
mkdir -p data/raw data/processed logs cache config notebooks

# Create .env file
cat > .env << 'EOF'
DEBUG=true
DATA_DIR=./data
CONFIG_DIR=./config
WORLD_BANK_FREE=true
COMTRADE_FREE=true
YAHOO_FINANCE_FREE=true
PYTHONPATH=/app:/app/src
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

forecasting:
  use_fundamentals: true
  ev_adjustment_weight: 0.3
  risk_adjustment_weight: 0.2
  rolling_window: 12
  forecast_horizon: 6
  confidence_levels: [0.1, 0.5, 0.9]

global_sources:
  ecb_enabled: true
  worldbank_enabled: true
  lme_enabled: true
  global_futures_enabled: true
EOF
fi

# Create sample notebook directory
mkdir -p notebooks
cat > notebooks/example_analysis.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Critical Materials Analysis Notebook\n",
    "\n",
    "This notebook provides examples for analyzing critical materials data."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Deploy with compose
echo "📦 Building and starting services..."
docker-compose up --build -d

echo "⏳ Waiting for services to start..."
sleep 15

# Test the application
echo "🧪 Testing application..."
if curl -s http://localhost:8501 > /dev/null; then
    echo "✅ App is healthy and running"
else
    echo "⚠️ App may be starting up, check logs with: docker-compose logs critical-materials-app"
fi

echo ""
echo "🎉 Deployment complete!"
echo "📊 Dashboard: http://localhost:8501"
echo "📓 Jupyter: http://localhost:8888 (if enabled)"
echo ""
echo "📋 Quick commands:"
echo "   View logs: docker-compose logs -f critical-materials-app"
echo "   Stop services: docker-compose down"
echo "   Restart: docker-compose restart"
echo "   Update: ./deploy-free-apis.sh"