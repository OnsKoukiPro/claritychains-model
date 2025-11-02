#!/bin/bash

echo "🚀 Deploying Integrated Critical Materials + AI Agent Platform"

# Check Docker
if ! docker version > /dev/null 2>&1; then
    echo "❌ Docker is not running"
    exit 1
fi

# Clean up any invalid files
echo "🧹 Cleaning up invalid files..."
rm -f agent/__init__  # Remove the invalid file without .py extension

# Create directories
mkdir -p data/raw data/processed logs cache config
mkdir -p agent_data/runs agent_data/uploaded_files
mkdir -p agent/tools agent/static

# Create static files
cat > agent/static/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Procurement AI Agent</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .status {
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .status.healthy {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .endpoints {
            margin-top: 20px;
        }
        .endpoint {
            background: #f8f9fa;
            padding: 10px;
            margin: 5px 0;
            border-left: 4px solid #007bff;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Procurement AI Agent API</h1>
        <div class="status healthy">
            ✅ Service is running and healthy
        </div>
        <div class="endpoints">
            <h3>Available Endpoints:</h3>
            <div class="endpoint">
                <strong>POST /api/add-offer</strong> - Upload procurement documents
            </div>
            <div class="endpoint">
                <strong>POST /api/analyze</strong> - Analyze uploaded offers
            </div>
            <div class="endpoint">
                <strong>POST /api/chat</strong> - Chat about analysis results
            </div>
            <div class="endpoint">
                <strong>GET /health</strong> - Health check
            </div>
        </div>
        <div style="margin-top: 20px;">
            <p>Visit <a href="/docs">/docs</a> for API documentation</p>
        </div>
    </div>
</body>
</html>
EOF

# Create a simple script.js
cat > agent/static/script.js << 'EOF'
// Basic frontend functionality
console.log('Procurement AI Agent frontend loaded');

document.addEventListener('DOMContentLoaded', function() {
    console.log('Document ready');
});
EOF

# Create a simple styles.css
cat > agent/static/styles.css << 'EOF'
/* Basic styles for the procurement agent frontend */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #333;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

.header {
    text-align: center;
    margin-bottom: 30px;
}

.health-status {
    padding: 15px;
    border-radius: 5px;
    margin: 20px 0;
}

.health-status.healthy {
    background-color: #e8f5e8;
    border: 1px solid #4caf50;
    color: #2e7d32;
}
EOF

# Ensure proper __init__.py files exist
touch agent/__init__.py
touch agent/tools/__init__.py

# Create .env file with agent API keys
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# Application settings
DEBUG=true
DATA_DIR=./data
CONFIG_DIR=./config

# Free API settings
WORLD_BANK_FREE=true
COMTRADE_FREE=true
YAHOO_FINANCE_FREE=true

# Agent API settings (REQUIRED for AI analysis)
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here
AGENT_API_URL=http://localhost:8000
EOF
    echo "⚠️  Created .env file - PLEASE ADD YOUR API KEYS!"
    echo "    Edit .env and add your OpenAI or Gemini API key"
fi

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

agent:
  enabled: true
  api_url: "http://localhost:8000"
  default_weights:
    tco: 25
    payment_terms: 10
    price_stability: 5
    lead_time: 20
    tech_specs: 25
    certifications: 5
    incoterms: 5
    warranty: 5
EOF
fi

# Create agent requirements with compatible versions
cat > agent/requirements.txt << 'EOF'
fastapi==0.109.0
uvicorn==0.27.0
python-multipart==0.0.6
langchain>=0.1.13
langchain-openai>=0.0.2
langchain-google-genai>=0.0.6
langchain-experimental>=0.0.55
markdown==3.5.1
python-dotenv==1.0.0
pandas>=2.0.0
openpyxl>=3.1.0
PyPDF2>=3.0.0
python-docx>=1.1.0
unstructured>=0.10.30
langchain-community>=0.0.10
requests>=2.31.0
EOF

# Create Dockerfile.agent with proper Python path
cat > Dockerfile.agent << 'EOF'
FROM python:3.10-slim

WORKDIR /app

# Set Python path to include current directory
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy agent requirements
COPY agent/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY agent/ .

# Create necessary directories
RUN mkdir -p /app/runs /app/uploaded_files /app/static

# Verify file structure
RUN echo "=== File structure in /app ===" && \
    ls -la /app/ && \
    echo "=== Checking static directory ===" && \
    ls -la /app/static/ && \
    echo "=== Checking Python packages ===" && \
    python -c "import langchain_experimental; print('langchain_experimental imported successfully')"

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

echo "📦 Building and starting all services..."
docker-compose down -v --remove-orphans
docker-compose build --no-cache
docker-compose up -d --force-recreate

# Try to pull images first to avoid build issues
echo "📥 Pulling base images..."
docker pull python:3.9-slim
docker pull python:3.10-slim

# Build and start services
echo "🏗️ Building services..."
docker-compose build --no-cache agent-api
docker-compose up -d

echo "⏳ Waiting for services to be ready..."
sleep 30

# Test services
echo "🔍 Testing services..."

# Test agent API
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ AI Agent API is healthy"
else
    echo "❌ AI Agent API failed to start"
    echo "Checking agent logs..."
    docker-compose logs agent-api --tail=50
fi

# Test main app
if curl -s http://localhost:8501 > /dev/null 2>&1; then
    echo "✅ Main dashboard is running"
else
    echo "⚠️  Main dashboard may not be ready yet"
fi

echo ""
echo "📊 Services available:"
echo "   - Main Dashboard: http://localhost:8501"
echo "   - AI Agent API: http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"
echo "   - Jupyter (optional): http://localhost:8888"
echo ""
echo "🎯 Next steps:"
echo "   1. Check that API keys are configured in .env"
echo "   2. Visit http://localhost:8501 for the main dashboard"
echo "   3. Go to 'AI Offer Analysis' tab to use procurement agent"
echo ""
echo "📋 View logs with:"
echo "   docker-compose logs -f critical-materials-app"
echo "   docker-compose logs -f agent-api"