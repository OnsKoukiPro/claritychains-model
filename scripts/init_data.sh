#!/bin/bash

# Initialize data directory structure and fetch initial data

echo "🚀 Initializing Critical Materials AI Platform..."

# Create directories
mkdir -p data/{raw,processed,templates}
mkdir -p logs cache

# Generate sample data if no real data exists
if [ ! -f "data/raw/prices.csv" ]; then
    echo "📊 Generating sample price data..."
    python scripts/generate_sample_data.py
fi

# Fetch real data if API keys are configured
if [ -n "$WORLD_BANK_API_KEY" ]; then
    echo "🌐 Fetching price data from World Bank..."
    python src/data_pipeline/price_fetcher.py
else
    echo "⚠️  WORLD_BANK_API_KEY not set, using sample data"
fi

echo "✅ Initialization complete!"