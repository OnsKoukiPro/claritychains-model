#!/usr/bin/env python3
"""
Generate sample data for development and testing
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path

def generate_sample_data():
    """Generate realistic sample data for all critical materials"""

    # Create data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    (data_dir / 'raw').mkdir(exist_ok=True)
    (data_dir / 'processed').mkdir(exist_ok=True)

    print("📊 Generating sample data...")

    # Generate price data (3 years of monthly data)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='M')
    materials = ['lithium', 'cobalt', 'nickel', 'copper']

    price_data = []
    for material in materials:
        # Base prices (realistic ranges in USD/tonne)
        base_prices = {
            'lithium': 15000,
            'cobalt': 35000,
            'nickel': 18000,
            'copper': 8000
        }

        base_price = base_prices[material]

        for i, date in enumerate(dates):
            # Add trend (increasing for battery materials)
            trend_factor = 1 + (i / len(dates)) * 0.8 if material in ['lithium', 'cobalt', 'nickel'] else 1 + (i / len(dates)) * 0.3

            # Add seasonality
            seasonal = np.sin(i * 2 * np.pi / 12) * 0.1

            # Add some randomness
            noise = np.random.normal(0, 0.15)

            # Calculate final price
            price = base_price * trend_factor * (1 + seasonal + noise)

            price_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'material': material,
                'price': round(max(price, base_price * 0.3), 2),  # Ensure positive
                'currency': 'USD',
                'unit': 'tonne'
            })

    prices_df = pd.DataFrame(price_data)
    prices_df.to_csv(data_dir / 'raw' / 'prices.csv', index=False)
    print(f"✅ Generated price data: {len(prices_df)} records")

    # Generate trade flow data
    countries = ['China', 'Chile', 'Australia', 'DRC', 'Indonesia', 'Canada', 'Russia', 'Peru']

    trade_data = []
    for material in materials:
        # Different country distributions for each material
        if material == 'lithium':
            distributors = {'Chile': 0.35, 'Australia': 0.25, 'China': 0.20, 'Argentina': 0.10, 'Others': 0.10}
        elif material == 'cobalt':
            distributors = {'DRC': 0.70, 'China': 0.15, 'Russia': 0.08, 'Canada': 0.05, 'Others': 0.02}
        elif material == 'nickel':
            distributors = {'Indonesia': 0.35, 'Philippines': 0.15, 'Russia': 0.12, 'Canada': 0.10, 'Australia': 0.08, 'Others': 0.20}
        else:  # copper
            distributors = {'Chile': 0.25, 'Peru': 0.12, 'China': 0.15, 'USA': 0.10, 'DRC': 0.08, 'Others': 0.30}

        for country, share in distributors.items():
            if country != 'Others':
                trade_data.append({
                    'date': '2024-01-01',
                    'exporter': country,
                    'material': material,
                    'value_usd': share * 1e9,  # Total ~$1B market
                    'volume_tonnes': share * 10000,
                    'importers': 'Global'
                })

    trade_df = pd.DataFrame(trade_data)
    trade_df.to_csv(data_dir / 'raw' / 'trade_flows.csv', index=False)
    print(f"✅ Generated trade flow data: {len(trade_df)} records")

    # Generate geopolitical events
    events_data = [
        {'date': '2024-01-15', 'country': 'DRC', 'material': 'cobalt', 'event_type': 'strike', 'severity': 0.7, 'description': 'Mining strike in Katanga region'},
        {'date': '2024-02-01', 'country': 'Chile', 'material': 'lithium', 'event_type': 'policy', 'severity': 0.6, 'description': 'New mining royalty legislation proposed'},
        {'date': '2024-02-20', 'country': 'Indonesia', 'material': 'nickel', 'event_type': 'export_ban', 'severity': 0.8, 'description': 'Export restrictions on raw nickel ore'},
        {'date': '2024-03-05', 'country': 'China', 'material': 'rare_earths', 'event_type': 'quota', 'severity': 0.5, 'description': 'Export quota reduction announced'},
    ]

    events_df = pd.DataFrame(events_data)
    events_df.to_csv(data_dir / 'raw' / 'geopolitical_events.csv', index=False)
    print(f"✅ Generated geopolitical events: {len(events_df)} records")

    print("🎉 Sample data generation complete!")
    print("📁 Files created:")
    print(f"   - data/raw/prices.csv ({len(prices_df)} records)")
    print(f"   - data/raw/trade_flows.csv ({len(trade_df)} records)")
    print(f"   - data/raw/geopolitical_events.csv ({len(events_df)} records)")

if __name__ == "__main__":
    generate_sample_data()