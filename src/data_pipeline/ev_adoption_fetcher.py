import pandas as pd
import requests
import yaml
from datetime import datetime
import logging
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)

class EVAdoptionFetcher:
    """
    Fetch and process EV adoption data using config-based settings
    """

    def __init__(self, config):
        self.config = config.get('ev_adoption', {})

        # Use config-based material intensity or fallback to defaults
        self.material_intensity = self.config.get('material_intensity', {
            'lithium': {'per_ev_kg': 8.0, 'growth_factor': 1.15, 'data_source': 'iea_2024'},
            'cobalt': {'per_ev_kg': 12.0, 'growth_factor': 1.10, 'data_source': 'bnef_2024'},
            'nickel': {'per_ev_kg': 40.0, 'growth_factor': 1.12, 'data_source': 'usgs_2024'},
            'copper': {'per_ev_kg': 80.0, 'growth_factor': 1.08, 'data_source': 'iea_2024'},
            'rare_earths': {'per_ev_kg': 1.0, 'growth_factor': 1.18, 'data_source': 'bnef_2024'}
        })

        # Use config-based price elasticity
        self.price_elasticity = self.config.get('price_elasticity', 0.3)

        # Use config-based scenarios with proper key names
        self.scenarios = self.config.get('scenarios', {
            'conservative': {'sales_2024': 14.0, 'sales_2030': 42.5, 'annual_growth': 0.20},
            'stated_policies': {'sales_2024': 14.0, 'sales_2030': 60.9, 'annual_growth': 0.28},
            'sustainable': {'sales_2024': 14.0, 'sales_2030': 112.5, 'annual_growth': 0.41}
        })

        logger.info(f"EV Adoption Fetcher initialized with {len(self.scenarios)} scenarios")

    def calculate_material_demand(self, material: str, scenario: str = 'stated_policies') -> pd.DataFrame:
        """
        Calculate material demand based on EV adoption projections
        """
        if material not in self.material_intensity:
            logger.error(f"No intensity data for material: {material}")
            return pd.DataFrame()

        # Get scenario data from config
        scenario_data = self.scenarios.get(scenario)
        if not scenario_data:
            logger.warning(f"Scenario {scenario} not found, using 'stated_policies'")
            scenario_data = self.scenarios['stated_policies']

        # Calculate demand using config-based parameters
        years = list(range(2024, 2031))
        demand_data = []

        intensity = self.material_intensity[material]

        # Use correct key names from config
        start_sales = scenario_data.get('sales_2024', 14.0) * 1e6  # Convert to vehicle count
        end_sales = scenario_data.get('sales_2030', 60.9) * 1e6
        annual_growth = scenario_data.get('annual_growth', 0.28)

        for i, year in enumerate(years):
            # Calculate EV sales for this year using exponential growth
            if year == 2024:
                ev_sales = start_sales
            else:
                growth_years = year - 2024
                ev_sales = start_sales * ((1 + annual_growth) ** growth_years)

            # Base demand from new EV sales
            base_demand = ev_sales * intensity['per_ev_kg'] / 1000  # Convert to metric tons

            # Apply growth factor for increasing battery sizes and market penetration
            years_from_2024 = max(0, year - 2024)
            growth_multiplier = intensity['growth_factor'] ** years_from_2024

            total_demand = base_demand * growth_multiplier

            demand_data.append({
                'year': year,
                'material': material,
                'ev_sales_millions': ev_sales / 1e6,
                'material_demand_tons': total_demand,
                'demand_growth_pct': (growth_multiplier - 1) * 100,
                'scenario': scenario,
                'data_source': intensity.get('data_source', 'unknown')
            })

        return pd.DataFrame(demand_data)

    def get_demand_forecast_adjustment(self, material: str) -> Dict:
        """
        Generate demand-based adjustment factors for price forecasts
        """
        try:
            # Calculate demand for the next year
            demand_data = self.calculate_material_demand(material, 'stated_policies')

            if demand_data.empty:
                return {'adjustment_factor': 1.0, 'demand_growth_pct': 0.0}

            # Get demand growth for forecast period (2024 to 2025)
            current_demand = demand_data[demand_data['year'] == 2024]['material_demand_tons']
            future_demand = demand_data[demand_data['year'] == 2025]['material_demand_tons']

            if current_demand.empty or future_demand.empty:
                return {'adjustment_factor': 1.0, 'demand_growth_pct': 0.0}

            current_val = current_demand.iloc[0]
            future_val = future_demand.iloc[0]

            if current_val > 0:
                demand_growth = (future_val - current_val) / current_val
            else:
                demand_growth = 0

            # Convert demand growth to price adjustment factor using price elasticity
            adjustment_factor = 1 + (demand_growth * self.price_elasticity)

            return {
                'adjustment_factor': adjustment_factor,
                'demand_growth_pct': demand_growth * 100,
                'material': material,
                'forecast_year': 2025,
                'price_elasticity': self.price_elasticity
            }

        except Exception as e:
            logger.error(f"Error calculating demand adjustment for {material}: {e}")
            return {'adjustment_factor': 1.0, 'demand_growth_pct': 0.0}

    def generate_ev_adoption_dashboard(self) -> Dict:
        """
        Generate comprehensive EV adoption dashboard data
        """
        materials = ['lithium', 'cobalt', 'nickel', 'copper', 'rare_earths']
        dashboard_data = {}

        for material in materials:
            try:
                # Calculate demand under different scenarios
                conservative_demand = self.calculate_material_demand(material, 'conservative')
                stated_demand = self.calculate_material_demand(material, 'stated_policies')
                sustainable_demand = self.calculate_material_demand(material, 'sustainable')

                # Get current adjustment factor
                adjustment = self.get_demand_forecast_adjustment(material)

                dashboard_data[material] = {
                    'demand_forecasts': {
                        'conservative': conservative_demand.to_dict('records') if not conservative_demand.empty else [],
                        'stated_policies': stated_demand.to_dict('records') if not stated_demand.empty else [],
                        'sustainable': sustainable_demand.to_dict('records') if not sustainable_demand.empty else []
                    },
                    'price_adjustment': adjustment,
                    'material_intensity': self.material_intensity.get(material, {}),
                    'last_updated': datetime.now().isoformat()
                }

            except Exception as e:
                logger.error(f"Failed to generate dashboard data for {material}: {e}")
                continue

        return dashboard_data

    def get_scenario_descriptions(self) -> Dict:
        """Get descriptions for all available scenarios"""
        descriptions = {}
        for scenario_name, scenario_data in self.scenarios.items():
            descriptions[scenario_name] = {
                'description': scenario_data.get('description', 'No description available'),
                'sales_2024': scenario_data.get('sales_2024', 0),
                'sales_2030': scenario_data.get('sales_2030', 0),
                'annual_growth': scenario_data.get('annual_growth', 0)
            }
        return descriptions