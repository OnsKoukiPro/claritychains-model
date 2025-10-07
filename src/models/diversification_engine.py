# src/models/diversification_engine.py
import pandas as pd
import numpy as np
from loguru import logger

class DiversificationEngine:
    """
    Supplier diversification optimizer using HHI and network analysis
    Recommends supplier allocation to reduce concentration risk
    """
    
    def __init__(self, config):
        self.config = config
        self.target_hhi = config['risk_thresholds']['hhi_max']
    
    def calculate_hhi(self, supplier_shares):
        """
        Calculate Herfindahl-Hirschman Index
        
        Args:
            supplier_shares: Dict or Series of {supplier: market_share}
            
        Returns:
            HHI value (0 to 1)
        """
        if isinstance(supplier_shares, dict):
            shares = pd.Series(supplier_shares)
        else:
            shares = supplier_shares
        
        # Normalize to sum to 1
        shares = shares / shares.sum()
        
        # HHI = sum of squared shares
        hhi = (shares ** 2).sum()
        
        return hhi
    
    def interpret_hhi(self, hhi):
        """Interpret HHI concentration level"""
        if hhi < 0.15:
            level = "LOW"
            risk = "Competitive market, low concentration risk"
        elif hhi < 0.25:
            level = "MODERATE"
            risk = "Some concentration, manageable risk"
        else:
            level = "HIGH"
            risk = "High concentration, significant supply chain risk"
        
        # Equivalent number of equal-sized suppliers
        n_equivalent = 1 / hhi if hhi > 0 else float('inf')
        
        return {
            'hhi': hhi,
            'level': level,
            'risk_description': risk,
            'n_equivalent_suppliers': n_equivalent
        }
    
    def analyze_current_suppliers(self, trade_df, material):
        """
        Analyze current supplier concentration
        
        Args:
            trade_df: Trade flows DataFrame from UN Comtrade
            material: Material name to analyze
            
        Returns:
            Analysis dict with HHI, top suppliers, risk scores
        """
        # Filter for material
        df = trade_df[trade_df['material'] == material].copy()
        
        # Aggregate by supplier country
        supplier_totals = df.groupby('partner_country')['trade_value_usd'].sum()
        supplier_totals = supplier_totals.sort_values(ascending=False)
        
        # Calculate market shares
        total_trade = supplier_totals.sum()
        market_shares = supplier_totals / total_trade
        
        # Calculate HHI
        hhi = self.calculate_hhi(market_shares)
        interpretation = self.interpret_hhi(hhi)
        
        # Top suppliers analysis
        top_5 = market_shares.head(5)
        cr4 = market_shares.head(4).sum()  # 4-firm concentration ratio
        
        # Geopolitical risk scoring (simplified)
        supplier_risk = self._score_suppliers(market_shares.to_dict())
        
        return {
            'material': material,
            'hhi': interpretation,
            'total_trade_value': total_trade,
            'n_suppliers': len(supplier_totals),
            'top_5_suppliers': top_5.to_dict(),
            'cr4': cr4,  # Top 4 concentration
            'supplier_risk_scores': supplier_risk
        }
    
    def _score_suppliers(self, supplier_shares):
        """
        Score suppliers by geopolitical and operational risk
        
        Risk factors:
        - Political stability
        - Regulatory environment
        - Trade policy risk
        - Infrastructure quality
        
        Simplified version - in production, pull from risk databases
        """
        # Simplified risk scores (0-10, higher = more risky)
        risk_scores = {
            'China': 7,  # High concentration, export control risk
            'Russia': 8,  # Sanctions risk
            'DRC': 9,  # Political instability, ESG concerns
            'Chile': 4,  # Stable but resource nationalism
            'Australia': 2,  # Low risk
            'USA': 2,
            'Canada': 2,
            'Indonesia': 5,
            'Philippines': 6,
            'Peru': 5,
            'Argentina': 6
        }
        
        scored = {}
        for supplier, share in supplier_shares.items():
            base_risk = risk_scores.get(supplier, 5)  # Default medium risk
            
            # Weight by market share (concentration amplifies risk)
            weighted_risk = base_risk * (1 + share)
            
            scored[supplier] = {
                'market_share': share,
                'base_risk_score': base_risk,
                'weighted_risk': weighted_risk
            }
        
        return scored
    
    def optimize_diversification(self, current_shares, target_hhi=None):
        """
        Generate optimal supplier allocation to hit target HHI
        
        Args:
            current_shares: Current supplier market shares (dict or Series)
            target_hhi: Target HHI (default from config)
            
        Returns:
            Recommended allocation and rebalancing steps
        """
        if target_hhi is None:
            target_hhi = self.target_hhi
        
        if isinstance(current_shares, dict):
            current = pd.Series(current_shares)
        else:
            current = current_shares.copy()
        
        # Normalize
        current = current / current.sum()
        current_hhi = self.calculate_hhi(current)
        
        if current_hhi <= target_hhi:
            logger.info(f"Current HHI ({current_hhi:.3f}) already meets target ({target_hhi:.3f})")
            return {
                'current_allocation': current.to_dict(),
                'recommended_allocation': current.to_dict(),
                'rebalancing_needed': False
            }
        
        # Greedy rebalancing: reduce top suppliers, increase smaller ones
        recommended = current.copy()
        
        max_iterations = 100
        for iteration in range(max_iterations):
            hhi = self.calculate_hhi(recommended)
            
            if hhi <= target_hhi:
                break
            
            # Find largest supplier
            largest_supplier = recommended.idxmax()
            largest_share = recommended[largest_supplier]
            
            # Find smallest suppliers (below median)
            median_share = recommended.median()
            small_suppliers = recommended[recommended < median_share]
            
            if len(small_suppliers) == 0:
                # All suppliers are equal, stop
                break
            
            # Transfer 1% from largest to smallest
            transfer_amount = 0.01
            
            if largest_share <= transfer_amount:
                break
            
            recommended[largest_supplier] -= transfer_amount
            
            # Distribute to small suppliers proportionally
            small_total = small_suppliers.sum()
            for supplier in small_suppliers.index:
                proportion = small_suppliers[supplier] / small_total
                recommended[supplier] += transfer_amount * proportion
        
        # Calculate rebalancing steps
        changes = recommended - current
        
        rebalancing_steps = []
        for supplier, change in changes.items():
            if abs(change) > 0.001:  # Only significant changes
                rebalancing_steps.append({
                    'supplier': supplier,
                    'current_share': current[supplier],
                    'recommended_share': recommended[supplier],
                    'change_pct': change * 100,
                    'action': 'INCREASE' if change > 0 else 'DECREASE'
                })
        
        # Sort by magnitude of change
        rebalancing_steps = sorted(
            rebalancing_steps, 
            key=lambda x: abs(x['change_pct']), 
            reverse=True
        )
        
        return {
            'current_hhi': current_hhi,
            'target_hhi': target_hhi,
            'recommended_hhi': self.calculate_hhi(recommended),
            'current_allocation': current.to_dict(),
            'recommended_allocation': recommended.to_dict(),
            'rebalancing_steps': rebalancing_steps,
            'rebalancing_needed': True,
            'convergence_iterations': iteration + 1
        }
    
    def find_alternative_suppliers(self, trade_df, material, min_capacity_pct=0.01):
        """
        Identify alternative/emerging suppliers beyond current top tier
        
        Args:
            trade_df: Trade flows DataFrame
            material: Material name
            min_capacity_pct: Minimum market share to consider (default 1%)
            
        Returns:
            List of alternative suppliers with growth trends
        """
        # Filter for material
        df = trade_df[trade_df['material'] == material].copy()
        
        # Analyze by supplier and year
        yearly = df.groupby(['partner_country', 'year'])['trade_value_usd'].sum().reset_index()
        
        suppliers_analysis = []
        
        for supplier in yearly['partner_country'].unique():
            supplier_data = yearly[yearly['partner_country'] == supplier]
            
            if len(supplier_data) < 2:
                continue
            
            # Calculate growth rate
            first_year = supplier_data['trade_value_usd'].iloc[0]
            last_year = supplier_data['trade_value_usd'].iloc[-1]
            
            if first_year > 0:
                cagr = ((last_year / first_year) ** (1 / (len(supplier_data) - 1)) - 1) * 100
            else:
                cagr = 0
            
            # Calculate current market share
            total_recent = yearly[yearly['year'] == yearly['year'].max()]['trade_value_usd'].sum()
            recent_value = supplier_data[supplier_data['year'] == supplier_data['year'].max()]['trade_value_usd'].values[0]
            market_share = recent_value / total_recent if total_recent > 0 else 0
            
            # Only include if meets minimum capacity
            if market_share >= min_capacity_pct:
                suppliers_analysis.append({
                    'supplier': supplier,
                    'current_market_share': market_share,
                    'recent_trade_value': recent_value,
                    'cagr_pct': cagr,
                    'growth_trend': 'growing' if cagr > 5 else 'stable' if cagr > -5 else 'declining',
                    'years_active': len(supplier_data)
                })
        
        # Sort by growth rate and market share
        suppliers_df = pd.DataFrame(suppliers_analysis)
        suppliers_df = suppliers_df.sort_values(['cagr_pct', 'current_market_share'], ascending=False)
        
        return suppliers_df
    
    def scenario_analysis(self, current_shares, scenarios):
        """
        Test diversification strategy under different supply disruption scenarios
        
        Args:
            current_shares: Current supplier allocation
            scenarios: List of dicts with {'name': str, 'disrupted_suppliers': list, 'impact_pct': float}
            
        Returns:
            Scenario impact analysis
        """
        if isinstance(current_shares, dict):
            current = pd.Series(current_shares)
        else:
            current = current_shares.copy()
        
        current = current / current.sum()
        
        results = []
        
        for scenario in scenarios:
            name = scenario['name']
            disrupted = scenario['disrupted_suppliers']
            impact = scenario['impact_pct']
            
            # Calculate exposure
            exposure = sum(current.get(s, 0) for s in disrupted)
            
            # Calculate supply loss
            supply_loss = exposure * impact
            
            # Remaining suppliers need to compensate
            remaining_suppliers = [s for s in current.index if s not in disrupted]
            remaining_capacity = 1 - exposure
            
            # Can remaining suppliers fill the gap?
            shortfall = supply_loss - remaining_capacity if supply_loss > remaining_capacity else 0
            
            results.append({
                'scenario': name,
                'disrupted_suppliers': ', '.join(disrupted),
                'exposure_pct': exposure * 100,
                'supply_loss_pct': supply_loss * 100,
                'remaining_capacity_pct': remaining_capacity * 100,
                'shortfall_pct': shortfall * 100,
                'severity': 'CRITICAL' if shortfall > 0.1 else 'HIGH' if supply_loss > 0.2 else 'MODERATE'
            })
        
        return pd.DataFrame(results)


# Usage Example
if __name__ == "__main__":
    import yaml
    
    # Load config
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Load trade data
    trade_df = pd.read_csv('data/raw/trade_flows.csv')
    trade_df['date'] = pd.to_datetime(trade_df['date'])
    
    # Initialize engine
    engine = DiversificationEngine(config)
    
    # Analyze current suppliers for lithium
    print("\n" + "="*60)
    print("CURRENT SUPPLIER ANALYSIS - LITHIUM")
    print("="*60)
    
    analysis = engine.analyze_current_suppliers(trade_df, 'lithium')
    
    print(f"\nMaterial: {analysis['material']}")
    print(f"Total Suppliers: {analysis['n_suppliers']}")
    print(f"Total Trade Value: ${analysis['total_trade_value']:,.0f}")
    print(f"\nHHI: {analysis['hhi']['hhi']:.3f}")
    print(f"Concentration Level: {analysis['hhi']['level']}")
    print(f"Risk: {analysis['hhi']['risk_description']}")
    print(f"Equivalent Suppliers: {analysis['hhi']['n_equivalent_suppliers']:.1f}")
    print(f"\nTop 4 Concentration (CR4): {analysis['cr4']:.1%}")
    
    print("\nTop 5 Suppliers:")
    for supplier, share in analysis['top_5_suppliers'].items():
        risk = analysis['supplier_risk_scores'][supplier]
        print(f"  {supplier:.<30} {share:>6.1%}  (Risk: {risk['base_risk_score']}/10)")
    
    # Optimize diversification
    print("\n" + "="*60)
    print("DIVERSIFICATION OPTIMIZATION")
    print("="*60)
    
    current_shares = analysis['top_5_suppliers']
    optimization = engine.optimize_diversification(current_shares, target_hhi=0.20)
    
    print(f"\nCurrent HHI: {optimization['current_hhi']:.3f}")
    print(f"Target HHI: {optimization['target_hhi']:.3f}")
    print(f"Recommended HHI: {optimization['recommended_hhi']:.3f}")
    print(f"Iterations: {optimization['convergence_iterations']}")
    
    print("\nRebalancing Steps:")
    for step in optimization['rebalancing_steps'][:10]:  # Top 10
        print(f"  {step['supplier']:.<25} {step['action']:>8}  "
              f"{step['current_share']:>6.1%} → {step['recommended_share']:>6.1%}  "
              f"({step['change_pct']:+.1f}%)")
    
    # Find alternative suppliers
    print("\n" + "="*60)
    print("ALTERNATIVE/EMERGING SUPPLIERS")
    print("="*60)
    
    alternatives = engine.find_alternative_suppliers(trade_df, 'lithium', min_capacity_pct=0.005)
    
    print("\nFastest Growing Suppliers:")
    print(alternatives.head(10).to_string(index=False))
    
    # Scenario analysis
    print("\n" + "="*60)
    print("SUPPLY DISRUPTION SCENARIOS")
    print("="*60)
    
    scenarios = [
        {
            'name': 'China Export Ban',
            'disrupted_suppliers': ['China'],
            'impact_pct': 1.0  # 100% disruption
        },
        {
            'name': 'DRC Political Crisis',
            'disrupted_suppliers': ['DRC'],
            'impact_pct': 0.7  # 70% disruption
        },
        {
            'name': 'Russia Sanctions Escalation',
            'disrupted_suppliers': ['Russia'],
            'impact_pct': 1.0
        },
        {
            'name': 'South America Regional Crisis',
            'disrupted_suppliers': ['Chile', 'Argentina', 'Peru'],
            'impact_pct': 0.5
        }
    ]
    
    scenario_results = engine.scenario_analysis(current_shares, scenarios)
    print("\n" + scenario_results.to_string(index=False))
