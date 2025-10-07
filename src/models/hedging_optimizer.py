# src/models/hedging_optimizer.py
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from loguru import logger

class HedgingOptimizer:
    """
    Optimize procurement timing and quantity allocation
    to minimize cost variance and exploit price signals
    """
    
    def __init__(self, config):
        self.config = config
        self.forecast_horizon = config['forecasting']['forecast_horizon']
        
    def optimize_purchase_schedule(self, forecast_df, total_quantity, 
                                   monthly_capacity=None):
        """
        Create optimal purchase schedule over forecast horizon
        
        Args:
            forecast_df: DataFrame with forecast_mean, forecast_p10, forecast_p90
            total_quantity: Total quantity needed (metric tons)
            monthly_capacity: Max purchase per month (optional constraint)
            
        Returns:
            DataFrame with optimal monthly purchases
        """
        n_months = len(forecast_df)
        
        if monthly_capacity is None:
            monthly_capacity = total_quantity  # No constraint
        
        # Extract price forecasts
        prices_mean = forecast_df['forecast_mean'].values
        prices_std = forecast_df['forecast_std'].values
        
        # Objective: Minimize expected cost + variance penalty
        def objective(quantities):
            expected_cost = np.sum(quantities * prices_mean)
            
            # Variance penalty (risk aversion)
            variance = np.sum((quantities ** 2) * (prices_std ** 2))
            risk_penalty = 0.5 * variance  # Risk aversion parameter
            
            return expected_cost + risk_penalty
        
        # Constraints
        constraints = [
            # Must purchase total quantity
            {'type': 'eq', 'fun': lambda x: np.sum(x) - total_quantity},
        ]
        
        # Bounds: 0 to monthly_capacity for each month
        bounds = [(0, monthly_capacity) for _ in range(n_months)]
        
        # Initial guess: equal distribution
        x0 = np.full(n_months, total_quantity / n_months)
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")
        
        # Build result DataFrame
        schedule = forecast_df.copy()
        schedule['optimal_quantity'] = result.x
        schedule['expected_cost'] = schedule['optimal_quantity'] * schedule['forecast_mean']
        schedule['cost_lower_bound'] = schedule['optimal_quantity'] * schedule['forecast_p10']
        schedule['cost_upper_bound'] = schedule['optimal_quantity'] * schedule['forecast_p90']
        
        # Calculate cumulative purchases
        schedule['cumulative_quantity'] = schedule['optimal_quantity'].cumsum()
        schedule['cumulative_cost'] = schedule['expected_cost'].cumsum()
        
        # Add recommendations
        schedule['recommendation'] = schedule.apply(self._get_recommendation, axis=1)
        
        summary = {
            'total_expected_cost': schedule['expected_cost'].sum(),
            'cost_at_risk_p90': schedule['cost_upper_bound'].sum(),
            'cost_savings_potential': schedule['cost_lower_bound'].sum(),
            'optimization_success': result.success,
            'variance': np.sum((result.x ** 2) * (prices_std ** 2))
        }
        
        return schedule, summary
    
    def _get_recommendation(self, row):
        """Generate human-readable recommendation"""
        quantity = row['optimal_quantity']
        
        if quantity == 0:
            return "SKIP - Wait for better prices"
        elif quantity < row['optimal_quantity'].mean() * 0.5:
            return "SMALL - Minimal purchase"
        elif quantity < row['optimal_quantity'].mean() * 1.5:
            return "MODERATE - Standard purchase"
        else:
            return "LARGE - Aggressive purchase (price dip expected)"
    
    def compare_strategies(self, forecast_df, total_quantity):
        """
        Compare different procurement strategies
        
        Returns comparison of:
        1. Optimal (from optimizer)
        2. Equal monthly splits
        3. Front-loaded (buy early)
        4. Back-loaded (buy late)
        """
        n_months = len(forecast_df)
        
        strategies = {}
        
        # Strategy 1: Optimal
        optimal_schedule, optimal_summary = self.optimize_purchase_schedule(
            forecast_df, 
            total_quantity
        )
        strategies['optimal'] = {
            'schedule': optimal_schedule,
            'expected_cost': optimal_summary['total_expected_cost'],
            'var': optimal_summary['variance']
        }
        
        # Strategy 2: Equal splits
        equal_quantities = np.full(n_months, total_quantity / n_months)
        equal_cost = np.sum(equal_quantities * forecast_df['forecast_mean'].values)
        equal_var = np.sum(
            (equal_quantities ** 2) * (forecast_df['forecast_std'].values ** 2)
        )
        strategies['equal_split'] = {
            'schedule': forecast_df.assign(quantity=equal_quantities),
            'expected_cost': equal_cost,
            'var': equal_var
        }
        
        # Strategy 3: Front-loaded (80% in first 2 months)
        front_quantities = np.zeros(n_months)
        front_quantities[0] = total_quantity * 0.5
        front_quantities[1] = total_quantity * 0.3
        front_quantities[2:] = total_quantity * 0.2 / (n_months - 2)
        front_cost = np.sum(front_quantities * forecast_df['forecast_mean'].values)
        front_var = np.sum(
            (front_quantities ** 2) * (forecast_df['forecast_std'].values ** 2)
        )
        strategies['front_loaded'] = {
            'schedule': forecast_df.assign(quantity=front_quantities),
            'expected_cost': front_cost,
            'var': front_var
        }
        
        # Strategy 4: Back-loaded (80% in last 2 months)
        back_quantities = np.zeros(n_months)
        back_quantities[-1] = total_quantity * 0.5
        back_quantities[-2] = total_quantity * 0.3
        back_quantities[:-2] = total_quantity * 0.2 / (n_months - 2)
        back_cost = np.sum(back_quantities * forecast_df['forecast_mean'].values)
        back_var = np.sum(
            (back_quantities ** 2) * (forecast_df['forecast_std'].values ** 2)
        )
        strategies['back_loaded'] = {
            'schedule': forecast_df.assign(quantity=back_quantities),
            'expected_cost': back_cost,
            'var': back_var
        }
        
        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'strategy': list(strategies.keys()),
            'expected_cost': [s['expected_cost'] for s in strategies.values()],
            'variance': [s['var'] for s in strategies.values()],
            'std_dev': [np.sqrt(s['var']) for s in strategies.values()]
        })
        
        # Calculate savings vs equal split
        baseline_cost = strategies['equal_split']['expected_cost']
        comparison['savings_vs_equal'] = baseline_cost - comparison['expected_cost']
        comparison['savings_pct'] = (
            comparison['savings_vs_equal'] / baseline_cost * 100
        )
        
        return comparison, strategies
    
    def calculate_var(self, schedule, confidence=0.95):
        """
        Calculate Value at Risk for procurement schedule
        
        Args:
            schedule: DataFrame with optimal_quantity and forecast_p10/p90
            confidence: VaR confidence level (default 95%)
            
        Returns:
            VaR estimate in USD
        """
        from scipy import stats
        
        # Use forecast bands to estimate distribution
        quantities = schedule['optimal_quantity'].values
        means = schedule['forecast_mean'].values
        stds = schedule['forecast_std'].values
        
        # Expected cost
        expected_cost = np.sum(quantities * means)
        
        # Variance (assuming independence)
        variance = np.sum((quantities ** 2) * (stds ** 2))
        std_dev = np.sqrt(variance)
        
        # VaR using normal approximation
        z_score = stats.norm.ppf(confidence)
        var = expected_cost + z_score * std_dev
        
        # CVaR (Conditional VaR) - expected loss beyond VaR
        cvar = expected_cost + std_dev * stats.norm.pdf(z_score) / (1 - confidence)
        
        return {
            'expected_cost': expected_cost,
            'std_dev': std_dev,
            f'var_{int(confidence*100)}': var,
            f'cvar_{int(confidence*100)}': cvar,
            'worst_case_cost': np.sum(quantities * schedule['forecast_p90'].values)
        }


# Usage Example
if __name__ == "__main__":
    import yaml
    
    # Load config
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Load forecast
    forecast = pd.read_csv('data/processed/forecast_lithium.csv')
    
    # Initialize optimizer
    optimizer = HedgingOptimizer(config)
    
    # Optimize for 1000 metric tons over 6 months
    schedule, summary = optimizer.optimize_purchase_schedule(
        forecast,
        total_quantity=1000,
        monthly_capacity=300  # Max 300 tons per month
    )
    
    print("\n" + "="*60)
    print("OPTIMAL PROCUREMENT SCHEDULE")
    print("="*60)
    print(schedule[['date', 'optimal_quantity', 'expected_cost', 'recommendation']])
    
    print("\n" + "="*60)
    print("SUMMARY METRICS")
    print("="*60)
    for key, value in summary.items():
        print(f"{key:.<40} {value:>15,.2f}" if isinstance(value, float) else f"{key:.<40} {value}")
    
    # Compare strategies
    print("\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60)
    comparison, _ = optimizer.compare_strategies(forecast, 1000)
    print(comparison.to_string(index=False))
    
    # Calculate VaR
    print("\n" + "="*60)
    print("RISK METRICS (Value at Risk)")
    print("="*60)
    var_metrics = optimizer.calculate_var(schedule, confidence=0.95)
    for key, value in var_metrics.items():
        print(f"{key:.<40} ${value:>15,.2f}")
