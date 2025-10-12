# src/models/baseline_forecaster.py
"""
Enhanced Baseline Statistical Forecaster for Critical Materials Prices

This module provides statistical baseline forecasting with fundamental adjustments:
- Rolling mean and standard deviation
- Momentum indicators
- Volatility regime detection
- Confidence interval predictions (P10, P50, P90)
- GDELT geopolitical risk integration
- EV adoption demand signals

Author: ClarityChain
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from loguru import logger
import warnings
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))

try:
    from data_pipeline.ev_adoption_fetcher import EVAdoptionFetcher
    from data_pipeline.gdelt_fetcher import GDELTFetcher
except ImportError as e:
    logger.warning(f"Could not import fundamental data fetchers: {e}")
    # Create placeholder classes for fallback
    class EVAdoptionFetcher:
        def __init__(self, config): pass
        def get_demand_forecast_adjustment(self, material):
            return {'adjustment_factor': 1.0, 'demand_growth': 0.0}

    class GDELTFetcher:
        def __init__(self, config): pass
        def fetch_events_for_material(self, material): return pd.DataFrame()
        def generate_risk_score(self, events, material):
            return {'risk_score': 0.0, 'risk_level': 'LOW', 'recent_events': 0}

warnings.filterwarnings('ignore')


class BaselineForecaster:
    """
    Enhanced statistical forecaster for commodity prices with fundamental adjustments

    Features:
    - Rolling statistics (mean, std, momentum)
    - Volatility regime classification
    - Multi-horizon forecasting with uncertainty bands
    - Trend detection
    - GDELT geopolitical risk integration
    - EV adoption demand signals

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
        - rolling_window: int, number of periods for rolling calculations
        - forecast_horizon: int, number of periods to forecast
        - confidence_levels: list, confidence levels for prediction intervals
        - min_data_points: int, minimum data required
    """

    def __init__(self, config: dict):
        self.config = config
        self.window = config['forecasting']['rolling_window']
        self.horizon = config['forecasting']['forecast_horizon']
        self.confidence_levels = config['forecasting']['confidence_levels']
        self.min_data_points = config['forecasting'].get('min_data_points', 24)

        # Enhanced forecasting settings
        self.use_fundamentals = config['forecasting'].get('use_fundamentals', True)
        self.ev_adjustment_weight = config['forecasting'].get('ev_adjustment_weight', 0.3)
        self.risk_adjustment_weight = config['forecasting'].get('risk_adjustment_weight', 0.2)

        # Initialize fundamental data fetchers
        self.gdelt_fetcher = GDELTFetcher(config)
        self.ev_fetcher = EVAdoptionFetcher(config)

        logger.info(f"Initialized Enhanced BaselineForecaster with window={self.window}, horizon={self.horizon}")
        logger.info(f"Fundamental adjustments: {self.use_fundamentals}")

    def fit_predict(self, prices_df: pd.DataFrame, material: str, use_fundamentals: bool = None) -> Dict:
        """
        Generate forecast with confidence bands for a specific material
        Enhanced with fundamental factors when use_fundamentals=True

        Parameters
        ----------
        prices_df : pd.DataFrame
            DataFrame with columns: date, material, price_usd
        material : str
            Material name to forecast
        use_fundamentals : bool, optional
            Whether to include fundamental adjustments (default from config)

        Returns
        -------
        dict
            Dictionary containing:
            - historical: DataFrame with historical data and indicators
            - forecast: DataFrame with forecast values and confidence bands
            - metrics: Dict with current metrics and diagnostics
            - model_info: Dict with model metadata
            - fundamentals: Dict with fundamental adjustment data (if used)
        """
        if use_fundamentals is None:
            use_fundamentals = self.use_fundamentals

        if use_fundamentals:
            return self.fit_predict_with_fundamentals(prices_df, material)
        else:
            return self._fit_predict_baseline(prices_df, material)

    def _fit_predict_baseline(self, prices_df: pd.DataFrame, material: str) -> Dict:
        """
        Original baseline forecasting without fundamental adjustments
        """
        logger.info(f"Generating baseline forecast for {material}")

        # Validate and prepare data
        df = self._prepare_data(prices_df, material)

        # Calculate rolling statistics
        df = self._calculate_rolling_stats(df)

        # Detect volatility regime
        df = self._classify_volatility_regime(df)

        # Calculate momentum indicators
        df = self._calculate_momentum(df)

        # Detect trend
        df = self._detect_trend(df)

        # Generate future forecasts
        forecast_df = self._generate_forecast(df)

        # Calculate performance metrics
        metrics = self._calculate_metrics(df)

        # Model metadata
        model_info = {
            'model_type': 'statistical_baseline',
            'version': '2.0.0',
            'material': material,
            'training_samples': len(df),
            'forecast_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'window': self.window,
            'horizon': self.horizon,
            'fundamental_adjustments': False
        }

        logger.success(f"Baseline forecast generated successfully for {material}")

        return {
            'historical': df,
            'forecast': forecast_df,
            'metrics': metrics,
            'model_info': model_info,
            'fundamentals': None
        }

    def fit_predict_with_fundamentals(self, prices_df: pd.DataFrame, material: str) -> Dict:
        """
        Enhanced forecasting with GDELT events and EV adoption signals

        Parameters
        ----------
        prices_df : pd.DataFrame
            DataFrame with columns: date, material, price_usd
        material : str
            Material name to forecast

        Returns
        -------
        dict
            Enhanced forecast results with fundamental adjustments
        """
        logger.info(f"Generating enhanced forecast with fundamentals for {material}")

        # Get baseline forecast
        baseline_result = self._fit_predict_baseline(prices_df, material)

        try:
            # Get EV demand adjustment
            ev_adjustment = self.ev_fetcher.get_demand_forecast_adjustment(material)

            # Get geopolitical risk events
            gdelt_events = self.gdelt_fetcher.fetch_events_for_material(material)
            risk_score = self.gdelt_fetcher.generate_risk_score(gdelt_events, material)

            # Adjust forecast based on fundamentals
            adjusted_forecast = self._adjust_forecast_with_fundamentals(
                baseline_result['forecast'],
                ev_adjustment,
                risk_score
            )

            # Update model info
            baseline_result['model_info']['fundamental_adjustments'] = True
            baseline_result['model_info']['ev_adjustment_factor'] = ev_adjustment.get('adjustment_factor', 1.0)
            baseline_result['model_info']['risk_score'] = risk_score.get('risk_score', 0.0)

            # Update results
            baseline_result['forecast'] = adjusted_forecast
            baseline_result['fundamentals'] = {
                'ev_adjustment': ev_adjustment,
                'geopolitical_risk': risk_score,
                'gdelt_events_count': len(gdelt_events),
                'adjustment_weights': {
                    'ev_weight': self.ev_adjustment_weight,
                    'risk_weight': self.risk_adjustment_weight
                }
            }

            logger.success(f"Enhanced forecast with fundamentals generated for {material}")

        except Exception as e:
            logger.error(f"Failed to apply fundamental adjustments for {material}: {e}")
            # Fall back to baseline forecast
            baseline_result['model_info']['fundamental_adjustments'] = False
            baseline_result['model_info']['fundamental_error'] = str(e)
            baseline_result['fundamentals'] = {
                'error': str(e),
                'fallback_to_baseline': True
            }

        return baseline_result

    def _adjust_forecast_with_fundamentals(self, forecast_df: pd.DataFrame,
                                         ev_adjustment: Dict,
                                         risk_score: Dict) -> pd.DataFrame:
        """
        Adjust statistical forecast with fundamental factors using weighted approach

        Parameters
        ----------
        forecast_df : pd.DataFrame
            Baseline forecast dataframe
        ev_adjustment : Dict
            EV adoption demand adjustment factors
        risk_score : Dict
            Geopolitical risk scores and events

        Returns
        -------
        pd.DataFrame
            Forecast adjusted for fundamental factors
        """
        adjusted_forecast = forecast_df.copy()

        # Apply EV demand adjustment with configurable weight
        demand_factor = ev_adjustment.get('adjustment_factor', 1.0)
        weighted_demand_factor = 1.0 + (demand_factor - 1.0) * self.ev_adjustment_weight

        # Apply geopolitical risk adjustment with configurable weight
        risk_factor = 1.0
        risk_level = risk_score.get('risk_level', 'LOW')

        if risk_level in ['HIGH', 'CRITICAL']:
            base_risk_adjustment = 1.1  # 10% upward adjustment for high risk
        elif risk_level == 'MEDIUM':
            base_risk_adjustment = 1.05  # 5% upward adjustment
        else:
            base_risk_adjustment = 1.0

        weighted_risk_factor = 1.0 + (base_risk_adjustment - 1.0) * self.risk_adjustment_weight

        # Combine adjustments
        combined_factor = weighted_demand_factor * weighted_risk_factor

        # Adjust forecast values
        forecast_columns = ['forecast_mean', 'forecast_p10', 'forecast_p50', 'forecast_p90']
        for col in forecast_columns:
            if col in adjusted_forecast.columns:
                adjusted_forecast[col] = adjusted_forecast[col] * combined_factor

        # Add adjustment metadata
        adjusted_forecast['fundamental_adjustment_factor'] = combined_factor
        adjusted_forecast['ev_demand_factor'] = weighted_demand_factor
        adjusted_forecast['risk_adjustment_factor'] = weighted_risk_factor
        adjusted_forecast['base_ev_factor'] = demand_factor
        adjusted_forecast['base_risk_factor'] = base_risk_adjustment

        logger.info(f"Applied fundamental adjustments: EV={weighted_demand_factor:.3f}, "
                   f"Risk={weighted_risk_factor:.3f}, Combined={combined_factor:.3f}")

        return adjusted_forecast

    def _prepare_data(self, prices_df: pd.DataFrame, material: str) -> pd.DataFrame:
        """Prepare and validate data for forecasting"""
        # Filter for specific material
        df = prices_df[prices_df['material'] == material].copy()

        if len(df) == 0:
            raise ValueError(f"No data found for material: {material}")

        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        # Check minimum data points
        if len(df) < self.min_data_points:
            logger.warning(
                f"Only {len(df)} data points available for {material} "
                f"(minimum recommended: {self.min_data_points})"
            )

        # Remove duplicates (keep last)
        df = df.drop_duplicates(subset=['date'], keep='last')

        # Handle missing values
        if df['price_usd'].isnull().any():
            logger.warning(f"Found {df['price_usd'].isnull().sum()} missing prices, interpolating...")
            df['price_usd'] = df['price_usd'].interpolate(method='linear')

        return df

    def _calculate_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling mean and standard deviation"""
        # Rolling mean
        df['rolling_mean'] = df['price_usd'].rolling(
            window=self.window,
            min_periods=max(1, self.window // 2)
        ).mean()

        # Rolling standard deviation
        df['rolling_std'] = df['price_usd'].rolling(
            window=self.window,
            min_periods=max(1, self.window // 2)
        ).std()

        # Rolling median (more robust to outliers)
        df['rolling_median'] = df['price_usd'].rolling(
            window=self.window,
            min_periods=max(1, self.window // 2)
        ).median()

        # Fill initial NaNs with expanding window
        df['rolling_mean'].fillna(
            df['price_usd'].expanding().mean(),
            inplace=True
        )
        df['rolling_std'].fillna(
            df['price_usd'].expanding().std(),
            inplace=True
        )
        df['rolling_median'].fillna(
            df['price_usd'].expanding().median(),
            inplace=True
        )

        return df

    def _classify_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify volatility into regimes: tight, neutral, loose"""
        # Coefficient of variation (CV = std / mean)
        df['volatility_cv'] = df['rolling_std'] / df['rolling_mean']

        # Volatility thresholds from config
        low_threshold = self.config['risk_thresholds'].get('volatility_moderate', 0.10)
        high_threshold = self.config['risk_thresholds'].get('volatility_high', 0.20)

        # Classify regime
        conditions = [
            df['volatility_cv'] <= low_threshold,
            (df['volatility_cv'] > low_threshold) & (df['volatility_cv'] <= high_threshold),
            df['volatility_cv'] > high_threshold
        ]
        choices = ['tight', 'neutral', 'loose']
        df['volatility_regime'] = np.select(conditions, choices, default='neutral')

        return df

    def _calculate_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators"""
        # Simple momentum (3-month rate of change)
        df['momentum_3m'] = df['price_usd'].pct_change(periods=3)

        # Z-score of momentum (standardized)
        df['momentum_zscore'] = (
            df['momentum_3m'] - df['momentum_3m'].rolling(self.window).mean()
        ) / df['momentum_3m'].rolling(self.window).std()

        # Momentum moving average
        df['momentum_ma'] = df['momentum_3m'].rolling(window=3).mean()

        # Rate of change acceleration
        df['momentum_acceleration'] = df['momentum_3m'].diff()

        return df

    def _detect_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect price trend using multiple indicators"""
        # Simple Moving Average crossover
        df['sma_short'] = df['price_usd'].rolling(window=3).mean()
        df['sma_long'] = df['price_usd'].rolling(window=12).mean()

        # Price relative to moving average
        df['price_vs_ma'] = (df['price_usd'] - df['rolling_mean']) / df['rolling_mean']

        # Linear regression trend
        df['trend_strength'] = df['price_usd'].rolling(
            window=self.window
        ).apply(self._calculate_trend_strength, raw=False)

        # Overall trend classification
        conditions = [
            (df['momentum_zscore'] > 0.5) & (df['price_vs_ma'] > 0.02),
            (df['momentum_zscore'] < -0.5) & (df['price_vs_ma'] < -0.02)
        ]
        choices = ['upward', 'downward']
        df['trend'] = np.select(conditions, choices, default='neutral')

        return df

    @staticmethod
    def _calculate_trend_strength(prices: pd.Series) -> float:
        """Calculate linear regression slope as trend strength"""
        if len(prices) < 3:
            return 0.0

        x = np.arange(len(prices))
        y = prices.values

        # Remove NaNs
        mask = ~np.isnan(y)
        if mask.sum() < 3:
            return 0.0

        x = x[mask]
        y = y[mask]

        # Linear regression
        slope, _, r_value, _, _ = stats.linregress(x, y)

        # Return slope normalized by mean price
        return slope / y.mean() if y.mean() != 0 else 0.0

    def _generate_forecast(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate future forecasts with confidence intervals"""
        # Get last available values
        last_date = df['date'].iloc[-1]
        last_price = df['price_usd'].iloc[-1]
        last_mean = df['rolling_mean'].iloc[-1]
        last_std = df['rolling_std'].iloc[-1]
        last_momentum_z = df['momentum_zscore'].iloc[-1]
        last_trend = df['trend'].iloc[-1]

        # Determine forecast adjustment based on momentum and trend
        momentum_adjustment = self._calculate_momentum_adjustment(
            last_momentum_z,
            last_trend
        )

        # Generate forecasts for each horizon
        forecasts = []

        for h in range(1, self.horizon + 1):
            # Forecast date
            forecast_date = last_date + pd.DateOffset(months=h)

            # Base forecast (with momentum adjustment)
            base_forecast = last_mean * (1 + momentum_adjustment)

            # Uncertainty grows with horizon (square root of time rule)
            forecast_std = last_std * np.sqrt(h)

            # Add mean reversion (prices tend to revert to mean over time)
            reversion_factor = 0.1 * h  # 10% per period
            reversion_adjustment = (last_mean - base_forecast) * min(reversion_factor, 0.5)
            forecast_mean = base_forecast + reversion_adjustment

            # Generate confidence intervals
            forecast_row = {
                'date': forecast_date,
                'horizon': h,
                'forecast_mean': forecast_mean,
                'forecast_std': forecast_std
            }

            # Add percentile forecasts
            for level in self.confidence_levels:
                z_score = stats.norm.ppf(level)
                forecast_value = forecast_mean + z_score * forecast_std
                forecast_row[f'forecast_p{int(level*100)}'] = max(0, forecast_value)  # Ensure non-negative

            forecasts.append(forecast_row)

        forecast_df = pd.DataFrame(forecasts)

        # Add metadata
        forecast_df['momentum_adjustment'] = momentum_adjustment
        forecast_df['last_trend'] = last_trend
        forecast_df['baseline_forecast_mean'] = forecast_df['forecast_mean']  # Store baseline for comparison

        return forecast_df

    def _calculate_momentum_adjustment(
        self,
        momentum_zscore: float,
        trend: str
    ) -> float:
        """
        Calculate forecast adjustment based on momentum

        Positive momentum -> forecast higher
        Negative momentum -> forecast lower
        Bounded to prevent extreme adjustments
        """
        # Base adjustment from z-score
        base_adjustment = np.clip(momentum_zscore * 0.02, -0.10, 0.10)  # ±10% max

        # Amplify if trend confirms momentum
        if (momentum_zscore > 0 and trend == 'upward') or \
           (momentum_zscore < 0 and trend == 'downward'):
            base_adjustment *= 1.5

        return base_adjustment

    def _calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate descriptive metrics for current state"""
        # Recent data (last window periods)
        recent_df = df.tail(self.window)

        # Current values
        current_price = df['price_usd'].iloc[-1]
        current_mean = df['rolling_mean'].iloc[-1]
        current_std = df['rolling_std'].iloc[-1]
        current_cv = df['volatility_cv'].iloc[-1]
        current_regime = df['volatility_regime'].iloc[-1]
        current_momentum = df['momentum_zscore'].iloc[-1]
        current_trend = df['trend'].iloc[-1]

        # Historical statistics
        avg_price = recent_df['price_usd'].mean()
        min_price = recent_df['price_usd'].min()
        max_price = recent_df['price_usd'].max()
        price_range = max_price - min_price

        # Price position in range
        if price_range > 0:
            price_percentile = (current_price - min_price) / price_range
        else:
            price_percentile = 0.5

        # Trend strength (average over recent period)
        trend_strength = recent_df['trend_strength'].mean()

        # Recent volatility
        recent_volatility = recent_df['price_usd'].pct_change().std()

        # Price velocity (rate of change)
        price_velocity = (current_price - avg_price) / avg_price if avg_price != 0 else 0

        return {
            'current_price': float(current_price),
            'rolling_mean': float(current_mean),
            'rolling_std': float(current_std),
            'coefficient_of_variation': float(current_cv),
            'volatility_regime': current_regime,
            'momentum_zscore': float(current_momentum),
            'trend': current_trend,
            'trend_strength': float(trend_strength),
            'price_vs_average': float(price_velocity),
            'price_percentile_in_range': float(price_percentile),
            'recent_min': float(min_price),
            'recent_max': float(max_price),
            'recent_volatility': float(recent_volatility),
            'data_quality': self._assess_data_quality(df)
        }

    def _assess_data_quality(self, df: pd.DataFrame) -> Dict:
        """Assess data quality metrics"""
        total_points = len(df)
        missing_points = df['price_usd'].isnull().sum()

        # Check for outliers (prices > 3 std from mean)
        z_scores = np.abs(stats.zscore(df['price_usd'].dropna()))
        outliers = (z_scores > 3).sum()

        # Data completeness
        completeness = (total_points - missing_points) / total_points if total_points > 0 else 0

        # Data recency (days since last update)
        last_date = df['date'].iloc[-1]
        days_old = (datetime.now() - last_date).days

        return {
            'total_data_points': int(total_points),
            'missing_values': int(missing_points),
            'outliers_detected': int(outliers),
            'completeness_pct': float(completeness * 100),
            'days_since_update': int(days_old),
            'quality_score': float(completeness * (1 - outliers/total_points if total_points > 0 else 0))
        }

    def backtest(
        self,
        prices_df: pd.DataFrame,
        material: str,
        test_periods: int = 6,
        use_fundamentals: bool = None
    ) -> Dict:
        """
        Backtest forecaster on historical data with optional fundamental adjustments

        Parameters
        ----------
        prices_df : pd.DataFrame
            Full price history
        material : str
            Material to test
        test_periods : int
            Number of periods to hold out for testing
        use_fundamentals : bool, optional
            Whether to use fundamental adjustments in backtest

        Returns
        -------
        dict
            Backtest results with error metrics
        """
        if use_fundamentals is None:
            use_fundamentals = self.use_fundamentals

        logger.info(f"Running backtest for {material} with {test_periods} test periods, fundamentals={use_fundamentals}")

        # Split data
        df = self._prepare_data(prices_df, material)
        train_df = df.iloc[:-test_periods]
        test_df = df.iloc[-test_periods:]

        # Generate forecast on training data
        result = self.fit_predict(
            train_df.rename(columns={'price_usd': 'price_usd'}),
            material,
            use_fundamentals=use_fundamentals
        )
        forecast = result['forecast']

        # Compare forecasts to actuals
        errors = []
        for idx, row in test_df.iterrows():
            actual = row['price_usd']
            forecast_row = forecast[forecast['horizon'] == (idx - len(train_df) + 1)]

            if len(forecast_row) > 0:
                predicted = forecast_row['forecast_mean'].iloc[0]
                error = actual - predicted
                pct_error = (error / actual) * 100 if actual != 0 else 0

                errors.append({
                    'date': row['date'],
                    'actual': actual,
                    'predicted': predicted,
                    'error': error,
                    'pct_error': pct_error,
                    'abs_pct_error': abs(pct_error)
                })

        errors_df = pd.DataFrame(errors)

        # Calculate metrics
        mae = errors_df['error'].abs().mean()
        rmse = np.sqrt((errors_df['error'] ** 2).mean())
        mape = errors_df['abs_pct_error'].mean()

        # Direction accuracy (did we get the trend right?)
        if len(errors_df) > 1:
            actual_direction = (test_df['price_usd'].diff() > 0).iloc[1:]
            predicted_direction = (errors_df['predicted'].diff() > 0).iloc[1:]
            direction_accuracy = (actual_direction == predicted_direction).mean()
        else:
            direction_accuracy = None

        logger.success(f"Backtest complete: MAPE={mape:.2f}%, RMSE={rmse:.2f}, Fundamentals={use_fundamentals}")

        return {
            'errors': errors_df,
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'direction_accuracy': float(direction_accuracy) if direction_accuracy is not None else None,
            'test_periods': test_periods,
            'used_fundamentals': use_fundamentals,
            'model_type': result['model_info']['model_type']
        }

    def compare_forecast_methods(self, prices_df: pd.DataFrame, material: str) -> Dict:
        """
        Compare baseline vs enhanced forecasting methods

        Parameters
        ----------
        prices_df : pd.DataFrame
            Price data
        material : str
            Material to analyze

        Returns
        -------
        dict
            Comparison results between baseline and enhanced forecasts
        """
        logger.info(f"Comparing forecast methods for {material}")

        # Generate both types of forecasts
        baseline_result = self.fit_predict(prices_df, material, use_fundamentals=False)
        enhanced_result = self.fit_predict(prices_df, material, use_fundamentals=True)

        # Compare key metrics
        baseline_forecast = baseline_result['forecast']
        enhanced_forecast = enhanced_result['forecast']

        comparison = {
            'material': material,
            'baseline': {
                'mean_forecast': baseline_forecast['forecast_mean'].mean(),
                'forecast_range': baseline_forecast['forecast_mean'].max() - baseline_forecast['forecast_mean'].min(),
                'model_type': baseline_result['model_info']['model_type']
            },
            'enhanced': {
                'mean_forecast': enhanced_forecast['forecast_mean'].mean(),
                'forecast_range': enhanced_forecast['forecast_mean'].max() - enhanced_forecast['forecast_mean'].min(),
                'model_type': enhanced_result['model_info']['model_type'],
                'fundamental_adjustment': enhanced_forecast['fundamental_adjustment_factor'].iloc[0] if 'fundamental_adjustment_factor' in enhanced_forecast.columns else 1.0
            },
            'difference': {
                'mean_change_pct': ((enhanced_forecast['forecast_mean'].mean() - baseline_forecast['forecast_mean'].mean()) / baseline_forecast['forecast_mean'].mean()) * 100,
                'adjustment_applied': 'fundamental_adjustment_factor' in enhanced_forecast.columns
            }
        }

        # Add fundamental info if available
        if enhanced_result['fundamentals'] and 'ev_adjustment' in enhanced_result['fundamentals']:
            comparison['enhanced']['ev_adjustment'] = enhanced_result['fundamentals']['ev_adjustment']
            comparison['enhanced']['risk_score'] = enhanced_result['fundamentals']['geopolitical_risk']

        return comparison

    def save_forecast(
        self,
        forecast_result: Dict,
        output_path: str,
        include_historical: bool = False
    ):
        """
        Save forecast results to CSV

        Parameters
        ----------
        forecast_result : dict
            Output from fit_predict()
        output_path : str
            Path to save CSV file
        include_historical : bool
            Whether to include historical data
        """
        forecast_df = forecast_result['forecast']

        if include_historical:
            historical_df = forecast_result['historical'][
                ['date', 'price_usd', 'rolling_mean', 'rolling_std', 'trend']
            ]
            historical_df['is_forecast'] = False

            forecast_export = forecast_df[['date', 'forecast_mean', 'forecast_p10', 'forecast_p90']]
            if 'fundamental_adjustment_factor' in forecast_export.columns:
                forecast_export['fundamental_adjustment'] = forecast_export['fundamental_adjustment_factor']

            forecast_export.columns = ['date', 'price_usd', 'lower_bound', 'upper_bound']
            forecast_export['is_forecast'] = True

            combined = pd.concat([historical_df, forecast_export], ignore_index=True)
            combined.to_csv(output_path, index=False)
        else:
            forecast_df.to_csv(output_path, index=False)

        logger.info(f"Forecast saved to {output_path}")


# ============================================================================
# Enhanced Usage Example / Main
# ============================================================================
if __name__ == "__main__":
    import yaml
    from pathlib import Path

    # Setup logging
    logger.add("logs/forecaster_enhanced.log", rotation="10 MB")

    print("="*70)
    print("Critical Materials AI - Enhanced Baseline Forecaster")
    print("="*70)
    print()

    # Load configuration
    config_path = Path('config/config.yaml')
    if not config_path.exists():
        logger.error("Configuration file not found: config/config.yaml")
        print("❌ Configuration file not found. Run setup.sh first.")
        exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load price data
    price_file = Path('data/processed/prices_clean.csv')

    if not price_file.exists():
        # Try raw data
        price_file = Path('data/raw/prices.csv')

        if not price_file.exists():
            # Use template data for demo
            print("⚠️  No price data found. Using template data for demonstration.")
            price_file = Path('data/templates/price_template.csv')

    if not price_file.exists():
        print("❌ No price data available. Please run data fetcher first:")
        print("   python src/data_pipeline/price_fetcher.py")
        exit(1)

    print(f"📊 Loading data from: {price_file}")
    prices = pd.read_csv(price_file)
    prices['date'] = pd.to_datetime(prices['date'])
    print(f"✓ Loaded {len(prices)} price records")
    print()

    # Initialize enhanced forecaster
    forecaster = BaselineForecaster(config)

    # Get list of materials
    materials = prices['material'].unique()
    print(f"📈 Available materials: {', '.join(materials)}")
    print()

    # Forecast for each material
    for material in materials:
        print(f"\n{'='*70}")
        print(f"Forecasting: {material.upper()}")
        print(f"{'='*70}\n")

        try:
            # Compare baseline vs enhanced forecasts
            comparison = forecaster.compare_forecast_methods(prices, material)

            # Generate enhanced forecast (with fundamentals)
            result = forecaster.fit_predict(prices, material, use_fundamentals=True)

            # Display metrics
            metrics = result['metrics']
            print("📊 Current Metrics:")
            print(f"  Current Price:        ${metrics['current_price']:,.2f}")
            print(f"  Rolling Mean (12m):   ${metrics['rolling_mean']:,.2f}")
            print(f"  Volatility (CV):      {metrics['coefficient_of_variation']:.2%}")
            print(f"  Regime:               {metrics['volatility_regime']}")
            print(f"  Momentum Z-score:     {metrics['momentum_zscore']:+.2f}")
            print(f"  Trend:                {metrics['trend']}")
            print(f"  Trend Strength:       {metrics['trend_strength']:+.4f}")

            # Display fundamental adjustments if applied
            if result['fundamentals'] and not result['fundamentals'].get('fallback_to_baseline', False):
                fundamentals = result['fundamentals']
                print("\n🎯 Fundamental Adjustments Applied:")
                print(f"  EV Demand Factor:    {fundamentals['ev_adjustment'].get('adjustment_factor', 1.0):.3f}")
                print(f"  Risk Score:          {fundamentals['geopolitical_risk'].get('risk_score', 0.0):.3f}")
                print(f"  Risk Level:          {fundamentals['geopolitical_risk'].get('risk_level', 'LOW')}")
                print(f"  Recent Events:       {fundamentals['gdelt_events_count']}")

            print()

            # Display forecast
            forecast = result['forecast']
            print("🔮 Enhanced Forecast (6 months):")
            print(f"{'Month':<10} {'Mean':>12} {'P10':>12} {'P90':>12} {'Adj Factor':>12}")
            print("-" * 60)

            for _, row in forecast.iterrows():
                date_str = row['date'].strftime('%Y-%m')
                adj_factor = row.get('fundamental_adjustment_factor', 1.0)
                print(
                    f"{date_str:<10} "
                    f"${row['forecast_mean']:>11,.0f} "
                    f"${row['forecast_p10']:>11,.0f} "
                    f"${row['forecast_p90']:>11,.0f} "
                    f"{adj_factor:>11.3f}"
                )
            print()

            # Show comparison results
            print("🔄 Forecast Method Comparison:")
            diff_pct = comparison['difference']['mean_change_pct']
            if abs(diff_pct) > 1.0:
                change_dir = "↑" if diff_pct > 0 else "↓"
                print(f"  Enhanced vs Baseline: {change_dir} {abs(diff_pct):.1f}% change")
            else:
                print("  Enhanced vs Baseline: Minimal change")
            print()

            # Save forecast
            output_path = f"data/processed/forecast_enhanced_{material}.csv"
            forecaster.save_forecast(result, output_path)
            print(f"✓ Enhanced forecast saved to: {output_path}")

            # Run backtest if enough data
            if len(prices[prices['material'] == material]) >= 30:
                print("\n🧪 Running backtest with fundamentals...")
                backtest_result = forecaster.backtest(prices, material, test_periods=6, use_fundamentals=True)

                print(f"  MAE:                  ${backtest_result['mae']:,.2f}")
                print(f"  RMSE:                 ${backtest_result['rmse']:,.2f}")
                print(f"  MAPE:                 {backtest_result['mape']:.2f}%")
                if backtest_result['direction_accuracy']:
                    print(f"  Direction Accuracy:   {backtest_result['direction_accuracy']:.1%}")
                print(f"  Used Fundamentals:    {backtest_result['used_fundamentals']}")

        except Exception as e:
            logger.error(f"Error forecasting {material}: {e}")
            print(f"❌ Error: {e}")
            continue

    print("\n" + "="*70)
    print("✓ Enhanced forecasting complete!")
    print("="*70)