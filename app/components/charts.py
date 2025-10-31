import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def create_price_trend_chart(prices_df, material=None):
    """Create a price trend chart for specific material or all materials"""

    if material:
        data = prices_df[prices_df['material'] == material]
        title = f"{material.title()} Price Trend"
    else:
        data = prices_df
        title = "Multi-Material Price Trends"

    fig = px.line(data, x='date', y='price', color='material', title=title)
    return fig

def create_source_breakdown_chart(prices_df):
    """Create a chart showing data source breakdown"""

    if 'source' in prices_df.columns:
        source_counts = prices_df['source'].value_counts()
        fig = px.pie(values=source_counts.values, names=source_counts.index,
                    title="Data Source Distribution")
        return fig
    return None

def create_forecast_chart(historical_data, forecast_data):
    """Create a forecast visualization with confidence intervals"""

    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_data['date'],
        y=historical_data['price'],
        name='Historical',
        line=dict(color='blue', width=2)
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_data['date'],
        y=forecast_data['forecast_mean'],
        name='Forecast',
        line=dict(color='orange', width=3, dash='dash')
    ))

    # Confidence interval
    if 'forecast_p10' in forecast_data.columns and 'forecast_p90' in forecast_data.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_data['date'], forecast_data['date'][::-1]]),
            y=pd.concat([forecast_data['forecast_p10'], forecast_data['forecast_p90'][::-1]]),
            fill='toself',
            fillcolor='rgba(255,165,0,0.2)',
            line=dict(color='rgba(255,165,0,0)'),
            name='80% Confidence Interval'
        ))

    fig.update_layout(
        title="Price Forecast with Confidence Intervals",
        xaxis_title="Date",
        yaxis_title="Price (USD/t)",
        hovermode='x unified'
    )

    return fig