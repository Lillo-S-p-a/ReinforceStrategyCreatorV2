import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional

def create_price_operations_chart(steps_df: pd.DataFrame, operations: List[Dict[str, Any]], template="plotly_dark") -> Tuple[go.Figure, bool]:
    """Creates a Plotly chart showing portfolio value and trading operations."""
    fig = go.Figure()

    # 1. Add Portfolio Value Line
    if not steps_df.empty and 'portfolio_value' in steps_df.columns:
        fig.add_trace(go.Scatter(
            x=steps_df.index,
            y=steps_df['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue')
        ))
    else:
        logging.warning("Steps DataFrame is empty or missing 'portfolio_value', cannot plot main line.")

    # 2. Prepare data for markers
    marker_data = {
        'ENTRY_LONG': {'x': [], 'y': [], 'text': [], 'color': 'green', 'symbol': 'triangle-up', 'size': 10},
        'EXIT_LONG': {'x': [], 'y': [], 'text': [], 'color': 'green', 'symbol': 'x', 'size': 12},
        'ENTRY_SHORT': {'x': [], 'y': [], 'text': [], 'color': 'red', 'symbol': 'triangle-down', 'size': 10},
        'EXIT_SHORT': {'x': [], 'y': [], 'text': [], 'color': 'red', 'symbol': 'x', 'size': 12},
    }

    # Create a dictionary to quickly look up portfolio values by timestamp
    portfolio_values = {}
    if not steps_df.empty and 'portfolio_value' in steps_df.columns:
        for timestamp, row in steps_df.iterrows():
            portfolio_values[timestamp] = row['portfolio_value']

    for op in operations:
        op_type = op.get('operation_type')
        if op_type in marker_data:
            timestamp = op.get('timestamp')
            price = op.get('price')
            size = op.get('size')
            
            if timestamp is not None and price is not None:
                # Find the closest timestamp in steps_df to get the portfolio value
                portfolio_value = None
                
                # Exact match
                if timestamp in portfolio_values:
                    portfolio_value = portfolio_values[timestamp]
                else:
                    # Find the closest timestamp
                    closest_timestamp = min(portfolio_values.keys(), key=lambda x: abs(x - timestamp), default=None)
                    if closest_timestamp is not None:
                        portfolio_value = portfolio_values[closest_timestamp]
                
                # Use portfolio value for y-coordinate if available, otherwise use price
                y_value = portfolio_value if portfolio_value is not None else price
                
                marker_data[op_type]['x'].append(timestamp)
                marker_data[op_type]['y'].append(y_value)
                hover_text = f"Type: {op_type}<br>Price: {price:.2f}<br>Size: {size or 'N/A'}<br>Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                marker_data[op_type]['text'].append(hover_text)

    # 3. Add Marker Traces
    markers_plotted = False
    for op_type, data in marker_data.items():
        if data['x']:
            markers_plotted = True
            fig.add_trace(go.Scatter(
                x=data['x'],
                y=data['y'],
                mode='markers',
                marker=dict(
                    color=data['color'],
                    symbol=data['symbol'],
                    size=data['size']
                ),
                name=op_type,
                text=data['text'],
                hoverinfo='text'
            ))

    # 4. Customize Layout
    fig.update_layout(
        title=f"Portfolio Value and Trading Operations",
        xaxis_title="Time",
        yaxis_title="Value / Price",
        hovermode="x unified",
        legend_title="Trace Type",
        height=500,
        template=template
    )

    return fig, markers_plotted

def create_drawdown_chart(steps_df: pd.DataFrame, template="plotly_dark") -> Optional[go.Figure]:
    """Creates a Drawdown chart based on portfolio values."""
    if steps_df.empty or 'portfolio_value' not in steps_df.columns:
        return None
    
    # Calculate drawdowns
    portfolio_values = steps_df['portfolio_value'].values
    peak = portfolio_values[0]
    drawdowns = []
    
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown_pct = (peak - value) / peak * 100 if peak > 0 else 0
        drawdowns.append(drawdown_pct)
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=steps_df.index,
        y=drawdowns,
        fill='tozeroy',
        mode='lines',
        line=dict(color='red'),
        name='Drawdown (%)'
    ))
    
    fig.update_layout(
        title="Portfolio Drawdown Over Time",
        xaxis_title="Time",
        yaxis_title="Drawdown (%)",
        yaxis=dict(tickformat='.2f', ticksuffix='%'),
        height=400,
        template=template
    )
    
    return fig

def create_trade_analysis_charts(trades: List[Dict[str, Any]], template="plotly_dark") -> Tuple[Optional[go.Figure], Optional[go.Figure], Optional[go.Figure]]:
    """Creates trade analysis charts: PnL distribution, holding time, and cumulative PnL."""
    
    if not trades:
        return None, None, None
    
    # Convert to DataFrame for easier analysis
    trades_df = pd.DataFrame(trades)
    
    # Calculate holding time in minutes
    if 'entry_time' in trades_df and 'exit_time' in trades_df:
        trades_df['holding_time'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 60
    
    # 1. PnL Distribution Chart
    fig_pnl_dist = px.histogram(
        trades_df, 
        x="pnl", 
        title="Trade PnL Distribution",
        labels={"pnl": "Profit/Loss ($)", "count": "Number of Trades"},
        color_discrete_sequence=['teal'],
        template=template
    )
    fig_pnl_dist.update_layout(height=350)
    
    # 2. Holding Time vs PnL Chart
    fig_holding_pnl = px.scatter(
        trades_df,
        x="holding_time",
        y="pnl",
        color="direction",
        title="Holding Time vs PnL",
        labels={"holding_time": "Holding Time (minutes)", "pnl": "Profit/Loss ($)", "direction": "Trade Direction"},
        template=template
    )
    fig_holding_pnl.update_layout(height=350)
    
    # 3. Cumulative PnL Chart
    trades_df = trades_df.sort_values('exit_time')
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
    
    fig_cum_pnl = px.line(
        trades_df,
        x="exit_time",
        y="cumulative_pnl",
        title="Cumulative PnL Over Time",
        labels={"exit_time": "Time", "cumulative_pnl": "Cumulative P&L ($)"},
        template=template
    )
    fig_cum_pnl.update_layout(height=350)
    
    return fig_pnl_dist, fig_holding_pnl, fig_cum_pnl

def create_action_analysis(steps_df: pd.DataFrame, template="plotly_dark") -> Tuple[Optional[go.Figure], Optional[go.Figure]]:
    """Create action distribution and action transition charts."""
    
    if steps_df.empty or 'action' not in steps_df.columns:
        return None, None
    
    # Ensure action is numeric for calculations
    steps_df['action'] = pd.to_numeric(steps_df['action'], errors='coerce')
    
    # Action distribution
    action_counts = steps_df['action'].value_counts().reset_index()
    action_counts.columns = ['action', 'count']
    
    fig_action_dist = px.pie(
        action_counts, 
        names='action', 
        values='count',
        title='Action Distribution',
        color_discrete_sequence=px.colors.sequential.Plasma_r,
        template=template
    )
    fig_action_dist.update_layout(height=350)
    
    # Action transitions (from one action to the next)
    steps_df['next_action'] = steps_df['action'].shift(-1)
    transitions = steps_df.dropna(subset=['next_action']).groupby(['action', 'next_action']).size().reset_index()
    transitions.columns = ['From', 'To', 'Count']
    
    # Converting to strings for better display
    transitions['From'] = transitions['From'].astype(str)
    transitions['To'] = transitions['To'].astype(str)
    
    # Create Sankey diagram for transitions
    fig_transitions = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=list(set(transitions['From'].tolist() + transitions['To'].tolist())),
            color="blue"
        ),
        link=dict(
            source=[list(set(transitions['From'].tolist() + transitions['To'].tolist())).index(x) for x in transitions['From']],
            target=[list(set(transitions['From'].tolist() + transitions['To'].tolist())).index(x) for x in transitions['To']],
            value=transitions['Count'],
        )
    )])
    
    fig_transitions.update_layout(
        title_text="Action Transitions",
        height=350,
        template=template
    )
    
    return fig_action_dist, fig_transitions

def create_reward_analysis(steps_df: pd.DataFrame, template="plotly_dark") -> Tuple[Optional[go.Figure], Optional[go.Figure]]:
    """Create reward distribution and running avg reward charts."""
    
    if steps_df.empty or 'reward' not in steps_df.columns:
        return None, None
    
    # Reward distribution histogram
    fig_reward_dist = px.histogram(
        steps_df, 
        x="reward", 
        title="Reward Distribution",
        labels={"reward": "Reward", "count": "Frequency"},
        color_discrete_sequence=['goldenrod'],
        template=template
    )
    fig_reward_dist.update_layout(height=350)
    
    # Running average reward
    steps_df['reward_cumsum'] = steps_df['reward'].cumsum()
    steps_df['step_number'] = range(1, len(steps_df) + 1)
    steps_df['running_avg_reward'] = steps_df['reward_cumsum'] / steps_df['step_number']
    
    fig_running_reward = px.line(
        steps_df,
        x=steps_df.index,
        y="running_avg_reward",
        title="Running Average Reward",
        labels={"index": "Time", "running_avg_reward": "Running Avg Reward"},
        template=template
    )
    fig_running_reward.update_layout(height=350)
    
    return fig_reward_dist, fig_running_reward

def create_model_parameter_radar(model_data: Dict[str, Any], template="plotly_dark") -> go.Figure:
    """Create a radar chart visualizing model parameters."""
    
    # Extract numeric parameters
    params = {}
    max_values = {
        "learning_rate": 0.01,
        "gamma": 1.0,
        "epsilon": 1.0,
        "batch_size": 128,
        "hidden_layer_size": 256,
        "memory_size": 20000
    }
    
    # Attempt to get hyperparameters, default to empty dict if not present
    hyperparams = model_data.get('hyperparameters', {})

    for param, max_val in max_values.items():
        # Check in hyperparameters first, then top-level model_data
        value = hyperparams.get(param, model_data.get(param))

        if value is not None:
            # Ensure value is numeric and max_val is not zero
            if isinstance(value, (int, float)) and max_val != 0:
                # Normalize to 0-1 scale, handle potential negative values if needed
                normalized_value = max(0, float(value)) / float(max_val) # Ensure float division and non-negative
                params[param] = min(normalized_value, 1.0) # Cap at 1.0
            else:
                logging.warning(f"Skipping non-numeric or invalid parameter '{param}' with value '{value}' for radar chart.")
                params[param] = 0 # Default to 0 if invalid or non-numeric
        else:
             # Parameter not found in either location
             logging.warning(f"Parameter '{param}' not found in model_data or hyperparameters for radar chart.")
             params[param] = 0 # Default to 0 if not found
    
    # Create radar chart data
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[params.get(p, 0) for p in max_values.keys()],
        theta=list(max_values.keys()),
        fill='toself',
        name='Model Parameters',
        line_color='#3366cc'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Model Parameter Profile",
        height=400,
        template=template
    )
    
    return fig