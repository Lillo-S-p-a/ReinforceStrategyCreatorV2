# Paper Trading Integration

This module provides paper trading capabilities for the model pipeline, allowing you to test trading models in a simulated environment before deploying them to live trading.

## Overview

The paper trading integration consists of two main components:

1. **TradingSimulationEngine**: A comprehensive trading simulation engine that handles order execution, portfolio management, and performance tracking.
2. **PaperTradingDeployer**: Manages the deployment of models to the paper trading environment and coordinates simulations.

## Features

### Trading Simulation Engine

- **Order Management**
  - Support for multiple order types: Market, Limit, Stop, Stop-Limit
  - Order validation and execution with realistic slippage and commission modeling
  - Pending order management and automatic execution based on market conditions

- **Portfolio Management**
  - Real-time position tracking with P&L calculations
  - Cash management with proper accounting for trades and commissions
  - Support for both long positions and optional short selling

- **Risk Management**
  - Position size limits (configurable as percentage of capital)
  - Daily stop-loss limits to prevent excessive losses
  - Automatic order rejection when risk limits are breached

- **Performance Tracking**
  - Comprehensive metrics: Total return, Sharpe ratio, Maximum drawdown
  - Trade statistics: Win rate, profit factor, average win/loss
  - Real-time portfolio value tracking

### Paper Trading Deployer

- **Model Deployment**
  - Seamless integration with the main deployment manager
  - Dedicated paper trading environment setup
  - Configuration management for simulation parameters

- **Simulation Management**
  - Start/stop simulations with full state persistence
  - Real-time status monitoring
  - Results archiving and reporting

## Usage

### Basic Example

```python
from src.deployment import (
    DeploymentManager,
    PaperTradingDeployer,
    TradingSimulationEngine
)
from src.models.registry import ModelRegistry
from src.artifact_store.local_adapter import LocalArtifactStore

# Initialize components
artifact_store = LocalArtifactStore(storage_root="./artifacts")
model_registry = ModelRegistry(artifact_store=artifact_store)
deployment_manager = DeploymentManager(
    model_registry=model_registry,
    artifact_store=artifact_store
)

# Create paper trading deployer
paper_trading_deployer = PaperTradingDeployer(
    deployment_manager=deployment_manager,
    model_registry=model_registry,
    artifact_store=artifact_store
)

# Deploy model to paper trading
simulation_config = {
    "initial_capital": 100000.0,
    "commission_rate": 0.001,  # 0.1%
    "slippage_rate": 0.0005,   # 0.05%
    "max_position_size": 0.1,  # 10% max per position
    "daily_stop_loss": 0.02,   # 2% daily stop
    "symbols": ["AAPL", "GOOGL", "MSFT"]
}

simulation_id = paper_trading_deployer.deploy_to_paper_trading(
    model_id="my_model",
    model_version="v1.0",
    simulation_config=simulation_config
)

# Start simulation
paper_trading_deployer.start_simulation(simulation_id)

# Process market updates
market_data = {"AAPL": 150.0, "GOOGL": 2600.0, "MSFT": 300.0}
paper_trading_deployer.process_market_update(simulation_id, market_data)

# Get status
status = paper_trading_deployer.get_simulation_status(simulation_id)
print(f"Portfolio Value: ${status['portfolio_value']:,.2f}")

# Stop and get results
results = paper_trading_deployer.stop_simulation(simulation_id)
```

### Configuration Options

#### Simulation Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_capital` | float | 100000.0 | Starting capital for simulation |
| `commission_rate` | float | 0.001 | Commission rate per trade (0.001 = 0.1%) |
| `slippage_rate` | float | 0.0005 | Simulated slippage rate |
| `max_position_size` | float | 0.1 | Maximum position size as fraction of capital |
| `daily_stop_loss` | float | 0.02 | Daily stop loss limit as fraction of capital |
| `enable_shorting` | bool | False | Whether to allow short selling |
| `symbols` | List[str] | ["AAPL", "GOOGL", "MSFT"] | Symbols to trade |
| `update_frequency` | str | "1min" | Frequency of market updates |

### Order Types

The simulation engine supports various order types:

```python
from src.deployment import Order, OrderType, OrderSide

# Market order
market_order = Order(
    order_id="order_001",
    symbol="AAPL",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=100
)

# Limit order
limit_order = Order(
    order_id="order_002",
    symbol="GOOGL",
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    quantity=10,
    price=2500.0
)

# Stop order
stop_order = Order(
    order_id="order_003",
    symbol="MSFT",
    side=OrderSide.SELL,
    order_type=OrderType.STOP,
    quantity=50,
    stop_price=295.0
)
```

### Performance Metrics

The simulation tracks comprehensive performance metrics:

```python
metrics = engine.get_performance_metrics()

# Available metrics:
# - total_return: Overall return percentage
# - sharpe_ratio: Risk-adjusted return metric
# - max_drawdown: Maximum peak-to-trough decline
# - win_rate: Percentage of profitable trades
# - profit_factor: Ratio of gross profit to gross loss
# - total_trades: Number of completed trades
# - winning_trades: Number of profitable trades
# - losing_trades: Number of losing trades
# - avg_win: Average profit on winning trades
# - avg_loss: Average loss on losing trades
```

## Integration with Model Pipeline

The paper trading module integrates seamlessly with the model pipeline:

1. **Model Training**: Train your model using the training engine
2. **Model Evaluation**: Evaluate model performance using historical data
3. **Paper Trading**: Deploy to paper trading for real-time simulation
4. **Production Deployment**: After successful paper trading, deploy to production

## Best Practices

1. **Realistic Configuration**: Set commission and slippage rates that match your actual trading environment
2. **Risk Management**: Always configure appropriate position size and stop-loss limits
3. **Sufficient Testing**: Run paper trading for an adequate period to assess model performance
4. **Market Conditions**: Test during various market conditions (trending, ranging, volatile)
5. **Performance Review**: Regularly review metrics and adjust strategy as needed

## Extending the Module

### Custom Order Execution

You can extend the order execution logic by subclassing `TradingSimulationEngine`:

```python
class CustomTradingEngine(TradingSimulationEngine):
    def _should_fill_order(self, order: Order, current_price: float) -> bool:
        # Add custom logic for order fills
        if order.order_type == OrderType.MARKET:
            # Custom market order logic
            return self._check_liquidity(order, current_price)
        return super()._should_fill_order(order, current_price)
```

### Custom Risk Management

Add custom risk management rules:

```python
class RiskManagedEngine(TradingSimulationEngine):
    def _validate_order(self, order: Order) -> None:
        super()._validate_order(order)
        
        # Add custom validation
        if self._check_correlation_risk(order):
            raise ValueError("Order violates correlation risk limits")
```

## Troubleshooting

### Common Issues

1. **Order Rejection**: Check risk limits and available capital
2. **No Trades Executed**: Verify model signals and order parameters
3. **Performance Issues**: Ensure market data updates are not too frequent

### Logging

Enable detailed logging for debugging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('src.deployment.paper_trading')
```

## Future Enhancements

- Real-time market data integration
- Advanced order types (OCO, bracket orders)
- Multi-asset portfolio optimization
- Integration with live brokers for seamless transition
- Advanced risk metrics (VaR, CVaR)
- Backtesting comparison tools