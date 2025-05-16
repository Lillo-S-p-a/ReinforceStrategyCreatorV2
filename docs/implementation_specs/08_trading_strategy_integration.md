# Trading Strategy Integration: Implementation Specification

## 1. Overview

This document specifies the implementation details for the Trading Strategy Integration module of the Trading Model Optimization Pipeline. This component is responsible for connecting trained machine learning models to actual trading logic, transforming model predictions into actionable trading signals, and providing integration with various execution platforms and brokers.

## 2. Component Responsibilities

The Trading Strategy Integration module is responsible for:

- Converting model predictions into concrete trading signals
- Implementing various trading strategy frameworks and logic
- Managing risk and position sizing
- Supporting both backtesting and live trading environments
- Providing integration with brokers and exchanges
- Tracking and managing trading state
- Generating alerts and notifications
- Recording trade execution data for analysis

## 3. Architecture

### 3.1 Overall Architecture

The Trading Strategy Integration module follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────┐
│     Strategy Manager            │  High-level API for trading operations
├─────────────────────────────────┤
│                                 │
│  ┌─────────────┐ ┌───────────┐  │
│  │   Signal    │ │ Position  │  │  Core trading strategy components
│  │  Generator  │ │  Manager  │  │
│  └─────────────┘ └───────────┘  │
│                                 │
│  ┌─────────────┐ ┌───────────┐  │
│  │   Risk      │ │   Order   │  │  Risk management and order handling
│  │  Manager    │ │  Manager  │  │
│  └─────────────┘ └───────────┘  │
│                                 │
├─────────────────────────────────┤
│                                 │
│  ┌─────────────┐ ┌───────────┐  │
│  │   Broker    │ │ Exchange  │  │  Platform integration layer
│  │ Integration │ │Integration│  │
│  └─────────────┘ └───────────┘  │
│                                 │
└─────────────────────────────────┘
```

### 3.2 Directory Structure

```
trading_optimization/
└── strategy/
    ├── __init__.py
    ├── manager.py                # High-level strategy manager
    ├── signal/
    │   ├── __init__.py
    │   ├── base.py               # Base signal generator
    │   ├── model_based.py        # ML model-based signals
    │   ├── technical.py          # Technical analysis signals
    │   ├── ensemble.py           # Ensemble signal generators
    │   └── filters.py            # Signal filtering utilities
    ├── position/
    │   ├── __init__.py
    │   ├── manager.py            # Position management
    │   ├── sizing.py             # Position sizing strategies
    │   └── tracker.py            # Position tracking
    ├── risk/
    │   ├── __init__.py
    │   ├── manager.py            # Risk management
    │   ├── limits.py             # Risk limits implementation
    │   ├── exposure.py           # Exposure calculation
    │   ├── diversification.py    # Portfolio diversification
    │   └── drawdown.py           # Drawdown protection
    ├── order/
    │   ├── __init__.py
    │   ├── manager.py            # Order management
    │   ├── types.py              # Order type definitions
    │   └── router.py             # Order routing logic
    ├── execution/
    │   ├── __init__.py
    │   ├── base.py               # Base execution handler
    │   ├── simulator.py          # Paper trading executor
    │   ├── backtest.py           # Backtest execution
    │   └── live.py               # Live trading execution
    ├── broker/
    │   ├── __init__.py
    │   ├── base.py               # Base broker interface
    │   ├── alpaca.py             # Alpaca Markets integration
    │   ├── interactive.py        # Interactive Brokers integration 
    │   ├── binance.py            # Binance integration
    │   └── factory.py            # Broker factory
    ├── exchange/
    │   ├── __init__.py
    │   ├── base.py               # Base exchange interface
    │   ├── binance.py            # Binance integration
    │   ├── coinbase.py           # Coinbase integration
    │   ├── ftx.py                # FTX integration
    │   └── factory.py            # Exchange factory
    ├── state/
    │   ├── __init__.py
    │   ├── manager.py            # Trading state manager
    │   ├── persistence.py        # State persistence
    │   └── recovery.py           # State recovery
    ├── notification/
    │   ├── __init__.py
    │   ├── alert.py              # Alert generation
    │   ├── channels.py           # Notification channels
    │   └── templates.py          # Message templates
    └── utils/
        ├── __init__.py
        ├── config.py             # Strategy configuration
        ├── metrics.py            # Real-time metrics
        ├── logging.py            # Strategy-specific logging
        └── validation.py         # Configuration validation
```

## 4. Core Components Design

### 4.1 Strategy Manager

The high-level interface for managing trading strategies:

```python
# manager.py
from typing import Dict, List, Any, Optional, Union, Callable
import os
import json
import uuid
import time
import threading
from datetime import datetime
import pandas as pd
import numpy as np

from trading_optimization.strategy.signal.base import SignalGenerator
from trading_optimization.strategy.position.manager import PositionManager
from trading_optimization.strategy.risk.manager import RiskManager
from trading_optimization.strategy.order.manager import OrderManager
from trading_optimization.strategy.execution.base import ExecutionHandler
from trading_optimization.strategy.broker.factory import BrokerFactory
from trading_optimization.strategy.exchange.factory import ExchangeFactory
from trading_optimization.strategy.state.manager import StateManager
from trading_optimization.strategy.notification.alert import AlertManager
from trading_optimization.models.interface import Model
from trading_optimization.config import ConfigManager

class StrategyManager:
    """
    High-level manager for trading strategies.
    Coordinates all aspects of strategy operation including signal generation,
    risk management, order execution, and integration with brokers/exchanges.
    """
    
    def __init__(self, config: Dict[str, Any], model: Optional[Model] = None):
        """
        Initialize strategy manager with configuration and optional model.
        
        Args:
            config: Strategy configuration dictionary
            model: Optional ML model for model-based strategies
        """
        self.config = config
        self.model = model
        
        # Generate unique ID for this strategy instance
        self.strategy_id = config.get('strategy_id', f"strategy_{uuid.uuid4().hex[:8]}")
        self.strategy_name = config.get('name', self.strategy_id)
        
        # Strategy metadata
        self.created_at = datetime.now().isoformat()
        self.description = config.get('description', '')
        self.version = config.get('version', '1.0')
        
        # Create signal generator
        self.signal_generator = self._create_signal_generator()
        
        # Create position manager
        position_config = config.get('position', {})
        self.position_manager = PositionManager(position_config)
        
        # Create risk manager
        risk_config = config.get('risk', {})
        self.risk_manager = RiskManager(risk_config)
        
        # Create order manager
        order_config = config.get('order', {})
        self.order_manager = OrderManager(order_config)
        
        # Create execution handler based on mode
        self.execution_mode = config.get('execution_mode', 'backtest')
        self.execution_handler = self._create_execution_handler()
        
        # Create broker/exchange connections if in live mode
        self.broker = None
        self.exchange = None
        if self.execution_mode == 'live':
            self._setup_trading_connections()
        
        # Create state manager
        state_config = config.get('state', {})
        state_config['strategy_id'] = self.strategy_id
        self.state_manager = StateManager(state_config)
        
        # Create alert manager
        alert_config = config.get('alerts', {})
        self.alert_manager = AlertManager(alert_config)
        
        # Strategy state
        self.is_running = False
        self.last_run_time = None
        self.run_count = 0
        self.run_thread = None
        self.stop_event = threading.Event()
        
        # Performance tracking
        self.start_equity = None
        self.current_equity = None
        self.peak_equity = None
        self.trades_executed = 0
        
        # Load initial state if available
        self._load_state()
    
    def _create_signal_generator(self) -> SignalGenerator:
        """
        Create signal generator based on configuration.
        
        Returns:
            SignalGenerator instance
        """
        from trading_optimization.strategy.signal.model_based import ModelBasedSignalGenerator
        from trading_optimization.strategy.signal.technical import TechnicalSignalGenerator
        from trading_optimization.strategy.signal.ensemble import EnsembleSignalGenerator
        
        signal_config = self.config.get('signal', {})
        signal_type = signal_config.get('type', 'model_based' if self.model else 'technical')
        
        if signal_type == 'model_based' and self.model:
            return ModelBasedSignalGenerator(signal_config, self.model)
        elif signal_type == 'technical':
            return TechnicalSignalGenerator(signal_config)
        elif signal_type == 'ensemble':
            return EnsembleSignalGenerator(signal_config)
        else:
            from trading_optimization.strategy.signal.base import BaseSignalGenerator
            return BaseSignalGenerator(signal_config)
    
    def _create_execution_handler(self) -> ExecutionHandler:
        """
        Create execution handler based on execution mode.
        
        Returns:
            ExecutionHandler instance
        """
        execution_config = self.config.get('execution', {})
        
        if self.execution_mode == 'backtest':
            from trading_optimization.strategy.execution.backtest import BacktestExecutionHandler
            return BacktestExecutionHandler(execution_config)
        elif self.execution_mode == 'paper':
            from trading_optimization.strategy.execution.simulator import PaperTradingExecutionHandler
            return PaperTradingExecutionHandler(execution_config)
        elif self.execution_mode == 'live':
            from trading_optimization.strategy.execution.live import LiveExecutionHandler
            return LiveExecutionHandler(execution_config)
        else:
            from trading_optimization.strategy.execution.base import BaseExecutionHandler
            return BaseExecutionHandler(execution_config)
    
    def _setup_trading_connections(self):
        """Set up connections to brokers and exchanges for live trading."""
        # Set up broker if configured
        broker_config = self.config.get('broker', {})
        if broker_config and broker_config.get('enabled', False):
            broker_factory = BrokerFactory()
            broker_type = broker_config.get('type', 'alpaca')
            self.broker = broker_factory.create_broker(broker_type, broker_config)
        
        # Set up exchange if configured
        exchange_config = self.config.get('exchange', {})
        if exchange_config and exchange_config.get('enabled', False):
            exchange_factory = ExchangeFactory()
            exchange_type = exchange_config.get('type', 'binance')
            self.exchange = exchange_factory.create_exchange(exchange_type, exchange_config)
    
    def _load_state(self):
        """Load persisted strategy state if available."""
        try:
            state = self.state_manager.load_state()
            if state:
                # Restore position manager state
                if 'positions' in state:
                    self.position_manager.restore_state(state['positions'])
                
                # Restore performance metrics
                if 'performance' in state:
                    perf = state['performance']
                    self.start_equity = perf.get('start_equity')
                    self.current_equity = perf.get('current_equity')
                    self.peak_equity = perf.get('peak_equity')
                    self.trades_executed = perf.get('trades_executed', 0)
                    
                # Log successful state restoration
                print(f"Restored strategy state for {self.strategy_id}")
                return True
        except Exception as e:
            print(f"Error loading strategy state: {str(e)}")
        
        return False
    
    def _save_state(self):
        """Save current strategy state."""
        try:
            # Build state dictionary
            state = {
                'strategy_id': self.strategy_id,
                'last_updated': datetime.now().isoformat(),
                'positions': self.position_manager.get_state(),
                'performance': {
                    'start_equity': self.start_equity,
                    'current_equity': self.current_equity,
                    'peak_equity': self.peak_equity,
                    'trades_executed': self.trades_executed
                }
            }
            
            # Save state
            self.state_manager.save_state(state)
        except Exception as e:
            print(f"Error saving strategy state: {str(e)}")
    
    def initialize(self, initial_capital: float = None):
        """
        Initialize the strategy with initial capital.
        
        Args:
            initial_capital: Optional initial capital amount.
                If None, we'll try to get it from broker or config.
        """
        # Determine initial capital
        if initial_capital is None:
            # Try to get from broker in live mode
            if self.execution_mode == 'live' and self.broker:
                try:
                    account_info = self.broker.get_account()
                    initial_capital = account_info.get('equity', 
                                      account_info.get('cash', 10000.0))
                except Exception as e:
                    print(f"Error fetching account info: {str(e)}")
                    # Fall back to config
                    initial_capital = self.config.get('initial_capital', 10000.0)
            else:
                # Use config value
                initial_capital = self.config.get('initial_capital', 10000.0)
        
        # Initialize components
        self.position_manager.initialize(initial_capital)
        self.order_manager.initialize()
        self.execution_handler.initialize(self)
        
        # Set initial equity values
        self.start_equity = initial_capital
        self.current_equity = initial_capital
        self.peak_equity = initial_capital
        
        # Set strategy as initialized but not running
        self.is_running = False
        print(f"Strategy {self.strategy_id} initialized with ${initial_capital:,.2f}")
    
    def run_once(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute strategy once with the provided data.
        
        Args:
            data: Dictionary with market data and other inputs
            
        Returns:
            Dictionary with results of this execution
        """
        start_time = time.time()
        
        # Track run metrics
        self.last_run_time = datetime.now().isoformat()
        self.run_count += 1
        
        # Initialize results dictionary
        results = {
            'run_id': f"run_{uuid.uuid4().hex[:8]}",
            'timestamp': self.last_run_time,
            'signals': {},
            'orders': [],
            'trades': [],
            'positions': {},
            'equity': self.current_equity
        }
        
        try:
            # 1. Generate signals
            signals = self.signal_generator.generate_signals(data)
            results['signals'] = signals
            
            # 2. Apply risk management
            filtered_signals = self.risk_manager.filter_signals(
                signals, 
                self.position_manager.get_positions(),
                data
            )
            results['filtered_signals'] = filtered_signals
            
            # 3. Convert signals to orders
            orders = []
            for asset, signal in filtered_signals.items():
                if signal['strength'] != 0:
                    # Get current position
                    current_position = self.position_manager.get_position(asset)
                    
                    # Calculate position size
                    size = self.position_manager.calculate_position_size(
                        asset=asset,
                        signal=signal,
                        price=signal.get('price', data.get('prices', {}).get(asset, 0)),
                        current_position=current_position
                    )
                    
                    # Create order
                    if size != 0:
                        order = self.order_manager.create_order(
                            asset=asset,
                            order_type=signal.get('order_type', 'market'),
                            direction=signal['direction'],
                            size=size,
                            price=signal.get('price'),
                            signal_metadata=signal
                        )
                        orders.append(order)
            
            results['orders'] = orders
            
            # 4. Execute orders
            for order in orders:
                trade = self.execution_handler.execute_order(order, data)
                if trade:
                    # Update position
                    self.position_manager.update_position(trade)
                    results['trades'].append(trade)
                    self.trades_executed += 1
            
            # 5. Update equity
            new_equity = self.position_manager.calculate_equity(data)
            self.current_equity = new_equity
            if self.current_equity > self.peak_equity:
                self.peak_equity = self.current_equity
            
            results['final_equity'] = self.current_equity
            results['positions'] = self.position_manager.get_positions()
            
            # 6. Check for alerts
            alerts = self.alert_manager.check_alerts(
                positions=self.position_manager.get_positions(),
                equity=self.current_equity,
                peak_equity=self.peak_equity,
                trades=results['trades']
            )
            results['alerts'] = alerts
            
            # 7. Save state periodically
            # In backtest mode, we'll save less frequently
            save_frequency = 100 if self.execution_mode == 'backtest' else 1
            if self.run_count % save_frequency == 0:
                self._save_state()
                
        except Exception as e:
            results['error'] = str(e)
            import traceback
            results['traceback'] = traceback.format_exc()
            print(f"Error in strategy execution: {str(e)}")
            
        # Add execution time
        results['execution_time'] = time.time() - start_time
        
        return results
    
    def run_continuous(self, data_stream, interval_seconds: int = 60):
        """
        Run strategy continuously with streaming data.
        
        Args:
            data_stream: Function or generator that provides market data
            interval_seconds: Seconds between strategy runs
        """
        if self.is_running:
            print("Strategy is already running")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        def run_loop():
            while not self.stop_event.is_set():
                try:
                    # Get latest data
                    if callable(data_stream):
                        data = data_stream()
                    else:
                        data = next(data_stream)
                    
                    # Run strategy
                    result = self.run_once(data)
                    
                    # Handle any alerts
                    if result.get('alerts'):
                        for alert in result['alerts']:
                            self.alert_manager.send_alert(alert)
                    
                    # Sleep until next interval
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    print(f"Error in continuous run loop: {str(e)}")
                    # Don't exit the loop on error, just continue
                    time.sleep(interval_seconds)
            
            print(f"Strategy {self.strategy_id} stopped")
        
        # Start run thread
        self.run_thread = threading.Thread(target=run_loop)
        self.run_thread.daemon = True
        self.run_thread.start()
        print(f"Strategy {self.strategy_id} started in continuous mode")
    
    def stop(self):
        """Stop the continuous execution of the strategy."""
        if not self.is_running:
            print("Strategy is not running")
            return
        
        self.stop_event.set()
        if self.run_thread:
            self.run_thread.join(timeout=5.0)
        
        self.is_running = False
        self._save_state()
        print(f"Strategy {self.strategy_id} stopped")
    
    def get_performance(self) -> Dict[str, Any]:
        """
        Get strategy performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        performance = {
            'strategy_id': self.strategy_id,
            'strategy_name': self.strategy_name,
            'start_time': self.created_at,
            'last_run_time': self.last_run_time,
            'run_count': self.run_count,
            'trades_executed': self.trades_executed,
            'initial_equity': self.start_equity,
            'current_equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'return_pct': ((self.current_equity / self.start_equity) - 1) * 100 if self.start_equity else 0,
            'drawdown_pct': ((self.peak_equity - self.current_equity) / self.peak_equity) * 100 
                            if self.peak_equity and self.peak_equity > 0 else 0
        }
        
        # Add additional metrics from position manager
        position_metrics = self.position_manager.get_metrics()
        performance.update(position_metrics)
        
        return performance
    
    def run_backtest(
        self,
        historical_data: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run strategy in backtest mode on historical data.
        
        Args:
            historical_data: DataFrame with historical market data
            start_date: Optional start date for backtest (format: 'YYYY-MM-DD')
            end_date: Optional end date for backtest (format: 'YYYY-MM-DD')
            
        Returns:
            Dictionary with backtest results
        """
        # Make sure we're in backtest mode
        original_mode = self.execution_mode
        if self.execution_mode != 'backtest':
            self.execution_mode = 'backtest'
            self.execution_handler = self._create_execution_handler()
        
        # Filter data by date range if specified
        if isinstance(historical_data, pd.DataFrame) and 'date' in historical_data.columns:
            if start_date:
                historical_data = historical_data[historical_data['date'] >= start_date]
            if end_date:
                historical_data = historical_data[historical_data['date'] <= end_date]
        
        # Initialize backtest
        self.initialize(self.config.get('initial_capital', 10000.0))
        
        # Track progress
        total_bars = len(historical_data)
        print(f"Starting backtest with {total_bars} bars of data")
        
        # Run strategy for each data point
        all_results = []
        equity_curve = [self.start_equity]
        all_trades = []
        
        start_time = time.time()
        
        for i, (timestamp, bar) in enumerate(historical_data.iterrows()):
            # Convert bar to dictionary
            data = bar.to_dict()
            data['timestamp'] = timestamp
            
            # Add progress tracking
            if i % 100 == 0:
                progress = (i / total_bars) * 100
                print(f"Backtest progress: {progress:.1f}% ({i}/{total_bars})")
            
            # Run strategy once
            result = self.run_once(data)
            all_results.append(result)
            
            # Track equity
            equity_curve.append(result['final_equity'])
            
            # Track trades
            if result['trades']:
                all_trades.extend(result['trades'])
        
        # Convert equity curve to numpy array for calculations
        equity_array = np.array(equity_curve)
        
        # Calculate returns
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # Calculate metrics
        total_return = (equity_array[-1] / equity_array[0]) - 1
        annualized_return = ((1 + total_return) ** (252 / total_bars)) - 1  # Assuming 252 trading days per year
        
        # Calculate drawdown
        peak = np.maximum.accumulate(equity_array)
        drawdown = (peak - equity_array) / peak
        max_drawdown = np.max(drawdown)
        
        # Calculate Sharpe ratio (assuming 0 risk-free rate)
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Calculate Sortino ratio
        negative_returns = returns[returns < 0]
        sortino_ratio = np.mean(returns) / np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 and np.std(negative_returns) > 0 else 0
        
        # Calculate win rate
        win_rate = len([t for t in all_trades if t.get('pnl', 0) > 0]) / len(all_trades) if all_trades else 0
        
        # Compile backtest results
        backtest_results = {
            'start_equity': self.start_equity,
            'final_equity': self.current_equity,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'annualized_return': annualized_return,
            'annualized_return_pct': annualized_return * 100,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'total_trades': len(all_trades),
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'equity_curve': equity_array.tolist(),
            'trades': all_trades[:100],  # Limit to first 100 trades in results
            'execution_time': time.time() - start_time
        }
        
        # Restore original mode if needed
        if original_mode != 'backtest':
            self.execution_mode = original_mode
            self.execution_handler = self._create_execution_handler()
        
        return backtest_results
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert strategy to dictionary representation.
        
        Returns:
            Dictionary representation of strategy
        """
        return {
            'strategy_id': self.strategy_id,
            'strategy_name': self.strategy_name,
            'description': self.description,
            'version': self.version,
            'created_at': self.created_at,
            'execution_mode': self.execution_mode,
            'is_running': self.is_running,
            'last_run_time': self.last_run_time,
            'run_count': self.run_count,
            'model_type': self.model.__class__.__name__ if self.model else None,
            'signal_generator_type': self.signal_generator.__class__.__name__,
            'performance': self.get_performance() if self.start_equity else None
        }
```

### 4.2 Signal Generation

Base signal generator and model-based implementation:

```python
# signal/base.py
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class SignalGenerator(ABC):
    """
    Abstract base class for signal generators.
    Signal generators convert market data into trading signals.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize signal generator with configuration.
        
        Args:
            config: Signal generator configuration
        """
        self.config = config
        
        # Common parameters
        self.signal_threshold = config.get('signal_threshold', 0.0)
        self.direction_threshold = config.get('direction_threshold', 0.0)
        self.signal_smoothing = config.get('signal_smoothing', 1)
    
    @abstractmethod
    def generate_signals(self, data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Generate trading signals from market data.
        
        Args:
            data: Dictionary with market data
            
        Returns:
            Dictionary mapping assets to signal dictionaries
        """
        pass
    
    def _strength_to_direction(self, strength: float) -> int:
        """
        Convert signal strength to direction.
        
        Args:
            strength: Signal strength value
            
        Returns:
            Direction: 1 (long), -1 (short), or 0 (no position)
        """
        if strength > self.direction_threshold:
            return 1
        elif strength < -self.direction_threshold:
            return -1
        else:
            return 0


class BaseSignalGenerator(SignalGenerator):
    """
    Basic signal generator implementation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base signal generator.
        
        Args:
            config: Signal generator configuration
        """
        super().__init__(config)
    
    def generate_signals(self, data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Generate trading signals from market data.
        Base implementation returns no signals.
        
        Args:
            data: Dictionary with market data
            
        Returns:
            Empty dictionary (no signals)
        """
        return {}
```

Model-based signal generator:

```python
# signal/model_based.py
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np

from trading_optimization.strategy.signal.base import SignalGenerator
from trading_optimization.models.interface import Model

class ModelBasedSignalGenerator(SignalGenerator):
    """
    Signal generator that uses ML model predictions.
    """
    
    def __init__(self, config: Dict[str, Any], model: Model):
        """
        Initialize model-based signal generator.
        
        Args:
            config: Signal generator configuration
            model: ML model for generating predictions
        """
        super().__init__(config)
        self.model = model
        
        # Model-specific parameters
        self.prediction_threshold = config.get('prediction_threshold', 0.0)
        self.feature_columns = config.get('feature_columns', None)
        self.output_mapping = config.get('output_mapping', None)
        self.lookback = config.get('lookback', 1)
        
        # Signal parameters
        self.long_threshold = config.get('long_threshold', 0.001)
        self.short_threshold = config.get('short_threshold', -0.001)
        self.signal_scale = config.get('signal_scale', 1.0)
    
    def generate_signals(self, data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Generate trading signals from market data using model predictions.
        
        Args:
            data: Dictionary with market data
            
        Returns:
            Dictionary mapping assets to signal dictionaries
        """
        signals = {}
        
        try:
            # Prepare features for model input
            X = self._prepare_features(data)
            
            if X is None:
                return {}
            
            # Get model predictions
            predictions = self.model.predict(X)
            
            # Handle potentially different prediction structures
            if isinstance(data.get('assets'), list):
                # Multi-asset case
                for i, asset in enumerate(data['assets']):
                    if isinstance(predictions, list) and len(predictions) == len(data['assets']):
                        prediction = predictions[i]
                    elif hasattr(predictions, 'shape') and predictions.shape[0] == len(data['assets']):
                        prediction = predictions[i]
                    else:
                        # Can't map predictions to assets
                        continue
                    
                    # Map prediction to signal
                    signals[asset] = self._prediction_to_signal(
                        prediction=prediction,
                        asset=asset,
                        data=data
                    )
            else:
                # Single asset case, treat 'asset' key or use default
                asset = data.get('asset', 'default')
                prediction = predictions[0] if isinstance(predictions, list) or \
                             (hasattr(predictions, 'shape') and predictions.shape[0] > 0) \
                             else predictions
                
                signals[asset] = self._prediction_to_signal(
                    prediction=prediction,
                    asset=asset,
                    data=data
                )
        except Exception as e:
            print(f"Error generating model-based signals: {str(e)}")
        
        return signals
    
    def _prepare_features(self, data: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Prepare features for model input.
        
        Args:
            data: Dictionary with market data
            
        Returns:
            Feature array for model input or None if insufficient data
        """
        if 'features' in data:
            # Features already provided
            features = data['features']
            return features
            
        elif isinstance(data.get('ohlcv'), pd.DataFrame):
            # OHLCV data in DataFrame format
            df = data['ohlcv']
            
            # Select feature columns if specified
            if self.feature_columns:
                features = df[self.feature_columns]
            else:
                # Use all columns except known metadata
                exclude_cols = ['asset', 'timestamp', 'date', 'time']
                features = df.drop(columns=[c for c in exclude_cols if c in df.columns])
            
            # Convert to numpy array if needed
            if isinstance(features, pd.DataFrame):
                features = features.values
                
            # Handle lookback if needed
            if self.lookback > 1 and len(features) >= self.lookback:
                # Reshape to (1, lookback, features) for sequence models
                features = features[-self.lookback:].reshape(1, self.lookback, -1)
            else:
                # Reshape to (1, features) for standard models
                features = features[-1:].reshape(1, -1)
                
            return features
            
        elif 'close' in data or 'price' in data:
            # Simple price data, construct basic features
            price = data.get('close', data.get('price', 0))
            
            # If historical prices available, add basic features
            if isinstance(price, list) or isinstance(price, np.ndarray):
                prices = np.array(price)
                if len(prices) > 1:
                    # Calculate returns
                    returns = np.diff(prices) / prices[:-1]
                    
                    # Basic feature set: price and return
                    features = np.column_stack((
                        prices[-self.lookback:], 
                        np.pad(returns, (1, 0))[-self.lookback:]
                    ))
                    
                    # Reshape based on lookback
                    if self.lookback > 1:
                        features = features.reshape(1, self.lookback, 2)
                    else:
                        features = features[-1:].reshape(1, -1)
                        
                    return features
            
            # Fallback to simple feature
            return np.array([[price]])
        
        return None
    
    def _prediction_to_signal(
        self, 
        prediction: Any, 
        asset: str, 
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert model prediction to trading signal.
        
        Args:
            prediction: Model prediction
            asset: Asset identifier
            data: Original market data
            
        Returns:
            Signal dictionary
        """
        # Get asset price if available
        price = data.get('prices', {}).get(
            asset, 
            data.get('close', data.get('price', 0))
        )
        
        # Handle different prediction formats
        signal_value = 0.0
        
        if self.output_mapping:
            # Use predefined mapping
            if isinstance(prediction, (int, float)):
                # Discrete class prediction
                signal_value = self.output_mapping.get(int(prediction), 0.0)
            elif hasattr(prediction, '__len__'):
                # Probability distribution or multiple outputs
                if len(prediction) == len(self.output_mapping):
                    # Weight by probabilities
                    for i, prob in enumerate(prediction):
                        signal_value += prob * self.output_mapping.get(i, 0.0)
        else:
            # Direct prediction to signal value
            if isinstance(prediction, (int, float)):
                signal_value = float(prediction)
            elif hasattr(prediction, '__len__'):
                # Use first value if multiple outputs
                signal_value = float(prediction[0]) if len(prediction) > 0 else 0.0
        
        # Apply scaling
        signal_value *= self.signal_scale
        
        # Determine direction
        if signal_value > self.long_threshold:
            direction = 1
        elif signal_value < self.short_threshold:
            direction = -1
        else:
            direction = 0
        
        # Calculate strength (normalized between -1 and 1)
        # This can be used for position sizing
        strength = max(min(signal_value, 1.0), -1.0)
        
        # Create signal dictionary
        return {
            'asset': asset,
            'direction': direction,
            'strength': strength,
            'raw_prediction': prediction,
            'signal_value': signal_value,
            'price': price,
            'timestamp': data.get('timestamp', None),
            'metadata': {
                'model_based': True,
                'confidence': abs(strength)
            }
        }
```

### 4.3 Position Management

Position manager for handling trading positions and sizing:

```python
# position/manager.py
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime

class Position:
    """Position class representing a trading position."""
    
    def __init__(
        self, 
        asset: str, 
        size: float = 0.0,
        avg_price: float = 0.0,
        unrealized_pnl: float = 0.0,
        realized_pnl: float = 0.0,
        open_time: Optional[str] = None
    ):
        """
        Initialize a position.
        
        Args:
            asset: Asset identifier
            size: Position size (positive for long, negative for short)
            avg_price: Average entry price
            unrealized_pnl: Unrealized profit/loss
            realized_pnl: Realized profit/loss
            open_time: ISO timestamp when position was opened
        """
        self.asset = asset
        self.size = size
        self.avg_price = avg_price
        self.unrealized_pnl = unrealized_pnl
        self.realized_pnl = realized_pnl
        self.open_time = open_time or datetime.now().isoformat()
        
        # Position tracking
        self.trades = []
        self.current_value = size * avg_price if size != 0 else 0.0
        self.initial_value = self.current_value
    
    def update(
        self, 
        trade: Dict[str, Any]
    ):
        """
        Update position with a new trade.
        
        Args:
            trade: Trade dictionary
        """
        # Extract trade info
        trade_size = trade.get('size', 0.0)
        trade_price = trade.get('price', 0.0)
        trade_direction = trade.get('direction', 0)
        
        # Adjust size for direction
        if trade_direction < 0:
            trade_size = -trade_size
        
        # Calculate trade value
        trade_value = abs(trade_size) * trade_price
        
        # Handle closing or reducing position
        if (self.size > 0 and trade_size < 0) or (self.size < 0 and trade_size > 0):
            # Calculate how much of position we're closing
            closed_size = min(abs(self.size), abs(trade_size))
            if self.size < 0:
                closed_size = -closed_size
            
            # Calculate realized P&L for closed portion
            closed_value = closed_size * trade_price
            cost_basis = closed_size * self.avg_price
            realized_pnl = closed_value - cost_basis
            
            # Add to realized P&L
            self.realized_pnl += realized_pnl

        # Update position
        old_size = self.size
        old_value = self.size * self.avg_price if self.size != 0 else 0.0
        
        # Set new position size
        new_size = self.size + trade_size
        
        # Calculate new average price
        if new_size != 0:
            if old_size == 0:
                # New position, use trade price
                self.avg_price = trade_price
            elif (old_size > 0 and trade_size > 0) or (old_size < 0 and trade_size < 0):
                # Adding to position, weighted average
                self.avg_price = (old_value + trade_value) / abs(new_size)
            else:
                # Reducing or flipping position
                if abs(trade_size) > abs(old_size):
                    # Position flipped, use trade price for remainder
                    self.avg_price = trade_price
        else:
            # Position closed
            self.avg_price = 0.0
        
        # Update size
        self.size = new_size
        
        # Update current value
        self.current_value = self.size * self.avg_price if self.size != 0 else 0.0
        
        # If position was opened
        if old_size == 0 and new_size != 0:
            self.open_time = trade.get('timestamp', datetime.now().isoformat())
            self.initial_value = self.current_value
        
        # Save trade
        self.trades.append(trade)
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized P&L at current price.
        
        Args:
            current_price: Current asset price
            
        Returns:
            Unrealized P&L
        """
        if self.size == 0 or current_price == 0:
            return 0.0
            
        current_value = self.size * current_price
        cost_basis = self.size * self.avg_price
        self.unrealized_pnl = current_value - cost_basis
        
        return self.unrealized_pnl
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert position to dictionary.
        
        Returns:
            Dictionary representation of position
        """
        return {
            'asset': self.asset,
            'size': self.size,
            'direction': 1 if self.size > 0 else (-1 if self.size < 0 else 0),
            'avg_price': self.avg_price,
            'current_value': self.current_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.unrealized_pnl + self.realized_pnl,
            'open_time': self.open_time,
            'trade_count': len(self.trades)
        }


class PositionManager:
    """
    Manages trading positions and position sizing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize position manager.
        
        Args:
            config: Position manager configuration
        """
        self.config = config
        
        # Position sizing parameters
        self.default_size_pct = config.get('default_size_pct', 0.1)  # 10% of capital
        self.max_position_pct = config.get('max_position_pct', 0.25)  # 25% of capital
        self.position_sizing = config.get('position_sizing', 'fixed_pct')
        self.scale_by_signal = config.get('scale_by_signal', True)
        self.scale_by_volatility = config.get('scale_by_volatility', False)
        
        # Exposure management
        self.max_gross_exposure = config.get('max_gross_exposure', 1.0)  # 100% of capital
        self.max_net_exposure = config.get('max_net_exposure', 0.8)  # 80% of capital
        
        # Position tracking
        self.positions = {}  # asset -> Position
        self.cash = 0.0
        self.initial_capital = 0.0
    
    def initialize(self, initial_capital: float):
        """
        Initialize position manager with initial capital.
        
        Args:
            initial_capital: Initial capital amount
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
    
    def get_position(self, asset: str) -> Position:
        """
        Get position for an asset.
        
        Args:
            asset: Asset identifier
            
        Returns:
            Position object (creates new one if needed)
        """
        if asset not in self.positions:
            self.positions[asset] = Position(asset)
        
        return self.positions[asset]
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all positions.
        
        Returns:
            Dictionary mapping assets to position dictionaries
        """
        return {asset: position.to_dict() for asset, position in self.positions.items()}
    
    def calculate_position_size(
        self,
        asset: str,
        signal: Dict[str, Any],
        price: float,
        current_position: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate position size for a trade.
        
        Args:
            asset: Asset identifier
            signal: Signal dictionary
            price: Current price
            current_position: Current position (if already exists)
            
        Returns:
            Position size (absolute value)
        """
        # Default to no position change
        if price <= 0:
            return 0.0
            
        # Get signal direction and strength
        direction = signal.get('direction', 0)
        strength = signal.get('strength', 0.0)
        
        # No signal, no position change
        if direction == 0 or strength == 0:
            return 0.0
        
        # Get current position
        if current_position is None:
            position = self.get_position(asset)
            current_position = position.to_dict()
        
        current_size = current_position.get('size', 0.0)
        current_direction = 1 if current_size > 0 else (-1 if current_size < 0 else 0)
        
        # Check if we're just closing position
        if current_direction != 0 and direction != current_direction:
            # Closing or reversing position
            return abs(current_size)
        
        # Calculate size based on position sizing method
        if self.position_sizing == 'fixed_pct':
            # Fixed percentage of capital
            size_pct = self.default_size_pct
            
            # Scale by signal strength if enabled
            if self.scale_by_signal:
                size_pct *= abs(strength)
                
            # Make sure we don't exceed max position size
            size_pct = min(size_pct, self.max_position_pct)
            
            # Calculate absolute size
            position_value = size_pct * self.calculate_equity({'prices': {asset: price}})
            size = position_value / price
            
            return size
            
        elif self.position_sizing == 'kelly':
            # Simple Kelly criterion implementation
            # Assumes win rate and average win/loss ratio are available
            win_rate = self.config.get('kelly_win_rate', 0.5)
            win_loss_ratio = self.config.get('kelly_win_loss_ratio', 1.0)
            
            # Kelly fraction = win_rate - (1 - win_rate) / win_loss_ratio
            kelly_fraction = win_rate - (1 - win_rate) / win_loss_ratio
            
            # Apply half-Kelly for safety
            kelly_fraction = max(0, kelly_fraction * 0.5)
            
            # Calculate position size
            position_value = kelly_fraction * self.calculate_equity({'prices': {asset: price}})
            size = position_value / price
            
            return size
            
        elif self.position_sizing == 'volatility':
            # Volatility-based position sizing
            volatility = signal.get('metadata', {}).get('volatility', 0.01)
            risk_per_trade = self.config.get('risk_per_trade', 0.01)  # 1% risk per trade
            
            # Calculate size based on volatility and risk
            risk_amount = risk_per_trade * self.calculate_equity({'prices': {asset: price}})
            
            # Translate volatility to price movement
            # Simple approach: volatility * price = expected move
            expected_move = volatility * price
            
            # Size = risk amount / expected move
            if expected_move > 0:
                size = risk_amount / expected_move
                return size
        
        # Default fallback
        return 0.0
    
    def update_position(self, trade: Dict[str, Any]):
        """
        Update position based on executed trade.
        
        Args:
            trade: Trade dictionary
        """
        asset = trade.get('asset', '')
        position = self.get_position(asset)
        
        # Update position
        position.update(trade)
        
        # Update cash
        trade_cost = trade.get('cost', trade.get('value', 0.0))
        self.cash -= trade_cost
    
    def calculate_equity(self, data: Dict[str, Any]) -> float:
        """
        Calculate current equity (cash + position values).
        
        Args:
            data: Dictionary with market data (including latest prices)
            
        Returns:
            Current equity value
        """
        # Start with cash
        equity = self.cash
        
        # Add position values
        for asset, position in self.positions.items():
            # Skip empty positions
            if position.size == 0:
                continue
                
            # Get current price
            price = 0.0
            if 'prices' in data and asset in data['prices']:
                price = data['prices'][asset]
            elif data.get('asset') == asset:
                price = data.get('close', data.get('price', 0.0))
            elif asset == 'default':
                price = data.get('close', data.get('price', 0.0))
                
            if price > 0:
                # Calculate unrealized P&L and add to position value
                position.calculate_unrealized_pnl(price)
                position_value = position.size * price
                equity += position_value
        
        return equity
    
    def get_exposure(self) -> Dict[str, float]:
        """
        Get current exposure metrics.
        
        Returns:
            Dictionary with exposure metrics
        """
        # Calculate gross and net exposure
        long_exposure = 0.0
        short_exposure = 0.0
        
        for position in self.positions.values():
            if position.size > 0:
                long_exposure += position.current_value
            elif position.size < 0:
                short_exposure += abs(position.current_value)
        
        total_equity = self.cash + long_exposure - short_exposure
        
        # Handle division by zero
        if total_equity <= 0:
            return {
                'gross_exposure': 0.0,
                'net_exposure': 0.0,
                'long_exposure': 0.0,
                'short_exposure': 0.0,
                'long_exposure_pct': 0.0,
                'short_exposure_pct': 0.0
            }
            
        return {
            'gross_exposure': (long_exposure + short_exposure) / total_equity,
            'net_exposure': (long_exposure - short_exposure) / total_equity,
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'long_exposure_pct': long_exposure / total_equity,
            'short_exposure_pct': short_exposure / total_equity
        }
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get position manager state for persistence.
        
        Returns:
            State dictionary
        """
        return {
            'cash': self.cash,
            'initial_capital': self.initial_capital,
            'positions': {
                asset: position.to_dict() for asset, position in self.positions.items()
            }
        }
    
    def restore_state(self, state: Dict[str, Any]):
        """
        Restore position manager from persisted state.
        
        Args:
            state: State dictionary
        """
        # Restore cash and initial capital
        self.cash = state.get('cash', self.cash)
        self.initial_capital = state.get('initial_capital', self.initial_capital)
        
        # Restore positions
        positions_data = state.get('positions', {})
        for asset, position_data in positions_data.items():
            position = Position(
                asset=asset,
                size=position_data.get('size', 0.0),
                avg_price=position_data.get('avg_price', 0.0),
                unrealized_pnl=position_data.get('unrealized_pnl', 0.0),
                realized_pnl=position_data.get('realized_pnl', 0.0),
                open_time=position_data.get('open_time')
            )
            self.positions[asset] = position
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get position manager metrics.
        
        Returns:
            Dictionary with metrics
        """
        # Calculate total P&L
        total_realized = 0.0
        total_unrealized = 0.0
        
        for position in self.positions.values():
            total_realized += position.realized_pnl
            total_unrealized += position.unrealized_pnl
            
        # Get exposure metrics
        exposure = self.get_exposure()
        
        return {
            'cash': self.cash,
            'realized_pnl': total_realized,
            'unrealized_pnl': total_unrealized,
            'total_pnl': total_realized + total_unrealized,
            'open_positions': sum(1 for p in self.positions.values() if p.size != 0),
            'long_positions': sum(1 for p in self.positions.values() if p.size > 0),
            'short_positions': sum(1 for p in self.positions.values() if p.size < 0),
            'gross_exposure': exposure['gross_exposure'],
            'net_exposure': exposure['net_exposure']
        }
```

### 4.4 Risk Management

Risk management implementation:

```python
# risk/manager.py
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime

class RiskManager:
    """
    Manages trading risk, including exposure limits, drawdown protection,
    and position sizing constraints.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize risk manager.
        
        Args:
            config: Risk manager configuration
        """
        self.config = config
        
        # Risk limits
        self.max_position_size = config.get('max_position_size', float('inf'))
        self.max_gross_exposure = config.get('max_gross_exposure', 1.0)  # 100% of equity
        self.max_net_exposure = config.get('max_net_exposure', 0.8)  # 80% of equity
        
        # Drawdown protection
        self.max_drawdown_pct = config.get('max_drawdown_pct', 0.2)  # 20% drawdown limit
        self.reduce_exposure_at_drawdown = config.get('reduce_exposure_at_drawdown', 0.1)  # 10%
        self.stop_trading_at_drawdown = config.get('stop_trading_at_drawdown', 0.15)  # 15%
        
        # Loss limits
        self.max_daily_loss_pct = config.get('max_daily_loss_pct', 0.03)  # 3% daily loss limit
        self.max_trade_loss_pct = config.get('max_trade_loss_pct', 0.01)  # 1% per trade
        
        # Real-time tracking
        self.peak_equity = 0.0
        self.daily_starting_equity = 0.0
        self.daily_realized_pnl = 0.0
        self.trade_count = 0
        self.last_update = datetime.now().isoformat()
        
        # Advanced risk parameters
        self.position_correlation_limit = config.get('position_correlation_limit', 0.7)
        self.sector_exposure_limit = config.get('sector_exposure_limit', 0.3)
        self.var_limit = config.get('var_limit', 0.05)  # 5% VaR limit
        
        # Risk state tracking
        self.current_day = datetime.now().date().isoformat()
        self.risk_state = 'normal'  # normal, reduced, stopped
        self.risk_events = []
    
    def filter_signals(
        self, 
        signals: Dict[str, Dict[str, Any]], 
        positions: Dict[str, Dict[str, Any]],
        data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Filter trading signals based on risk management rules.
        
        Args:
            signals: Dictionary mapping assets to signal dictionaries
            positions: Dictionary mapping assets to position dictionaries
            data: Dictionary with market data
            
        Returns:
            Filtered signals (signals that pass risk checks)
        """
        filtered_signals = {}
        
        # Check if risk state allows trading
        if self.risk_state == 'stopped':
            return {}  # No trading allowed
            
        # Check daily loss limit
        if self._check_daily_loss_limit(data.get('equity', 0.0)):
            return {}  # Daily loss limit hit
        
        # Update drawdown state
        self._check_drawdown(data.get('equity', 0.0))
        
        # Calculate current exposure
        exposure = self._calculate_exposure(positions)
        gross_exposure = exposure.get('gross_exposure', 0.0)
        net_exposure = exposure.get('net_exposure', 0.0)
        
        # Apply exposure scaling based on risk state
        max_gross = self.max_gross_exposure
        max_net = self.max_net_exposure
        
        if self.risk_state == 'reduced':
            # Scale down exposure limits in reduced risk mode
            scale_factor = 0.5  # 50% of normal exposure
            max_gross *= scale_factor
            max_net *= scale_factor
        
        # Process each signal
        for asset, signal in signals.items():
            # Don't allow signal flipping (changing from long to short or vice versa)
            # when we're in reduced risk state
            current_position = positions.get(asset, {})
            current_direction = current_position.get('direction', 0)
            signal_direction = signal.get('direction', 0)
            
            if self.risk_state == 'reduced' and current_direction != 0 and \
               signal_direction != 0 and current_direction != signal_direction:
                # In reduced risk mode, only allow reducing positions, not flipping
                signal_copy = signal.copy()
                signal_copy['direction'] = 0  # Set to flat (close only)
                signal_copy['strength'] = 0.0
                filtered_signals[asset] = signal_copy
                continue
            
            # Check if this signal would breach exposure limits
            signal_exposure = self._estimate_signal_exposure(
                signal, 
                positions,
                data.get('equity', 0.0)
            )
            
            new_gross = gross_exposure + signal_exposure['gross_change']
            new_net = net_exposure + signal_exposure['net_change']
            
            if new_gross > max_gross:
                # Gross exposure limit would be breached
                # Allow position reduction but not increase
                if signal_exposure['gross_change'] > 0:
                    continue
            
            if abs(new_net) > max_net:
                # Net exposure limit would be breached
                # Allow position reduction but not increase in same direction
                if abs(new_net) > abs(net_exposure):
                    continue
            
            # Signal passes risk checks
            filtered_signals[asset] = signal
        
        return filtered_signals
    
    def _calculate_exposure(
        self, 
        positions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate current gross and net exposure.
        
        Args:
            positions: Dictionary mapping assets to position dictionaries
            
        Returns:
            Dictionary with exposure metrics
        """
        # Calculate exposure
        long_value = 0.0
        short_value = 0.0
        total_equity = 0.0
        
        # Get total equity value and position values
        for position in positions.values():
            size = position.get('size', 0.0)
            value = position.get('current_value', 0.0)
            
            if size > 0:
                long_value += value
            elif size < 0:
                short_value += abs(value)
        
        # Get cash component (assuming it's included in position data)
        cash = positions.get('_cash', {}).get('size', 0.0)
        total_equity = cash + long_value - short_value
        
        # Calculate exposure ratios
        if total_equity > 0:
            gross_exposure = (long_value + short_value) / total_equity
            net_exposure = (long_value - short_value) / total_equity
            long_exposure = long_value / total_equity
            short_exposure = short_value / total_equity
        else:
            # Default to max exposure if equity is zero or negative
            gross_exposure = 1.0
            net_exposure = 0.0
            long_exposure = 0.5
            short_exposure = 0.5
            
        return {
            'gross_exposure': gross_exposure,
            'net_exposure': net_exposure,
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'total_equity': total_equity,
            'long_value': long_value,
            'short_value': short_value
        }
    
    def _estimate_signal_exposure(
        self, 
        signal: Dict[str, Any], 
        positions: Dict[str, Dict[str, Any]],
        equity: float
    ) -> Dict[str, float]:
        """
        Estimate how a signal would change exposure if executed.
        
        Args:
            signal: Signal dictionary
            positions: Dictionary mapping assets to position dictionaries
            equity: Current equity value
            
        Returns:
            Dictionary with exposure change metrics
        """
        asset = signal.get('asset', '')
        direction = signal.get('direction', 0)
        strength = signal.get('strength', 0.0)
        
        # Get current position
        position = positions.get(asset, {})
        current_size = position.get('size', 0.0)
        current_direction = position.get('direction', 0)
        
        # No change in direction, no change in exposure
        if direction == current_direction:
            return {'gross_change': 0.0, 'net_change': 0.0}
        
        # Estimate size of trade
        price = signal.get('price', 1.0)  # Default to 1.0 if no price
        if price <= 0:
            price = 1.0
            
        # Simple estimation using position sizing defaults
        default_size_pct = 0.1  # 10% of equity
        estimated_size = default_size_pct * equity / price
        
        # Scale by signal strength
        estimated_size *= abs(strength)
        
        # Calculate exposure change
        gross_change = 0.0
        net_change = 0.0
        
        if current_direction == 0:
            # Opening new position
            gross_change = estimated_size * price / equity
            net_change = gross_change if direction > 0 else -gross_change
        elif direction == 0:
            # Closing position
            gross_change = -abs(current_size * price / equity)
            net_change = -current_size * price / equity
        else:
            # Flipping position
            gross_change = 0.0  # Net change in gross exposure is 0
            
            # Net exposure flips from one side to the other
            current_exposure = current_size * price / equity
            new_exposure = estimated_size * price / equity * direction
            net_change = new_exposure - current_exposure
        
        return {
            'gross_change': gross_change,
            'net_change': net_change
        }
    
    def _check_drawdown(self, current_equity: float) -> bool:
        """
        Check drawdown and update risk state if needed.
        
        Args:
            current_equity: Current equity value
            
        Returns:
            True if drawdown limits exceeded, False otherwise
        """
        # Update peak equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Calculate drawdown
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
        else:
            drawdown = 0.0
        
        # Check against limits
        if drawdown > self.max_drawdown_pct:
            # Maximum drawdown exceeded, stop trading
            if self.risk_state != 'stopped':
                self.risk_state = 'stopped'
                self._log_risk_event('max_drawdown', drawdown)
            return True
        elif drawdown > self.stop_trading_at_drawdown:
            # Stop trading drawdown exceeded
            if self.risk_state != 'stopped':
                self.risk_state = 'stopped'
                self._log_risk_event('stop_trading_drawdown', drawdown)
            return True
        elif drawdown > self.reduce_exposure_at_drawdown:
            # Reduce exposure drawdown exceeded
            if self.risk_state != 'reduced' and self.risk_state != 'stopped':
                self.risk_state = 'reduced'
                self._log_risk_event('reduce_exposure_drawdown', drawdown)
        else:
            # Drawdown within acceptable limits
            if self.risk_state != 'normal':
                # If conditions have improved, restore to normal
                self.risk_state = 'normal'
                self._log_risk_event('normal_risk', drawdown)
        
        return False
    
    def _check_daily_loss_limit(self, current_equity: float) -> bool:
        """
        Check daily loss limit and update risk state if needed.
        
        Args:
            current_equity: Current equity value
            
        Returns:
            True if daily loss limit exceeded, False otherwise
        """
        # Check if this is a new day
        today = datetime.now().date().isoformat()
        if today != self.current_day:
            # Reset daily tracking
            self.current_day = today
            self.daily_starting_equity = current_equity
            self.daily_realized_pnl = 0.0
            return False
        
        # If daily starting equity not set, set it now
        if self.daily_starting_equity <= 0:
            self.daily_starting_equity = current_equity
            return False
        
        # Calculate daily return
        daily_return = (current_equity / self.daily_starting_equity) - 1
        
        # Check against daily loss limit
        if daily_return < -self.max_daily_loss_pct:
            # Daily loss limit exceeded
            if self.risk_state != 'stopped':
                self.risk_state = 'stopped'
                self._log_risk_event('daily_loss_limit', daily_return)
            return True
        
        return False
    
    def _log_risk_event(self, event_type: str, value: float):
        """
        Log a risk management event.
        
        Args:
            event_type: Type of risk event
            value: Value associated with the event
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'value': value,
            'risk_state': self.risk_state
        }
        
        self.risk_events.append(event)
        
        # Print event for debugging
        print(f"Risk event: {event_type}, value: {value}, state: {self.risk_state}")
```

### 4.5 Order Management

Order management classes:

```python
# order/manager.py
from typing import Dict, List, Any, Optional, Union
import uuid
from datetime import datetime

class OrderManager:
    """
    Manages creation, tracking, and validation of trading orders.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize order manager.
        
        Args:
            config: Order manager configuration
        """
        self.config = config
        
        # Order type defaults
        self.default_order_type = config.get('default_order_type', 'market')
        self.default_time_in_force = config.get('default_time_in_force', 'day')
        self.use_stop_loss = config.get('use_stop_loss', False)
        self.default_stop_loss_pct = config.get('default_stop_loss_pct', 0.05)  # 5%
        self.use_take_profit = config.get('use_take_profit', False)
        self.default_take_profit_pct = config.get('default_take_profit_pct', 0.1)  # 10%
        
        # Order tracking
        self.orders = {}  # order_id -> order
        self.active_orders = {}  # order_id -> order (for open orders)
    
    def initialize(self):
        """Initialize order manager."""
        self.orders = {}
        self.active_orders = {}
    
    def create_order(
        self,
        asset: str,
        order_type: str,
        direction: int,
        size: float,
        price: Optional[float] = None,
        signal_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new order.
        
        Args:
            asset: Asset identifier
            order_type: Order type (market, limit, etc.)
            direction: Direction (1 for buy, -1 for sell)
            size: Order size
            price: Optional price for limit orders
            signal_metadata: Optional additional data from signal
            
        Returns:
            Dictionary with order details
        """
        # Generate unique order ID
        order_id = f"order_{uuid.uuid4().hex[:8]}"
        
        # Adjust size based on direction
        size = abs(size)  # Ensure size is positive
        
        # Create order object
        order = {
            'order_id': order_id,
            'asset': asset,
            'type': order_type,
            'direction': direction,
            'size': size,
            'price': price,
            'status': 'created',
            'time_in_force': self.default_time_in_force,
            'created_at': datetime.now().isoformat()
        }
        
        # Add stop loss if enabled
        if self.use_stop_loss and price and price > 0:
            stop_loss_pct = self.default_stop_loss_pct
            if signal_metadata and 'stop_loss_pct' in signal_metadata:
                stop_loss_pct = signal_metadata['stop_loss_pct']
                
            if direction > 0:
                # Long position - stop loss below entry
                stop_price = price * (1 - stop_loss_pct)
            else:
                # Short position - stop loss above entry
                stop_price = price * (1 + stop_loss_pct)
                
            order['stop_loss'] = {
                'price': stop_price,
                'pct': stop_loss_pct
            }
        
        # Add take profit if enabled
        if self.use_take_profit and price and price > 0:
            take_profit_pct = self.default_take_profit_pct
            if signal_metadata and 'take_profit_pct' in signal_metadata:
                take_profit_pct = signal_metadata['take_profit_pct']
                
            if direction > 0:
                # Long position - take profit above entry
                take_price = price * (1 + take_profit_pct)
            else:
                # Short position - take profit below entry
                take_price = price * (1 - take_profit_pct)
                
            order['take_profit'] = {
                'price': take_price,
                'pct': take_profit_pct
            }
        
        # Add signal metadata if provided
        if signal_metadata:
            order['signal_metadata'] = signal_metadata
        
        # Track order
        self.orders[order_id] = order
        self.active_orders[order_id] = order
        
        return order
    
    def cancel_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Cancel an active order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Updated order dictionary or None if not found
        """
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            order['status'] = 'cancelled'
            order['cancelled_at'] = datetime.now().isoformat()
            
            # Remove from active orders
            del self.active_orders[order_id]
            
            return order
        
        return None
    
    def update_order_status(
        self, 
        order_id: str,
        status: str,
        execution_data: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update order status after execution or rejection.
        
        Args:
            order_id: Order ID to update
            status: New status (filled, partial, rejected)
            execution_data: Optional execution details
            
        Returns:
            Updated order dictionary or None if not found
        """
        if order_id not in self.orders:
            return None
            
        order = self.orders[order_id]
        order['status'] = status
        order['updated_at'] = datetime.now().isoformat()
        
        # Add execution data if provided
        if execution_data:
            order['execution'] = execution_data
            
            # Store fill price and quantity
            if 'fill_price' in execution_data:
                order['fill_price'] = execution_data['fill_price']
            if 'fill_quantity' in execution_data:
                order['fill_quantity'] = execution_data['fill_quantity']
        
        # Remove from active orders if status is terminal
        if status in ['filled', 'rejected', 'cancelled']:
            if order_id in self.active_orders:
                del self.active_orders[order_id]
        
        return order
    
    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get order by ID.
        
        Args:
            order_id: Order ID to retrieve
            
        Returns:
            Order dictionary or None if not found
        """
        return self.orders.get(order_id)
    
    def get_active_orders(self, asset: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get all active orders, optionally filtered by asset.
        
        Args:
            asset: Optional asset to filter by
            
        Returns:
            Dictionary mapping order IDs to order dictionaries
        """
        if asset:
            return {
                order_id: order 
                for order_id, order in self.active_orders.items()
                if order['asset'] == asset
            }
        
        return dict(self.active_orders)
```

### 4.6 Execution Handling

Base execution handler and live trading implementation:

```python
# execution/base.py
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import uuid

class ExecutionHandler:
    """
    Base class for order execution handlers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize execution handler.
        
        Args:
            config: Execution handler configuration
        """
        self.config = config
    
    def initialize(self, strategy_manager: Any):
        """
        Initialize execution handler with strategy manager reference.
        
        Args:
            strategy_manager: StrategyManager instance
        """
        self.strategy_manager = strategy_manager
    
    def execute_order(
        self, 
        order: Dict[str, Any],
        data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Execute an order and return trade result.
        
        Args:
            order: Order dictionary
            data: Market data dictionary
            
        Returns:
            Trade dictionary if successful, None otherwise
        """
        # Base implementation does nothing
        return None


class BaseExecutionHandler(ExecutionHandler):
    """
    Basic implementation of execution handler for testing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base execution handler.
        
        Args:
            config: Execution handler configuration
        """
        super().__init__(config)
    
    def execute_order(
        self, 
        order: Dict[str, Any],
        data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Execute an order using a simple simulation.
        
        Args:
            order: Order dictionary
            data: Market data dictionary
            
        Returns:
            Trade dictionary if successful, None otherwise
        """
        # Extract order details
        asset = order.get('asset', '')
        direction = order.get('direction', 0)
        size = order.get('size', 0.0)
        order_type = order.get('type', 'market')
        
        # Ignore invalid orders
        if asset == '' or direction == 0 or size <= 0:
            return None
        
        # Get execution price
        price = order.get('price', 0.0)
        
        # If no price specified or this is a market order, use current price
        if price <= 0 or order_type == 'market':
            if 'prices' in data and asset in data['prices']:
                price = data['prices'][asset]
            elif data.get('asset') == asset:
                price = data.get('close', data.get('price', 0.0))
            elif asset == 'default':
                price = data.get('close', data.get('price', 0.0))
        
        # If still no valid price, can't execute
        if price <= 0:
            return None
            
        # Generate trade ID
        trade_id = f"trade_{uuid.uuid4().hex[:8]}"
        
        # Create trade result
        trade = {
            'trade_id': trade_id,
            'order_id': order.get('order_id', ''),
            'asset': asset,
            'direction': direction,
            'size': size,
            'price': price,
            'value': size * price,
            'timestamp': data.get('timestamp', datetime.now().isoformat()),
            'type': order.get('type', 'market'),
            'commission': 0.0  # No commission in base implementation
        }
        
        # Mark order as filled
        self.strategy_manager.order_manager.update_order_status(
            order_id=order.get('order_id', ''),
            status='filled',
            execution_data={
                'fill_price': price,
                'fill_quantity': size,
                'timestamp': trade['timestamp']
            }
        )
        
        return trade
```

Live execution handler:

```python
# execution/live.py
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import uuid
import time

from trading_optimization.strategy.execution.base import ExecutionHandler

class LiveExecutionHandler(ExecutionHandler):
    """
    Live execution handler for real trading.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize live execution handler.
        
        Args:
            config: Execution handler configuration
        """
        super().__init__(config)
        
        # Trading parameters
        self.max_retry_count = config.get('max_retry_count', 3)
        self.retry_delay_seconds = config.get('retry_delay_seconds', 1.0)
        self.slippage_model = config.get('slippage_model', 'fixed')
        self.fixed_slippage_bps = config.get('fixed_slippage_bps', 5)  # 0.05%
        self.commission_rate = config.get('commission_rate', 0.001)  # 0.1%
    
    def execute_order(
        self, 
        order: Dict[str, Any],
        data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Execute an order through broker or exchange.
        
        Args:
            order: Order dictionary
            data: Market data dictionary
            
        Returns:
            Trade dictionary if successful, None otherwise
        """
        # Extract order details
        asset = order.get('asset', '')
        direction = order.get('direction', 0)
        size = order.get('size', 0.0)
        order_type = order.get('type', 'market')
        
        # Ignore invalid orders
        if asset == '' or direction == 0 or size <= 0:
            return None
        
        # Get execution service
        broker = self.strategy_manager.broker
        exchange = self.strategy_manager.exchange
        
        execution_service = broker if broker is not None else exchange
        
        # If no execution service available, use basic simulation
        if execution_service is None:
            # Fall back to parent class implementation
            return super().execute_order(order, data)
            
        # Try to execute the order
        retry_count = 0
        executed_order = None
        
        while retry_count < self.max_retry_count:
            try:
                # Send order to execution service
                if order_type == 'market':
                    executed_order = execution_service.place_market_order(
                        symbol=asset,
                        side='buy' if direction > 0 else 'sell',
                        quantity=size
                    )
                elif order_type == 'limit':
                    price = order.get('price', 0.0)
                    if price <= 0:
                        # Invalid price, skip this order
                        break
                    
                    executed_order = execution_service.place_limit_order(
                        symbol=asset,
                        side='buy' if direction > 0 else 'sell',
                        quantity=size,
                        price=price
                    )
                
                # If order was placed successfully, stop retrying
                if executed_order:
                    break
                    
            except Exception as e:
                print(f"Error executing order: {str(e)}")
                
            # Increment retry counter
            retry_count += 1
            
            # Wait before retrying
            if retry_count < self.max_retry_count:
                time.sleep(self.retry_delay_seconds)
        
        # If order failed to execute, return None
        if not executed_order:
            # Update order status to rejected
            self.strategy_manager.order_manager.update_order_status(
                order_id=order.get('order_id', ''),
                status='rejected',
                execution_data={
                    'error': 'Failed to execute after retries',
                    'timestamp': datetime.now().isoformat()
                }
            )
            return None
        
        # Extract execution details
        exec_price = executed_order.get('executed_price', 
                    executed_order.get('price', order.get('price', 0.0)))
        exec_size = executed_order.get('executed_quantity', 
                    executed_order.get('quantity', size))
        
        # Calculate commission
        commission = exec_price * exec_size * self.commission_rate
        
        # Generate trade ID
        trade_id = f"trade_{uuid.uuid4().hex[:8]}"
        
        # Create trade result
        trade = {
            'trade_id': trade_id,
            'order_id': order.get('order_id', ''),
            'execution_id': executed_order.get('id', ''),
            'asset': asset,
            'direction': direction,
            'size': exec_size,
            'price': exec_price,
            'value': exec_size * exec_price,
            'commission': commission,
            'cost': (exec_size * exec_price) + commission if direction > 0 else (exec_size * exec_price) - commission,
            'timestamp': executed_order.get('timestamp', datetime.now().isoformat()),
            'type': order.get('type', 'market')
        }
        
        # Update order status
        self.strategy_manager.order_manager.update_order_status(
            order_id=order.get('order_id', ''),
            status='filled',
            execution_data={
                'fill_price': exec_price,
                'fill_quantity': exec_size,
                'commission': commission,
                'timestamp': trade['timestamp'],
                'execution_id': executed_order.get('id', ''),
                'execution_details': executed_order
            }
        )
        
        return trade
```

### 4.7 Broker Integration

Base broker interface and example implementation:

```python
# broker/base.py
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod

class Broker(ABC):
    """
    Abstract base class for broker integrations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize broker with configuration.
        
        Args:
            config: Broker configuration
        """
        self.config = config
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to broker API.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_account(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dictionary with account details
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[str, Any]:
        """
        Get current positions.
        
        Returns:
            Dictionary mapping assets to position details
        """
        pass
    
    @abstractmethod
    def place_market_order(
        self, 
        symbol: str, 
        side: str, 
        quantity: float
    ) -> Dict[str, Any]:
        """
        Place market order.
        
        Args:
            symbol: Asset symbol
            side: Order side ('buy' or 'sell')
            quantity: Order quantity
            
        Returns:
            Dictionary with order details
        """
        pass
    
    @abstractmethod
    def place_limit_order(
        self, 
        symbol: str, 
        side: str, 
        quantity: float,
        price: float
    ) -> Dict[str, Any]:
        """
        Place limit order.
        
        Args:
            symbol: Asset symbol
            side: Order side ('buy' or 'sell')
            quantity: Order quantity
            price: Limit price
            
        Returns:
            Dictionary with order details
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get order status.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Dictionary with order status
        """
        pass
```

Example broker implementation (Alpaca):

```python
# broker/alpaca.py
from typing import Dict, List, Any, Optional, Union
import os
import time
from datetime import datetime
from urllib.parse import urljoin
import requests

from trading_optimization.strategy.broker.base import Broker

class AlpacaBroker(Broker):
    """
    Integration with Alpaca Markets API.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Alpaca broker.
        
        Args:
            config: Broker configuration
        """
        super().__init__(config)
        
        # API credentials
        self.api_key = config.get('api_key', os.environ.get('ALPACA_API_KEY', ''))
        self.api_secret = config.get('api_secret', os.environ.get('ALPACA_API_SECRET', ''))
        
        # API endpoints
        self.paper_trading = config.get('paper_trading', True)
        if self.paper_trading:
            self.base_url = 'https://paper-api.alpaca.markets'
        else:
            self.base_url = 'https://api.alpaca.markets'
        
        self.api_version = 'v2'
        self.session = requests.Session()
        self.session.headers.update({
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret
        })
        
        self.account_info = None
        self.connected = False
    
    def connect(self) -> bool:
        """
        Connect to Alpaca API.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Test connection by getting account info
            account = self.get_account()
            self.account_info = account
            self.connected = True
            return True
        except Exception as e:
            print(f"Error connecting to Alpaca API: {str(e)}")
            self.connected = False
            return False
    
    def _api_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make an API request to Alpaca.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Optional request data
            
        Returns:
            API response as dictionary
        """
        url = urljoin(f"{self.base_url}/{self.api_version}/", endpoint)
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, params=data)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            error_msg = f"API error: {e}"
            try:
                error_msg += f", {response.json()}"
            except:
                pass
            raise Exception(error_msg)
        except Exception as e:
            raise Exception(f"API request failed: {str(e)}")
    
    def get_account(self) -> Dict[str, Any]:
        """
        Get account information from Alpaca.
        
        Returns:
            Dictionary with account details
        """
        return self._api_request('GET', 'account')
    
    def get_positions(self) -> Dict[str, Any]:
        """
        Get current positions from Alpaca.
        
        Returns:
            Dictionary mapping symbols to position details
        """
        positions = self._api_request('GET', 'positions')
        
        # Convert to dictionary keyed by symbol
        positions_dict = {}
        for position in positions:
            symbol = position.get('symbol')
            if symbol:
                positions_dict[symbol] = position
        
        return positions_dict
    
    def place_market_order(
        self, 
        symbol: str, 
        side: str, 
        quantity: float
    ) -> Dict[str, Any]:
        """
        Place a market order with Alpaca.
        
        Args:
            symbol: Asset symbol
            side: Order side ('buy' or 'sell')
            quantity: Order quantity
            
        Returns:
            Dictionary with order details
        """
        data = {
            'symbol': symbol,
            'qty': str(quantity),
            'side': side.lower(),
            'type': 'market',
            'time_in_force': 'day'
        }
        
        order = self._api_request('POST', 'orders', data)
        
        # Get filled details if the order was filled immediately
        if order.get('status') == 'filled':
            time.sleep(1)  # Brief delay to ensure order details are available
            order = self.get_order_status(order.get('id'))
        
        return order
    
    def place_limit_order(
        self, 
        symbol: str, 
        side: str, 
        quantity: float,
        price: float
    ) -> Dict[str, Any]:
        """
        Place a limit order with Alpaca.
        
        Args:
            symbol: Asset symbol
            side: Order side ('buy' or 'sell')
            quantity: Order quantity
            price: Limit price
            
        Returns:
            Dictionary with order details
        """
        data = {
            'symbol': symbol,
            'qty': str(quantity),
            'side': side.lower(),
            'type': 'limit',
            'limit_price': str(price),
            'time_in_force': 'day'
        }
        
        return self._api_request('POST', 'orders', data)
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order with Alpaca.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation successful, False otherwise
        """
        try:
            self._api_request('DELETE', f'orders/{order_id}')
            return True
        except Exception as e:
            print(f"Error cancelling order {order_id}: {str(e)}")
            return False
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get order status from Alpaca.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Dictionary with order status
        """
        return self._api_request('GET', f'orders/{order_id}')
```

### 4.8 Exchange Integration

Base exchange interface and example implementation:

```python
# exchange/base.py
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod

class Exchange(ABC):
    """
    Abstract base class for crypto exchange integrations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize exchange with configuration.
        
        Args:
            config: Exchange configuration
        """
        self.config = config
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to exchange API.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_balance(self) -> Dict[str, Any]:
        """
        Get account balance.
        
        Returns:
            Dictionary mapping assets to balances
        """
        pass
    
    @abstractmethod
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker information for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with ticker information
        """
        pass
    
    @abstractmethod
    def place_market_order(
        self, 
        symbol: str, 
        side: str, 
        quantity: float
    ) -> Dict[str, Any]:
        """
        Place market order.
        
        Args:
            symbol: Trading pair symbol
            side: Order side ('buy' or 'sell')
            quantity: Order quantity
            
        Returns:
            Dictionary with order details
        """
        pass
    
    @abstractmethod
    def place_limit_order(
        self, 
        symbol: str, 
        side: str, 
        quantity: float,
        price: float
    ) -> Dict[str, Any]:
        """
        Place limit order.
        
        Args:
            symbol: Trading pair symbol
            side: Order side ('buy' or 'sell')
            quantity: Order quantity
            price: Limit price
            
        Returns:
            Dictionary with order details
        """
        pass
    
    @abstractmethod
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancel order.
        
        Args:
            symbol: Trading pair symbol
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_order_status(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        Get order status.
        
        Args:
            symbol: Trading pair symbol
            order_id: Order ID to check
            
        Returns:
            Dictionary with order status
        """
        pass
```

Example exchange implementation (Binance):

```python
# exchange/binance.py
from typing import Dict, List, Any, Optional, Union
import os
import time
import hmac
import hashlib
from urllib.parse import urlencode
import requests
from datetime import datetime

from trading_optimization.strategy.exchange.base import Exchange

class BinanceExchange(Exchange):
    """
    Integration with Binance API.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Binance exchange.
        
        Args:
            config: Exchange configuration
        """
        super().__init__(config)
        
        # API credentials
        self.api_key = config.get('api_key', os.environ.get('BINANCE_API_KEY', ''))
        self.api_secret = config.get('api_secret', os.environ.get('BINANCE_API_SECRET', ''))
        
        # API endpoints
        self.use_testnet = config.get('use_testnet', True)
        if self.use_testnet:
            self.base_url = 'https://testnet.binance.vision'
        else:
            self.base_url = 'https://api.binance.com'
        
        self.session = requests.Session()
        self.session.headers.update({
            'X-MBX-APIKEY': self.api_key
        })
        
        self.connected = False
        
        # Symbol information cache
        self.symbol_info = {}
    
    def connect(self) -> bool:
        """
        Connect to Binance API and test connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Test API connection
            response = self._api_request('GET', '/api/v3/ping')
            
            # Get exchange information
            exchange_info = self._api_request('GET', '/api/v3/exchangeInfo')
            
            # Cache symbol information
            for symbol_data in exchange_info.get('symbols', []):
                symbol = symbol_data.get('symbol')
                if symbol:
                    self.symbol_info[symbol] = symbol_data
            
            self.connected = True
            return True
        except Exception as e:
            print(f"Error connecting to Binance API: {str(e)}")
            self.connected = False
            return False
    
    def _api_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict] = None,
        signed: bool = False
    ) -> Dict[str, Any]:
        """
        Make an API request to Binance.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Optional request parameters
            signed: Whether request needs signature
            
        Returns:
            API response as dictionary
        """
        url = f"{self.base_url}{endpoint}"
        
        # Copy params to avoid modifying the original
        request_params = params.copy() if params else {}
        
        # Add timestamp for signed requests
        if signed:
            request_params['timestamp'] = int(time.time() * 1000)
            
            # Create query string
            query_string = urlencode(request_params)
            
            # Create signature
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Add signature to params
            request_params['signature'] = signature
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, params=request_params)
            elif method.upper() == 'POST':
                response = self.session.post(url, params=request_params)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url, params=request_params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            error_msg = f"API error: {e}"
            try:
                error_msg += f", {response.json()}"
            except:
                pass
            raise Exception(error_msg)
        except Exception as e:
            raise Exception(f"API request failed: {str(e)}")
    
    def get_balance(self) -> Dict[str, Any]:
        """
        Get account balance from Binance.
        
        Returns:
            Dictionary mapping assets to balances
        """
        account = self._api_request('GET', '/api/v3/account', signed=True)
        
        # Convert to dictionary keyed by asset
        balances = {}
        for balance in account.get('balances', []):
            asset = balance.get('asset')
            if asset:
                balances[asset] = {
                    'free': float(balance.get('free', 0)),
                    'locked': float(balance.get('locked', 0))
                }
        
        return balances
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker information for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with ticker information
        """
        return self._api_request('GET', '/api/v3/ticker/price', params={'symbol': symbol})
    
    def place_market_order(
        self, 
        symbol: str, 
        side: str, 
        quantity: float
    ) -> Dict[str, Any]:
        """
        Place a market order with Binance.
        
        Args:
            symbol: Trading pair symbol
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity
            
        Returns:
            Dictionary with order details
        """
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': 'MARKET',
            'quantity': quantity
        }
        
        return self._api_request('POST', '/api/v3/order', params=params, signed=True)
    
    def place_limit_order(
        self, 
        symbol: str, 
        side: str, 
        quantity: float,
        price: float
    ) -> Dict[str, Any]:
        """
        Place a limit order with Binance.
        
        Args:
            symbol: Trading pair symbol
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity
            price: Limit price
            
        Returns:
            Dictionary with order details
        """
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': 'LIMIT',
            'timeInForce': 'GTC',
            'quantity': quantity,
            'price': price
        }
        
        return self._api_request('POST', '/api/v3/order', params=params, signed=True)
    
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancel an order with Binance.
        
        Args:
            symbol: Trading pair symbol
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation successful, False otherwise
        """
        try:
            self._api_request(
                'DELETE', 
                '/api/v3/order', 
                params={'symbol': symbol, 'orderId': order_id}, 
                signed=True
            )
            return True
        except Exception as e:
            print(f"Error cancelling order {order_id}: {str(e)}")
            return False
    
    def get_order_status(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        Get order status from Binance.
        
        Args:
            symbol: Trading pair symbol
            order_id: Order ID to check
            
        Returns:
            Dictionary with order status
        """
        return self._api_request(
            'GET', 
            '/api/v3/order', 
            params={'symbol': symbol, 'orderId': order_id}, 
            signed=True
        )
```

## 5. Configuration

### 5.1 Strategy Configuration Schema

```yaml
# Example strategy configuration
strategy:
  # Basic strategy information
  strategy_id: "lstm_trend_strategy"
  name: "LSTM-based Trend Strategy"
  description: "Uses LSTM model to predict price trends for trading decisions"
  version: "1.0"
  
  # Execution mode: backtest, paper, or live
  execution_mode: "backtest"  
  initial_capital: 100000.0
  
  # Signal generation configuration
  signal:
    type: "model_based"
    prediction_threshold: 0.0
    lookback: 20
    signal_smoothing: 1
    # Thresholds for generating trades
    long_threshold: 0.01   # 1% predicted increase
    short_threshold: -0.01 # 1% predicted decrease
    signal_scale: 3.0      # Scale signal strength
  
  # Position management configuration
  position:
    position_sizing: "fixed_pct"
    default_size_pct: 0.1  # 10% of capital per position
    max_position_pct: 0.25 # Maximum 25% in single position
    scale_by_signal: true  # Scale position size by signal strength
  
  # Risk management configuration
  risk:
    max_gross_exposure: 1.0  # Maximum 100% gross exposure
    max_net_exposure: 0.8    # Maximum 80% net exposure
    max_drawdown_pct: 0.2    # Stop at 20% drawdown
    reduce_exposure_at_drawdown: 0.1  # Reduce at 10% drawdown
    stop_trading_at_drawdown: 0.15    # Stop at 15% drawdown
    max_daily_loss_pct: 0.03  # 3% daily loss limit
  
  # Order configuration
  order:
    default_order_type: "market"
    default_time_in_force: "day"
    use_stop_loss: true
    default_stop_loss_pct: 0.05  # 5% stop loss
    use_take_profit: true
    default_take_profit_pct: 0.1  # 10% take profit
  
  # Execution configuration for live/paper trading
  execution:
    max_retry_count: 3
    retry_delay_seconds: 1.0
    commission_rate: 0.001  # 0.1% commission
  
  # Broker configuration for live trading
  broker:
    enabled: false
    type: "alpaca"
    api_key: ""  # Replace with actual key or use environment variable
    api_secret: ""  # Replace with actual secret or use environment variable
    paper_trading: true  # Use paper trading API
  
  # Exchange configuration for crypto trading
  exchange:
    enabled: false
    type: "binance"
    api_key: ""  # Replace with actual key or use environment variable
    api_secret: ""  # Replace with actual secret or use environment variable
    use_testnet: true  # Use testnet API
  
  # State persistence
  state:
    persistence_enabled: true
    persistence_dir: "./strategy_state"
    backup_frequency_minutes: 60
  
  # Alert configuration
  alerts:
    enabled: true
    channels: ["console", "email"]  # Alert channels to use
    email_recipients: ["user@example.com"]
    # Alert triggers
    triggers:
      drawdown_trigger: 0.1  # 10% drawdown
      profit_trigger: 0.1    # 10% profit
      trade_count_trigger: 10  # After 10 trades
```

## 6. Usage Examples

### 6.1 Basic Strategy Setup

```python
# Example of setting up a basic trading strategy

from trading_optimization.config import ConfigManager
from trading_optimization.models.manager import ModelManager
from trading_optimization.strategy.manager import StrategyManager

# Load configuration
config = ConfigManager.instance()
strategy_config = config.get('strategy', {})
model_config = config.get('models', {})

# Create model manager
model_manager = ModelManager(model_config)

# Load pre-trained model
model_id = 'lstm_price_predictor'
model = model_manager.load_model(model_id)

# Create strategy with model
strategy = StrategyManager(strategy_config, model)

# Initialize strategy
strategy.initialize(initial_capital=100000.0)

# Run strategy once with sample data
data = {
    'asset': 'BTC/USD',
    'timestamp': '2023-01-01T12:00:00Z',
    'close': 45000.0,
    'open': 44800.0,
    'high': 45200.0,
    'low': 44700.0,
    'volume': 100.5,
    'features': model_manager.prepare_features('BTC/USD')  # Get features for model
}

result = strategy.run_once(data)

# Print result
print(f"Strategy execution result:")
print(f"- Signals: {result['signals']}")
print(f"- Orders: {len(result['orders'])}")
print(f"- Trades: {len(result['trades'])}")
print(f"- Equity: ${result['equity']:.2f}")
```

### 6.2 Backtesting a Strategy

```python
# Example of backtesting a trading strategy

from trading_optimization.config import ConfigManager
from trading_optimization.models.manager import ModelManager
from trading_optimization.data.interface import DataManager
from trading_optimization.strategy.manager import StrategyManager

# Load configuration
config = ConfigManager.instance()
strategy_config = config.get('strategy', {})
model_config = config.get('models', {})
data_config = config.get('data', {})

# Create managers
model_manager = ModelManager(model_config)
data_manager = DataManager(data_config)

# Load pre-trained model
model_id, model = model_manager.load_model('lstm_price_predictor')

# Get historical data for backtesting
historical_data = data_manager.load_historical_data(
    symbol='BTC/USD',
    start_date='2022-01-01',
    end_date='2022-12-31',
    timeframe='1d'
)

# Create strategy with model
strategy_config['execution_mode'] = 'backtest'
strategy = StrategyManager(strategy_config, model)

# Run backtest
backtest_results = strategy.run_backtest(
    historical_data=historical_data,
    start_date='2022-01-01',
    end_date='2022-12-31'
)

# Print backtest summary
print("Backtest Results:")
print(f"Initial Capital: ${backtest_results['start_equity']:.2f}")
print(f"Final Equity: ${backtest_results['final_equity']:.2f}")
print(f"Total Return: {backtest_results['total_return_pct']:.2f}%")
print(f"Annualized Return: {backtest_results['annualized_return_pct']:.2f}%")
print(f"Max Drawdown: {backtest_results['max_drawdown_pct']:.2f}%")
print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
print(f"Sortino Ratio: {backtest_results['sortino_ratio']:.2f}")
print(f"Win Rate: {backtest_results['win_rate_pct']:.2f}%")
print(f"Total Trades: {backtest_results['total_trades']}")
```

### 6.3 Live Trading with Broker Integration

```python
# Example of live trading with broker integration

from trading_optimization.config import ConfigManager
from trading_optimization.models.manager import ModelManager
from trading_optimization.data.live import LiveDataStream
from trading_optimization.strategy.manager import StrategyManager

# Load configuration with API keys
config = ConfigManager.instance()
strategy_config = config.get('strategy', {})
model_config = config.get('models', {})
data_config = config.get('data', {})

# Update broker configuration for live trading
strategy_config['execution_mode'] = 'live'
strategy_config['broker']['enabled'] = True
strategy_config['broker']['paper_trading'] = True  # Use paper trading for safety

# Create model manager and load model
model_manager = ModelManager(model_config)
model_id, model = model_manager.load_model('lstm_price_predictor')

# Create live data stream
data_stream = LiveDataStream(
    symbols=['AAPL', 'MSFT', 'GOOG'],
    interval='1m',
    config=data_config
)

# Create strategy with model
strategy = StrategyManager(strategy_config, model)

# Initialize strategy
strategy.initialize()

# Start data stream
data_stream.start()

# Run strategy continuously
try:
    strategy.run_continuous(
        data_stream=data_stream.get_latest_data,
        interval_seconds=60  # Run every minute
    )
    
    # Keep main thread alive
    import time
    while True:
        # Print performance metrics every hour
        performance = strategy.get_performance()
        print(f"Strategy Performance:")
        print(f"Return: {performance['return_pct']:.2f}%")
        print(f"Drawdown: {performance['drawdown_pct']:.2f}%")
        print(f"Open Positions: {performance['open_positions']}")
        print(f"----------------------------")
        
        time.sleep(3600)  # Sleep for an hour
        
except KeyboardInterrupt:
    # Stop strategy and data stream on keyboard interrupt
    strategy.stop()
    data_stream.stop()
    print("Strategy stopped")
```

## 7. Implementation Prerequisites

Before implementing this component, ensure:

1. Project structure is set up
2. Configuration management system is implemented
3. Data management module is implemented
4. Model training module is implemented
5. Model evaluation infrastructure is implemented
6. Required libraries are installed:
   - numpy
   - pandas
   - requests
   - websocket-client (for live data streams)
   - python-binance (for Binance integration)
   - alpaca-trade-api (for Alpaca integration)

## 8. Implementation Sequence

1. Set up directory structure for strategy components
2. Implement signal generators
3. Develop position management system
4. Create risk management system
5. Build order management components
6. Implement execution handlers
7. Develop broker/exchange interfaces
8. Create notification system
9. Implement state persistence/recovery mechanisms
10. Build high-level strategy manager
11. Add comprehensive testing and debugging utilities
12. Create example notebooks and documentation

## 9. Testing Strategy

### 9.1 Unit Tests

```python
# Example unit tests for signal generators

import unittest
import numpy as np
import pandas as pd

from trading_optimization.strategy.signal.model_based import ModelBasedSignalGenerator
from trading_optimization.strategy.signal.technical import TechnicalSignalGenerator

# Simple mock model for testing
class MockModel:
    def __init__(self, prediction_value=0.02):
        self.prediction_value = prediction_value
        
    def predict(self, X):
        if isinstance(X, np.ndarray):
            return np.array([self.prediction_value] * X.shape[0])
        return np.array([self.prediction_value])

class TestSignalGenerators(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        # Create sample data
        self.data = {
            'asset': 'BTC/USD',
            'timestamp': '2023-01-01T12:00:00Z',
            'close': 45000.0,
            'open': 44800.0,
            'high': 45200.0,
            'low': 44700.0,
            'volume': 100.5,
            'features': np.array([[44800.0, 45000.0, 45200.0, 44700.0, 100.5]])
        }
        
        # Create model and configs
        self.model = MockModel(prediction_value=0.02)  # 2% positive prediction
        
        self.model_config = {
            'long_threshold': 0.01,
            'short_threshold': -0.01,
            'signal_scale': 1.0
        }
        
        self.technical_config = {
            'indicator': 'moving_average',
            'fast_period': 5,
            'slow_period': 20
        }
    
    def test_model_based_signal_generator(self):
        """Test model-based signal generator."""
        generator = ModelBasedSignalGenerator(self.model_config, self.model)
        
        # Generate signals
        signals = generator.generate_signals(self.data)
        
        # Check that we have one signal
        self.assertEqual(len(signals), 1)
        
        # Check that signals contains the correct asset
        self.assertIn('BTC/USD', signals)
        
        # Check signal direction and strength
        signal = signals['BTC/USD']
        self.assertEqual(signal['direction'], 1)  # Long signal
        self.assertGreater(signal['strength'], 0)  # Positive strength
        
        # Change model prediction to negative
        generator.model = MockModel(prediction_value=-0.02)
        
        # Generate signals again
        signals = generator.generate_signals(self.data)
        signal = signals['BTC/USD']
        
        # Check for short signal
        self.assertEqual(signal['direction'], -1)  # Short signal
        self.assertLess(signal['strength'], 0)  # Negative strength
    
    def test_technical_signal_generator(self):
        """Test technical signal generator."""
        # Needs historical data for moving averages
        historical_data = pd.DataFrame({
            'close': [44000.0, 44100.0, 44200.0, 44300.0, 44400.0, 44500.0, 44600.0, 
                     44700.0, 44800.0, 44900.0, 45000.0, 45100.0, 45200.0, 45300.0,
                     45400.0, 45500.0, 45600.0, 45700.0, 45800.0, 45900.0, 46000.0],
            'open': [43900.0] * 21,
            'high': [44100.0] * 21,
            'low': [43800.0] * 21,
            'volume': [100.0] * 21
        })
        
        # Update data with historical OHLCV
        data_with_history = dict(self.data)
        data_with_history['ohlcv'] = historical_data
        
        # Create generator
        generator = TechnicalSignalGenerator(self.technical_config)
        
        # Generate signals
        signals = generator.generate_signals(data_with_history)
        
        # Check that we have one signal
        self.assertEqual(len(signals), 1)
        
        # Check that signals contains the correct asset
        self.assertIn('BTC/USD', signals)
        
        # Since prices are consistently rising, should be a long signal
        signal = signals['BTC/USD']
        self.assertEqual(signal['direction'], 1)  # Long signal
```

### 9.2 Integration Tests

```python
# Example integration tests for strategy manager

import unittest
import os
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime

from trading_optimization.strategy.manager import StrategyManager
from trading_optimization.strategy.signal.base import BaseSignalGenerator

# Simple custom signal generator for testing
class TestSignalGenerator(BaseSignalGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.signals_to_generate = config.get('signals_to_generate', {})
    
    def generate_signals(self, data):
        # Simply return the pre-configured signals
        return self.signals_to_generate

class TestStrategyManager(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create a temp directory for state persistence
        cls.temp_dir = tempfile.mkdtemp()
        
        # Create sample historical data
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
        prices = np.linspace(100, 120, len(dates))
        
        # Add some volatility
        np.random.seed(42)
        prices += np.random.normal(0, 1, len(dates))
        
        # Create DataFrame
        cls.historical_data = pd.DataFrame({
            'date': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000, 5000, len(dates))
        })
        
        # Sample signals
        cls.test_signals = {
            'AAPL': {
                'asset': 'AAPL',
                'direction': 1,
                'strength': 0.8,
                'price': 150.0,
                'timestamp': datetime.now().isoformat()
            },
            'MSFT': {
                'asset': 'MSFT',
                'direction': -1,
                'strength': -0.6,
                'price': 250.0,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        # Remove the temp directory
        import shutil
        shutil.rmtree(cls.temp_dir)
    
    def test_strategy_initialization(self):
        """Test strategy initialization."""
        # Create basic config
        config = {
            'strategy_id': 'test_strategy',
            'name': 'Test Strategy',
            'initial_capital': 10000.0,
            'execution_mode': 'backtest',
            'signal': {
                'type': 'custom',
                'signals_to_generate': self.test_signals
            },
            'state': {
                'persistence_enabled': True,
                'persistence_dir': self.temp_dir
            }
        }
        
        # Create strategy
        strategy = StrategyManager(config)
        
        # Initialize with specific capital
        strategy.initialize(initial_capital=20000.0)
        
        # Check initialization
        self.assertEqual(strategy.strategy_id, 'test_strategy')
        self.assertEqual(strategy.strategy_name, 'Test Strategy')
        self.assertEqual(strategy.execution_mode, 'backtest')
        
        # Check that position manager is initialized with correct capital
        metrics = strategy.get_performance()
        self.assertEqual(metrics['initial_equity'], 20000.0)
    
    def test_strategy_run_once(self):
        """Test running strategy once."""
        # Create config with pre-defined signals
        config = {
            'strategy_id': 'test_strategy',
            'name': 'Test Strategy',
            'initial_capital': 10000.0,
            'execution_mode': 'backtest',
            'signal': {
                'type': 'custom',
                'signals_to_generate': self.test_signals
            },
            'position': {
                'position_sizing': 'fixed_pct',
                'default_size_pct': 0.1
            },
            'risk': {
                'max_position_size': 1000.0
            }
        }
        
        # Create custom signal generator
        from trading_optimization.strategy.signal.factory import register_signal_generator
        register_signal_generator('custom', TestSignalGenerator)
        
        # Create strategy
        strategy = StrategyManager(config)
        strategy.initialize()
        
        # Run strategy once
        data = {
            'prices': {
                'AAPL': 150.0,
                'MSFT': 250.0
            }
        }
        
        result = strategy.run_once(data)
        
        # Check that signals were generated
        self.assertEqual(len(result['signals']), 2)
        self.assertIn('AAPL', result['signals'])
        self.assertIn('MSFT', result['signals'])
        
        # Check that orders were created
        self.assertGreater(len(result['orders']), 0)
        
        # Check that trades were executed
        self.assertGreater(len(result['trades']), 0)
        
        # Check that equity was updated
        self.assertEqual(result['final_equity'], result['equity'])
    
    def test_strategy_backtest(self):
        """Test strategy backtesting."""
        # Create config
        config = {
            'strategy_id': 'test_backtest',
            'name': 'Test Backtest Strategy',
            'initial_capital': 10000.0,
            'execution_mode': 'backtest',
            'signal': {
                'type': 'technical',
                'indicator': 'moving_average',
                'fast_period': 5,
                'slow_period': 10
            },
            'position': {
                'position_sizing': 'fixed_pct',
                'default_size_pct': 0.2
            }
        }
        
        # Create strategy
        strategy = StrategyManager(config)
        
        # Run backtest
        results = strategy.run_backtest(
            historical_data=self.historical_data
        )
        
        # Check that backtest ran successfully
        self.assertIn('equity_curve', results)
        self.assertIn('total_return_pct', results)
        self.assertIn('sharpe_ratio', results)
        
        # Check that we have an equity curve
        self.assertEqual(len(results['equity_curve']), len(self.historical_data) + 1)
        
        # Check that initial and final equity values make sense
        self.assertEqual(results['start_equity'], 10000.0)
        self.assertNotEqual(results['final_equity'], 10000.0)  # Should have changed
        
        # Check that trades were generated
        self.assertGreater(results['total_trades'], 0)
```

## 10. Integration with Other Components

The Trading Strategy Integration module integrates with:

1. **Model Training Module**: To use trained models for signal generation
2. **Data Management Module**: To access market data for trading decisions
3. **Model Evaluation Infrastructure**: To share evaluation metrics with the strategy
4. **Configuration System**: To access broker credentials and other settings

Integration occurs primarily through the StrategyManager, which connects model predictions, market data, and trading infrastructure.

## 11. Extension Points

The module is designed to be easily extended:

1. **New Signal Generators**:
   - Implement custom signal generators by inheriting from SignalGenerator
   - Register them with the signal generator factory

2. **New Brokers/Exchanges**:
   - Implement new broker/exchange integrations by inheriting from Broker/Exchange
   - Register them with the respective factory class

3. **Custom Position Sizing**:
   - Add new position sizing methods to the PositionManager

4. **Risk Management Strategies**:
   - Implement additional risk management techniques in RiskManager

5. **Notification Channels**:
   - Add new notification channels to the alert system