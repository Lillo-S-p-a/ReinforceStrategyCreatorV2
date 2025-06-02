"""Paper trading deployment and simulation components."""

import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from collections import defaultdict

from ..models.base import ModelBase
from ..models.registry import ModelRegistry
from ..artifact_store.base import ArtifactStore, ArtifactType
from .manager import DeploymentManager, DeploymentStatus


class OrderType(Enum):
    """Types of trading orders."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Side of trading order."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Status of trading order."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    quantity: float
    average_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    def update_price(self, new_price: float) -> None:
        """Update position with new market price."""
        self.current_price = new_price
        self.unrealized_pnl = (new_price - self.average_price) * self.quantity


class TradingSimulationEngine:
    """Simulates trading execution and portfolio management.
    
    This engine handles:
    - Order execution with simulated fills
    - Portfolio state management
    - Performance tracking
    - Risk management rules
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,  # 0.1% per trade
        slippage_rate: float = 0.0005,  # 0.05% slippage
        max_position_size: float = 0.1,  # Max 10% of capital per position
        daily_stop_loss: float = 0.02,  # 2% daily stop loss
        enable_shorting: bool = False
    ):
        """Initialize the trading simulation engine.
        
        Args:
            initial_capital: Starting capital for simulation
            commission_rate: Commission rate per trade
            slippage_rate: Simulated slippage rate
            max_position_size: Maximum position size as fraction of capital
            daily_stop_loss: Daily stop loss as fraction of capital
            enable_shorting: Whether to allow short selling
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_position_size = max_position_size
        self.daily_stop_loss = daily_stop_loss
        self.enable_shorting = enable_shorting
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        
        # Performance tracking
        self.portfolio_values: List[Tuple[datetime, float]] = [(datetime.now(), initial_capital)]
        self.daily_pnl: List[Tuple[datetime, float]] = []
        self.trades: List[Dict[str, Any]] = []
        
        # Risk management state
        self.daily_start_value = initial_capital
        self.daily_loss = 0.0
        self.risk_limit_hit = False
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def submit_order(self, order: Order) -> str:
        """Submit an order for execution.
        
        Args:
            order: Order to submit
            
        Returns:
            Order ID
            
        Raises:
            ValueError: If order validation fails
        """
        # Validate order
        self._validate_order(order)
        
        # Check risk limits
        if self.risk_limit_hit:
            order.status = OrderStatus.REJECTED
            self.logger.warning(f"Order {order.order_id} rejected due to risk limit")
            return order.order_id
        
        # Add to pending orders
        self.orders[order.order_id] = order
        self.logger.info(f"Order submitted: {order.order_id}")
        
        return order.order_id
    
    def process_market_data(self, market_data: Dict[str, float]) -> None:
        """Process market data and execute pending orders.
        
        Args:
            market_data: Dictionary of symbol -> price
        """
        # Update position prices
        for symbol, position in self.positions.items():
            if symbol in market_data:
                position.update_price(market_data[symbol])
        
        # Process pending orders
        filled_orders = []
        for order_id, order in list(self.orders.items()):
            if order.status == OrderStatus.PENDING and order.symbol in market_data:
                current_price = market_data[order.symbol]
                
                # Check if order should be filled
                if self._should_fill_order(order, current_price):
                    self._execute_order(order, current_price)
                    filled_orders.append(order_id)
        
        # Remove filled orders from pending
        for order_id in filled_orders:
            self.order_history.append(self.orders.pop(order_id))
        
        # Update portfolio value
        portfolio_value = self.get_portfolio_value(market_data)
        self.portfolio_values.append((datetime.now(), portfolio_value))
        
        # Check daily stop loss
        self._check_daily_stop_loss(portfolio_value)
    
    def get_portfolio_value(self, market_data: Dict[str, float]) -> float:
        """Calculate total portfolio value.
        
        Args:
            market_data: Current market prices
            
        Returns:
            Total portfolio value
        """
        positions_value = sum(
            position.quantity * market_data.get(position.symbol, position.current_price)
            for position in self.positions.values()
        )
        return self.cash + positions_value
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        return self.positions.copy()
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if len(self.portfolio_values) < 2:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0
            }
        
        # Calculate returns
        values = np.array([v[1] for v in self.portfolio_values])
        returns = np.diff(values) / values[:-1]
        
        # Total return
        total_return = (values[-1] - values[0]) / values[0]
        
        # Sharpe ratio (assuming 252 trading days)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
        
        # Trade statistics
        winning_trades = [t for t in self.trades if t["pnl"] > 0]
        losing_trades = [t for t in self.trades if t["pnl"] < 0]
        
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0.0
        
        # Profit factor
        gross_profit = sum(t["pnl"] for t in winning_trades)
        gross_loss = abs(sum(t["pnl"] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": len(self.trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "avg_win": np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0.0,
            "avg_loss": np.mean([t["pnl"] for t in losing_trades]) if losing_trades else 0.0
        }
    
    def reset_daily_limits(self) -> None:
        """Reset daily risk limits."""
        if self.portfolio_values:
            self.daily_start_value = self.portfolio_values[-1][1]
        self.daily_loss = 0.0
        self.risk_limit_hit = False
        self.logger.info("Daily risk limits reset")
    
    def _validate_order(self, order: Order) -> None:
        """Validate order parameters."""
        # Check position size limit
        order_value = order.quantity * (order.price or 0)
        if order_value > self.cash * self.max_position_size:
            raise ValueError(f"Order size exceeds maximum position size limit")
        
        # Check if shorting is allowed
        if not self.enable_shorting and order.side == OrderSide.SELL:
            position = self.positions.get(order.symbol)
            if not position or position.quantity < order.quantity:
                raise ValueError("Short selling not allowed")
        
        # Validate order type parameters
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and order.price is None:
            raise ValueError(f"{order.order_type.value} order requires price")
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price is None:
            raise ValueError(f"{order.order_type.value} order requires stop price")
    
    def _should_fill_order(self, order: Order, current_price: float) -> bool:
        """Check if order should be filled at current price."""
        if order.order_type == OrderType.MARKET:
            return True
        
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY:
                return current_price <= order.price
            else:
                return current_price >= order.price
        
        elif order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY:
                return current_price >= order.stop_price
            else:
                return current_price <= order.stop_price
        
        elif order.order_type == OrderType.STOP_LIMIT:
            # Stop condition must be met first
            if order.side == OrderSide.BUY:
                if current_price >= order.stop_price:
                    return current_price <= order.price
            else:
                if current_price <= order.stop_price:
                    return current_price >= order.price
        
        return False
    
    def _execute_order(self, order: Order, market_price: float) -> None:
        """Execute an order at market price."""
        # Calculate fill price with slippage
        slippage_multiplier = 1 + self.slippage_rate if order.side == OrderSide.BUY else 1 - self.slippage_rate
        fill_price = market_price * slippage_multiplier
        
        # Calculate commission
        commission = order.quantity * fill_price * self.commission_rate
        
        # Update order
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_fill_price = fill_price
        order.filled_at = datetime.now()
        order.commission = commission
        order.slippage = abs(fill_price - market_price) * order.quantity
        
        # Update cash
        if order.side == OrderSide.BUY:
            self.cash -= (order.quantity * fill_price + commission)
        else:
            self.cash += (order.quantity * fill_price - commission)
        
        # Update positions
        self._update_position(order)
        
        # Record trade
        self.trades.append({
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "fill_price": fill_price,
            "commission": commission,
            "timestamp": order.filled_at,
            "pnl": 0.0  # Will be calculated when position is closed
        })
        
        self.logger.info(
            f"Order {order.order_id} filled: {order.side.value} {order.quantity} "
            f"{order.symbol} @ {fill_price:.2f}"
        )
    
    def _update_position(self, order: Order) -> None:
        """Update position based on filled order."""
        symbol = order.symbol
        
        if symbol not in self.positions:
            # New position
            if order.side == OrderSide.BUY:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=order.filled_quantity,
                    average_price=order.average_fill_price,
                    current_price=order.average_fill_price
                )
        else:
            position = self.positions[symbol]
            
            if order.side == OrderSide.BUY:
                # Adding to position
                total_cost = (position.quantity * position.average_price + 
                             order.filled_quantity * order.average_fill_price)
                position.quantity += order.filled_quantity
                position.average_price = total_cost / position.quantity
            else:
                # Reducing position
                if order.filled_quantity >= position.quantity:
                    # Position closed
                    realized_pnl = (order.average_fill_price - position.average_price) * position.quantity
                    position.realized_pnl += realized_pnl
                    
                    # Update last trade with PnL
                    if self.trades:
                        self.trades[-1]["pnl"] = realized_pnl
                    
                    # Remove position
                    del self.positions[symbol]
                else:
                    # Partial close
                    realized_pnl = (order.average_fill_price - position.average_price) * order.filled_quantity
                    position.realized_pnl += realized_pnl
                    position.quantity -= order.filled_quantity
                    
                    # Update last trade with PnL
                    if self.trades:
                        self.trades[-1]["pnl"] = realized_pnl
    
    def _check_daily_stop_loss(self, current_value: float) -> None:
        """Check if daily stop loss has been hit."""
        daily_loss = (self.daily_start_value - current_value) / self.daily_start_value
        
        if daily_loss >= self.daily_stop_loss:
            self.risk_limit_hit = True
            self.logger.warning(
                f"Daily stop loss hit: {daily_loss:.2%} loss "
                f"(limit: {self.daily_stop_loss:.2%})"
            )
            
            # Cancel all pending orders
            for order in self.orders.values():
                if order.status == OrderStatus.PENDING:
                    order.status = OrderStatus.CANCELLED
                    self.logger.info(f"Order {order.order_id} cancelled due to risk limit")


class PaperTradingDeployer:
    """Handles deployment of models to paper trading environment.
    
    This class manages:
    - Setting up paper trading simulations
    - Loading and running deployed models
    - Tracking paper trading performance
    - Integration with the deployment manager
    """
    
    def __init__(
        self,
        deployment_manager: DeploymentManager,
        model_registry: ModelRegistry,
        artifact_store: ArtifactStore,
        paper_trading_root: Optional[Union[str, Path]] = None
    ):
        """Initialize the paper trading deployer.
        
        Args:
            deployment_manager: Main deployment manager
            model_registry: Registry for accessing models
            artifact_store: Store for artifacts
            paper_trading_root: Root directory for paper trading (default: ./paper_trading)
        """
        self.deployment_manager = deployment_manager
        self.model_registry = model_registry
        self.artifact_store = artifact_store
        
        # Set paper trading root directory
        if paper_trading_root is None:
            paper_trading_root = Path.cwd() / "paper_trading"
        self.paper_trading_root = Path(paper_trading_root)
        self.paper_trading_root.mkdir(parents=True, exist_ok=True)
        
        # Active simulations
        self.active_simulations: Dict[str, Dict[str, Any]] = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def deploy_to_paper_trading(
        self,
        model_id: str,
        model_version: Optional[str] = None,
        simulation_config: Optional[Dict[str, Any]] = None,
        deployment_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Deploy a model to paper trading environment.
        
        Args:
            model_id: ID of the model to deploy
            model_version: Specific model version
            simulation_config: Configuration for the simulation
            deployment_config: Deployment configuration
            
        Returns:
            Paper trading deployment ID
        """
        # Default simulation config
        if simulation_config is None:
            simulation_config = {}
        
        sim_config = {
            "initial_capital": simulation_config.get("initial_capital", 100000.0),
            "commission_rate": simulation_config.get("commission_rate", 0.001),
            "slippage_rate": simulation_config.get("slippage_rate", 0.0005),
            "max_position_size": simulation_config.get("max_position_size", 0.1),
            "daily_stop_loss": simulation_config.get("daily_stop_loss", 0.02),
            "enable_shorting": simulation_config.get("enable_shorting", False),
            "data_source": simulation_config.get("data_source", "simulated"),
            "symbols": simulation_config.get("symbols", ["AAPL", "GOOGL", "MSFT"]),
            "update_frequency": simulation_config.get("update_frequency", "1min")
        }
        
        # Deploy using deployment manager
        deployment_id = self.deployment_manager.deploy(
            model_id=model_id,
            target_environment="paper_trading",
            model_version=model_version,
            deployment_config=deployment_config,
            strategy="direct"
        )
        
        # Get deployment info
        deployment_info = self.deployment_manager.get_deployment_status(deployment_id)
        
        # Initialize simulation
        simulation_id = f"sim_{deployment_id}"
        simulation = self._initialize_simulation(
            simulation_id=simulation_id,
            deployment_info=deployment_info,
            simulation_config=sim_config
        )
        
        # Store active simulation
        self.active_simulations[simulation_id] = simulation
        
        self.logger.info(
            f"Model {model_id} deployed to paper trading "
            f"(deployment_id: {deployment_id}, simulation_id: {simulation_id})"
        )
        
        return simulation_id
    
    def start_simulation(self, simulation_id: str) -> None:
        """Start a paper trading simulation.
        
        Args:
            simulation_id: ID of the simulation to start
        """
        if simulation_id not in self.active_simulations:
            raise ValueError(f"Simulation {simulation_id} not found")
        
        simulation = self.active_simulations[simulation_id]
        simulation["status"] = "running"
        simulation["started_at"] = datetime.now()
        
        self.logger.info(f"Started paper trading simulation {simulation_id}")
    
    def stop_simulation(self, simulation_id: str) -> Dict[str, Any]:
        """Stop a paper trading simulation and return results.
        
        Args:
            simulation_id: ID of the simulation to stop
            
        Returns:
            Simulation results including performance metrics
        """
        if simulation_id not in self.active_simulations:
            raise ValueError(f"Simulation {simulation_id} not found")
        
        simulation = self.active_simulations[simulation_id]
        simulation["status"] = "stopped"
        simulation["stopped_at"] = datetime.now()
        
        # Get final performance metrics
        engine = simulation["engine"]
        metrics = engine.get_performance_metrics()
        
        # Save results
        results = {
            "simulation_id": simulation_id,
            "deployment_id": simulation["deployment_id"],
            "model_id": simulation["model_id"],
            "started_at": simulation["started_at"].isoformat(),
            "stopped_at": simulation["stopped_at"].isoformat(),
            "duration_hours": (simulation["stopped_at"] - simulation["started_at"]).total_seconds() / 3600,
            "performance_metrics": metrics,
            "final_portfolio_value": engine.portfolio_values[-1][1] if engine.portfolio_values else 0,
            "total_trades": len(engine.trades),
            "configuration": simulation["config"]
        }
        
        # Save results to file
        results_file = self.paper_trading_root / f"{simulation_id}_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Save to artifact store
        artifact_id = self.artifact_store.save_artifact(
            artifact_path=results_file,
            artifact_type=ArtifactType.REPORT,
            metadata={
                "simulation_id": simulation_id,
                "model_id": simulation["model_id"],
                "type": "paper_trading_results"
            },
            tags=["paper_trading", simulation["model_id"]]
        )
        
        self.logger.info(
            f"Stopped paper trading simulation {simulation_id}. "
            f"Results saved to {results_file} (artifact_id: {artifact_id})"
        )
        
        return results
    
    def get_simulation_status(self, simulation_id: str) -> Dict[str, Any]:
        """Get the status of a paper trading simulation.
        
        Args:
            simulation_id: ID of the simulation
            
        Returns:
            Simulation status information
        """
        if simulation_id not in self.active_simulations:
            raise ValueError(f"Simulation {simulation_id} not found")
        
        simulation = self.active_simulations[simulation_id]
        engine = simulation["engine"]
        
        # Get current metrics
        metrics = engine.get_performance_metrics()
        
        return {
            "simulation_id": simulation_id,
            "status": simulation["status"],
            "model_id": simulation["model_id"],
            "started_at": simulation.get("started_at"),
            "current_metrics": metrics,
            "portfolio_value": engine.portfolio_values[-1][1] if engine.portfolio_values else 0,
            "positions": {
                symbol: {
                    "quantity": pos.quantity,
                    "average_price": pos.average_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl
                }
                for symbol, pos in engine.get_positions().items()
            },
            "pending_orders": len(engine.orders),
            "total_trades": len(engine.trades)
        }
    
    def process_market_update(
        self,
        simulation_id: str,
        market_data: Dict[str, float]
    ) -> None:
        """Process market data update for a simulation.
        
        Args:
            simulation_id: ID of the simulation
            market_data: Dictionary of symbol -> price
        """
        if simulation_id not in self.active_simulations:
            raise ValueError(f"Simulation {simulation_id} not found")
        
        simulation = self.active_simulations[simulation_id]
        if simulation["status"] != "running":
            self.logger.warning(f"Simulation {simulation_id} is not running")
            return
        
        engine = simulation["engine"]
        model = simulation["model"]
        
        # Get model predictions/signals
        # This is a simplified example - actual implementation would depend on model interface
        signals = self._get_model_signals(model, market_data, engine.get_positions())
        
        # Submit orders based on signals
        for signal in signals:
            order = Order(
                order_id=f"order_{simulation_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                symbol=signal["symbol"],
                side=OrderSide(signal["side"]),
                order_type=OrderType(signal.get("order_type", "market")),
                quantity=signal["quantity"],
                price=signal.get("price"),
                stop_price=signal.get("stop_price")
            )
            
            try:
                engine.submit_order(order)
            except ValueError as e:
                self.logger.error(f"Failed to submit order: {e}")
        
        # Process market data
        engine.process_market_data(market_data)
    
    def _initialize_simulation(
        self,
        simulation_id: str,
        deployment_info: Dict[str, Any],
        simulation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Initialize a paper trading simulation."""
        # Create simulation directory
        sim_dir = self.paper_trading_root / simulation_id
        sim_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize trading engine
        engine = TradingSimulationEngine(
            initial_capital=simulation_config["initial_capital"],
            commission_rate=simulation_config["commission_rate"],
            slippage_rate=simulation_config["slippage_rate"],
            max_position_size=simulation_config["max_position_size"],
            daily_stop_loss=simulation_config["daily_stop_loss"],
            enable_shorting=simulation_config["enable_shorting"]
        )
        
        # Load model (simplified - actual implementation would load from deployment)
        model = self._load_deployed_model(deployment_info)
        
        # Create simulation record
        simulation = {
            "simulation_id": simulation_id,
            "deployment_id": deployment_info["deployment_id"],
            "model_id": deployment_info["model_id"],
            "model_version": deployment_info["model_version"],
            "status": "initialized",
            "config": simulation_config,
            "engine": engine,
            "model": model,
            "created_at": datetime.now()
        }
        
        # Save initial state
        state_file = sim_dir / "simulation_state.json"
        with open(state_file, "w") as f:
            json.dump({
                "simulation_id": simulation_id,
                "deployment_id": deployment_info["deployment_id"],
                "model_id": deployment_info["model_id"],
                "config": simulation_config,
                "status": "initialized",
                "created_at": simulation["created_at"].isoformat()
            }, f, indent=2)
        
        return simulation
    
    def _load_deployed_model(self, deployment_info: Dict[str, Any]) -> Any:
        """Load a deployed model.
        
        This is a placeholder - actual implementation would load
        the model from the deployment path.
        """
        # In a real implementation, this would:
        # 1. Navigate to deployment_info["deployment_path"]
        # 2. Load the model using appropriate method
        # 3. Return the loaded model instance
        
        # For now, return a mock model
        class MockModel:
            def predict(self, features):
                # Simple mock prediction
                return np.random.choice(["buy", "sell", "hold"], p=[0.3, 0.3, 0.4])
        
        return MockModel()
    
    def _get_model_signals(
        self,
        model: Any,
        market_data: Dict[str, float],
        positions: Dict[str, Position]
    ) -> List[Dict[str, Any]]:
        """Get trading signals from model.
        
        This is a simplified example - actual implementation would
        depend on the specific model interface and strategy.
        """
        signals = []
        
        # Mock signal generation
        for symbol, price in market_data.items():
            # Get model prediction (simplified)
            prediction = model.predict({"symbol": symbol, "price": price})
            
            if prediction == "buy" and symbol not in positions:
                signals.append({
                    "symbol": symbol,
                    "side": "buy",
                    "quantity": 100,  # Fixed quantity for simplicity
                    "order_type": "market"
                })
            elif prediction == "sell" and symbol in positions:
                position = positions[symbol]
                signals.append({
                    "symbol": symbol,
                    "side": "sell",
                    "quantity": position.quantity,
                    "order_type": "market"
                })
        
        return signals