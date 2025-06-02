"""Unit tests for paper trading components."""

import pytest
from datetime import datetime
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, MagicMock, patch

from src.deployment.paper_trading import (
    TradingSimulationEngine,
    PaperTradingDeployer,
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    Position
)
from src.deployment.manager import DeploymentManager, DeploymentStatus
from src.models.registry import ModelRegistry
from src.artifact_store.base import ArtifactStore, ArtifactType


class TestTradingSimulationEngine:
    """Test cases for TradingSimulationEngine."""
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = TradingSimulationEngine(
            initial_capital=50000.0,
            commission_rate=0.002,
            slippage_rate=0.001
        )
        
        assert engine.initial_capital == 50000.0
        assert engine.cash == 50000.0
        assert engine.commission_rate == 0.002
        assert engine.slippage_rate == 0.001
        assert len(engine.positions) == 0
        assert len(engine.orders) == 0
    
    def test_submit_market_order(self):
        """Test submitting a market order."""
        engine = TradingSimulationEngine()
        
        order = Order(
            order_id="test_order_1",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        order_id = engine.submit_order(order)
        assert order_id == "test_order_1"
        assert "test_order_1" in engine.orders
        assert engine.orders["test_order_1"].status == OrderStatus.PENDING
    
    def test_submit_limit_order(self):
        """Test submitting a limit order."""
        engine = TradingSimulationEngine()
        
        order = Order(
            order_id="test_order_2",
            symbol="GOOGL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=50,
            price=2500.0
        )
        
        order_id = engine.submit_order(order)
        assert order_id == "test_order_2"
        assert engine.orders["test_order_2"].price == 2500.0
    
    def test_order_validation_position_size(self):
        """Test order validation for position size limit."""
        engine = TradingSimulationEngine(
            initial_capital=10000.0,
            max_position_size=0.1  # 10% max
        )
        
        # Order exceeding position size limit
        order = Order(
            order_id="test_order_3",
            symbol="TSLA",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=200.0  # Total value: 20,000 > 10% of 10,000
        )
        
        with pytest.raises(ValueError, match="exceeds maximum position size"):
            engine.submit_order(order)
    
    def test_order_validation_short_selling(self):
        """Test order validation for short selling restriction."""
        engine = TradingSimulationEngine(enable_shorting=False)
        
        # Attempt to sell without position
        order = Order(
            order_id="test_order_4",
            symbol="MSFT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=50
        )
        
        with pytest.raises(ValueError, match="Short selling not allowed"):
            engine.submit_order(order)
    
    def test_process_market_data_fills_market_order(self):
        """Test market order execution."""
        engine = TradingSimulationEngine(commission_rate=0.001)
        
        # Submit buy order
        order = Order(
            order_id="test_order_5",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        engine.submit_order(order)
        
        # Process market data
        market_data = {"AAPL": 150.0}
        engine.process_market_data(market_data)
        
        # Check order was filled
        assert "test_order_5" not in engine.orders  # Moved to history
        assert len(engine.order_history) == 1
        filled_order = engine.order_history[0]
        assert filled_order.status == OrderStatus.FILLED
        assert filled_order.filled_quantity == 100
        
        # Check position created
        assert "AAPL" in engine.positions
        position = engine.positions["AAPL"]
        assert position.quantity == 100
        assert position.average_price > 150.0  # Due to slippage
        
        # Check cash reduced
        assert engine.cash < engine.initial_capital
    
    def test_process_market_data_fills_limit_order(self):
        """Test limit order execution."""
        engine = TradingSimulationEngine()
        
        # Submit limit buy order
        order = Order(
            order_id="test_order_6",
            symbol="GOOGL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=10,
            price=2600.0
        )
        engine.submit_order(order)
        
        # Market price above limit - should not fill
        market_data = {"GOOGL": 2650.0}
        engine.process_market_data(market_data)
        assert "test_order_6" in engine.orders
        
        # Market price at limit - should fill
        market_data = {"GOOGL": 2600.0}
        engine.process_market_data(market_data)
        assert "test_order_6" not in engine.orders
        assert len(engine.order_history) == 1
    
    def test_position_update_and_pnl(self):
        """Test position updates and P&L calculation."""
        engine = TradingSimulationEngine()
        
        # Create initial position
        engine.positions["AAPL"] = Position(
            symbol="AAPL",
            quantity=100,
            average_price=150.0,
            current_price=150.0
        )
        
        # Update with new market price
        market_data = {"AAPL": 155.0}
        engine.process_market_data(market_data)
        
        position = engine.positions["AAPL"]
        assert position.current_price == 155.0
        assert position.unrealized_pnl == 500.0  # (155-150) * 100
    
    def test_close_position_with_pnl(self):
        """Test closing a position and calculating realized P&L."""
        engine = TradingSimulationEngine()
        
        # Create position
        engine.positions["MSFT"] = Position(
            symbol="MSFT",
            quantity=50,
            average_price=300.0,
            current_price=300.0
        )
        engine.cash = 85000.0  # Adjusted for position
        
        # Submit sell order to close
        order = Order(
            order_id="test_order_7",
            symbol="MSFT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=50
        )
        engine.submit_order(order)
        
        # Process at higher price
        market_data = {"MSFT": 310.0}
        engine.process_market_data(market_data)
        
        # Position should be closed
        assert "MSFT" not in engine.positions
        
        # Check trade recorded with P&L
        assert len(engine.trades) == 1
        trade = engine.trades[0]
        assert trade["pnl"] > 0  # Profitable trade
    
    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation."""
        engine = TradingSimulationEngine(initial_capital=100000.0)
        
        # Simulate some portfolio value changes
        engine.portfolio_values = [
            (datetime.now(), 100000.0),
            (datetime.now(), 102000.0),
            (datetime.now(), 101000.0),
            (datetime.now(), 103000.0)
        ]
        
        # Add some trades
        engine.trades = [
            {"pnl": 500.0},
            {"pnl": -200.0},
            {"pnl": 700.0},
            {"pnl": 1000.0}
        ]
        
        metrics = engine.get_performance_metrics()
        
        assert metrics["total_return"] == 0.03  # 3% return
        assert metrics["total_trades"] == 4
        assert metrics["winning_trades"] == 3
        assert metrics["losing_trades"] == 1
        assert metrics["win_rate"] == 0.75
        assert metrics["profit_factor"] > 1.0
    
    def test_daily_stop_loss(self):
        """Test daily stop loss risk management."""
        engine = TradingSimulationEngine(
            initial_capital=100000.0,
            daily_stop_loss=0.02  # 2% daily stop
        )
        
        # Simulate loss
        engine.portfolio_values.append((datetime.now(), 97900.0))  # 2.1% loss
        engine._check_daily_stop_loss(97900.0)
        
        assert engine.risk_limit_hit is True
        
        # Try to submit order after risk limit hit
        order = Order(
            order_id="test_order_8",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        order_id = engine.submit_order(order)
        
        # Order should be rejected
        assert engine.orders[order_id].status == OrderStatus.REJECTED
    
    def test_reset_daily_limits(self):
        """Test resetting daily risk limits."""
        engine = TradingSimulationEngine()
        engine.risk_limit_hit = True
        engine.daily_loss = 0.025
        
        engine.reset_daily_limits()
        
        assert engine.risk_limit_hit is False
        assert engine.daily_loss == 0.0


class TestPaperTradingDeployer:
    """Test cases for PaperTradingDeployer."""
    
    @pytest.fixture
    def mock_deployment_manager(self):
        """Create mock deployment manager."""
        mock = Mock(spec=DeploymentManager)
        mock.deploy.return_value = "deploy_test_model_paper_trading_20250529_120000"
        mock.get_deployment_status.return_value = {
            "deployment_id": "deploy_test_model_paper_trading_20250529_120000",
            "model_id": "test_model",
            "model_version": "v1.0",
            "target_environment": "paper_trading",
            "status": "deployed"
        }
        return mock
    
    @pytest.fixture
    def mock_model_registry(self):
        """Create mock model registry."""
        return Mock(spec=ModelRegistry)
    
    @pytest.fixture
    def mock_artifact_store(self):
        """Create mock artifact store."""
        mock = Mock(spec=ArtifactStore)
        mock.save_artifact.return_value = "artifact_123"
        return mock
    
    @pytest.fixture
    def paper_trading_deployer(self, mock_deployment_manager, mock_model_registry, mock_artifact_store):
        """Create PaperTradingDeployer instance with mocks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            deployer = PaperTradingDeployer(
                deployment_manager=mock_deployment_manager,
                model_registry=mock_model_registry,
                artifact_store=mock_artifact_store,
                paper_trading_root=temp_dir
            )
            yield deployer
    
    def test_initialization(self, paper_trading_deployer):
        """Test deployer initialization."""
        assert paper_trading_deployer.paper_trading_root.exists()
        assert len(paper_trading_deployer.active_simulations) == 0
    
    def test_deploy_to_paper_trading(self, paper_trading_deployer, mock_deployment_manager):
        """Test deploying model to paper trading."""
        simulation_id = paper_trading_deployer.deploy_to_paper_trading(
            model_id="test_model",
            model_version="v1.0",
            simulation_config={
                "initial_capital": 50000.0,
                "symbols": ["AAPL", "GOOGL"]
            }
        )
        
        # Check deployment manager was called
        mock_deployment_manager.deploy.assert_called_once()
        call_args = mock_deployment_manager.deploy.call_args
        assert call_args.kwargs["model_id"] == "test_model"
        assert call_args.kwargs["target_environment"] == "paper_trading"
        
        # Check simulation created
        assert simulation_id.startswith("sim_deploy_test_model")
        assert simulation_id in paper_trading_deployer.active_simulations
        
        simulation = paper_trading_deployer.active_simulations[simulation_id]
        assert simulation["status"] == "initialized"
        assert simulation["config"]["initial_capital"] == 50000.0
        assert simulation["config"]["symbols"] == ["AAPL", "GOOGL"]
    
    def test_start_simulation(self, paper_trading_deployer):
        """Test starting a simulation."""
        # First deploy
        simulation_id = paper_trading_deployer.deploy_to_paper_trading(
            model_id="test_model",
            model_version="v1.0"
        )
        
        # Start simulation
        paper_trading_deployer.start_simulation(simulation_id)
        
        simulation = paper_trading_deployer.active_simulations[simulation_id]
        assert simulation["status"] == "running"
        assert "started_at" in simulation
    
    def test_stop_simulation(self, paper_trading_deployer, mock_artifact_store):
        """Test stopping a simulation."""
        # Deploy and start
        simulation_id = paper_trading_deployer.deploy_to_paper_trading(
            model_id="test_model",
            model_version="v1.0"
        )
        paper_trading_deployer.start_simulation(simulation_id)
        
        # Add some mock data to engine
        simulation = paper_trading_deployer.active_simulations[simulation_id]
        engine = simulation["engine"]
        engine.portfolio_values.append((datetime.now(), 101000.0))
        engine.trades.append({"pnl": 1000.0})
        
        # Stop simulation
        results = paper_trading_deployer.stop_simulation(simulation_id)
        
        assert results["simulation_id"] == simulation_id
        assert results["model_id"] == "test_model"
        assert "performance_metrics" in results
        assert results["total_trades"] == 1
        
        # Check results saved
        results_file = paper_trading_deployer.paper_trading_root / f"{simulation_id}_results.json"
        assert results_file.exists()
        
        # Check artifact store called
        mock_artifact_store.save_artifact.assert_called_once()
    
    def test_get_simulation_status(self, paper_trading_deployer):
        """Test getting simulation status."""
        # Deploy
        simulation_id = paper_trading_deployer.deploy_to_paper_trading(
            model_id="test_model",
            model_version="v1.0"
        )
        
        # Get status
        status = paper_trading_deployer.get_simulation_status(simulation_id)
        
        assert status["simulation_id"] == simulation_id
        assert status["status"] == "initialized"
        assert status["model_id"] == "test_model"
        assert "current_metrics" in status
        assert "portfolio_value" in status
        assert "positions" in status
    
    def test_process_market_update(self, paper_trading_deployer):
        """Test processing market data update."""
        # Deploy and start
        simulation_id = paper_trading_deployer.deploy_to_paper_trading(
            model_id="test_model",
            model_version="v1.0"
        )
        paper_trading_deployer.start_simulation(simulation_id)
        
        # Mock model to return buy signal
        simulation = paper_trading_deployer.active_simulations[simulation_id]
        mock_model = Mock()
        mock_model.predict.return_value = "buy"
        simulation["model"] = mock_model
        
        # Process market update
        market_data = {"AAPL": 150.0, "GOOGL": 2600.0}
        paper_trading_deployer.process_market_update(simulation_id, market_data)
        
        # Check model was called
        assert mock_model.predict.called
        
        # Check orders were submitted
        engine = simulation["engine"]
        assert len(engine.orders) > 0 or len(engine.order_history) > 0
    
    def test_invalid_simulation_id(self, paper_trading_deployer):
        """Test operations with invalid simulation ID."""
        with pytest.raises(ValueError, match="Simulation invalid_id not found"):
            paper_trading_deployer.start_simulation("invalid_id")
        
        with pytest.raises(ValueError, match="Simulation invalid_id not found"):
            paper_trading_deployer.stop_simulation("invalid_id")
        
        with pytest.raises(ValueError, match="Simulation invalid_id not found"):
            paper_trading_deployer.get_simulation_status("invalid_id")


class TestIntegration:
    """Integration tests for paper trading components."""
    
    def test_full_paper_trading_workflow(self):
        """Test complete paper trading workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create real instances (with mocks for external dependencies)
            mock_model_registry = Mock(spec=ModelRegistry)
            mock_artifact_store = Mock(spec=ArtifactStore)
            mock_artifact_store.save_artifact.return_value = "artifact_123"
            
            deployment_manager = DeploymentManager(
                model_registry=mock_model_registry,
                artifact_store=mock_artifact_store,
                deployment_root=Path(temp_dir) / "deployments"
            )
            
            paper_trading_deployer = PaperTradingDeployer(
                deployment_manager=deployment_manager,
                model_registry=mock_model_registry,
                artifact_store=mock_artifact_store,
                paper_trading_root=Path(temp_dir) / "paper_trading"
            )
            
            # Mock the deployment manager's package loading
            with patch.object(deployment_manager.packager, 'package_model', return_value="package_123"):
                with patch.object(mock_artifact_store, 'load_artifact') as mock_load:
                    # Create a mock package file
                    mock_package_path = Path(temp_dir) / "mock_package.tar.gz"
                    mock_package_path.touch()
                    mock_load.return_value = str(mock_package_path)
                    
                    # Deploy to paper trading
                    simulation_id = paper_trading_deployer.deploy_to_paper_trading(
                        model_id="test_model",
                        model_version="v1.0",
                        simulation_config={
                            "initial_capital": 100000.0,
                            "symbols": ["AAPL", "GOOGL", "MSFT"]
                        }
                    )
                    
                    # Start simulation
                    paper_trading_deployer.start_simulation(simulation_id)
                    
                    # Process some market updates
                    for i in range(5):
                        market_data = {
                            "AAPL": 150.0 + i,
                            "GOOGL": 2600.0 + i * 10,
                            "MSFT": 300.0 + i * 2
                        }
                        paper_trading_deployer.process_market_update(simulation_id, market_data)
                    
                    # Get status
                    status = paper_trading_deployer.get_simulation_status(simulation_id)
                    assert status["status"] == "running"
                    
                    # Stop simulation
                    results = paper_trading_deployer.stop_simulation(simulation_id)
                    assert results["simulation_id"] == simulation_id
                    assert "performance_metrics" in results