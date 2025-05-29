"""Example script demonstrating paper trading deployment and simulation.

This example shows how to:
1. Deploy a model to paper trading environment
2. Start a simulation with custom configuration
3. Process market data updates
4. Monitor simulation performance
5. Stop simulation and retrieve results
"""

import asyncio
import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import required components
from src.deployment import (
    DeploymentManager,
    PaperTradingDeployer,
    Order,
    OrderType,
    OrderSide
)
from src.models.registry import ModelRegistry
from src.artifact_store.local_adapter import LocalArtifactStore


def generate_market_data(symbols: List[str], base_prices: Dict[str, float]) -> Dict[str, float]:
    """Generate simulated market data with random walk."""
    market_data = {}
    for symbol in symbols:
        # Random walk: +/- 0.5% change
        change = random.uniform(-0.005, 0.005)
        new_price = base_prices[symbol] * (1 + change)
        market_data[symbol] = round(new_price, 2)
        base_prices[symbol] = new_price
    return market_data


def create_mock_model():
    """Create a simple mock trading model for demonstration."""
    class SimpleTrendModel:
        def __init__(self):
            self.price_history = {}
            self.positions = set()
        
        def predict(self, features: Dict) -> str:
            """Simple trend-following strategy."""
            symbol = features["symbol"]
            current_price = features["price"]
            
            # Initialize price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            self.price_history[symbol].append(current_price)
            
            # Keep only last 10 prices
            if len(self.price_history[symbol]) > 10:
                self.price_history[symbol].pop(0)
            
            # Need at least 5 prices for decision
            if len(self.price_history[symbol]) < 5:
                return "hold"
            
            # Calculate simple moving averages
            prices = self.price_history[symbol]
            sma_short = sum(prices[-3:]) / 3  # 3-period SMA
            sma_long = sum(prices[-5:]) / 5   # 5-period SMA
            
            # Generate signals
            if sma_short > sma_long * 1.001 and symbol not in self.positions:
                self.positions.add(symbol)
                return "buy"
            elif sma_short < sma_long * 0.999 and symbol in self.positions:
                self.positions.remove(symbol)
                return "sell"
            else:
                return "hold"
    
    return SimpleTrendModel()


async def run_paper_trading_example():
    """Run a complete paper trading example."""
    
    # Setup directories
    base_dir = Path("./paper_trading_example_output")
    base_dir.mkdir(exist_ok=True)
    
    # Initialize components
    logger.info("Initializing components...")
    
    # Create artifact store
    artifact_store = LocalArtifactStore(
        storage_root=base_dir / "artifacts"
    )
    
    # Create model registry
    model_registry = ModelRegistry(
        artifact_store=artifact_store
    )
    
    # Create deployment manager
    deployment_manager = DeploymentManager(
        model_registry=model_registry,
        artifact_store=artifact_store,
        deployment_root=base_dir / "deployments"
    )
    
    # Create paper trading deployer
    paper_trading_deployer = PaperTradingDeployer(
        deployment_manager=deployment_manager,
        model_registry=model_registry,
        artifact_store=artifact_store,
        paper_trading_root=base_dir / "paper_trading"
    )
    
    # Configuration for paper trading
    simulation_config = {
        "initial_capital": 100000.0,
        "commission_rate": 0.001,  # 0.1% commission
        "slippage_rate": 0.0005,   # 0.05% slippage
        "max_position_size": 0.2,  # Max 20% per position
        "daily_stop_loss": 0.05,   # 5% daily stop loss
        "enable_shorting": False,
        "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN"],
        "update_frequency": "1min"
    }
    
    try:
        # Step 1: Register a mock model
        logger.info("Registering mock model...")
        model = create_mock_model()
        
        # In a real scenario, you would save and register the model properly
        # For this example, we'll mock the registration
        model_id = "trend_following_v1"
        model_version = "1.0.0"
        
        # Mock model registration by creating a simple model package
        model_dir = base_dir / "models" / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model metadata
        metadata = {
            "model_id": model_id,
            "model_version": model_version,
            "model_type": "trend_following",
            "created_at": datetime.now().isoformat()
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Step 2: Deploy model to paper trading
        logger.info(f"Deploying model {model_id} to paper trading...")
        
        # Override the model loading in paper trading deployer
        paper_trading_deployer._load_deployed_model = lambda x: model
        
        simulation_id = paper_trading_deployer.deploy_to_paper_trading(
            model_id=model_id,
            model_version=model_version,
            simulation_config=simulation_config
        )
        
        logger.info(f"Deployment successful. Simulation ID: {simulation_id}")
        
        # Step 3: Start the simulation
        logger.info("Starting paper trading simulation...")
        paper_trading_deployer.start_simulation(simulation_id)
        
        # Step 4: Run simulation for a period
        logger.info("Running simulation...")
        
        # Initial prices
        base_prices = {
            "AAPL": 150.0,
            "GOOGL": 2600.0,
            "MSFT": 300.0,
            "AMZN": 3200.0
        }
        
        # Simulate 100 market updates (e.g., 100 minutes)
        for i in range(100):
            # Generate market data
            market_data = generate_market_data(
                simulation_config["symbols"],
                base_prices
            )
            
            # Process market update
            paper_trading_deployer.process_market_update(
                simulation_id,
                market_data
            )
            
            # Log progress every 10 updates
            if (i + 1) % 10 == 0:
                status = paper_trading_deployer.get_simulation_status(simulation_id)
                logger.info(
                    f"Update {i + 1}/100 - "
                    f"Portfolio Value: ${status['portfolio_value']:,.2f}, "
                    f"Positions: {len(status['positions'])}, "
                    f"Total Trades: {status['total_trades']}"
                )
            
            # Small delay to simulate real-time updates
            await asyncio.sleep(0.1)
        
        # Step 5: Get final status
        logger.info("\nGetting final simulation status...")
        final_status = paper_trading_deployer.get_simulation_status(simulation_id)
        
        logger.info("Final Status:")
        logger.info(f"  Portfolio Value: ${final_status['portfolio_value']:,.2f}")
        logger.info(f"  Total Trades: {final_status['total_trades']}")
        logger.info(f"  Current Positions: {len(final_status['positions'])}")
        
        if final_status['positions']:
            logger.info("\nOpen Positions:")
            for symbol, position in final_status['positions'].items():
                logger.info(
                    f"  {symbol}: {position['quantity']} shares @ "
                    f"${position['average_price']:.2f} "
                    f"(P&L: ${position['unrealized_pnl']:.2f})"
                )
        
        # Performance metrics
        metrics = final_status['current_metrics']
        logger.info("\nPerformance Metrics:")
        logger.info(f"  Total Return: {metrics['total_return']:.2%}")
        logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"  Win Rate: {metrics['win_rate']:.2%}")
        logger.info(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        
        # Step 6: Stop simulation and save results
        logger.info("\nStopping simulation and saving results...")
        results = paper_trading_deployer.stop_simulation(simulation_id)
        
        # Save detailed results
        results_file = base_dir / f"simulation_results_{simulation_id}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        
        # Display summary
        logger.info("\n" + "="*50)
        logger.info("PAPER TRADING SIMULATION COMPLETE")
        logger.info("="*50)
        logger.info(f"Simulation ID: {simulation_id}")
        logger.info(f"Duration: {results['duration_hours']:.2f} hours")
        logger.info(f"Initial Capital: ${simulation_config['initial_capital']:,.2f}")
        logger.info(f"Final Value: ${results['final_portfolio_value']:,.2f}")
        logger.info(f"Total Return: {results['performance_metrics']['total_return']:.2%}")
        logger.info(f"Total Trades: {results['total_trades']}")
        
    except Exception as e:
        logger.error(f"Error during paper trading simulation: {e}")
        raise
    
    finally:
        # Cleanup
        logger.info("\nCleaning up...")
        # In a real scenario, you might want to keep the results
        # For this example, we'll just log that cleanup would happen


def main():
    """Main entry point."""
    logger.info("Starting Paper Trading Example")
    logger.info("="*50)
    
    # Run the async example
    asyncio.run(run_paper_trading_example())
    
    logger.info("\nExample completed successfully!")


if __name__ == "__main__":
    main()