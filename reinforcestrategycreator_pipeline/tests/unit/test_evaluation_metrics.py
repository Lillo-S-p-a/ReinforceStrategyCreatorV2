"""Unit tests for the metrics calculator."""

import pytest
import numpy as np
import pandas as pd

from src.evaluation.metrics import MetricsCalculator


class TestMetricsCalculator:
    """Test cases for MetricsCalculator."""
    
    @pytest.fixture
    def metrics_calculator(self):
        """Create a metrics calculator instance."""
        config = {
            "risk_free_rate": 0.02,
            "trading_days_per_year": 252,
            "sharpe_window_size": None
        }
        return MetricsCalculator(config)
    
    @pytest.fixture
    def sample_portfolio_values(self):
        """Create sample portfolio values."""
        # Start at 10000, end at 11000 (10% gain)
        values = [10000]
        for i in range(99):
            # Add some volatility
            change = np.random.normal(10, 50)
            new_value = max(values[-1] + change, 1000)  # Prevent negative
            values.append(new_value)
        
        # Ensure final value is 11000 for predictable PnL
        values[-1] = 11000
        return values
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trade data."""
        return [
            {"pnl": 100},
            {"pnl": -50},
            {"pnl": 200},
            {"pnl": -30},
            {"pnl": 150},
            {"pnl": -20},
            {"pnl": 80},
            {"pnl": 120},
            {"pnl": -40},
            {"pnl": 90}
        ]
    
    def test_initialization(self, metrics_calculator):
        """Test metrics calculator initialization."""
        assert metrics_calculator.risk_free_rate == 0.02
        assert metrics_calculator.trading_days_per_year == 252
        assert metrics_calculator.sharpe_window_size is None
        assert len(metrics_calculator.available_metrics) > 0
    
    def test_calculate_pnl(self, metrics_calculator, sample_portfolio_values):
        """Test PnL calculation."""
        pnl = metrics_calculator.calculate_pnl(sample_portfolio_values)
        assert pnl == 1000.0  # 11000 - 10000
    
    def test_calculate_pnl_percentage(self, metrics_calculator, sample_portfolio_values):
        """Test PnL percentage calculation."""
        pnl_pct = metrics_calculator.calculate_pnl_percentage(sample_portfolio_values)
        assert pnl_pct == 10.0  # (1000 / 10000) * 100
    
    def test_calculate_total_return(self, metrics_calculator, sample_portfolio_values):
        """Test total return calculation."""
        total_return = metrics_calculator.calculate_total_return(sample_portfolio_values)
        assert total_return == 0.1  # 10% as decimal
    
    def test_calculate_sharpe_ratio(self, metrics_calculator):
        """Test Sharpe ratio calculation."""
        # Create returns with known properties
        daily_returns = np.array([0.001] * 100)  # 0.1% daily return
        sharpe = metrics_calculator.calculate_sharpe_ratio(daily_returns)
        
        # With 0.1% daily return and no volatility, Sharpe should be very high
        assert sharpe > 0
        
        # Test with volatile returns
        volatile_returns = np.random.normal(0.001, 0.02, 100)
        sharpe_volatile = metrics_calculator.calculate_sharpe_ratio(volatile_returns)
        assert isinstance(sharpe_volatile, float)
    
    def test_calculate_sortino_ratio(self, metrics_calculator):
        """Test Sortino ratio calculation."""
        # Returns with downside risk
        returns = np.array([0.01, -0.02, 0.015, -0.01, 0.02, -0.005])
        sortino = metrics_calculator.calculate_sortino_ratio(returns)
        assert isinstance(sortino, float)
        
        # Test with no downside risk
        positive_returns = np.array([0.01, 0.02, 0.015, 0.005])
        sortino_positive = metrics_calculator.calculate_sortino_ratio(positive_returns)
        assert sortino_positive == float('inf')
    
    def test_calculate_max_drawdown(self, metrics_calculator):
        """Test maximum drawdown calculation."""
        # Create portfolio with known drawdown
        values = [10000, 11000, 12000, 10000, 11000, 13000, 11000]
        max_dd = metrics_calculator.calculate_max_drawdown(values)
        
        # Max drawdown from 13000 to 11000 = 2000/13000 ≈ 0.1538
        assert pytest.approx(max_dd, 0.01) == 0.1538
        
        # Test with no drawdown
        increasing_values = [10000, 11000, 12000, 13000]
        max_dd_none = metrics_calculator.calculate_max_drawdown(increasing_values)
        assert max_dd_none == 0.0
    
    def test_calculate_calmar_ratio(self, metrics_calculator):
        """Test Calmar ratio calculation."""
        values = [10000, 11000, 12000, 10000, 11000]
        returns = np.diff(values) / values[:-1]
        
        calmar = metrics_calculator.calculate_calmar_ratio(values, returns)
        assert isinstance(calmar, float)
        
        # Test with no drawdown
        increasing_values = [10000, 11000, 12000, 13000]
        increasing_returns = np.diff(increasing_values) / increasing_values[:-1]
        calmar_no_dd = metrics_calculator.calculate_calmar_ratio(
            increasing_values, increasing_returns
        )
        assert calmar_no_dd == 0.0  # No drawdown means Calmar is 0
    
    def test_calculate_win_rate(self, metrics_calculator, sample_trades):
        """Test win rate calculation."""
        win_rate = metrics_calculator.calculate_win_rate(sample_trades)
        
        # 6 winning trades out of 10
        assert win_rate == 0.6
        
        # Test empty trades
        assert metrics_calculator.calculate_win_rate([]) == 0.0
    
    def test_calculate_profit_factor(self, metrics_calculator, sample_trades):
        """Test profit factor calculation."""
        profit_factor = metrics_calculator.calculate_profit_factor(sample_trades)
        
        # Gross profit: 100 + 200 + 150 + 80 + 120 + 90 = 740
        # Gross loss: 50 + 30 + 20 + 40 = 140
        # Profit factor: 740 / 140 ≈ 5.29
        assert pytest.approx(profit_factor, 0.01) == 5.29
        
        # Test with no losses
        winning_trades = [{"pnl": 100}, {"pnl": 200}]
        pf_no_loss = metrics_calculator.calculate_profit_factor(winning_trades)
        assert pf_no_loss == float('inf')
    
    def test_calculate_average_win(self, metrics_calculator, sample_trades):
        """Test average win calculation."""
        avg_win = metrics_calculator.calculate_average_win(sample_trades)
        
        # Winning trades: 100, 200, 150, 80, 120, 90
        # Average: 740 / 6 ≈ 123.33
        assert pytest.approx(avg_win, 0.01) == 123.33
    
    def test_calculate_average_loss(self, metrics_calculator, sample_trades):
        """Test average loss calculation."""
        avg_loss = metrics_calculator.calculate_average_loss(sample_trades)
        
        # Losing trades: 50, 30, 20, 40
        # Average: 140 / 4 = 35
        assert avg_loss == 35.0
    
    def test_calculate_expectancy(self, metrics_calculator, sample_trades):
        """Test expectancy calculation."""
        expectancy = metrics_calculator.calculate_expectancy(sample_trades)
        
        # Win rate: 0.6, Avg win: 123.33, Avg loss: 35
        # Expectancy: (0.6 * 123.33) - (0.4 * 35) = 74 - 14 = 60
        assert pytest.approx(expectancy, 0.1) == 60.0
    
    def test_calculate_volatility(self, metrics_calculator):
        """Test volatility calculation."""
        # Constant returns should have low volatility
        constant_returns = np.array([0.01] * 100)
        vol = metrics_calculator.calculate_volatility(constant_returns)
        assert vol == 0.0
        
        # Variable returns
        variable_returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        vol_variable = metrics_calculator.calculate_volatility(variable_returns)
        assert vol_variable > 0
    
    def test_calculate_downside_deviation(self, metrics_calculator):
        """Test downside deviation calculation."""
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02, -0.03])
        downside_dev = metrics_calculator.calculate_downside_deviation(returns)
        
        # Should only consider negative returns: -0.02, -0.01, -0.03
        negative_returns = np.array([-0.02, -0.01, -0.03])
        expected = np.std(negative_returns)
        assert pytest.approx(downside_dev, 0.0001) == expected
    
    def test_calculate_value_at_risk(self, metrics_calculator):
        """Test Value at Risk calculation."""
        # Create returns with known distribution
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 1000)  # Mean 0, std 0.02
        
        var_95 = metrics_calculator.calculate_value_at_risk(returns, 0.95)
        assert var_95 > 0
        
        # VaR at 95% should be around 1.65 * std dev for normal distribution
        expected_var = 1.65 * 0.02
        assert pytest.approx(var_95, 0.01) == expected_var
    
    def test_calculate_conditional_value_at_risk(self, metrics_calculator):
        """Test Conditional Value at Risk calculation."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 1000)
        
        cvar_95 = metrics_calculator.calculate_conditional_value_at_risk(returns, 0.95)
        var_95 = metrics_calculator.calculate_value_at_risk(returns, 0.95)
        
        # CVaR should be greater than VaR (worse losses)
        assert cvar_95 > var_95
    
    def test_calculate_all_metrics(self, metrics_calculator, sample_portfolio_values, sample_trades):
        """Test calculating all metrics at once."""
        returns = np.diff(sample_portfolio_values) / sample_portfolio_values[:-1]
        
        all_metrics = metrics_calculator.calculate_all_metrics(
            portfolio_values=sample_portfolio_values,
            returns=returns,
            trades=sample_trades,
            trades_count=10,
            win_rate=0.6
        )
        
        # Check that all expected metrics are present
        expected_metrics = [
            "pnl", "pnl_percentage", "total_return", "sharpe_ratio",
            "sortino_ratio", "max_drawdown", "win_rate", "trades_count"
        ]
        
        for metric in expected_metrics:
            assert metric in all_metrics
        
        # Verify specific values
        assert all_metrics["pnl"] == 1000.0
        assert all_metrics["pnl_percentage"] == 10.0
        assert all_metrics["win_rate"] == 0.6
        assert all_metrics["trades_count"] == 10
    
    def test_calculate_all_metrics_with_requested(self, metrics_calculator, sample_portfolio_values):
        """Test calculating only requested metrics."""
        requested = ["pnl", "sharpe_ratio"]
        
        metrics = metrics_calculator.calculate_all_metrics(
            portfolio_values=sample_portfolio_values,
            requested_metrics=requested
        )
        
        # Only requested metrics should be present
        assert "pnl" in metrics
        assert "sharpe_ratio" in metrics
        assert "sortino_ratio" not in metrics
        assert "max_drawdown" not in metrics
    
    def test_edge_cases(self, metrics_calculator):
        """Test edge cases for various metrics."""
        # Empty portfolio values
        assert metrics_calculator.calculate_pnl([]) == 0.0
        assert metrics_calculator.calculate_pnl([10000]) == 0.0
        
        # Zero initial value
        assert metrics_calculator.calculate_pnl_percentage([0, 100]) == 0.0
        
        # Empty returns
        assert metrics_calculator.calculate_sharpe_ratio([]) == 0.0
        assert metrics_calculator.calculate_sharpe_ratio([0.01]) == 0.0
        
        # Zero standard deviation
        assert metrics_calculator.calculate_sharpe_ratio([0.01, 0.01, 0.01]) == 0.0
        
        # Empty trades
        assert metrics_calculator.calculate_profit_factor([]) == 0.0
        assert metrics_calculator.calculate_average_win([]) == 0.0
        assert metrics_calculator.calculate_average_loss([]) == 0.0
    
    def test_sharpe_window_size(self):
        """Test Sharpe ratio with window size."""
        config = {
            "risk_free_rate": 0.02,
            "trading_days_per_year": 252,
            "sharpe_window_size": 20
        }
        calc = MetricsCalculator(config)
        
        # Create 100 returns
        returns = np.random.normal(0.001, 0.02, 100)
        
        # Calculate Sharpe with window
        sharpe = calc.calculate_sharpe_ratio(returns)
        
        # Should only use last 20 returns
        expected_sharpe = calc.calculate_sharpe_ratio(returns[-20:])
        calc.sharpe_window_size = None  # Reset for comparison
        full_sharpe = calc.calculate_sharpe_ratio(returns)
        
        # Windowed Sharpe should be different from full Sharpe
        assert sharpe != full_sharpe