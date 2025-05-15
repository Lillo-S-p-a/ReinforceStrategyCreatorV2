# ReinforceStrategyCreatorV2: System Setup

This document provides comprehensive instructions for setting up and configuring the ReinforceStrategyCreatorV2 system for different environments (development, testing, and production).

## Prerequisites

Before setting up the ReinforceStrategyCreatorV2 system, ensure you have the following prerequisites installed:

### Operating System
- Any modern Linux distribution (recommended)
- macOS
- Windows with WSL (Windows Subsystem for Linux) for optimal compatibility

### Software Requirements
- **Python**: Version 3.12 or higher (required by `pyproject.toml`)
- **Poetry**: Dependency management tool (version 1.6.0 or higher recommended)
  - Installation instructions: [https://python-poetry.org/docs/#installation](https://python-poetry.org/docs/#installation)
- **Docker & Docker Compose**: Required for database services
  - Docker version 24.0 or higher
  - Docker Compose version 2.20 or higher
- **Git**: For version control and project management

### Hardware Recommendations
- **CPU**: Minimum 4 cores (8+ cores recommended for faster training)
- **RAM**: Minimum 8GB (16GB+ recommended)
- **GPU**: Optional but recommended for accelerated training (CUDA-compatible)
- **Storage**: At least 10GB of free disk space

## Development Environment Setup

Follow these steps to set up a development environment:

### 1. Clone the Repository

```bash
git clone <repository-url> ReinforceStrategyCreatorV2
cd ReinforceStrategyCreatorV2
```

### 2. Install Dependencies

Use Poetry to install all dependencies:

```bash
poetry install
```

This will create a virtual environment and install all dependencies specified in `pyproject.toml`.

### 3. Configure Environment Variables

Create a `.env` file in the project root directory with the following content (or copy from an existing `.env.example` and modify as needed):

```bash
# Regular PostgreSQL (trading_db) connection string for the application
DATABASE_URL=postgresql://postgres:mysecretpassword@localhost:5434/trading_db
API_KEY=test-key-123

# TimescaleDB (marketdata) credentials
TIMESCALEDB_USER=user
TIMESCALEDB_PASSWORD=password
TIMESCALEDB_DB=marketdata

# PostgreSQL (trading_db) credentials
POSTGRES_USER=postgres
POSTGRES_PASSWORD=mysecretpassword
POSTGRES_DB=trading_db
```

Modify the credentials as needed for your local environment.

### 4. Start Database Services

Start the required database services using Docker Compose:

```bash
docker-compose up -d
```

This will start:
- TimescaleDB container on port 5433 (for market data with time-series extensions)
- PostgreSQL container on port 5434 (for trading data)

Wait until both containers are fully initialized and healthy before proceeding.

### 5. Initialize the Database Schema

Run the database initialization script to create necessary tables and schema:

```bash
poetry run python init_db.py
```

This script will:
- Connect to the PostgreSQL database using the credentials from your `.env` file
- Create all the required tables defined in the database models
- Output confirmation when the schema initialization is complete

### 6. Running the Application Components

#### 6.1. Training a Model

For regular training run:

```bash
./run_train.sh
```

For debug training with enhanced logging:

```bash
./run_debug_train.sh
```

The training process will:
- Fetch historical data from Yahoo Finance
- Calculate technical indicators
- Initialize the reinforcement learning environment
- Train an RL agent using Ray/RLlib
- Log training progress and store model results in the database

#### 6.2. Running the Dashboard

To visualize and analyze training results:

```bash
poetry run python run_dashboard.py
```

This will start the Streamlit-based dashboard on a local port (typically http://localhost:8501).

#### 6.3. Running the API Server

To start the FastAPI server:

```bash
poetry run uvicorn reinforcestrategycreator.api.main:app --reload
```

The API will be available at http://localhost:8000, with interactive documentation at http://localhost:8000/docs.

## Testing/Staging Environment Setup

The testing/staging environment follows the same setup as the development environment with the following adjustments:

1. Use separate database instances with different credentials
2. Modify the `.env` file to point to testing databases
3. Consider using dedicated hardware resources for more intensive testing scenarios

For existing test database integration, modify your `.env` file accordingly:

```bash
# Testing environment database connection
DATABASE_URL=postgresql://postgres:test_password@test-db-host:5434/trading_db_test
```

## Production Environment Setup

For production deployment, the following additional considerations apply:

### 1. Security Enhancements

- Use strong, unique passwords for all database connections
- Implement proper network security measures (firewalls, VPNs)
- Store sensitive credentials using a secure secret management system
- Use HTTPS for all API communications

### 2. Database Configuration

- Configure database services with appropriate resource allocations based on expected load
- Set up regular database backups
- Implement proper monitoring and alerting

### 3. Scaling Considerations

- For higher workloads, consider separating the database services onto dedicated hardware
- Configure Ray for distributed training across multiple machines if needed
- Implement load balancing for API services if required

### 4. Deployment Process

For deploying trained models to production:

1. Train models in development or testing environment
2. Export successful models using the model export functionality
3. Load models in the production environment
4. Connect to live data feeds (instead of historical data)
5. Implement proper monitoring and risk management systems

## Configuration Management

### Environment Variables

The system uses the following environment variables that should be configured in the `.env` file:

| Variable | Purpose | Example Value |
|----------|---------|---------------|
| DATABASE_URL | Main PostgreSQL connection string for the application | postgresql://postgres:mysecretpassword@localhost:5434/trading_db |
| API_KEY | API key for external services or authentication | test-key-123 |
| TIMESCALEDB_USER | Username for TimescaleDB connections | user |
| TIMESCALEDB_PASSWORD | Password for TimescaleDB connections | password |
| TIMESCALEDB_DB | Database name for TimescaleDB | marketdata |
| POSTGRES_USER | Username for PostgreSQL connections | postgres |
| POSTGRES_PASSWORD | Password for PostgreSQL connections | mysecretpassword |
| POSTGRES_DB | Database name for PostgreSQL | trading_db |

### Training Configuration

The training configuration is defined in the training scripts (`train.py` and `train_debug.py`). Key configurable parameters include:

#### Data Configuration:
- `TICKER`: Stock symbol to train on (e.g., "SPY")
- `START_DATE`, `END_DATE`: Date range for historical data

#### Environment Parameters:
- `ENV_INITIAL_BALANCE`: Initial portfolio balance
- `ENV_TRANSACTION_FEE_PERCENT`: Transaction fee as a percentage
- `ENV_WINDOW_SIZE`: Size of observation window
- `ENV_SHARPE_WINDOW_SIZE`: Window size for Sharpe ratio calculation
- `ENV_DRAWDOWN_PENALTY`: Penalty for drawdowns
- `ENV_TRADING_PENALTY`: Penalty for excessive trading
- `ENV_RISK_FRACTION`: Position sizing as fraction of portfolio
- `ENV_STOP_LOSS_PCT`: Stop loss percentage
- `ENV_USE_SHARPE_RATIO`: Whether to use Sharpe ratio in reward
- `ENV_NORMALIZATION_WINDOW_SIZE`: Window size for data normalization

#### Agent Hyperparameters:
- `AGENT_LEARNING_RATE`: Learning rate for neural network
- `AGENT_GAMMA`: Discount factor
- `AGENT_BUFFER_SIZE`: Size of replay buffer
- `AGENT_TRAIN_BATCH_SIZE`: Batch size for training
- `AGENT_TARGET_NETWORK_UPDATE_FREQ_TIMESTEPS`: Target network update frequency
- `AGENT_INITIAL_EPSILON`: Initial exploration rate
- `AGENT_FINAL_EPSILON`: Final exploration rate
- `AGENT_EPSILON_TIMESTEPS`: Timesteps for epsilon decay
- `NUM_ROLLOUT_WORKERS`: Number of parallel workers for training

## Troubleshooting

### Common Issues

1. **Database Connection Errors**:
   - Verify that Docker containers are running (`docker ps`)
   - Check if PostgreSQL and TimescaleDB ports are accessible
   - Ensure your `.env` file has the correct credentials

2. **Poetry Installation Issues**:
   - Ensure you have Python 3.12+ installed
   - Try reinstalling Poetry following the official documentation
   - Check dependency conflicts with `poetry show --tree`

3. **Training Script Errors**:
   - Verify that all dependencies are correctly installed
   - Check that data fetching is working (network connectivity to Yahoo Finance)
   - Ensure database schema is properly initialized

4. **Dashboard Not Showing Expected Results**:
   - Verify that training runs have completed successfully
   - Check database connection from the dashboard service
   - Look for errors in dashboard logs

### Getting Help

If you encounter issues not covered here, please:
- Check the project issues on the repository
- Refer to the API and component documentation in `docs/`
- Reach out to the project maintainers