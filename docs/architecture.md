# ReinforceStrategyCreatorV2: System Architecture

## High-Level Architecture

The ReinforceStrategyCreatorV2 is a comprehensive reinforcement learning (RL) framework for developing, training, and evaluating automated trading strategies. The system architecture follows a modular design pattern with clear separation of concerns across components.

### System Overview Diagram

```mermaid
graph TD
    subgraph External ["External Systems"]
        YF["Yahoo Finance API"]
        IB["Interactive Brokers API"]
    end

    subgraph Core ["Core Components"]
        DF["DataFetcher"]
        TA["TechnicalAnalyzer"]
        TE["TradingEnvironment"]
        RA["RL Agent (StrategyAgent)"]
        MC["Metrics Calculator"]
        CB["Callbacks"]
    end

    subgraph Infrastructure ["Infrastructure"]
        DB[(Database)]
        API["REST API"]
        Dashboard["Analytics Dashboard"]
    end

    subgraph Training ["Training Loop"]
        TL["Training Scripts"]
    end
    
    subgraph Optimization ["Hyperparameter Optimization"]
        HPO["Ray Tune"]
        ASHA["ASHA Scheduler"]
        VIS["Visualization"]
    end
    
    subgraph Evaluation ["Model Evaluation"]
        ME["Evaluation System"]
        BM["Benchmark Strategies"]
        EVAL_VIS["Performance Visualization"]
    end
    
    subgraph PaperTrading ["Paper Trading"]
        PT["Paper Trading System"]
        IBC["IB Client"]
        INF["Inference Module"]
        EXP["Model Export"]
    end

    %% External connections
    YF -->|OHLCV Data| DF
    IB <-->|Market Data, Orders| IBC

    %% Data processing flow
    DF -->|Market Data| TA
    TA -->|Processed Data + Indicators| TL

    %% Training loop
    TL -->|Configure| TE
    TL -->|Configure| RA
    TL -->|Log Results| DB
    TE <-->|State, Action, Reward| RA
    TE -->|Metrics| MC
    RA -->|Decision| TE
    CB -->|Episode Info| DB
    
    %% Hyperparameter optimization
    HPO -->|Configure| TL
    HPO -->|Evaluate| ME
    ASHA -->|Schedule| HPO
    HPO -->|Results| VIS
    
    %% Model evaluation
    RA -->|Trained Model| ME
    ME -->|Compare| BM
    ME -->|Results| EVAL_VIS
    ME -->|Metrics| DB
    
    %% Paper trading
    RA -->|Best Model| EXP
    EXP -->|Exported Model| INF
    INF -->|Predictions| PT
    PT <-->|Orders, Data| IBC
    PT -->|Performance| DB

    %% Infrastructure connections
    DB <-->|Query/Store| API
    API -->|Data| Dashboard
    Dashboard -->|Queries| API

    %% Style
    classDef external fill:#f9f,stroke:#333,stroke-width:2px;
    classDef core fill:#bbf,stroke:#333,stroke-width:1px;
    classDef infrastructure fill:#bfb,stroke:#333,stroke-width:1px;
    classDef training fill:#fbb,stroke:#333,stroke-width:1px;
    classDef optimization fill:#fbf,stroke:#333,stroke-width:1px;
    classDef evaluation fill:#bff,stroke:#333,stroke-width:1px;
    classDef papertrading fill:#ffb,stroke:#333,stroke-width:1px;
    class External external;
    class Core core;
    class Infrastructure infrastructure;
    class Training training;
    class Optimization optimization;
    class Evaluation evaluation;
    class PaperTrading papertrading;
```

### Architecture Description

ReinforceStrategyCreatorV2 follows a modular, service-oriented architecture designed around the reinforcement learning workflow. The system is composed of:

1. **Data Pipeline**: Responsible for acquiring and preprocessing financial data.
2. **RL Core**: The reinforcement learning implementation including environment and agent.
3. **Storage Layer**: Database models and persistence capabilities.
4. **API Layer**: RESTful API for retrieving training results and metrics.
5. **Visualization Layer**: Dashboard for analyzing agent performance.
6. **Optimization Framework**: Hyperparameter optimization for model improvement.
7. **Evaluation System**: Comprehensive model evaluation and benchmark comparison.
8. **Paper Trading Integration**: Export and deployment of models to paper trading environments.

The architecture is designed to be both scalable and extensible, allowing for:
- Addition of new data sources beyond Yahoo Finance
- Integration of different ML/RL algorithms
- Extension with new technical indicators
- Deployment of successful strategies to production environments
- Systematic improvement of model performance through hyperparameter optimization
- Rigorous evaluation against benchmark strategies
- Seamless transition from training to paper trading

## Components and Modules

### Data Fetcher

The Data Fetcher module is responsible for retrieving historical financial data from external sources.

**Key Responsibilities:**
- Fetch OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance
- Handle request errors and network issues gracefully
- Validate and clean retrieved data

**Key Functions:**
- `fetch_historical_data(ticker, start_date, end_date)`: Retrieves market data for a specific ticker and date range

**Interactions:**
- Inputs data to the Technical Analyzer for feature engineering
- Provides raw data for the Trading Environment

```mermaid
sequenceDiagram
    participant TS as Training Script
    participant DF as DataFetcher
    participant YF as Yahoo Finance API
    participant TA as TechnicalAnalyzer
    
    TS->>DF: fetch_historical_data(ticker, start_date, end_date)
    DF->>YF: HTTP request for OHLCV data
    YF-->>DF: Raw market data
    DF->>DF: Validate and clean data
    DF-->>TS: Formatted DataFrame
    TS->>TA: calculate_indicators(df)
```

### Technical Analyzer

The Technical Analyzer processes raw market data and calculates a variety of technical indicators that serve as inputs to the reinforcement learning agent.

**Key Responsibilities:**
- Calculate standard technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Handle data normalization for machine learning consumption
- Calculate volatility metrics

**Key Functions:**
- `calculate_indicators(data)`: Processes raw OHLCV data and adds calculated technical indicators

**Technical Indicators Implemented:**
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ADX (Average Directional Index)
- Aroon Indicators
- ATR (Average True Range)
- Historical Volatility

### Trading Environment

The Trading Environment module is a Gymnasium-compatible environment that simulates market interactions for reinforcement learning.

**Key Responsibilities:**
- Implement the OpenAI Gym interface (`reset()`, `step()`, etc.)
- Simulate market interactions with realistic constraints
- Calculate rewards based on trading performance
- Track portfolio metrics during episodes
- Handle position sizing and risk management

**Key Functions:**
- `reset()`: Initialize or reset the environment state
- `step(action)`: Execute trading action and return new state, reward, and termination status
- `_execute_trade_action(action)`: Handle the mechanics of executing trades
- `_calculate_reward()`: Calculate the agent's reward based on trading performance
- `_get_observation()`: Construct the state observation for the agent

**Core Features:**
- Support for different position sizing methods (fixed fractional, all-in)
- Implementation of stop-loss and take-profit mechanisms
- Transaction fees and slippage simulation
- Support for different reward functions (P&L, Sharpe ratio)

### RL Agent (StrategyAgent)

The StrategyAgent is the reinforcement learning agent responsible for learning and implementing trading strategies.

**Key Responsibilities:**
- Implement a Deep Q-Network (DQN) for trading decision making
- Provide mechanisms for exploration and exploitation
- Process observations and select actions
- Learn from experiences through replay memory

**Key Functions:**
- `select_action(state)`: Choose an action based on the current state using epsilon-greedy policy
- `learn()`: Sample from replay memory and update the Q-network
- `remember(state, action, reward, next_state, done)`: Store experiences in replay memory
- `update_target_model()`: Sync target network with main network

**Core Features:**
- Experience replay for more stable learning
- Target network for reducing overoptimistic value estimates
- Epsilon-greedy exploration strategy
- Neural network architecture for approximating Q-values

```mermaid
sequenceDiagram
    participant TE as TradingEnvironment
    participant RA as RL Agent (StrategyAgent)
    
    TE->>TE: reset()
    loop For each step in episode
        TE->>RA: state = _get_observation()
        RA->>RA: select_action(state)
        RA-->>TE: action
        TE->>TE: _execute_trade_action(action)
        TE->>TE: Calculate new portfolio value
        TE->>TE: reward = _calculate_reward()
        TE->>TE: next_state = _get_observation()
        TE-->>RA: next_state, reward, terminated, info
        RA->>RA: remember(state, action, reward, next_state, done)
        RA->>RA: learn() (batch learning from replay memory)
        RA->>RA: _update_target_if_needed()
    end
```

### Hyperparameter Optimization Framework

The Hyperparameter Optimization Framework systematically explores different model configurations to find optimal settings for trading performance.

**Key Responsibilities:**
- Define search spaces for hyperparameters
- Execute training runs with different configurations
- Evaluate model performance using key metrics
- Select the best configuration for further refinement
- Visualize optimization results

**Key Components:**
- Ray Tune integration for distributed hyperparameter search
- ASHA (Asynchronous Successive Halving Algorithm) scheduler
- Search space definition for environment, model, and training parameters
- Evaluation metrics calculation (PnL, Sharpe ratio, drawdown, win rate)
- Visualization tools for analyzing results

**Core Features:**
- Efficient exploration of large hyperparameter spaces
- Early stopping of underperforming configurations
- Correlation analysis between parameters and performance
- Parallel execution of training runs
- Comprehensive logging and visualization

### Model Evaluation System

The Model Evaluation System provides a comprehensive framework for assessing model performance on test data and comparing against benchmark strategies.

**Key Responsibilities:**
- Evaluate trained models on unseen market data
- Calculate comprehensive performance metrics
- Compare against benchmark trading strategies
- Visualize performance and trading behavior
- Provide insights for model improvement

**Key Components:**
- Test data preparation and preprocessing
- Multiple evaluation episodes execution
- Benchmark strategy implementation (Buy and Hold, Moving Average Crossover)
- Performance metrics calculation
- Visualization tools for performance analysis

**Core Features:**
- Out-of-sample testing on recent market data
- Benchmark comparison with traditional strategies
- Detailed trade analysis
- Portfolio value tracking
- Performance visualization

### Paper Trading Integration

The Paper Trading Integration enables the deployment of trained models to paper trading environments, with specific support for Interactive Brokers.

**Key Responsibilities:**
- Export trained models in a format suitable for inference
- Connect to Interactive Brokers API
- Fetch real-time market data
- Execute trades based on model predictions
- Track and log trading performance

**Key Components:**
- Model export and serialization
- Interactive Brokers client implementation
- Trading logic and risk management
- Market data processing
- Performance monitoring and logging

**Core Features:**
- PyTorch model export and inference
- Interactive Brokers API integration
- Real-time market data processing
- Trading hours and days enforcement
- Position sizing and risk management
- Performance tracking and reporting

### API Layer

The API layer exposes training results and metrics through a RESTful interface.

**Key Responsibilities:**
- Provide access to training run data
- Expose episode metrics and details
- Enable querying and filtering of training results
- Access to hyperparameter optimization results
- Retrieve model evaluation metrics

**Key Components:**
- FastAPI framework for RESTful endpoints
- Router modules for organizing endpoint groups
- Dependency injection for database sessions
- Pydantic schemas for data validation and serialization

**Main Endpoints:**
- `/api/v1/episodes/{episode_id}`: Get details for a specific episode
- `/api/v1/episodes/{episode_id}/steps/`: Get steps for a specific episode
- `/api/v1/episodes/{episode_id}/trades/`: Get trades for a specific episode
- `/api/v1/episodes/{episode_id}/operations/`: Get trading operations for a specific episode
- `/api/v1/episodes/{episode_id}/model/`: Get model parameters for a specific episode
- `/api/v1/optimization/runs/`: Get hyperparameter optimization runs
- `/api/v1/optimization/runs/{run_id}/results/`: Get results for a specific optimization run
- `/api/v1/evaluation/runs/`: Get model evaluation runs
- `/api/v1/evaluation/runs/{run_id}/metrics/`: Get metrics for a specific evaluation run

### Dashboard

The Dashboard provides a visual interface for analyzing agent performance and trade decisions.

**Key Responsibilities:**
- Visualize trading performance and metrics
- Display decision-making patterns
- Analyze trade outcomes
- Compare different strategies

**Key Components:**
- Streamlit-based interactive web interface
- Performance metrics visualization
- Price and operation charts
- Decision analysis tools
- Model management capabilities

**Main Features:**
- Episode selection and comparison
- Trading operations visualization
- Portfolio performance tracking
- Action distribution analysis
- Reward analysis

### Database Integration

The system uses an SQLAlchemy-based database model for storing training runs, episodes, steps, trades, and operations.

**Key Responsibilities:**
- Define database schema
- Handle database connections
- Provide ORM for data access
- Support logging and querying of training results

**Core Components:**
- SQLAlchemy ORM models
- Database session management
- Transaction handling

## Technology Stack

### Programming Languages
- **Python 3.12**: Primary language for all components

### Machine Learning / RL Frameworks
- **PyTorch**: Deep learning framework for neural network implementation
- **Ray/RLlib**: Distributed reinforcement learning framework
- **Gymnasium**: Environment interface for reinforcement learning

### Data Processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **pandas-ta**: Technical analysis library
- **ta**: Additional technical indicators library

### API Framework
- **FastAPI**: Modern, high-performance web framework for building APIs
- **Uvicorn**: ASGI server for FastAPI

### Database
- **PostgreSQL**: Main database system (via SQLAlchemy)
- **SQLAlchemy**: ORM for database operations

### Dashboard/Visualization
- **Streamlit**: Interactive dashboard framework
- **Plotly**: Interactive visualization library
- **Matplotlib**: Static visualization library

### Data Sources
- **yfinance**: Yahoo Finance API client

### Development and Testing
- **pytest**: Testing framework
- **ruff**: Linting and code formatting
- **httpx**: HTTP client for API testing

## Data Models and Data Flow

### Database Schema

The system uses a relational database model implemented with SQLAlchemy:

```mermaid
erDiagram
    TrainingRun ||--o{ Episode : "contains"
    Episode ||--o{ Step : "contains"
    Episode ||--o{ Trade : "contains"
    Episode ||--o{ TradingOperation : "contains"
    Step ||--o{ TradingOperation : "contains"

    TrainingRun {
        string run_id PK
        datetime start_time
        datetime end_time
        json parameters
        string status
        string notes
    }
    
    Episode {
        int episode_id PK
        string run_id FK
        string rllib_episode_id
        datetime start_time
        datetime end_time
        float initial_portfolio_value
        float final_portfolio_value
        string status
        float pnl
        float sharpe_ratio
        float max_drawdown
        float total_reward
        int total_steps
        float win_rate
    }
    
    Step {
        int step_id PK
        int episode_id FK
        datetime timestamp
        float portfolio_value
        float reward
        float asset_price
        string action
        string position
    }
    
    Trade {
        int trade_id PK
        int episode_id FK
        datetime entry_time
        datetime exit_time
        float entry_price
        float exit_price
        float quantity
        string direction
        float pnl
        float costs
    }
    
    TradingOperation {
        int operation_id PK
        int step_id FK
        int episode_id FK
        datetime timestamp
        enum operation_type
        float size
        float price
    }
```

### Data Flow Diagram

The following diagram illustrates the primary data flow within the system:

```mermaid
flowchart TD
    subgraph DataAcquisition["Data Acquisition"]
        YF[Yahoo Finance API]
        DF[DataFetcher]
        IB[Interactive Brokers API]
    end
    
    subgraph FeatureEngineering["Feature Engineering"]
        TA[TechnicalAnalyzer]
    end
    
    subgraph TrainingLoop["Training Loop"]
        TL[Training Script]
        TE[TradingEnvironment]
        RA[RL Agent]
        CB[Callbacks]
    end
    
    subgraph Storage["Storage"]
        DB[(Database)]
        MODEL_STORE[Model Storage]
    end
    
    subgraph Analysis["Analysis"]
        API[FastAPI]
        DASH[Dashboard]
        MC[Metrics Calculator]
    end
    
    subgraph Optimization["Hyperparameter Optimization"]
        HPO[Ray Tune]
        ASHA[ASHA Scheduler]
        SEARCH[Search Space]
        OPT_VIS[Optimization Visualization]
    end
    
    subgraph Evaluation["Model Evaluation"]
        EVAL[Evaluation System]
        BENCH[Benchmark Comparison]
        PERF_VIS[Performance Visualization]
    end
    
    subgraph PaperTrading["Paper Trading"]
        PT_SYS[Paper Trading System]
        IB_CLIENT[IB Client]
        INFERENCE[Inference Module]
        EXPORT[Model Export]
    end
    
    YF -->|OHLCV Data| DF
    DF -->|Raw Market Data| TA
    TA -->|Price + Indicators| TL
    
    TL -->|Training Config| TE
    TL -->|Agent Config| RA
    
    TE <-->|State, Action, Reward| RA
    TE -->|Episode Metrics| CB
    CB -->|Training Data| DB
    
    DB <-->|Query/Store| API
    API -->|Training Results| DASH
    DASH -->|User Queries| API
    TE -->|Raw Metrics| MC
    MC -->|Calculated Metrics| CB
    
    %% Hyperparameter optimization flow
    HPO -->|Configure| TL
    SEARCH -->|Parameter Space| HPO
    HPO -->|Results| DB
    HPO -->|Best Config| MODEL_STORE
    ASHA -->|Scheduling| HPO
    DB -->|Optimization Results| OPT_VIS
    
    %% Evaluation flow
    MODEL_STORE -->|Trained Model| EVAL
    DF -->|Test Data| EVAL
    BENCH -->|Benchmark Strategies| EVAL
    EVAL -->|Evaluation Results| DB
    EVAL -->|Performance Data| PERF_VIS
    
    %% Paper trading flow
    MODEL_STORE -->|Best Model| EXPORT
    EXPORT -->|Exported Model| INFERENCE
    INFERENCE -->|Predictions| PT_SYS
    IB -->|Market Data| PT_SYS
    PT_SYS -->|Orders| IB_CLIENT
    IB_CLIENT -->|Execution| IB
    PT_SYS -->|Trading Results| DB
```

## External System Integrations

### Yahoo Finance API
- **Purpose**: Fetch historical price and volume data for financial assets
- **Integration Method**: Via the `yfinance` Python library
- **Data Retrieved**: OHLCV (Open, High, Low, Close, Volume) data
- **Interaction Pattern**: Pull-based, fetched at the beginning of each training run

### Interactive Brokers API
- **Purpose**: Execute paper trading operations and fetch real-time market data
- **Integration Method**: Via the Interactive Brokers API (ibapi) Python client
- **Data Retrieved**: Real-time market data, account information, positions, order status
- **Data Sent**: Trading orders (market, limit, etc.)
- **Interaction Pattern**: Bidirectional, event-driven communication
- **Key Components**:
  - **IBClient**: Custom wrapper around the EClient/EWrapper classes
  - **Contract**: Representation of financial instruments
  - **Order**: Definition of trading orders
  - **Position Management**: Tracking and managing trading positions

## Dependencies

The system relies on the following key dependencies as defined in `pyproject.toml`:

### Core Dependencies
- **Python**: ^3.12
- **yfinance**: ^0.2.58 - Yahoo Finance API wrapper
- **requests**: ^2.32.3 - HTTP library
- **pandas**: ^2.2.3 - Data manipulation and analysis
- **pandas-ta**: ^0.3.14b0 - Technical analysis library
- **ta**: ^0.11.0 - Additional technical indicators
- **numpy**: ^1.26.0 - Numerical computing
- **gymnasium**: ^1.0.0 - RL environment interface
- **matplotlib**: ^3.8.4 - Plotting results
- **streamlit**: ^1.34.0 - Dashboard
- **sqlalchemy**: ^2.0.30 - Database ORM
- **psycopg2-binary**: ^2.9.9 - PostgreSQL driver
- **fastapi**: ^0.115.12 - API framework
- **uvicorn**: ^0.34.2 - ASGI server
- **scikit-learn**: ^1.6.1 - Machine learning utilities
- **plotly**: ^6.0.1 - Interactive visualization
- **python-dotenv**: ^1.0.1 - Environment variable management
- **torch**: ^2.3.0 - Deep learning framework
- **torchvision**: ^0.18.0 - PyTorch computer vision
- **torchaudio**: ^2.3.0 - PyTorch audio processing
- **ray**: ^2.46.0 (with rllib extras) - Distributed RL framework
- **ray[tune]**: ^2.46.0 - Ray Tune for hyperparameter optimization
- **seaborn**: ^0.13.0 - Statistical data visualization
- **ibapi**: ^9.81.1 - Interactive Brokers API client
- **protobuf**: ~3.20.0 - Protocol buffers

### Development Dependencies
- **pytest**: ^8.2.2 - Testing framework
- **ruff**: ^0.4.4 - Linting and formatting
- **httpx**: ^0.28.1 - HTTP client for testing

## Deployment Architecture

The system is primarily designed for research and development environments, with components that can be deployed in various configurations:

1. **Local Development**:
   - All components run on a single machine
   - Database can be SQLite or local PostgreSQL
   - Dashboard and API run on local ports

2. **Research Environment**:
   - Training components run on high-performance compute resources
   - Results stored in a central database
   - Dashboard and API deployed as web services for team access

3. **Production Deployment** (for executing trained models):
   - Trained models exported to production environment
   - API deployed as a service for integration with trading platforms
   - Database focused on logging live trading actions and performance