# 9. Deployment (`src/deployment/`)

Once a trading strategy model has been trained and evaluated satisfactorily, the next step is to deploy it. The pipeline provides support for different deployment modes, managed by components likely within `reinforcestrategycreator_pipeline/src/deployment/`. The architect's outline notes that details of the `DeploymentManager` are a potential area for deeper documentation.

### 9.1. Deployment Modes
The `deployment.mode` parameter in `pipeline.yaml` determines how the trained model is deployed. Common modes include:

*   **`paper_trading`:** In this mode (default in the base `pipeline.yaml`), the model makes trading decisions based on live or simulated live market data, but no real capital is at risk. Trades are simulated, and performance is tracked to assess how the strategy would perform in a live environment without financial exposure. This is a crucial step before committing to live trading.
*   **`live_trading`:** In this mode, the model connects to a brokerage API and executes real trades with actual capital. This mode requires careful configuration of API keys, risk management parameters, and robust error handling.

### 9.2. Configuration for Deployment
The `deployment` section in `pipeline.yaml` contains parameters critical for both paper and live trading:

*   `api_endpoint`: The endpoint of the brokerage or trading platform API (e.g., `"${TRADING_API_ENDPOINT}"`). This is essential for live trading and may also be used by some paper trading simulators.
*   `api_key`: The API key for authenticating with the trading platform (e.g., `"${TRADING_API_KEY}"`). Secure handling of this key (e.g., via environment variables) is paramount.
*   `max_positions`: The maximum number of open positions the strategy is allowed to hold simultaneously (e.g., `10`).
*   `position_size`: The amount of capital or fraction of portfolio to allocate to each new position (e.g., `0.1` for 10% of available capital).
*   `risk_limit`: A predefined risk threshold, which could be a maximum percentage loss per trade or per day (e.g., `0.02` for a 2% limit). The deployment manager should enforce this limit.
*   `update_frequency`: How often the deployed model fetches new market data, re-evaluates its strategy, and potentially makes new trading decisions (e.g., `"1h"`, `"5min"`, `"1d"`).

A dedicated `DeploymentManager` class would typically handle the logic for connecting to trading APIs, managing orders, tracking positions, and enforcing risk limits based on these configurations.