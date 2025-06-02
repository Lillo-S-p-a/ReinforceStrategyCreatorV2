Setting Up Deployment Environments
==================================

This section describes how to set up different environments for deploying and testing your trained trading models using the ReinforceStrategyCreator Pipeline.

General Prerequisites
---------------------

Before setting up any deployment environment, ensure you have:

1.  **Python Environment**: A working Python environment (e.g., version 3.9+). Using a dedicated virtual environment (e.g., via ``python -m venv .venv`` and activating it) is **strongly recommended** to manage dependencies and avoid conflicts.
2.  **Installed Dependencies**: All required Python packages installed. You can typically install these from the ``requirements.txt`` file located in the ``reinforcestrategycreator_pipeline`` directory:

    .. code-block:: bash

       pip install -r requirements.txt

3.  **Pipeline Configuration**: Base and environment-specific configurations set up as described in the :doc:`../../user_guide/configuration` guide.

Local Deployment / Simulation
-----------------------------

A local deployment setup is primarily used for:
*   Testing the model's inference logic with historical or simulated data.
*   Debugging the ``DeploymentManager`` and custom deployment hooks.
*   Running backtests that simulate a deployment scenario.

**Setup Steps**:

1.  **Configuration**:
    *   In your ``pipeline.yaml`` or an environment-specific override (e.g., ``configs/environments/local_simulation.yaml``), you might define a specific ``deployment.mode`` like "local_simulation" or "backtest_deployment".
    *   Ensure the ``data`` configuration section (or relevant parts of ``data.yaml``) points to local historical data files (e.g., CSVs) or a mock/simulated API if you are not using live data for this local test.
2.  **Model**: Have a trained model available, either as a saved checkpoint or registered in the ``ModelRegistry``.
3.  **Execution Script**: You would typically write a Python script that:
    *   Initializes ``ConfigManager``, ``ArtifactStore``, ``ModelRegistry``, and ``DeploymentManager``.
    *   Uses ``DeploymentManager`` to "deploy" a model. In a local simulation context, this might involve loading the model and preparing it for inference.
    *   Feeds data to the model and simulates trade execution based on its predictions.

Refer to examples like ``examples/deployment_example.py`` (if available and relevant for local simulation) or adapt the ``TrainingEngine`` examples to focus on inference with a loaded model.

Paper Trading Environment
-------------------------

Paper trading allows you to test your model with live or delayed market data using a simulated brokerage account, without risking real capital.

**Setup Steps**:

1.  **Brokerage Account**:
    *   Sign up for a paper trading account with a supported brokerage that provides an API.
    *   Obtain your API key, API secret, and the API endpoint URL for paper trading. **Keep these credentials secure.**

2.  **Environment Variables**:
    *   It is strongly recommended to provide your brokerage API credentials via environment variables rather than hardcoding them in configuration files.
    *   The pipeline configuration (e.g., ``deployment`` section in ``pipeline.yaml``) typically expects placeholders like ``${TRADING_API_ENDPOINT}`` and ``${TRADING_API_KEY}``.
    *   Set these environment variables in your system before running the deployment script:

        .. code-block:: bash

           export TRADING_API_ENDPOINT="your_broker_paper_trading_api_endpoint"
           export TRADING_API_KEY="your_paper_trading_api_key"
           export TRADING_API_SECRET="your_paper_trading_api_secret" # If required

3.  **Pipeline Configuration for Paper Trading**:
    *   Create or use an environment-specific configuration file (e.g., ``configs/environments/paper_trading.yaml``).
    *   In this file, or by setting the main ``pipeline.environment`` to "paper_trading", ensure the ``deployment`` section is configured for paper trading:

        .. code-block:: yaml

           # Example: configs/environments/paper_trading.yaml
           deployment:
             mode: "paper_trading"
             # The following are resolved from environment variables by default
             # api_endpoint: "${TRADING_API_ENDPOINT}"
             # api_key: "${TRADING_API_KEY}"
             # api_secret: "${TRADING_API_SECRET}" # If your broker needs it

             # Paper trading specific parameters
             initial_capital: 100000.0  # Example starting capital
             commission_rate: 0.001   # Example commission
             slippage_rate: 0.0005    # Example slippage
             # ... other parameters as shown in examples/paper_trading_example.py

4.  **Model**: Ensure the model you want to deploy is trained and registered in the ``ModelRegistry`` or available as a loadable artifact.

5.  **Running Paper Trading**:
    *   Use a script similar to ``examples/paper_trading_example.py``. This script will::

        *   Initialize ``ConfigManager``, ``ArtifactStore``, ``ModelRegistry``, ``DeploymentManager``, and ``PaperTradingDeployer``.
        *   Use ``PaperTradingDeployer.deploy_to_paper_trading()`` to deploy your chosen model with the simulation configuration.
        *   Start and manage the simulation (e.g., ``start_simulation()``, ``process_market_update()``, ``stop_simulation()``).

Live Trading Environment
------------------------

Setting up a live trading environment involves similar steps to paper trading but with critical differences regarding real capital and security.

**CAUTION: Deploying to a live trading environment involves real financial risk. Proceed with extreme caution and ensure thorough testing in paper trading first.**

1.  **Brokerage Account**: A live trading account with API access.
2.  **API Credentials**: Live trading API key, secret, and endpoint. **These must be protected with utmost care.**
3.  **Environment Variables**: Set environment variables for live trading credentials, distinct from paper trading ones (e.g., ``LIVE_TRADING_API_KEY``).
4.  **Pipeline Configuration for Live Trading**:
    *   Use a dedicated environment configuration (e.g., ``configs/environments/production_live.yaml``).
    *   Set ``deployment.mode: "live_trading"``.
    *   Reference the live trading environment variables for API credentials.
    *   Carefully review and set parameters like ``max_positions``, ``position_size``, and ``risk_limit`` according to your risk management strategy.
5.  **Security**:
    *   Ensure the execution environment is highly secure.
    *   Restrict access to API keys and configuration files using appropriate file permissions and secrets management practices.
    *   Implement robust error handling, retry mechanisms, and fail-safes within your deployment scripts and any custom logic interfacing with the live broker. The ``DeploymentManager`` and ``PaperTradingDeployer`` provide foundational capabilities, but live trading often requires additional custom safeguards.
6.  **Monitoring**: Ensure comprehensive monitoring and alerting are active (see :doc:`monitoring_models`). This includes not only model performance but also system health, API connectivity, and execution logs.

Due to the risks involved, detailed steps for live trading setup should be developed with strict adherence to security best practices and after extensive validation in paper trading. The ``dev-python`` and ``lead-security`` teams should be consulted for guidance on live trading deployments.