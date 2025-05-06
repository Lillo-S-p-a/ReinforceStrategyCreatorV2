# 📊 RL Trader Dashboard

Welcome! This dashboard helps you visualize and understand the performance of a Reinforcement Learning (RL) trading strategy. Think of it as your window into how the AI agent is learning to trade! ✨

---

## 🚀 Getting Started: Running the Dashboard

Ready to explore? Here’s how to launch the dashboard:

> **💡 Quick Tip:**
> This dashboard uses Python and requires some setup the first time. If you haven't already, you might need to install the project's tools using [Poetry](https://python-poetry.org/) by running `poetry install` in your terminal within the project folder.

1.  **Open your terminal** (command prompt) in the project directory.
2.  **Run this command:**
    ```bash
    streamlit run dashboard/main.py
    ```
3.  **That's it!** The dashboard should automatically open in your web browser.

---

---

## 💾 TimescaleDB Setup (for Market Data)

This project uses TimescaleDB (a PostgreSQL extension for time-series data) running in a Docker container to store historical market data.

### Starting the Database Service

1.  **Ensure Docker is running.**
2.  **Navigate to the project root directory** in your terminal.
3.  **Run the following command** to start the TimescaleDB service (and any other services defined in `docker-compose.yml`):
    ```bash
    docker compose up -d timescaledb
    ```
    This will start the `timescaledb_market_data` container in detached mode. The data is persisted in a Docker volume named `timescaledb_data`.

### Environment Variables

The TimescaleDB service requires the following environment variables. You should create a `.env` file in the project root if it doesn't exist and add these variables:

```env
# .env file
DATABASE_URL=postgresql://postgres:mysecretpassword@localhost:5432/trading_db
API_KEY=test-key-123

# TimescaleDB Connection Details
TIMESCALEDB_USER=user
TIMESCALEDB_PASSWORD=password
TIMESCALEDB_DB=marketdata
```

*   `TIMESCALEDB_USER`: The username for the database (default: `user`).
*   `TIMESCALEDB_PASSWORD`: The password for the database user (default: `password`).
*   `TIMESCALEDB_DB`: The name of the database (default: `marketdata`).

The `docker-compose.yml` file is configured to use these environment variables or the specified defaults.

### Connecting to the Database

*   **From other Docker services (on the same `app-network`):**
    Use the service name `timescaledb` as the host and port `5432`.
    Connection string example: `postgresql://user:password@timescaledb:5432/marketdata`
    (Replace `user`, `password`, and `marketdata` with the actual values from your `.env` file if you changed the defaults).

*   **From the host machine (e.g., for `psql` or a DB GUI):**
    Use `localhost` as the host and port `5433` (as mapped in `docker-compose.yml`).
    Connection string example: `postgresql://user:password@localhost:5433/marketdata`
## 📈 What You Can See

The dashboard provides several views to analyze the trading agent's behavior and results:

*   **Overall Performance:** Get a summary of key metrics like total profit/loss and win rate.
*   **Episode Deep Dive:** Select specific training episodes (periods) to see detailed charts:
    *   **Portfolio Value:** Watch how the agent's account balance changes over time. 💰
    *   **Actions Taken:** See when the agent decided to Buy (🔼), Sell (🔽), or Hold (⏸️).
    *   **Market Data:** Compare the agent's actions against the actual price movements.
*   **Action Analysis:** Understand the agent's trading patterns and decision-making tendencies.

Explore the different sections using the sidebar navigation!

---

Enjoy analyzing the trading strategies!