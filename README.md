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