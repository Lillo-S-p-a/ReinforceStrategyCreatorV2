# train.py
import numpy as np
import pandas as pd
import logging
from reinforcestrategycreator.data_fetcher import fetch_historical_data # Corrected import
from reinforcestrategycreator.technical_analyzer import calculate_indicators # Corrected import
from reinforcestrategycreator.trading_environment import TradingEnv # Corrected import
from reinforcestrategycreator.rl_agent import StrategyAgent

# --- Configuration ---
TICKER = "SPY"
START_DATE = "2020-01-01"
END_DATE = "2023-12-31"
TRAINING_EPISODES = 10 # Start with a small number for testing
SHARPE_WINDOW_SIZE = 100 # Example value, adjust if needed based on env implementation

# Agent Hyperparameters (Example values, use defaults or tune later)
STATE_SIZE = None # Will be determined from env
ACTION_SIZE = None # Will be determined from env
AGENT_MEMORY_SIZE = 2000
AGENT_BATCH_SIZE = 32
AGENT_GAMMA = 0.95
AGENT_EPSILON = 1.0
AGENT_EPSILON_DECAY = 0.995
AGENT_EPSILON_MIN = 0.01
AGENT_LEARNING_RATE = 0.001
AGENT_TARGET_UPDATE_FREQ = 5 # Example value

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info(f"Starting training for {TICKER} from {START_DATE} to {END_DATE}")

    # --- 1. Data Pipeline ---
    logging.info("Fetching historical data...")
    # fetcher = DataFetcher() # Removed instantiation
    try:
        # Call the function directly
        data = fetch_historical_data(TICKER, START_DATE, END_DATE)
        if data.empty:
            logging.error("Failed to fetch data or data is empty.")
            return
        logging.info(f"Data fetched successfully: {data.shape[0]} rows")
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return

    # Rename columns to lowercase for consistency (Moved outside except block)
    logging.info(f"Original columns: {data.columns}") # Log original columns
    # Handle MultiIndex by accessing the first element of the tuple
    data.columns = [col[0].lower() for col in data.columns]
    logging.info(f"Renamed columns to lowercase: {list(data.columns)}")

    logging.info("Adding technical indicators...")
    # analyzer = TechnicalAnalyzer() # Removed instantiation
    try:
        # Call the function directly and use the correct name
        data_with_indicators = calculate_indicators(data.copy()) # Use copy to avoid modifying original
        if data_with_indicators.empty or data_with_indicators.isnull().values.any():
             logging.warning("Data after adding indicators is empty or contains NaNs. Check indicator calculations and data range.")
             # Decide how to handle NaNs, e.g., dropna or adjust start date
             initial_rows = len(data_with_indicators)
             data_with_indicators.dropna(inplace=True)
             rows_dropped = initial_rows - len(data_with_indicators)
             if rows_dropped > 0:
                 logging.warning(f"Dropped {rows_dropped} rows containing NaNs after indicator calculation.")
             if data_with_indicators.empty:
                 logging.error("Data is empty after dropping NaNs from indicators.")
                 return
        logging.info("Technical indicators added successfully.")
    except Exception as e:
        logging.error(f"Error adding technical indicators: {e}")
        return

    # --- 2. Environment Setup ---
    logging.info("Initializing trading environment...")
    try:
        env = TradingEnv(data_with_indicators, sharpe_window_size=SHARPE_WINDOW_SIZE) # Corrected class name
        STATE_SIZE = env.observation_space.shape[0]
        ACTION_SIZE = env.action_space.n
        logging.info(f"Environment initialized. State size: {STATE_SIZE}, Action size: {ACTION_SIZE}")
    except Exception as e:
        logging.error(f"Error initializing environment: {e}")
        return

    # --- 3. Agent Setup ---
    logging.info("Initializing RL agent...")
    try:
        agent = StrategyAgent(
            state_size=STATE_SIZE,
            action_size=ACTION_SIZE,
            memory_size=AGENT_MEMORY_SIZE,
            batch_size=AGENT_BATCH_SIZE,
            gamma=AGENT_GAMMA,
            epsilon=AGENT_EPSILON,
            epsilon_decay=AGENT_EPSILON_DECAY,
            epsilon_min=AGENT_EPSILON_MIN,
            learning_rate=AGENT_LEARNING_RATE,
            target_update_freq=AGENT_TARGET_UPDATE_FREQ
        )
        logging.info("Agent initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing agent: {e}")
        return

    # --- 4. Training Loop ---
    logging.info(f"Starting training loop for {TRAINING_EPISODES} episodes...")
    for episode in range(TRAINING_EPISODES):
        state, info = env.reset()
        # Reshape state for the agent's Keras model (expects batch dimension)
        state = np.reshape(state, [1, STATE_SIZE])
        total_reward = 0
        done = False
        step_count = 0

        while not done:
            # Agent selects action
            action = agent.select_action(state)

            # Environment steps
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Reshape next_state for the agent
            next_state = np.reshape(next_state, [1, STATE_SIZE])

            # Agent remembers the transition
            agent.remember(state, action, reward, next_state, done)

            # Update state
            state = next_state
            total_reward += reward
            step_count += 1

            # Agent learns (if enough memory samples)
            if len(agent.memory) > AGENT_BATCH_SIZE:
                agent.learn()

            if done:
                logging.info(f"Episode: {episode+1}/{TRAINING_EPISODES}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}, Steps: {step_count}")
                # Optional: Add more info like portfolio value from `info` dict if available

    logging.info("Training finished.")
    # Optional: Save the trained model
    # agent.save_model("spy_rl_model.keras") # Or .weights.h5
    # logging.info("Agent model saved.")

if __name__ == "__main__":
    main()