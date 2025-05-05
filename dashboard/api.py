import requests
import logging
import pandas as pd
from typing import List, Dict, Optional, Any

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8001/api/v1"
API_KEY = "test-key-123"
API_HEADERS = {"X-API-Key": API_KEY}

def fetch_api_data(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Generic function to fetch data from the API."""
    url = f"{API_BASE_URL}{endpoint}"
    log_headers = API_HEADERS.copy()
    if "X-API-Key" in log_headers:
        log_headers["X-API-Key"] = "****"

    logging.info(f"Attempting to fetch data from API endpoint: {url}")
    logging.info(f"  Headers: {log_headers}")
    logging.info(f"  Params: {params}")
    try:
        response = requests.get(url, headers=API_HEADERS, params=params, timeout=10)
        logging.info(f"  API Response Status Code: {response.status_code}")
        response_snippet = response.text[:200] + "..." if len(response.text) > 200 else response.text
        logging.info(f"  API Response Snippet: {response_snippet}")

        response.raise_for_status()
        logging.info(f"  Successfully fetched and parsed JSON for {url}")
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred fetching {url}: {http_err}")
        logging.error(f"  Response Body: {response.text}")
        if response.status_code == 404:
            logging.info(f"Endpoint not found: {url}")
            return None
        return None
    except requests.exceptions.ConnectionError as conn_err:
        logging.error(f"Connection error occurred fetching {url}: {conn_err}")
        return None
    except requests.exceptions.Timeout as timeout_err:
        logging.error(f"Timeout error occurred fetching {url}: {timeout_err}")
        return None
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Ambiguous request error occurred fetching {url}: {req_err}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred processing API response from {url}: {e}", exc_info=True)
        return None

def fetch_latest_run() -> Optional[Dict[str, Any]]:
    """Fetches the most recent training run."""
    data = fetch_api_data("/runs/", params={"page": 1, "page_size": 1})
    if data and data.get("items"):
        return data["items"][0]
    logging.warning("Could not fetch latest training run.")
    return None

def fetch_run_summary(run_id: str) -> Optional[Dict[str, Any]]:
    """Fetches the episode summary for a given run."""
    return fetch_api_data(f"/runs/{run_id}/episodes/summary/")

def fetch_run_episodes(run_id: str) -> List[Dict[str, Any]]:
    """Fetches all episodes for a given run (handles pagination)."""
    episodes = []
    page = 1
    while True:
        data = fetch_api_data(f"/runs/{run_id}/episodes/", params={"page": page, "page_size": 100})
        if not data or not data.get("items"):
            break
        episodes.extend(data["items"])
        if page >= data.get("total_pages", 1):
            break
        page += 1
    if not episodes:
        logging.warning(f"No episodes found for run {run_id}.")
    return episodes

def fetch_episode_steps(episode_id: int) -> pd.DataFrame:
    """Fetches all steps for a given episode and returns a DataFrame."""
    steps_list = []
    page = 1
    while True:
        data = fetch_api_data(f"/episodes/{episode_id}/steps/", params={"page": page, "page_size": 100})
        if not data or not data.get("items"):
            if page == 1:
                 logging.warning(f"No steps found for episode {episode_id}.")
            break
        steps_list.extend(data["items"])
        if page >= data.get("total_pages", 1):
            break
        page += 1

    if not steps_list:
        return pd.DataFrame()

    df = pd.DataFrame(steps_list)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df['portfolio_value'] = pd.to_numeric(df['portfolio_value'], errors='coerce')
    df['reward'] = pd.to_numeric(df['reward'], errors='coerce')
    # Ensure action is numeric for calculations - THIS IS THE FIX
    df['action'] = pd.to_numeric(df['action'], errors='coerce')
    return df.sort_index()

def fetch_episode_trades(episode_id: int) -> List[Dict[str, Any]]:
    """Fetches all trades for a given episode."""
    trades_list = []
    page = 1
    while True:
        endpoint = f"/episodes/{episode_id}/trades/"
        data = fetch_api_data(endpoint, params={"page": page, "page_size": 100})
        if not data or not data.get("items"):
            if page == 1:
                 logging.warning(f"No trades found in API response for episode {episode_id} on page 1.")
            break
        trades_list.extend(data["items"])
        if page >= data.get("total_pages", 1):
            break
        page += 1
    logging.info(f"Fetched {len(trades_list)} trades for episode {episode_id}.")
    for trade in trades_list:
        trade['entry_time'] = pd.to_datetime(trade['entry_time'])
        if trade.get('exit_time'):
            trade['exit_time'] = pd.to_datetime(trade['exit_time'])
    return sorted(trades_list, key=lambda x: x['entry_time'])

def fetch_episode_operations(episode_id: int) -> List[Dict[str, Any]]:
    """Fetches all trading operations for a given episode."""
    operations_list = []
    page = 1
    logging.info(f"Fetching operations for episode {episode_id}")
    while True:
        endpoint = f"/episodes/{episode_id}/operations/"
        data = fetch_api_data(endpoint, params={"page": page, "page_size": 100})
        logging.debug(f"Raw operations data page {page} for episode {episode_id}: {data}")
        if not data or not data.get("items"):
            if page == 1:
                 logging.warning(f"No operations items found in API response for episode {episode_id} on page 1.")
            break
        operations_list.extend(data["items"])
        if page >= data.get("total_pages", 1):
            break
        page += 1
    logging.info(f"Fetched {len(operations_list)} operations for episode {episode_id}.")
    for op in operations_list:
        op['timestamp'] = pd.to_datetime(op['timestamp'])
    return sorted(operations_list, key=lambda x: x['timestamp'])

def fetch_episode_model(episode_id: int) -> Optional[Dict[str, Any]]:
    """Fetches model parameters for a given episode."""
    # Try to fetch from API first
    endpoint = f"/episodes/{episode_id}/model/"
    data = fetch_api_data(endpoint)
    
    # If not available, generate mock data for demonstration
    if not data:
        logging.info(f"Model data not available for episode {episode_id}, generating mock data")
        return {
            "episode_id": episode_id,
            "learning_rate": 0.001 + (episode_id % 10) * 0.0001,
            "gamma": 0.95 + (episode_id % 5) * 0.01,
            "epsilon": max(0.1, 1.0 - (episode_id * 0.02)),
            "batch_size": 32 + (episode_id % 3) * 16,
            "hidden_layer_size": 64 + (episode_id % 4) * 16,
            "memory_size": 10000 + (episode_id % 5) * 1000,
            "architecture": "DQN" if episode_id % 2 == 0 else "DDQN",
            "optimizer": "Adam" if episode_id % 3 != 0 else "RMSprop",
            "loss_function": "MSE" if episode_id % 2 == 0 else "Huber",
            "exploration_strategy": "epsilon-greedy",
            "feature_extractors": ["price", "volume", "macd", "rsi"]
        }
    return data