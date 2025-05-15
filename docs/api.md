# ReinforceStrategyCreator Metrics API Reference

## API Overview

### Base URL

The base URL for all API endpoints is:

```
/api/v1
```

When running the API locally, the full URL would typically be:

```
http://localhost:8000/api/v1
```

### Authentication

All API endpoints require authentication using an API key. The API key must be included in the request header:

```
X-API-Key: your_api_key_here
```

The API key is loaded from the `API_KEY` environment variable on the server. If the API key is missing or invalid, the API will return a `401 Unauthorized` error.

### Error Handling

The API uses standard HTTP status codes to indicate the success or failure of requests:

- `200 OK`: Request successful
- `401 Unauthorized`: Missing or invalid API key
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

Error responses include a JSON body with a `detail` field providing more information about the error:

```json
{
  "detail": "Error message here"
}
```

### Pagination

Many endpoints that return lists of items support pagination. The pagination parameters are:

- `page`: Page number (integer, ≥1, default=1)
- `page_size`: Number of items per page (integer, typically 1-100, default=20)

Paginated responses follow this structure:

```json
{
  "total_items": 150,
  "total_pages": 8,
  "current_page": 1,
  "page_size": 20,
  "items": [
    // Array of items
  ]
}
```

### OpenAPI Documentation

When the API is running, interactive API documentation is available at:

- Swagger UI: `/docs`
- ReDoc: `/redoc`

## Health Check Endpoint

### GET /

Root endpoint for health check.

**Response (200 OK)**

```json
{
  "status": "ok",
  "message": "Welcome to ReinforceStrategyCreator Metrics API"
}
```

## Training Runs Endpoints

### GET /api/v1/runs/

Retrieve a paginated list of training runs, optionally filtered by date range and status.

**Query Parameters**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| page | integer | No | Page number (≥1, default=1) |
| page_size | integer | No | Items per page (1-100, default=20) |
| start_date | date | No | Filter runs started on or after this date (YYYY-MM-DD) |
| end_date | date | No | Filter runs started on or before this date (YYYY-MM-DD) |
| status | string | No | Filter runs by status (e.g., 'completed', 'running') |

**Success Response (200 OK)**

```json
{
  "total_items": 25,
  "total_pages": 2,
  "current_page": 1,
  "page_size": 20,
  "items": [
    {
      "run_id": "RUN-SPY-20250505113632-48ccd3c0",
      "start_time": "2025-05-05T11:36:32",
      "end_time": "2025-05-05T12:45:10",
      "parameters": {
        "learning_rate": 0.001,
        "batch_size": 64,
        "gamma": 0.99
      },
      "status": "completed",
      "notes": "Production model training"
    }
  ]
}
```

### GET /api/v1/runs/{run_id}

Retrieve details for a specific training run by its ID.

**Path Parameters**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| run_id | string | Yes | The ID of the training run to retrieve |

**Success Response (200 OK)**

```json
{
  "run_id": "RUN-SPY-20250505113632-48ccd3c0",
  "start_time": "2025-05-05T11:36:32",
  "end_time": "2025-05-05T12:45:10",
  "parameters": {
    "learning_rate": 0.001,
    "batch_size": 64,
    "gamma": 0.99,
    "symbol": "SPY",
    "model_type": "PPO"
  },
  "status": "completed",
  "notes": "Production model training"
}
```

**Error Response (404 Not Found)**

```json
{
  "detail": "Training run with ID 'RUN-SPY-20250505113632-48ccd3c0' not found"
}
```

### GET /api/v1/runs/{run_id}/episodes/

Retrieve a paginated list of episodes for a specific training run, optionally filtered by performance metrics and date range.

**Path Parameters**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| run_id | string | Yes | The ID of the training run whose episodes to retrieve |

**Query Parameters**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| page | integer | No | Page number (≥1, default=1) |
| page_size | integer | No | Items per page (1-100, default=20) |
| min_pnl | float | No | Filter episodes with PnL greater than or equal to this value |
| max_sharpe | float | No | Filter episodes with Sharpe Ratio less than or equal to this value |
| start_date | date | No | Filter episodes started on or after this date (YYYY-MM-DD) |
| end_date | date | No | Filter episodes started on or before this date (YYYY-MM-DD) |

**Success Response (200 OK)**

```json
{
  "total_items": 100,
  "total_pages": 5,
  "current_page": 1,
  "page_size": 20,
  "items": [
    {
      "episode_id": 47,
      "run_id": "RUN-SPY-20250505113632-48ccd3c0",
      "start_time": "2025-05-05T11:40:15",
      "end_time": "2025-05-05T11:42:30",
      "initial_portfolio_value": 10000.0,
      "final_portfolio_value": 10250.5,
      "pnl": 250.5,
      "sharpe_ratio": 1.45,
      "max_drawdown": -50.0,
      "total_reward": 25.05,
      "total_steps": 240,
      "win_rate": 0.65
    }
  ]
}
```

**Error Response (404 Not Found)**

```json
{
  "detail": "Training run with ID 'RUN-SPY-20250505113632-48ccd3c0' not found"
}
```

### GET /api/v1/runs/{run_id}/episodes/summary/

Calculate and retrieve aggregated performance metrics for all episodes within a specific training run.

**Path Parameters**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| run_id | string | Yes | The ID of the training run whose episodes to summarize |

**Success Response (200 OK)**

```json
{
  "run_id": "RUN-SPY-20250505113632-48ccd3c0",
  "total_episodes": 50,
  "average_pnl": 175.25,
  "median_pnl": 165.5,
  "average_sharpe_ratio": 1.25,
  "median_sharpe_ratio": 1.2,
  "average_max_drawdown": -75.5,
  "median_max_drawdown": -65.0,
  "average_win_rate": 0.62,
  "median_win_rate": 0.60
}
```

**Error Response (404 Not Found)**

```json
{
  "detail": "Training run with ID 'RUN-SPY-20250505113632-48ccd3c0' not found"
}
```

## Episodes Endpoints

### GET /api/v1/episodes/ids

Retrieve a list of all distinct episode IDs available in the database, sorted in descending order (newest first).

**Success Response (200 OK)**

```json
{
  "episode_ids": [50, 49, 48, 47, 46]
}
```

**Error Response (500 Internal Server Error)**

```json
{
  "detail": "Internal server error fetching episode IDs"
}
```

### GET /api/v1/episodes/{episode_id}

Retrieve details for a specific episode by its ID.

**Path Parameters**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| episode_id | integer | Yes | The ID of the episode to retrieve |

**Success Response (200 OK)**

```json
{
  "episode_id": 47,
  "run_id": "RUN-SPY-20250505113632-48ccd3c0",
  "start_time": "2025-05-05T11:40:15",
  "end_time": "2025-05-05T11:42:30",
  "initial_portfolio_value": 10000.0,
  "final_portfolio_value": 10250.5,
  "pnl": 250.5,
  "sharpe_ratio": 1.45,
  "max_drawdown": -50.0,
  "total_reward": 25.05,
  "total_steps": 240,
  "win_rate": 0.65
}
```

**Error Response (404 Not Found)**

```json
{
  "detail": "Episode with ID 47 not found"
}
```

### GET /api/v1/episodes/{episode_id}/steps/

Retrieve a paginated list of steps for a specific episode.

**Path Parameters**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| episode_id | integer | Yes | The ID of the episode whose steps to retrieve |

**Query Parameters**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| page | integer | No | Page number (≥1, default=1) |
| page_size | integer | No | Items per page (1-100, default=20) |

**Success Response (200 OK)**

```json
{
  "total_items": 240,
  "total_pages": 12,
  "current_page": 1,
  "page_size": 20,
  "items": [
    {
      "step_id": 1024,
      "episode_id": 47,
      "timestamp": "2025-05-05T11:40:15",
      "portfolio_value": 10000.0,
      "reward": 0.0,
      "action": "hold",
      "position": "flat",
      "asset_price": 452.75
    }
  ]
}
```

**Error Response (404 Not Found)**

```json
{
  "detail": "Episode with ID 47 not found"
}
```

### GET /api/v1/episodes/{episode_id}/trades/

Retrieve a paginated list of trades for a specific episode.

**Path Parameters**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| episode_id | integer | Yes | The ID of the episode whose trades to retrieve |

**Query Parameters**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| page | integer | No | Page number (≥1, default=1) |
| page_size | integer | No | Items per page (1-100, default=20) |

**Success Response (200 OK)**

```json
{
  "total_items": 15,
  "total_pages": 1,
  "current_page": 1,
  "page_size": 20,
  "items": [
    {
      "trade_id": 128,
      "episode_id": 47,
      "entry_time": "2025-05-05T11:40:30",
      "exit_time": "2025-05-05T11:41:15",
      "entry_price": 450.50,
      "exit_price": 452.25,
      "quantity": 10.0,
      "direction": "long",
      "pnl": 17.5,
      "costs": 2.5
    }
  ]
}
```

**Error Response (404 Not Found)**

```json
{
  "detail": "Episode with ID 47 not found"
}
```

### GET /api/v1/episodes/{episode_id}/operations/

Retrieve a paginated list of trading operations for a specific episode, ordered by timestamp.

**Path Parameters**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| episode_id | integer | Yes | The ID of the episode whose operations to retrieve |

**Query Parameters**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| page | integer | No | Page number (≥1, default=1) |
| page_size | integer | No | Items per page (1-1000, default=100) |

**Success Response (200 OK)**

```json
{
  "total_items": 30,
  "total_pages": 1,
  "current_page": 1,
  "page_size": 100,
  "items": [
    {
      "operation_id": 256,
      "step_id": 1030,
      "timestamp": "2025-05-05T11:40:30",
      "operation_type": "ENTRY_LONG",
      "size": 10.0,
      "price": 450.50
    }
  ]
}
```

Note: The `operation_type` field is an enum with the following possible values:
- `ENTRY_LONG`: Enter a long position
- `EXIT_LONG`: Exit a long position
- `ENTRY_SHORT`: Enter a short position
- `EXIT_SHORT`: Exit a short position
- `HOLD`: No position change

**Error Response (404 Not Found)**

```json
{
  "detail": "Episode with ID 47 not found"
}
```

### GET /api/v1/episodes/{episode_id}/model/

Retrieve the training parameters (model configuration) associated with a specific episode's training run.

**Path Parameters**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| episode_id | integer | Yes | The ID of the episode whose model parameters to retrieve |

**Success Response (200 OK)**

```json
{
  "parameters": {
    "learning_rate": 0.001,
    "batch_size": 64,
    "gamma": 0.99,
    "symbol": "SPY",
    "model_type": "PPO",
    "nn_layers": [256, 256],
    "activation": "tanh"
  }
}
```

**Error Response (404 Not Found)**

```json
{
  "detail": "Episode with ID 47 not found"
}
```

or

```json
{
  "detail": "Training run associated with episode 47 not found"
}