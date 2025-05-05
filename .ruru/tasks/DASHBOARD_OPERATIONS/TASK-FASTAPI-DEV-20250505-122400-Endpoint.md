+++
id = "TASK-FASTAPI-DEV-20250505-122400-Endpoint"
title = "Implement API Endpoint for Trading Operations per Episode"
status = "🟢 Done"
type = "🌟 Feature" # Sub-task of a feature
priority = "🔼 High"
created_date = "2025-05-05"
updated_date = "2025-05-05" # Keeping date, time updated implicitly
assigned_to = "framework-fastapi"
coordinator = "TASK-BE-LEAD-20250505-122100" # This Backend Lead task
parent_task = "TASK-BE-LEAD-20250505-122100"
depends_on = ["TASK-FASTAPI-DEV-20250505-122200-Schema"] # Depends on the schema being created
related_docs = [
    "reinforcestrategycreator/db_models.py",
    "reinforcestrategycreator/api/routers/episodes.py",
    "reinforcestrategycreator/api/schemas/operations.py",
    "reinforcestrategycreator/api/dependencies.py",
    ".ruru/tasks/DASHBOARD_OPERATIONS/TASK-BE-LEAD-20250505-122100.md",
    ".ruru/tasks/DASHBOARD_OPERATIONS/TASK-FASTAPI-DEV-20250505-122200-Schema.md"
    ]
tags = ["backend", "api", "fastapi", "sqlalchemy", "database", "dashboard", "operations", "pagination"]
template_schema_doc = ".ruru/templates/toml-md/01_mdtm_feature.README.md"
+++

# Implement API Endpoint for Trading Operations per Episode

## Description ✍️

Implement the FastAPI endpoint `GET /api/v1/episodes/{episode_id}/operations/` to retrieve a paginated list of trading operations for a specific episode.

*   **Parent Task:** `TASK-BE-LEAD-20250505-122100`
*   **Schema Reference:** `reinforcestrategycreator.api.schemas.TradingOperationRead` (created in `TASK-FASTAPI-DEV-20250505-122200-Schema`)
*   **Router File:** `reinforcestrategycreator/api/routers/episodes.py`
*   **DB Model:** `reinforcestrategycreator.db_models.TradingOperation`

## Acceptance Criteria ✅

*   - [✅] A new route function (e.g., `read_episode_operations`) is added to `reinforcestrategycreator/api/routers/episodes.py`.
*   - [✅] The route decorator is configured for `GET /episodes/{episode_id}/operations/`.
*   - [✅] The `response_model` is set to `schemas.PaginatedResponse[schemas.TradingOperationRead]`. *(Note: Used PaginatedResponse for consistency)*
*   - [✅] The endpoint accepts `episode_id: int` as a path parameter.
*   - [✅] The endpoint accepts optional query parameters `page: int = 1` and `page_size: int = Query(100, ge=1, le=1000)`.
*   - [✅] The endpoint uses the injected SQLAlchemy `Session` dependency (`db: DBSession`).
*   - [✅] The endpoint uses the API key dependency (`api_key: str = Depends(get_api_key)`). *(Note: Used str as per existing code)*
*   - [✅] Database query retrieves `TradingOperation` objects filtered by the provided `episode_id`.
*   - [✅] Results are ordered by `timestamp` (ascending).
*   - [✅] Pagination is correctly implemented using `offset()` and `limit()` based on `page` and `page_size`. (`offset = (page - 1) * limit`)
*   - [✅] If no operations are found for the `episode_id`, an empty `items` list within `PaginatedResponse` is returned (HTTP 200).
*   - [✅] If the specified `episode_id` does not correspond to an existing `Episode` in the database, raise an `HTTPException` with `status_code=404` and detail "Episode not found".
*   - [✅] Necessary imports are added (`TradingOperation`, `Episode`, `TradingOperationRead`). *(Note: Others were present or implicit)*

## Implementation Notes / Sub-Tasks 📝

*   - [✅] Add imports to `episodes.py`: `TradingOperation`, `TradingOperationRead`, `Episode`.
*   - [✅] Define the new route function `read_episode_operations` with the correct signature and decorator.
*   - [✅] **First**, query the `Episode` table to check if `episode_id` exists. If not, raise `HTTPException(status_code=404, detail="Episode not found")`.
*   - [✅] Implement the SQLAlchemy query for `TradingOperation`, applying filtering, ordering, offset, and limit.
*   - [✅] Return the list of operations (within `PaginatedResponse`).

## Diagrams 📊 (Optional)

See parent task `TASK-BE-LEAD-20250505-122100`.

## AI Prompt Log 🤖 (Optional)

N/A

## Review Notes 👀 (For Reviewer)

*   Verify endpoint path, method, and response model.
*   Check pagination logic calculation (`offset`, `limit`).
*   Confirm correct filtering by `episode_id` and ordering by `timestamp`.
*   Ensure 404 handling for non-existent `episode_id` is implemented *before* querying operations.
*   Verify API key dependency is present.
*   Check imports.

## Key Learnings 💡 (Optional - Fill upon completion)

N/A