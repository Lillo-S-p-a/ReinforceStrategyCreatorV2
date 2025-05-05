+++
id = "TASK-FASTAPI-DEV-20250505-122200-Schema"
title = "Create Pydantic Schema for Trading Operations"
status = "🟢 Done"
type = "🌟 Feature" # Sub-task of a feature
priority = "🔼 High"
created_date = "2025-05-05"
updated_date = "2025-05-05" # Date is already correct, ensuring it stays
assigned_to = "framework-fastapi"
coordinator = "TASK-BE-LEAD-20250505-122100" # This Backend Lead task
parent_task = "TASK-BE-LEAD-20250505-122100"
depends_on = [] # Depends on db_models.py which already exists
related_docs = [
    "reinforcestrategycreator/db_models.py",
    "reinforcestrategycreator/api/schemas/",
    ".ruru/tasks/DASHBOARD_OPERATIONS/TASK-BE-LEAD-20250505-122100.md"
    ]
tags = ["backend", "api", "fastapi", "pydantic", "schema", "dashboard", "operations"]
template_schema_doc = ".ruru/templates/toml-md/01_mdtm_feature.README.md"
+++

# Create Pydantic Schema for Trading Operations

## Description ✍️

Create a Pydantic schema (`TradingOperationRead`) to represent a trading operation when returned by the API. This schema will be used as the `response_model` for the new endpoint retrieving operations for an episode.

*   **Parent Task:** `TASK-BE-LEAD-20250505-122100` (Create API Endpoint for Trading Operations per Episode)
*   **Model Reference:** `reinforcestrategycreator.db_models.TradingOperation` and `reinforcestrategycreator.db_models.OperationType`

## Acceptance Criteria ✅

*   - [✅] A new file `reinforcestrategycreator/api/schemas/operations.py` is created.
*   - [✅] The file contains a Pydantic schema named `TradingOperationRead` inheriting from `pydantic.BaseModel`.
*   - [✅] The schema includes the following fields with correct types:
    *   `operation_id: int`
    *   `step_id: int`
    *   `timestamp: datetime` (from `datetime` module)
    *   `operation_type: OperationType` (imported from `reinforcestrategycreator.db_models`)
    *   `size: float`
    *   `price: float`
*   - [✅] Necessary imports (`datetime`, `OperationType`, `BaseModel`) are included.
*   - [✅] The new schema `TradingOperationRead` is exposed by adding `from .operations import TradingOperationRead` to `reinforcestrategycreator/api/schemas/__init__.py`.

## Implementation Notes / Sub-Tasks 📝

*   - [✅] Create the file `reinforcestrategycreator/api/schemas/operations.py`.
*   - [✅] Define the `TradingOperationRead` class within the new file.
*   - [✅] Add the required imports.
*   - [✅] Update `reinforcestrategycreator/api/schemas/__init__.py` to include the new schema.

## Diagrams 📊 (Optional)

N/A

## AI Prompt Log 🤖 (Optional)

N/A

## Review Notes 👀 (For Reviewer)

*   Verify field names and types match the `TradingOperation` model and requirements.
*   Confirm correct import and usage of `OperationType` enum.
*   Check that the schema is correctly exposed in `schemas/__init__.py`.

## Key Learnings 💡 (Optional - Fill upon completion)

N/A