import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

# Assuming your FastAPI app instance is accessible, e.g., from main
# Adjust the import path as necessary
from reinforcestrategycreator.api.main import app
from reinforcestrategycreator import db_models
from reinforcestrategycreator.api.dependencies import get_api_key # For overriding dependency

# Dummy API Key for testing
TEST_API_KEY = "test-key-123"

# --- Test Client Setup ---
# This might be provided by a fixture in conftest.py in a real setup
# For now, we instantiate it directly.
client = TestClient(app)

# --- Dependency Override ---
# Override the API key dependency for testing
async def override_get_api_key():
    return TEST_API_KEY

app.dependency_overrides[get_api_key] = override_get_api_key


# --- Test Database Fixture ---
# This is a placeholder. A real setup would likely use pytest fixtures
# (e.g., pytest-fastapi-deps, or custom fixtures in conftest.py)
# to manage test database sessions and data rollback.
@pytest.fixture(scope="function")
def test_db_session():
    # In a real scenario, this would yield a configured test session
    # and handle cleanup/rollback. For now, it's conceptual.
    # from reinforcestrategycreator.db_utils import SessionLocal # Example import
    # db = SessionLocal()
    # try:
    #     yield db
    # finally:
    #     db.rollback() # Ensure clean state
    #     db.close()
    # For this example, we'll pass None and handle it in the test
    # This WILL NOT WORK without a proper fixture setup.
    yield None # Placeholder

# --- Test Function ---

def test_list_episode_steps_includes_asset_price(test_db_session: Session | None):
    """
    Test that the /episodes/{episode_id}/steps/ endpoint returns steps
    including the asset_price field.
    """
    # --- Test Data Setup ---
    # This part requires a working test_db_session fixture
    if test_db_session is None:
        pytest.skip("Skipping test: Requires a working test database session fixture.")
        return # Added return to satisfy type checker

    # Create a dummy Training Run
    test_run = db_models.TrainingRun(
        run_id="test_run_for_episode_api",
        symbol="TEST",
        start_date="2023-01-01",
        end_date="2023-01-10",
        parameters={"lr": 0.001},
        git_hash="test_hash",
        version="1.0"
    )
    test_db_session.add(test_run)
    test_db_session.flush() # Assign run_id

    # Create a dummy Episode linked to the Training Run
    test_episode = db_models.Episode(
        episode_id=999, # Use a distinct ID for testing
        run_id=test_run.run_id,
        start_balance=10000.0,
        end_balance=10500.0,
        total_reward=500.0,
        total_steps=10,
        # Add other required fields if any
    )
    test_db_session.add(test_episode)
    test_db_session.flush() # Assign episode_id if auto-generated (though we set it)

    # Create a dummy Step linked to the Episode
    test_step = db_models.Step(
        episode_id=test_episode.episode_id,
        step_index=1,
        timestamp="2023-01-01T10:00:00", # Example timestamp
        action=1, # Buy
        reward=10.5,
        portfolio_value=10010.50,
        asset_price=150.75 # The crucial field to test
        # Add other required fields if any
    )
    test_db_session.add(test_step)
    test_db_session.commit() # Commit test data

    # --- API Call ---
    episode_id_to_test = test_episode.episode_id
    response = client.get(
        f"/api/v1/episodes/{episode_id_to_test}/steps/",
        headers={"X-API-Key": TEST_API_KEY}
    )

    # --- Assertions ---
    assert response.status_code == 200
    response_data = response.json()

    assert "items" in response_data
    assert isinstance(response_data["items"], list)
    assert len(response_data["items"]) > 0 # Should find the step we added

    first_step = response_data["items"][0]
    assert "asset_price" in first_step
    assert first_step["asset_price"] == test_step.asset_price # Verify the value

    # --- Cleanup (handled by fixture ideally) ---
    # test_db_session.delete(test_step)
    # test_db_session.delete(test_episode)
    # test_db_session.delete(test_run)
    # test_db_session.commit()

# Add more tests for other episode endpoints as needed...