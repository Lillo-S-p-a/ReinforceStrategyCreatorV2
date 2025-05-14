+++
id = "TASK-LEADBACKEND-20250514-222900"
title = "Ensure Complete Episode Metric Logging for All Episodes"
status = "🟡 To Do"
type = "🌟 Feature"
assigned_to = "lead-backend"
coordinator = "RooCommander-Session-20250514-222900" # Using a session-based ID for RooCommander
created_date = "2025-05-14"
updated_date = "2025-05-14"
priority = "High"
complexity = "Medium"
tags = ["database", "logging", "rllib", "callbacks", "episode-lifecycle"]
related_docs = [
    "reinforcestrategycreator/callbacks.py",
    "train_debug.py",
    "reinforcestrategycreator/trading_environment.py"
]
blocked_by = []
blocks = []
# --- User Story (Optional) ---
# As a user, I want all episodes, including those active at the end of a training run,
# to have their final metrics correctly logged to the database,
# so that I have a complete record of training performance.
+++

# Description

Currently, RL training episodes that are still active when a training run concludes (e.g., due to `NUM_TRAINING_ITERATIONS` being met or `algo.stop()` being called) do not have their final metrics (PNL, Sharpe ratio, status, etc.) fully logged to the database. They remain in a 'started' state with NULL values for most metrics.

For example, in debug run `RLlibDBG-SPY-20250514202212-b7567af3`, the first episode (DB ID 2027) was fully logged, but the second episode (DB ID 2028) was left incomplete because the single training iteration finished before it naturally terminated.

This task is to investigate and implement a robust solution to ensure comprehensive logging for all episodes.

# Acceptance Criteria

1.  All episodes initiated during a training run have their final status updated from 'started' to 'completed' (or 'truncated', 'error', as appropriate) and all relevant performance metrics (PNL, Sharpe ratio, max drawdown, win rate, total steps, total reward, final portfolio value, end time) correctly recorded in the `episodes` database table.
2.  This logging should occur reliably for episodes that complete naturally during training *and* for episodes that are effectively terminated because the overall training process is ending.
3.  The solution should be robust for both short debug runs (e.g., 1 iteration that might spawn multiple concurrent episodes) and longer training sessions.
4.  The `check_episode_details.py` script, when run after a test training session, should report all episodes for that session as fully populated with key metrics.
5.  The solution should minimize impact on training performance if possible.

# Checklist

-   [✅] **Research**: Review RLlib documentation for configuration options like `batch_mode: "complete_episodes"`, `callbacks_batch_mode`, or other settings that might influence episode completion and data availability at the end of training iterations or `algo.stop()`.
-   [ ] **Investigate Callbacks**:
    -   [✅] Determine if the `on_train_result(algorithm, result, **kwargs)` callback can be used to inspect the state of ongoing episodes and trigger final logging if the training is about to end. (Finding: No direct access to *active* episodes, only those completed in the last iteration.)
    -   [✅] Explore if `on_algorithm_stop(algorithm, **kwargs)` (if available and suitably triggered) or a similar shutdown hook can be leveraged to process any remaining active episodes. (Finding: Called too late, workers shutting down; not suitable for processing active episodes.)
-   [✅] **Environment Interaction**: Assess if the `TradingEnv` needs to be queried directly (if accessible from a suitable callback at shutdown) for final states of unterminated episodes. (Finding: Not feasible *from a shutdown callback*. However, interaction with workers/environments *before* `algo.stop()` as part of a graceful shutdown in the main script is the way forward.)
-   [✅] **Design Solution**: Propose a preferred approach (e.g., modifying existing callbacks, adding logic to the main training script around `algo.stop()`, or a combination).
-   [✅] **Implement Solution**: Make necessary code changes in [`reinforcestrategycreator/callbacks.py`](reinforcestrategycreator/callbacks.py), [`train_debug.py`](train_debug.py)/[`train.py`](train.py), or other relevant files.
-   [ ] **Testing**:
    -   [ ] Conduct test runs with `train_debug.py` (e.g., `NUM_TRAINING_ITERATIONS = 1` but ensuring conditions where multiple episodes might be active).
    -   [ ] Use `check_episode_details.py` to verify that all episodes from the test run are fully logged with non-NULL metrics.
    -   [ ] Verify that `completed_trades` are also logged for these episodes.
-   [ ] **Documentation**: Briefly document the chosen solution, its rationale, and any important considerations or limitations in the task log or related project documentation.

# Notes & Logs
(Specialist to add logs and notes here during task execution)

**2025-05-14 (lead-backend):**
*   **Research Findings (Perplexity):**
    *   `batch_mode: "complete_episodes"` (likely current setting) discards in-progress episodes when training stops. This is the probable cause of incomplete logging.
    *   `batch_mode: "truncate_episodes"` would capture partial data but metrics might be incomplete/bootstrapped.
    *   A "Graceful Shutdown Pattern" is recommended: allow active episodes to finish naturally before `algo.stop()` is called. This seems the most promising direction.
    *   `callbacks_batch_mode` influences what callbacks see but doesn't solve the core issue of discarded episodes.
*   **Callback Investigation Plan:**
    *   Current `callbacks.py` uses `on_episode_start`, `on_episode_end`, `on_sample_end`, `on_episode_step`.
    *   Will investigate adding `on_train_result` to see if it's possible to detect the end of training and access active episodes.
    *   Will investigate adding `on_algorithm_stop` as a potential hook for final processing of active episodes.
*   **Callback Investigation Findings (Perplexity):**
    *   `on_train_result`: Invoked after each training iteration. Provides access to episodes *completed* during that iteration. Does **not** provide direct access to episodes still active/ongoing.
    *   `on_algorithm_stop`: Invoked when `algo.stop()` is called, just before full shutdown. By this time, workers are already shutting down, and there's no reliable way to access or process episodes that were active when `algo.stop()` was initiated.
    *   Conclusion: Neither callback is suitable for directly processing and finalizing metrics for episodes that are active when training concludes. The "Graceful Shutdown Pattern" or modifications to the main training script logic around `algo.stop()` are more viable.
*   **Graceful Shutdown Pattern Investigation (Perplexity):**
    *   A "Graceful Shutdown" involves signaling workers to finish current episodes but not start new ones, waiting for completion, then calling `algo.stop()`.
    *   This requires custom logic, potentially involving:
        *   Custom methods on `RolloutWorker`s or `TradingEnv` to signal "stop accepting new episodes".
        *   Polling workers (e.g., `algo.workers.remote_workers()`) or using a synchronization mechanism to check if all active episodes are done.
        *   This interaction would happen in the main training script ([`train_debug.py`](train_debug.py)/[`train.py`](train.py)) *before* `algo.stop()` is called.
    *   The existing `on_episode_end` callback in [`callbacks.py`](reinforcestrategycreator/callbacks.py) should then naturally log these episodes as they complete during the draining phase.
*   **Design Proposal (Graceful Shutdown in Main Training Script):**
    1.  **Modify `TradingEnv`**:
        *   Add a `self.graceful_shutdown_signaled = False` flag.
        *   Add a method `signal_graceful_shutdown(self)` that sets this flag to `True`.
        *   In the `step()` method of `TradingEnv`, if `self.graceful_shutdown_signaled` is `True`, the environment should aim to terminate the episode quickly but cleanly. This might mean:
            *   No new trades.
            *   Liquidate existing positions (if simple and quick, otherwise just hold).
            *   Return `terminated=True` or `truncated=True` in the `step()` result after a few steps or a very short internal countdown.
            *   Ensure all final metrics are calculated and available in the `info` dict returned by `step()`.
    2.  **Modify Main Training Script (`train_debug.py`, `train.py`)**:
        *   Locate the section where training iterations complete (e.g., after the `for i in range(NUM_TRAINING_ITERATIONS): algo.train()` loop).
        *   **Before** calling `algo.stop()`:
            *   Log that graceful shutdown is being initiated.
            *   Call `algo.workers.foreach_env(lambda env: env.signal_graceful_shutdown())` to set the flag in all environment instances.
            *   Execute a "draining loop":
                *   Run `algo.train()` for a small, fixed number of additional iterations (e.g., 2-5, configurable) or for a short duration (e.g., 10-30 seconds). This allows RLlib to collect rollouts from the environments that are now trying to terminate.
                *   The existing `on_episode_end` callback in `callbacks.py` will be triggered as these episodes end, logging their data.
            *   Include a timeout mechanism for this draining phase to prevent indefinite hanging.
    3.  **Call `algo.stop()`**: After the draining loop/timeout, proceed with the original `algo.stop()` call.
    4.  **Configuration**: Add configuration options (e.g., `GRACEFUL_SHUTDOWN_DRAIN_ITERATIONS`, `GRACEFUL_SHUTDOWN_TIMEOUT`) to control the behavior.
*   **Implementation Summary:**
    *   Modified `reinforcestrategycreator/trading_environment.py`:
        *   Added `graceful_shutdown_signaled` flag (initialized to `False`, reset in `reset()`).
        *   Added `signal_graceful_shutdown(self)` method to set the flag.
        *   Updated `step()` method:
            *   Checks `graceful_shutdown_signaled`. If true, sets `terminated=True` for the current step.
            *   Adjusted price fetching for the very last step to avoid index out of bounds.
            *   Updated logging in `step()` to reflect graceful shutdown status.
    *   Modified `train_debug.py`:
        *   Added `GRACEFUL_SHUTDOWN_DRAIN_ITERATIONS` and `GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS` constants.
        *   Before `algo.stop()`:
            *   Logs initiation of graceful shutdown.
            *   Calls `algo.workers.foreach_env(lambda env: env.signal_graceful_shutdown())` (with a check for local worker if `num_remote_workers == 0`).
            *   Loops `GRACEFUL_SHUTDOWN_DRAIN_ITERATIONS` times, calling `algo.train()` in each iteration to process episodes.
            *   Includes a timeout for this draining phase.
            *   Updated logging in the `finally` block.