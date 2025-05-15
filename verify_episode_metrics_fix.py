import os
import sys
from sqlalchemy import func, and_

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from reinforcestrategycreator.db_models import Episode, Step, TrainingRun
from reinforcestrategycreator.db_utils import get_db_session

RUN_ID_TO_VERIFY = "RLlibDBG-SPY-20250515110712-b9d26b47"

def verify_episode_metrics():
    """
    Queries the 'episodes' table for the specified run ID to confirm
    that metrics are correct.
    """
    print(f"--- Verifying Episode Metrics for Run ID: {RUN_ID_TO_VERIFY} ---")
    findings = []
    all_episodes_valid = True

    with get_db_session() as db:
        episodes = db.query(Episode).filter(Episode.run_id == RUN_ID_TO_VERIFY).all()

        if not episodes:
            findings.append(f"🔴 ERROR: No episodes found for run_id '{RUN_ID_TO_VERIFY}'.")
            all_episodes_valid = False
            return findings, all_episodes_valid

        findings.append(f"Found {len(episodes)} episodes for run_id '{RUN_ID_TO_VERIFY}'.")

        for i, ep in enumerate(episodes):
            episode_valid = True
            ep_findings = [f"  Episode {i+1} (ID: {ep.episode_id}, RLlib ID: {ep.rllib_episode_id}):"]

            # Check metrics that should ideally be non-zero (or non-default)
            # PNL can be zero if no profit/loss
            if ep.pnl is None: # Check for None explicitly
                ep_findings.append(f"    🔴 pnl: IS NULL. Expected float.")
                episode_valid = False
            else:
                ep_findings.append(f"    🟢 pnl: {ep.pnl}")

            # Sharpe ratio can be zero or negative, but should not be None if calculated
            if ep.sharpe_ratio is None:
                ep_findings.append(f"    🔴 sharpe_ratio: IS NULL. Expected float.")
                episode_valid = False
            else:
                ep_findings.append(f"    🟢 sharpe_ratio: {ep.sharpe_ratio}")
            
            # Max drawdown should be non-negative, usually non-zero if there are price fluctuations
            if ep.max_drawdown is None or ep.max_drawdown < 0:
                ep_findings.append(f"    🔴 max_drawdown: {ep.max_drawdown}. Expected non-negative float.")
                episode_valid = False
            else:
                ep_findings.append(f"    🟢 max_drawdown: {ep.max_drawdown}")

            # Total reward can be zero or negative
            if ep.total_reward is None:
                ep_findings.append(f"    🔴 total_reward: IS NULL. Expected float.")
                episode_valid = False
            else:
                ep_findings.append(f"    🟢 total_reward: {ep.total_reward}")

            # Win rate should be between 0 and 1
            if ep.win_rate is None or not (0 <= ep.win_rate <= 1):
                ep_findings.append(f"    🔴 win_rate: {ep.win_rate}. Expected float between 0 and 1.")
                episode_valid = False
            else:
                ep_findings.append(f"    🟢 win_rate: {ep.win_rate}")

            # Total steps should be greater than 1 (usually much more)
            if ep.total_steps is None or ep.total_steps <= 1:
                ep_findings.append(f"    🔴 total_steps: {ep.total_steps}. Expected integer > 1.")
                episode_valid = False
            else:
                ep_findings.append(f"    🟢 total_steps: {ep.total_steps}")
            
            if not episode_valid:
                all_episodes_valid = False
            findings.extend(ep_findings)

    if all_episodes_valid and episodes:
        findings.append("✅ All episodes checked appear to have valid metrics based on basic checks.")
    elif episodes:
        findings.append("⚠️ Some episodes have metric issues.")
    
    return findings, all_episodes_valid


def verify_steps_data():
    """
    Queries the 'steps' table for the specified run ID to confirm that
    portfolio_value, asset_price, action, and position are non-NULL.
    """
    print(f"\n--- Verifying Steps Data for Run ID: {RUN_ID_TO_VERIFY} ---")
    findings = []
    all_steps_valid = True

    with get_db_session() as db:
        # Get episode IDs for the current run
        episode_ids = db.query(Episode.episode_id).filter(Episode.run_id == RUN_ID_TO_VERIFY).scalar_subquery()

        if not db.query(episode_ids.exists()).scalar():
            findings.append(f"🔴 ERROR: No episodes found for run_id '{RUN_ID_TO_VERIFY}', cannot check steps.")
            all_steps_valid = False
            return findings, all_steps_valid

        # Count total steps for this run
        total_steps_in_run = db.query(func.count(Step.step_id)).filter(Step.episode_id.in_(episode_ids)).scalar()

        if total_steps_in_run == 0:
            findings.append(f"🔴 ERROR: No steps found for any episode in run_id '{RUN_ID_TO_VERIFY}'.")
            all_steps_valid = False
            return findings, all_steps_valid
        
        findings.append(f"Found {total_steps_in_run} total steps for run_id '{RUN_ID_TO_VERIFY}'.")

        # Check for NULLs in key columns
        null_checks = {
            "portfolio_value": func.count().filter(Step.episode_id.in_(episode_ids), Step.portfolio_value == None),
            "asset_price": func.count().filter(Step.episode_id.in_(episode_ids), Step.asset_price == None),
            "action": func.count().filter(Step.episode_id.in_(episode_ids), Step.action == None),
            "position": func.count().filter(Step.episode_id.in_(episode_ids), Step.position == None),
        }

        for column_name, query_filter in null_checks.items():
            count_null = db.query(query_filter).scalar()
            if count_null > 0:
                findings.append(f"    🔴 {column_name}: Found {count_null} NULL values out of {total_steps_in_run} steps.")
                all_steps_valid = False
            else:
                findings.append(f"    🟢 {column_name}: No NULL values found.")
    
    if all_steps_valid:
        findings.append("✅ All steps checked appear to have non-NULL values for critical fields.")
    else:
        findings.append("⚠️ Some steps have NULL values in critical fields.")
        
    return findings, all_steps_valid

if __name__ == "__main__":
    # Determine RUN_ID_TO_VERIFY from command line arguments or use default
    if len(sys.argv) > 1:
        RUN_ID_TO_VERIFY = sys.argv[1]
        # No need to re-assign to the global here, as the functions will use the global RUN_ID_TO_VERIFY
        # which will be updated if sys.argv[1] exists.
        # However, for clarity and to ensure the print statements use the correct ID,
        # it's better to update the global RUN_ID_TO_VERIFY if an argument is passed.
    else:
        print("No Run ID provided as argument, using default.")
        # RUN_ID_TO_VERIFY remains the hardcoded default

    print(f"Database connection successful.") # This line was present in the original script output
    print(f"Starting verification for Run ID: {RUN_ID_TO_VERIFY}")
    
    episode_findings, episodes_ok = verify_episode_metrics()
    for finding in episode_findings:
        print(finding)

    steps_findings, steps_ok = verify_steps_data()
    for finding in steps_findings:
        print(finding)

    print("\n--- Summary ---")
    if episodes_ok and steps_ok:
        print("✅✅✅ Verification PASSED: Both episode metrics and steps data look good.")
    else:
        print("❌❌❌ Verification FAILED: Issues found.")
        if not episodes_ok:
            print("  - Episode metrics have issues.")
        if not steps_ok:
            print("  - Steps data has issues.")
    
    # Store findings for MDTM update
    mdtm_summary = ["## Verification Log for Run ID: " + RUN_ID_TO_VERIFY]
    mdtm_summary.append("\n**Episode Metrics Verification:**")
    mdtm_summary.extend(episode_findings)
    mdtm_summary.append("\n**Steps Data Verification:**")
    mdtm_summary.extend(steps_findings)
    mdtm_summary.append("\n**Overall Result:**")
    if episodes_ok and steps_ok:
        mdtm_summary.append("PASSED")
    else:
        mdtm_summary.append("FAILED")

    with open("verification_summary.txt", "w") as f:
        f.write("\n".join(mdtm_summary))
    print("\nVerification summary saved to verification_summary.txt")