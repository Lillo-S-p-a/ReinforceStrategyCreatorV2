# Removed sys.path manipulation - handled by run_dashboard.py
import streamlit as st
import pandas as pd
import logging
import os
from typing import List, Dict, Any

# Import modules from our package
# Handle imports for both direct execution and module imports
# Use absolute imports assuming run_dashboard.py sets the path correctly
from dashboard.api import (
    fetch_latest_run, fetch_run_summary, fetch_run_episodes,
    fetch_episode_steps, fetch_episode_trades, fetch_episode_operations,
    fetch_episode_model
)
from dashboard.analysis import (
    analyze_decision_making, analyze_why_episode_performed,
    generate_episode_summary, calculate_additional_metrics
)
from dashboard.visualization import (
    create_price_operations_chart, create_drawdown_chart,
    create_trade_analysis_charts, create_action_analysis,
    create_reward_analysis, create_model_parameter_radar
)
from dashboard.model_management import (
    save_model_to_production, get_saved_production_models
)
from dashboard.utils import format_metric, ACTION_MAP

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Streamlit App ---
def main():
    st.set_page_config(layout="wide", page_title="Enhanced RL Trading Dashboard v2", page_icon="📈")
    st.title("📈 Enhanced RL Trading Agent Performance Dashboard")

    # Dashboard sidebar for config options
    st.sidebar.title("Configuration")
    theme = st.sidebar.selectbox("Dashboard Theme", ["Dark", "Light"], index=0)
    plot_template = "plotly_dark" if theme == "Dark" else "plotly_white"

    # Fetch Latest Run Info
    latest_run = fetch_latest_run()

    if latest_run:
        run_id = latest_run.get("run_id")
        st.sidebar.info(f"Displaying data for Run ID: {run_id}")
        with st.sidebar.expander("Run Parameters", expanded=False):
            st.json(latest_run.get("parameters", {}))

        # --- Production Model Management ---
        with st.sidebar.expander("Production Model Management", expanded=True):
            production_models = get_saved_production_models()
            st.write(f"**{len(production_models)} models saved for production**")
            
            if production_models:
                # Display a table of top 3 saved models
                model_table = []
                for i, model in enumerate(production_models[:3]):
                    metrics = model.get('metrics', {})
                    model_table.append({
                        "Filename": model.get('filename', 'Unknown'),
                        "Episode": model.get('episode_id', 'Unknown'),
                        "PnL": metrics.get('pnl', 'N/A'),
                        "Win Rate": metrics.get('win_rate', 'N/A'),
                        "Saved At": model.get('saved_at', 'Unknown')
                    })
                
                st.dataframe(pd.DataFrame(model_table))

        # --- Overall Summary ---
        st.header("📊 Run Summary")
        summary = fetch_run_summary(run_id)
        # st.write(f"DEBUG [main.py]: summary value = {summary}") # DEBUGGING REMOVED

        if summary:
            with st.expander("📊 Metrics Explanation"):
                st.markdown("""
                - **PnL**: Profit and Loss - The total profit or loss achieved by the agent.
                - **Win Rate**: Percentage of trades that were profitable.
                - **Sharpe Ratio**: Risk-adjusted return metric. Higher is better, values > 1 are good.
                - **Max Drawdown**: Largest percentage drop from peak to trough. Lower is better.
                - **Sortino Ratio**: Similar to Sharpe but only penalizes downside volatility.
                - **Calmar Ratio**: Return divided by maximum drawdown. Higher is better.
                """)
            
            col1, col2, col3, col4 = st.columns(4)
            # FIX: Use correct keys from API response
            with col1:
                # Use 'average_pnl' instead of 'pnl'
                st.metric("Avg PnL", format_metric(summary.get('average_pnl'), "pnl"))
                # Use 'average_win_rate' instead of 'win_rate'
                st.metric("Avg Win Rate", format_metric(summary.get('average_win_rate'), "percentage"))
            with col2:
                # Use 'average_sharpe_ratio' instead of 'sharpe_ratio'
                st.metric("Avg Sharpe Ratio", format_metric(summary.get('average_sharpe_ratio'), "ratio"))
                 # Use 'average_max_drawdown' instead of 'max_drawdown'
                st.metric("Avg Max Drawdown", format_metric(summary.get('average_max_drawdown'), "percentage"))
            with col3:
                 # API provides 'total_episodes', not 'total_trades'
                st.metric("Total Episodes", format_metric(summary.get('total_episodes'), "count"))
                 # API provides 'median_pnl', let's display that instead of non-existent 'avg_trade_pnl'
                st.metric("Median PnL", format_metric(summary.get('median_pnl'), "pnl"))
            with col4:
                 # API doesn't provide best episode info in summary, display medians instead
                st.metric("Median Sharpe", format_metric(summary.get('median_sharpe_ratio'), "ratio"))
                st.metric("Median Win Rate", format_metric(summary.get('median_win_rate'), "percentage"))
        else:
            st.warning("No summary data available for this run.")

        # --- Episode Selection ---
        st.header("🔍 Episode Analysis")
        episodes = fetch_run_episodes(run_id)
        
        # Log to help with debugging
        st.write(f"Found {len(episodes)} episodes")
        
        if episodes:
            # Convert to DataFrame for easier manipulation
            episodes_df = pd.DataFrame(episodes)
            # Log the column names to help with debugging
            st.write(f"Episode data columns: {', '.join(episodes_df.columns)}")
            
            # Add a column for episode selection display
            episodes_df['display_name'] = episodes_df.apply(
                lambda row: f"Episode {row['episode_id']} (PnL: {format_metric(row.get('pnl'), 'pnl')})", 
                axis=1
            )
            
            # Sort by episode_id for consistent ordering
            episodes_df = episodes_df.sort_values('episode_id')
            
            # Create a selectbox for episode selection
            # Extract episode_ids
            episode_ids = episodes_df['episode_id'].tolist()
            
            # Set default index to 0 (first episode)
            default_index = 0
            
            # Create selectbox with default selection
            selected_episode_id = st.selectbox(
                "Select Episode to Analyze:",
                episode_ids,
                index=default_index,
                format_func=lambda x: episodes_df.loc[episodes_df['episode_id'] == x, 'display_name'].iloc[0]
            )
            
            # Log the selected episode ID for debugging
            st.write(f"Selected Episode ID: {selected_episode_id}")
            
            # Display selected episode details
            if selected_episode_id:
                selected_episode = episodes_df.loc[episodes_df['episode_id'] == selected_episode_id].iloc[0]
                
                # Fetch episode data
                steps_df = fetch_episode_steps(selected_episode_id)
                trades = fetch_episode_trades(selected_episode_id)
                operations = fetch_episode_operations(selected_episode_id)
                model_data = fetch_episode_model(selected_episode_id)
                
                # Calculate additional metrics
                additional_metrics = calculate_additional_metrics(steps_df, trades)
                
                # Display episode metrics
                st.subheader(f"Episode {selected_episode_id} Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("PnL", format_metric(selected_episode.get('pnl'), "pnl"))
                    st.metric("Win Rate", format_metric(selected_episode.get('win_rate'), "percentage"))
                with col2:
                    st.metric("Sharpe Ratio", format_metric(selected_episode.get('sharpe_ratio'), "ratio"))
                    st.metric("Max Drawdown", format_metric(selected_episode.get('max_drawdown'), "percentage"))
                with col3:
                    st.metric("Sortino Ratio", format_metric(additional_metrics.get('sortino_ratio'), "ratio"))
                    st.metric("Calmar Ratio", format_metric(additional_metrics.get('calmar_ratio'), "ratio"))
                with col4:
                    st.metric("Volatility", format_metric(additional_metrics.get('volatility'), "percentage"))
                    st.metric("Avg Trade Duration", f"{additional_metrics.get('avg_trade_duration', 0):.1f} min")
                
                # Price and Operations Chart
                st.subheader("Price and Trading Operations")
                fig_price_ops, has_markers = create_price_operations_chart(steps_df, operations, template=plot_template)
                st.plotly_chart(fig_price_ops, use_container_width=True)
                
                if not has_markers:
                    st.info("No trading operations found for this episode.")
                
                # Drawdown Chart
                fig_drawdown = create_drawdown_chart(steps_df, template=plot_template)
                if fig_drawdown:
                    st.plotly_chart(fig_drawdown, use_container_width=True)
                
                # Decision Analysis
                decision_analysis = analyze_decision_making(steps_df)
                with st.expander("📈 Detailed Decision Analysis", expanded=False):
                    if decision_analysis:
                        st.subheader("Decision Making Patterns")
                        
                        # Action consistency
                        st.write(f"**Action Change Rate:** {decision_analysis.get('action_change_rate', 0):.2f}% - How often the agent changes its action")
                        
                        # Price responsiveness
                        st.write(f"**Buy on Price Up Rate:** {decision_analysis.get('buy_on_price_up_rate', 0):.2f}% - How often the agent buys when price is rising")
                        st.write(f"**Sell on Price Down Rate:** {decision_analysis.get('sell_on_price_down_rate', 0):.2f}% - How often the agent sells when price is falling")
                        
                        # Trend alignment
                        if 'buy_in_uptrend_rate' in decision_analysis:
                            st.write(f"**Buy in Uptrend Rate:** {decision_analysis.get('buy_in_uptrend_rate', 0):.2f}% - How often the agent buys during uptrends")
                            st.write(f"**Sell in Downtrend Rate:** {decision_analysis.get('sell_in_downtrend_rate', 0):.2f}% - How often the agent sells during downtrends")
                        
                        # Decision timing
                        if 'action_future_return_correlation' in decision_analysis:
                            st.write(f"**Action-Future Return Correlation:** {decision_analysis.get('action_future_return_correlation', 0):.3f} - How well actions correlate with future returns")
                        
                        # Decision contexts
                        if 'best_decision_context' in decision_analysis and 'worst_decision_context' in decision_analysis:
                            st.subheader("Decision Contexts")
                            
                            context_col1, context_col2 = st.columns(2)
                            
                            with context_col1:
                                st.write("**Best Decision Context:**")
                                best = decision_analysis['best_decision_context']
                                st.write(f"- Typical Action: {best.get('typical_action')}")
                                st.write(f"- Avg Price: {best.get('avg_price', 0):.2f}")
                                st.write(f"- Avg Price Change: {best.get('avg_price_change', 0)*100:.2f}%")
                                st.write(f"- Avg Reward: {best.get('avg_reward', 0):.4f}")
                            
                            with context_col2:
                                st.write("**Worst Decision Context:**")
                                worst = decision_analysis['worst_decision_context']
                                st.write(f"- Typical Action: {worst.get('typical_action')}")
                                st.write(f"- Avg Price: {worst.get('avg_price', 0):.2f}")
                                st.write(f"- Avg Price Change: {worst.get('avg_price_change', 0)*100:.2f}%")
                                st.write(f"- Avg Reward: {worst.get('avg_reward', 0):.4f}")
                    else:
                        st.info("Insufficient data for decision analysis.")
                
                # Trade Analysis
                with st.expander("📊 Trade Analysis", expanded=False):
                    if trades:
                        st.subheader("Trade Analysis")
                        
                        # Create trade analysis charts
                        fig_pnl_dist, fig_holding_pnl, fig_cum_pnl = create_trade_analysis_charts(trades, template=plot_template)
                        
                        if fig_pnl_dist and fig_holding_pnl and fig_cum_pnl:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.plotly_chart(fig_pnl_dist, use_container_width=True)
                            with col2:
                                st.plotly_chart(fig_holding_pnl, use_container_width=True)
                            
                            st.plotly_chart(fig_cum_pnl, use_container_width=True)
                    else:
                        st.info("No trades found for this episode.")
                
                # Model Analysis
                # Add logging to check model_data before rendering this section
                logging.info(f"Preparing Model Analysis section for episode {selected_episode_id}. Model data available: {bool(model_data)}")
                if model_data:
                    logging.info(f"Model data keys for episode {selected_episode_id}: {list(model_data.keys())}")

                # Move the 'if model_data' check outside the expander
                if model_data:
                    with st.expander("🧠 Model Analysis", expanded=False):
                        # Add episode ID to subheader for clarity
                        st.subheader(f"Model Parameters (Episode {selected_episode_id})")

                        model_col1, model_col2 = st.columns(2)

                        with model_col1:
                            st.subheader("Parameters") # Add a subheader for clarity
                            # Attempt to extract hyperparameters, default to empty dict if not found
                            # Extract parameters from the nested 'parameters' key provided by the API
                            params_to_display = model_data.get('parameters', {})

                            if params_to_display:
                                # Filter out any remaining complex types just in case
                                params_to_display_filtered = {
                                    k: v for k, v in params_to_display.items() if not isinstance(v, (list, dict))
                                }
                                if params_to_display_filtered:
                                    # Display parameters using st.text to avoid DataFrame/Arrow issues with mixed types
                                    for key, value in params_to_display_filtered.items():
                                        st.text(f"{key}: {value}")
                                else:
                                     st.info("No simple parameters found to display.") # Clarified message
                            else:
                                st.info("No parameters found in model data.")

                        with model_col2:
                            # Display radar chart
                            fig_radar = create_model_parameter_radar(model_data, template=plot_template)
                            st.plotly_chart(fig_radar, use_container_width=True)

                        # Feature extractors
                        if 'feature_extractors' in model_data:
                            st.subheader("Feature Extractors")
                            st.write(", ".join(model_data['feature_extractors']))

                        # Save model button (Temporarily commented out for debugging reactivity)
                        # if st.button("Save Model to Production"):
                        #     # Combine episode metrics with additional metrics
                        #     metrics = {
                        #         'pnl': selected_episode.get('pnl'),
                        #         'win_rate': selected_episode.get('win_rate'),
                        #         'sharpe_ratio': selected_episode.get('sharpe_ratio'),
                        #         'max_drawdown': selected_episode.get('max_drawdown'),
                        #         'sortino_ratio': additional_metrics.get('sortino_ratio'),
                        #         'calmar_ratio': additional_metrics.get('calmar_ratio'),
                        #         'volatility': additional_metrics.get('volatility')
                        #     }
                        #
                        #     saved_path = save_model_to_production(selected_episode_id, run_id, model_data, metrics)
                        #     if saved_path:
                        #         st.success(f"Model saved to production: {saved_path}")
                        #     else:
                        #         st.error("Failed to save model to production.")
                # This else corresponds to the outer 'if model_data:' check
                else:
                    st.info(f"No model data available for episode {selected_episode_id}.")
                
                # Market Adaptation
                with st.expander("🔄 Market Adaptation", expanded=False):
                    # Action distribution analysis
                    fig_action_dist, fig_transitions = create_action_analysis(steps_df, template=plot_template)
                    
                    if fig_action_dist and fig_transitions:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(fig_action_dist, use_container_width=True)
                        with col2:
                            st.plotly_chart(fig_transitions, use_container_width=True)
                    
                    # Reward analysis
                    fig_reward_dist, fig_running_reward = create_reward_analysis(steps_df, template=plot_template)
                    
                    if fig_reward_dist and fig_running_reward:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(fig_reward_dist, use_container_width=True)
                        with col2:
                            st.plotly_chart(fig_running_reward, use_container_width=True)
                
                # Performance Analysis
                st.subheader("Why This Episode Performed As It Did")
                performance_analysis = analyze_why_episode_performed(
                    selected_episode.to_dict(), steps_df, trades
                )
                
                if performance_analysis and 'summary' in performance_analysis:
                    st.write(performance_analysis['summary'])
                else:
                    st.info("Insufficient data for performance analysis.")
        else:
            st.warning("No episodes found for this run.")
    else:
        st.error("Could not fetch latest run data. Is the API server running?")

if __name__ == "__main__":
    main()