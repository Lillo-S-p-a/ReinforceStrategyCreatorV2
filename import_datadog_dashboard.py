import json
import os
import logging
import argparse
from datadog_api_client.v1 import ApiClient, ApiException, Configuration
from datadog_api_client.v1.api import dashboards_api
from datadog_api_client.v1.models import Dashboard

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def import_dashboard_from_file(file_path: str):
    """
    Imports a Datadog dashboard definition from a JSON file.

    Args:
        file_path (str): The path to the JSON file containing the dashboard definition.
    """
    # Ensure API key and App key are set in environment variables
    dd_api_key = os.environ.get("DATADOG_API_KEY")
    dd_app_key = os.environ.get("DATADOG_APP_KEY")
    dd_site = os.environ.get("DATADOG_SITE", "datadoghq.com") # Default to US1, use "datadoghq.eu" for EU1 etc.

    if not dd_api_key or not dd_app_key:
        logger.error("DATADOG_API_KEY and/or DATADOG_APP_KEY environment variables are not set.")
        logger.error("Please set them before running the script.")
        return

    # Load dashboard definition from JSON file
    try:
        with open(file_path, 'r') as f:
            dashboard_json = json.load(f)
        logger.info(f"Successfully loaded dashboard definition from {file_path}")
    except FileNotFoundError:
        logger.error(f"Error: Dashboard file not found at {file_path}")
        return
    except json.JSONDecodeError:
        logger.error(f"Error: Could not decode JSON from {file_path}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading {file_path}: {e}")
        return

    # Configure API client
    configuration = Configuration()
    configuration.api_key["apiKeyAuth"] = dd_api_key
    configuration.api_key["appKeyAuth"] = dd_app_key
    configuration.server_variables["site"] = dd_site # Important for non-US1 sites

    api_client = ApiClient(configuration)
    dashboards_api_instance = dashboards_api.DashboardsApi(api_client)

    # Create Dashboard object from the loaded JSON
    # The Dashboard model expects specific fields. We need to map our JSON to these.
    # The JSON we generated is already in the format expected by the API for creating a dashboard.
    # So, we can pass the dictionary directly as the body.
    
    dashboard_body = Dashboard(
        title=dashboard_json.get("title", "Untitled Dashboard"),
        description=dashboard_json.get("description", ""),
        widgets=dashboard_json.get("widgets", []),
        layout_type=dashboard_json.get("layout_type", "ordered"), # Ensure this matches your JSON
        template_variables=dashboard_json.get("template_variables", []),
        notify_list=dashboard_json.get("notify_list", []),
        reflow_type=dashboard_json.get("reflow_type", "auto")
        # Add other top-level dashboard properties from your JSON if needed
    )

    try:
        logger.info(f"Attempting to create dashboard '{dashboard_body.title}' in Datadog...")
        api_response = dashboards_api_instance.create_dashboard(body=dashboard_body)
        logger.info(f"Dashboard '{api_response.title}' created successfully!")
        logger.info(f"Dashboard ID: {api_response.id}")
        logger.info(f"Dashboard URL: {api_response.url}")
        if hasattr(api_response, 'author_handle'): # Check if attribute exists
             logger.info(f"Author: {api_response.author_handle}")

    except ApiException as e:
        logger.error(f"Exception when calling DashboardsApi->create_dashboard: {e}\n")
        if hasattr(e, 'body') and e.body:
            error_body = e.body
            # If e.body is a string, try to parse it as JSON
            if isinstance(error_body, (str, bytes, bytearray)):
                try:
                    error_details = json.loads(error_body)
                except json.JSONDecodeError:
                    logger.error(f"Could not parse error body as JSON: {error_body}")
                    error_details = None
            else: # Assume it's already a dict
                error_details = error_body

            if isinstance(error_details, dict) and 'errors' in error_details:
                logger.error("Datadog API Errors:")
                for err in error_details['errors']:
                    logger.error(f"- {err}")
            elif error_details: # If it's not None but not the expected dict structure
                logger.error(f"Error details from API: {error_details}")
            # else: logger.error(f"Raw error body (if not parsed): {e.body}") # Redundant if already logged


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Import a Datadog dashboard from a JSON file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python import_datadog_dashboard.py dashboard_ml_engineer.json
  python import_datadog_dashboard.py dashboard_quant_analyst.json
        """
    )
    
    # Add positional argument for dashboard filename
    parser.add_argument(
        'dashboard_filename',
        help='Path to the JSON file containing the dashboard definition'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    logger.info(f"Starting dashboard import process for: {args.dashboard_filename}")
    import_dashboard_from_file(args.dashboard_filename)
    logger.info("Dashboard import process finished.")