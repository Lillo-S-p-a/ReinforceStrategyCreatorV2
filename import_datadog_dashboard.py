import json
import os
import logging
import argparse
from datadog_api_client.v1 import ApiClient, ApiException, Configuration
from datadog_api_client.v1.api import dashboards_api
from datadog_api_client.v1.models import (
    Dashboard,
    Widget,
    WidgetDefinition,
    WidgetLayout,
    ToplistWidgetRequest,
    TimeseriesWidgetRequest,
    DistributionWidgetRequest,
    TableWidgetRequest,
    WidgetConditionalFormat,
    WidgetMarker
    # Removed WidgetRequestStyle, WidgetStyle as they are not explicitly used yet
    # Removed enum type comments as they are not directly imported/used
)
from dotenv import load_dotenv

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def import_dashboard_from_file(file_path: str):
    """
    Imports a Datadog dashboard definition from a JSON file.

    Args:
        file_path (str): The path to the JSON file containing the dashboard definition.
    """
    # Load environment variables from .env file
    load_dotenv()

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
    
    # --- BEGIN MODIFICATION TO REMOVE title_size FROM ALL GROUP WIDGETS ---
    # This modification is done on the raw dictionary before converting to models
    if dashboard_json and "widgets" in dashboard_json and \
       isinstance(dashboard_json["widgets"], list):
        for i, widget_data_dict in enumerate(dashboard_json["widgets"]):
            if isinstance(widget_data_dict, dict) and "definition" in widget_data_dict and \
               isinstance(widget_data_dict["definition"], dict):
                if widget_data_dict["definition"].get("type") == "group":
                    if "title_size" in widget_data_dict["definition"]:
                        logger.info(f"Found 'title_size' in group widget definition at index {i}. Removing it.")
                        del widget_data_dict["definition"]["title_size"]
    else:
        logger.warning("Dashboard JSON has no widgets or is not structured as expected. Skipping 'title_size' removal for all widgets.")
    # --- END MODIFICATION ---

    def _create_widget_models_from_list(widget_dicts: list) -> list[Widget]:
        """
        Recursively creates a list of Widget model instances from a list of widget dictionaries.
        Handles nested widgets within group definitions.
        """
        widget_models = []
        if not isinstance(widget_dicts, list):
            logger.warning(f"Expected a list of widget dictionaries, got {type(widget_dicts)}. Returning empty list.")
            return []

        for widget_dict in widget_dicts:
            if not isinstance(widget_dict, dict):
                logger.warning(f"Expected widget data to be a dictionary, got {type(widget_dict)}. Skipping.")
                continue

            layout_dict = widget_dict.get("layout")
            definition_dict = widget_dict.get("definition")

            if not layout_dict or not isinstance(layout_dict, dict):
                logger.warning(f"Widget missing layout or layout is not a dict: {widget_dict.get('id', 'N/A')}. Skipping.")
                continue
            if not definition_dict or not isinstance(definition_dict, dict):
                logger.warning(f"Widget missing definition or definition is not a dict: {widget_dict.get('id', 'N/A')}. Skipping.")
                continue
            
            try:
                layout_model = WidgetLayout(**layout_dict)
                
                # Process definition, especially for nested widgets in groups
                current_definition_dict = dict(definition_dict) # Work with a copy
                widget_type = current_definition_dict.get("type")

                if widget_type == "group":
                    nested_widget_dicts = current_definition_dict.get("widgets", [])
                    # Recursively convert nested widgets
                    current_definition_dict["widgets"] = _create_widget_models_from_list(nested_widget_dicts)
                
                # Prepare definition dictionary, ensuring nested complex objects are model instances
                processed_definition_dict = dict(current_definition_dict) # Work with a copy

                if "requests" in processed_definition_dict and isinstance(processed_definition_dict["requests"], list):
                    requests_data = processed_definition_dict["requests"]
                    model_requests = []
                    for req_dict_item in requests_data: # Renamed req_dict to req_dict_item to avoid conflict
                        if not isinstance(req_dict_item, dict):
                            logger.warning(f"Request item is not a dict for widget ID {widget_dict.get('id', 'N/A')}. Skipping request item.")
                            continue
                        
                        # Make a copy of the request dictionary to modify
                        current_req_dict = dict(req_dict_item)

                        # Handle conditional_formats for query_table
                        if widget_type == "query_table" and "conditional_formats" in current_req_dict and isinstance(current_req_dict["conditional_formats"], list):
                            current_req_dict["conditional_formats"] = [WidgetConditionalFormat(**cf) for cf in current_req_dict["conditional_formats"]]
                        
                        # Handle markers for timeseries
                        if widget_type == "timeseries" and "markers" in current_req_dict and isinstance(current_req_dict["markers"], list):
                            current_req_dict["markers"] = [WidgetMarker(**marker) for marker in current_req_dict["markers"]]

                        # Instantiate appropriate request model based on widget type
                        if widget_type == "toplist":
                            model_requests.append(ToplistWidgetRequest(**current_req_dict))
                        elif widget_type == "timeseries":
                            # Specifically remove 'metadata' if it exists, as 'alias' within it is problematic
                            if 'metadata' in current_req_dict:
                                logger.info(f"Removing 'metadata' from timeseries request for widget ID {widget_dict.get('id', 'N/A')} to avoid 'alias' issue.")
                                del current_req_dict['metadata']
                            model_requests.append(TimeseriesWidgetRequest(**current_req_dict))
                        elif widget_type == "distribution":
                            model_requests.append(DistributionWidgetRequest(**current_req_dict))
                        elif widget_type == "query_table":
                            model_requests.append(TableWidgetRequest(**current_req_dict))
                        # For other widget types (heatmap, event_timeline, alert_value, check_status),
                        # their 'requests' might be simpler (e.g., just a query string) or might
                        # not be a list of complex request objects.
                        # The generic WidgetDefinition should handle these if the structure matches.
                        # If they also have specific request models, those would need to be added here.
                        else:
                             # Fallback for widget types whose requests don't have specific models yet
                             # or are simple enough not to need them (e.g. heatmap's request is a list of dicts with 'q')
                             model_requests.append(current_req_dict)
                    processed_definition_dict["requests"] = model_requests
                
                # For GroupWidgetDefinition, 'widgets' should already be a list of Widget models
                # from the recursive call (this is handled by current_definition_dict["widgets"] = ... above)
                
                definition_model = WidgetDefinition(**processed_definition_dict)
                
                # 'id' is usually read-only and not set during creation.
                # The Widget model might not even have 'id' as an init parameter.
                # If it does, and it's optional, we can omit it.
                # If it's required, this might need adjustment based on the model's __init__.
                # For now, we assume 'id' is not passed to Widget constructor.
                widget_model = Widget(definition=definition_model, layout=layout_model)
                widget_models.append(widget_model)
            except Exception as e:
                logger.error(f"Error creating widget model for widget ID {widget_dict.get('id', 'N/A')}: {e}. Skipping this widget.")
        return widget_models

    processed_widget_models = _create_widget_models_from_list(dashboard_json.get("widgets", []))

    dashboard_body = Dashboard(
        title=dashboard_json.get("title", "Untitled Dashboard"),
        description=dashboard_json.get("description", ""),
        widgets=processed_widget_models, # Use the list of Widget model instances
        layout_type=dashboard_json.get("layout_type", "ordered"),
        template_variables=dashboard_json.get("template_variables", []),
        notify_list=dashboard_json.get("notify_list", []),
        reflow_type=dashboard_json.get("reflow_type", "auto")
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