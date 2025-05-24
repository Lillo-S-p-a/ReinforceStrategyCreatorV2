import logging
import pandas as pd
from reinforcestrategycreator.backtesting.data import DataManager

# Configure basic logging to see warnings/errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_pipeline():
    logger.info("Starting data pipeline test...")
    try:
        data_manager = DataManager(
            asset="SPY",
            start_date="2020-01-01",
            end_date="2023-01-01",
            test_ratio=0.2
        )

        logger.info("Attempting to fetch and process data...")
        processed_data = data_manager.fetch_data()

        if processed_data is not None and not processed_data.empty:
            logger.info("Data fetched and processed successfully.")
            logger.info(f"Number of rows: {len(processed_data)}")
            logger.info(f"Columns: {processed_data.columns.tolist()}")
            logger.info("First 5 rows of processed data:")
            print(processed_data.head().to_string())
            
            # Check if indicator columns are present (example: RSI_14, ADX_14)
            expected_indicators = ['RSI_14', 'ADX_14']
            missing_indicators = [col for col in expected_indicators if col not in processed_data.columns]
            if not missing_indicators:
                logger.info(f"Successfully found example indicator columns: {expected_indicators}")
            else:
                logger.warning(f"Missing some expected indicator columns: {missing_indicators}")

        else:
            logger.error("Failed to fetch or process data. Resulting DataFrame is None or empty.")

    except Exception as e:
        logger.error(f"An error occurred during the data pipeline test: {str(e)}", exc_info=True)

if __name__ == "__main__":
    test_data_pipeline()
    logger.info("Data pipeline test finished.")