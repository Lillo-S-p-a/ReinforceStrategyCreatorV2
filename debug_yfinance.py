import yfinance as yf
import pandas as pd
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ticker = "SPY"
start_date = "2020-01-01"
end_date = "2023-01-01"

logger.info(f"Attempting to download data for {ticker} from {start_date} to {end_date}")
try:
    data = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        progress=True # Enable progress to see if it hangs
    )

    if data.empty:
        logger.warning(f"yf.download returned an EMPTY DataFrame for {ticker}")
    else:
        logger.info(f"yf.download returned a DataFrame with {len(data)} rows.")
        logger.info(f"Columns: {data.columns.tolist()}")
        logger.info("First 5 rows:")
        print(data.head().to_string())
        # Check for standard OHLC columns (case-insensitive)
        expected_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = []
        present_cols = {}
        for ecol in expected_cols:
            found_col = next((col for col in data.columns if isinstance(col, str) and col.lower() == ecol), None)
            if found_col:
                present_cols[ecol] = found_col
            else:
                missing_cols.append(ecol)
        
        if missing_cols:
            logger.warning(f"Missing expected columns (case-insensitive): {missing_cols}")
        else:
            logger.info(f"All expected OHLCV columns are present. Mapped: {present_cols}")

except Exception as e:
    logger.error(f"An exception occurred during yf.download: {str(e)}", exc_info=True)

logger.info("Debug script finished.")