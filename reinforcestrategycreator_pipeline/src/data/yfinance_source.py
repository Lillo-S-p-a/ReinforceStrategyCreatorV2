"""Data source for fetching data from Yahoo Finance using yfinance library."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import time
import pandas as pd
import yfinance as yf
from reinforcestrategycreator_pipeline.src.monitoring.logger import get_logger

from .base import DataSource, DataSourceMetadata


class YFinanceDataSource(DataSource):
    """
    Data source for loading financial data from Yahoo Finance.
    """

    def __init__(self, source_id: str, config: Dict[str, Any]):
        """
        Initialize YFinanceDataSource.

        Args:
            source_id: Unique identifier for this data source.
            config: Configuration dictionary containing:
                - tickers: String or List of stock tickers (e.g., "SPY" or ["SPY", "AAPL"]).
                - period: Data period to download (e.g., "1y", "5d", "max").
                         Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
                - interval: Data interval (e.g., "1d", "1wk", "1mo").
                            Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 
                                             1d, 5d, 1wk, 1mo, 3mo.
                - start_date: Start date string (YYYY-MM-DD). Overrides period if both provided.
                - end_date: End date string (YYYY-MM-DD).
                - auto_adjust: Automatically adjust OHLC data (default: True).
                - prepost: Include pre/post market data (default: False).
        """
        super().__init__(source_id, config)
        self.logger = get_logger(self.__class__.__name__)
        self.tickers: Union[str, List[str]] = self.config.get("tickers", [])
        self.period: Optional[str] = self.config.get("period")
        self.interval: str = self.config.get("interval", "1d")
        self.start_date: Optional[str] = self.config.get("start_date")
        self.end_date: Optional[str] = self.config.get("end_date")
        self.auto_adjust: bool = self.config.get("auto_adjust", True)
        self.prepost: bool = self.config.get("prepost", False)

        self.validate_config()
        self.logger.info(f"YFinanceDataSource '{self.source_id}' initialized with config: {self.config}")

    def validate_config(self) -> bool:
        """Validate the data source configuration."""
        if not self.tickers:
            raise ValueError(f"YFinanceDataSource '{self.source_id}' requires 'tickers' in config.")
        if not isinstance(self.tickers, (str, list)):
            raise ValueError(f"'tickers' must be a string or a list of strings for YFinanceDataSource '{self.source_id}'.")
        if isinstance(self.tickers, list) and not all(isinstance(t, str) for t in self.tickers):
            raise ValueError(f"All items in 'tickers' list must be strings for YFinanceDataSource '{self.source_id}'.")
        
        self.logger.debug(f"Configuration for YFinanceDataSource '{self.source_id}' validated.")
        return True

    def load_data(self, **kwargs) -> pd.DataFrame:
        """
        Load data from Yahoo Finance.

        Args:
            **kwargs: Can override config parameters like 'tickers', 'period', 
                      'interval', 'start_date', 'end_date'.
        
        Returns:
            DataFrame containing the loaded financial data.
        """
        tickers_to_load = kwargs.get("tickers", self.tickers)
        period_to_load = kwargs.get("period", self.period)
        interval_to_load = kwargs.get("interval", self.interval)
        start_date_to_load = kwargs.get("start_date", self.start_date)
        end_date_to_load = kwargs.get("end_date", self.end_date)
        auto_adjust_to_load = kwargs.get("auto_adjust", self.auto_adjust)
        prepost_to_load = kwargs.get("prepost", self.prepost)

        if isinstance(tickers_to_load, list):
            ticker_str_log = ", ".join(tickers_to_load)
        else:
            ticker_str_log = tickers_to_load
            
        self.logger.info(
            f"Loading data for ticker(s): {ticker_str_log}, "
            f"period: {period_to_load}, interval: {interval_to_load}, "
            f"start: {start_date_to_load}, end: {end_date_to_load}"
        )
        self.update_lineage("load_data_attempt", {
            "tickers": tickers_to_load, "period": period_to_load, "interval": interval_to_load,
            "start_date": start_date_to_load, "end_date": end_date_to_load,
            "auto_adjust": auto_adjust_to_load, "prepost": prepost_to_load
        })

        try:
            # yfinance expects a space-separated string for multiple tickers or a list
            yf_tickers_param = tickers_to_load
            if isinstance(tickers_to_load, list):
                 yf_tickers_param = " ".join(tickers_to_load)


            data: pd.DataFrame = yf.download(
                tickers=yf_tickers_param, # Use the processed tickers parameter
                start=start_date_to_load,
                end=end_date_to_load,
                period=period_to_load if not (start_date_to_load or end_date_to_load) else None,
                interval=interval_to_load,
                auto_adjust=auto_adjust_to_load,
                prepost=prepost_to_load,
                progress=False,
                # show_errors=True, # Removed, yfinance typically shows errors by default
                group_by='ticker' if isinstance(tickers_to_load, list) and len(tickers_to_load) > 1 else None
            )

            if data.empty:
                error_msg = (
                    f"No data returned for tickers: {tickers_to_load}. "
                    f"This could be due to: (1) Invalid ticker symbols, "
                    f"(2) No data available for the specified date range, "
                    f"(3) Market holidays/weekends, or (4) Yahoo Finance API issues. "
                    f"Parameters used: period={period_to_load}, interval={interval_to_load}, "
                    f"start={start_date_to_load}, end={end_date_to_load}"
                )
                self.logger.error(error_msg)
                self.update_lineage("load_data_empty", {
                    "tickers": tickers_to_load,
                    "params_used": self.config,
                    "error_details": error_msg
                })
                raise ValueError(f"YFinanceDataSource failed to load data: {error_msg}")

            # If multiple tickers, yfinance returns a MultiIndex for columns.
            # If single ticker (even as a list of one), it might return flat or MultiIndex.
            # We want to ensure a consistent flat DataFrame if only one ticker was requested.
            if isinstance(tickers_to_load, str) or (isinstance(tickers_to_load, list) and len(tickers_to_load) == 1):
                if isinstance(data.columns, pd.MultiIndex):
                    # If it's a single ticker but columns are MultiIndex, flatten them.
                    # e.g. ('Open', 'SPY') -> 'Open'
                    # The ticker name is usually the first element in the list/string.
                    actual_ticker_name = tickers_to_load if isinstance(tickers_to_load, str) else tickers_to_load[0]
                    if actual_ticker_name in data.columns.get_level_values(0): # Check if ticker is the first level
                         data = data.xs(actual_ticker_name, level=0, axis=1)
                    elif actual_ticker_name in data.columns.get_level_values(1): # Check if ticker is the second level
                         data = data.xs(actual_ticker_name, level=1, axis=1)


            self.logger.info(f"Successfully loaded data for {ticker_str_log}. Shape: {data.shape}")
            self.update_lineage("load_data_success", {
                "tickers": tickers_to_load, "shape": list(data.shape), "columns": list(data.columns)
            })
            return data
        except Exception as e:
            if "No data returned" in str(e):
                # Re-raise our custom ValueError with detailed context
                raise
            else:
                # Handle other exceptions (network issues, API errors, etc.)
                enhanced_error_msg = (
                    f"Error loading data for {ticker_str_log} from Yahoo Finance: {e}. "
                    f"This could be due to network connectivity issues, Yahoo Finance API problems, "
                    f"or invalid parameters. Please check your internet connection and verify "
                    f"the ticker symbols and date ranges."
                )
                self.logger.error(enhanced_error_msg, exc_info=True)
                self.update_lineage("load_data_error", {
                    "tickers": tickers_to_load,
                    "error": str(e),
                    "enhanced_error": enhanced_error_msg
                })
                raise ValueError(f"YFinanceDataSource failed to load data: {enhanced_error_msg}") from e

    def get_schema(self) -> Dict[str, str]:
        """
        Get the schema of the data source by fetching a small sample.
        """
        self.logger.debug(f"Attempting to get schema for YFinanceDataSource '{self.source_id}'")
        try:
            sample_tickers_for_schema = self.tickers
            if isinstance(sample_tickers_for_schema, list):
                if not sample_tickers_for_schema:
                    self.logger.warning("Tickers list is empty, cannot fetch schema sample.")
                    return {}
                sample_tickers_for_schema = sample_tickers_for_schema[0] 
            
            if not sample_tickers_for_schema:
                 self.logger.warning("Ticker is empty, cannot fetch schema sample.")
                 return {}

            df_sample = yf.download(
                tickers=sample_tickers_for_schema,
                period="5d", 
                interval="1d",
                progress=False,
                # show_errors=False, # Removed
                auto_adjust=self.auto_adjust,
                prepost=self.prepost
            )
            if df_sample.empty:
                self.logger.warning(f"Could not fetch sample data for schema for {sample_tickers_for_schema}")
                return {}
            
            if isinstance(df_sample.columns, pd.MultiIndex):
                 # If single ticker sample resulted in MultiIndex, flatten it
                if sample_tickers_for_schema in df_sample.columns.get_level_values(0):
                    df_sample = df_sample.xs(sample_tickers_for_schema, level=0, axis=1)
                elif sample_tickers_for_schema in df_sample.columns.get_level_values(1):
                     df_sample = df_sample.xs(sample_tickers_for_schema, level=1, axis=1)


            schema = {}
            for col, dtype in df_sample.dtypes.items():
                schema[str(col)] = str(dtype)
            
            self.logger.debug(f"Schema for {sample_tickers_for_schema}: {schema}")
            return schema
            
        except Exception as e:
            self.logger.error(f"Error fetching schema for YFinanceDataSource '{self.source_id}': {e}", exc_info=True)
            return {}

    def get_metadata(self) -> DataSourceMetadata:
        """
        Get metadata about the data source.
        """
        # Ensure _metadata is initialized by calling super's get_metadata
        # if it hasn't been called or if we want to refresh parts of it.
        # However, YFinanceDataSource might want to set its own source_type.
        
        if self._metadata is None:
            # Initialize basic metadata structure, similar to base class,
            # but with yfinance specific source_type.
            # The lineage dictionary will be initialized as empty by DataSourceMetadata.
            self._metadata = DataSourceMetadata(
                source_id=self.source_id,
                source_type="yfinance", # YFinance specific
                version="1.0.0", # Default version or manage as needed
                created_at=datetime.now(), # Or a more persistent creation time
                schema=self.get_schema(),
                properties=self.config.copy(),
                # lineage is initialized to {} by default in DataSourceMetadata
            )
        else:
            # If metadata already exists, ensure its schema and properties are up-to-date
            # and source_type is correct.
            self._metadata.schema = self.get_schema()
            self._metadata.properties = self.config.copy()
            self._metadata.source_type = "yfinance" # Ensure it's set

        # The update_lineage method in the base class will handle the "operations" list.
        return self._metadata

    def test_connection(self) -> bool:
        """
        Test if data can be fetched for a sample ticker.
        """
        self.logger.info(f"Testing connection for YFinanceDataSource '{self.source_id}'")
        try:
            sample_tickers_for_test = self.tickers
            if isinstance(sample_tickers_for_test, list):
                if not sample_tickers_for_test:
                    self.logger.warning("Cannot test connection: tickers list is empty.")
                    return False
                sample_tickers_for_test = sample_tickers_for_test[0]
            
            if not sample_tickers_for_test:
                self.logger.warning("Cannot test connection: ticker is empty.")
                return False

            test_data = yf.download(
                tickers=sample_tickers_for_test,
                period="1d",
                interval="1d",
                progress=False
                # show_errors=True # Removed
            )
            if not test_data.empty:
                self.logger.info(f"Connection test successful for {sample_tickers_for_test}.")
                return True
            else:
                self.logger.warning(f"Connection test failed for {sample_tickers_for_test}: No data returned.")
                return False
        except Exception as e:
            self.logger.error(f"Connection test failed for YFinanceDataSource '{self.source_id}': {e}", exc_info=True)
            return False