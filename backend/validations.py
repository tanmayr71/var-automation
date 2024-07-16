import logging
from xbbg import blp

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def validate_tickers(tickers):
    """
    Validate tickers using Bloomberg API by attempting to fetch minimal historical data.
    """
    valid_tickers = []
    invalid_tickers = []

    for ticker in tickers:
        try:
            logging.debug(f"Validating ticker: {ticker}")
            # Attempt to fetch minimal historical data to validate the ticker
            response = blp.bdh(tickers=ticker, flds='PX_LAST', start_date='2024-07-02', end_date='2024-07-03')
            logging.debug(f"Response for ticker {ticker}: {response}")
            if not response.empty:
                valid_tickers.append(ticker)
            else:
                invalid_tickers.append(ticker)
        except Exception as e:
            logging.error(f"Error validating ticker {ticker}: {e}")
            invalid_tickers.append(ticker)
    
    logging.debug(f"Valid tickers: {valid_tickers}")
    logging.debug(f"Invalid tickers: {invalid_tickers}")

    return valid_tickers, invalid_tickers