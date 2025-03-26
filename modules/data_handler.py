import yfinance as yf
import pandas as pd

def fetch_stock_data(symbols, years):
    """
    Fetch historical stock data for the given symbols.
    :param symbols: List of stock symbols (e.g., ["AAPL"] or ["JPM", "BAC"])
    :param years: Number of years of historical data to fetch
    :return: DataFrame containing stock data
    """
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=years)

    data_frames = []
    for symbol in symbols:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        stock_data['Symbol'] = symbol
        data_frames.append(stock_data)

    combined_data = pd.concat(data_frames)
    return combined_data.reset_index()