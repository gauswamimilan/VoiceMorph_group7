
# %load_ext lab_black
import requests
from datetime import datetime, date, timedelta
import pytz
import urllib
import pandas as pd
import numpy as np
import math
import yfinance as yf

white_pixel = 255
black_pixel = 0
color_red = 76
color_green = 150

def symbol_search(symbol):
    symbol = urllib.parse.quote_plus(symbol).upper()
    request_url = f"https://priceapi.moneycontrol.com/techCharts/symbols?symbol={symbol}"
    result = requests.get(request_url)
    if result.status_code != 200 or result.text == "":
        return False
    result = result.json()
    if "ticker" in result:
        return urllib.parse.quote_plus(result["ticker"]).upper()
    return False


def get_stock_market_data_yf(symbol="RELIANCE.NS", time_interval="5m", period="60d"):
    stock_ticker = yf.Ticker(symbol)
    stock_data = stock_ticker.history(period=period, interval=time_interval)

    # rename columns of dataframe
    stock_data["symbol"] = symbol
    stock_data = stock_data.reset_index()
    stock_data = stock_data.rename(
        columns={
            "Datetime": "date_time",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    stock_data = stock_data.set_index(stock_data["date_time"]).drop(columns=["date_time", "Dividends", "Stock Splits"])
    return stock_data


def get_stock_market_data(symbol="RELIANCE", time_interval=5, from_time=None, to_time=None, no_of_days=60):
    input_sumbol = symbol
    from_time = from_time if from_time is not None else math.ceil(
        datetime.timestamp(
            datetime.now() - timedelta(days=no_of_days)
        )
    )
    to_time = to_time if to_time is not None else math.ceil(
        datetime.timestamp(
            datetime.now()
        )
    )
    request_data = {
        "symbol": urllib.parse.quote_plus(symbol).upper(),
        "resolution": str(time_interval),
        "from": from_time,
        "to": to_time,
    }
    symbol = symbol_search(symbol)
    request_data["symbol"] = symbol if symbol is not False else request_data["symbol"]
    url_list = [
        "https://priceapi.moneycontrol.com/techCharts/indianMarket/stock/history?symbol={symbol}&resolution={resolution}&from={from}&to={to}&currencyCode=INR",
        "https://priceapi.moneycontrol.com/techCharts/history?symbol={symbol}&resolution={resolution}&from={from}&to={to}&currencyCode=INR",
    ]
    for single_url in url_list:
        request_url = single_url
        request_url = request_url.format(**request_data)
        result = requests.get(request_url)
        if result.status_code != 200:
            continue
        
        
        print(result.status_code)

        result = result.json()
        if result["s"] == "no_data":
            print(symbol, result["s"])
            continue
        if result["s"] == "error" or "t" not in result:
            print(symbol, result["s"])
            continue
        result_t = []
        for i, single_time in enumerate(result["t"]):
            temp_json = {
                "symbol": input_sumbol,
                "open": result["o"][i],
                "high": result["h"][i],
                "close": result["c"][i],
                "low": result["l"][i],  
                "volume": result["v"][i],
                "date_time": datetime.fromtimestamp(result["t"][i])
            }
            result_t.append(temp_json)
        df = pd.DataFrame(result_t)
        df.set_index(pd.DatetimeIndex(df["date_time"]), inplace=True)
        df.drop(["date_time"], axis = 1, inplace = True)
        return df
    return False

def calculate_image_color(single_row, no_of_pixel=22, color_red=76, color_green=150):
    start_end = [int(single_row["open"] * 22), int(single_row["close"] * 22)]
    start_end.sort()
    for i in range(1, no_of_pixel - 1):
        if i >= start_end[0] and i <= start_end[1]:
            if single_row["open_gt_close"] == 1:
                single_row[f"pixel_{i}"] = color_red
            else:
                single_row[f"pixel_{i}"] = color_green
        else:
            single_row[f"pixel_{i}"] = black_pixel
    return single_row


def create_candles_image_from_ohcl(df, no_of_pixel=22, max_pixel_value=255, color_red=76, color_green=150):
    df["open"] = (df["high"] - df["open"]) / (df["high"] - df["low"])
    df["close"] = (df["high"] - df["close"]) / (df["high"] - df["low"])
    df["high"] = 1
    df["low"] = 1
    df["open_gt_close"] = df["open"] > df["close"]
    for i in range(0, no_of_pixel):
        df[f"pixel_{i}"] = max_pixel_value
    df.dropna(inplace=True)
    df = df.apply(calculate_image_color, axis=1)
    return df


def create_candles_image_from_ohcl_numpy(df, no_of_pixel=22, max_pixel_value=255, color_red=76, color_green=150):
    date_range = []
    df["open"] = (df["high"] - df["open"]) / (df["high"] - df["low"])
    df["close"] = (df["high"] - df["close"]) / (df["high"] - df["low"])
    df["high"] = 1
    df["low"] = 1
    df["open_gt_close"] = df["open"] > df["close"]

    result_numpy = []
    for index, row in df.iterrows():
        row = row.fillna(white_pixel)
        date_range.append(index)
        start_end = [int(row["open"] * 22), int(row["close"] * 22)]
        start_end.sort()

        tmp_np = np.zeros((no_of_pixel, 1)) * max_pixel_value
        for i in range(1, no_of_pixel - 1):
            if i >= start_end[0] and i <= start_end[1]:
                if row["open_gt_close"] == 1:
                    tmp_np[i][0] = color_red
                else:
                    tmp_np[i][0] = color_green
            else:
                tmp_np[i][0] = white_pixel
        result_numpy.append(tmp_np)
    numpy_image = np.hstack(result_numpy)
    return date_range, numpy_image
