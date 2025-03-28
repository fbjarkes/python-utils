
import json
import logging
import os

from multiprocessing import Pool
from typing import Dict, List, Optional, Union


import numpy as np
import pandas as pd

from .decorators import try_except
from .functional import pipe


logger = logging.getLogger(__name__)


def filter_rth(df: pd.DataFrame, start_time='09:30', end_time='16:00') -> pd.DataFrame:
    if start_time and end_time and not df.empty and df.attrs['timeframe'] not in ['day', 'week', 'month']:
        if df.attrs['timeframe'] == '60min':
            start_time = '09:00'
        return df.between_time(start_time, end_time, inclusive='left')
    else:
        return df


def filter_date(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    # TODO: raise exception if start/end outside of df?
    if start and end and not df.empty:
        df = df[start:end]
        return df
    if end and not df.empty:
        df = df[:end]
        # logger.debug(f"Filtering date to {end}, last={df.index[-1]}, OHLC={df.iloc[-1].values}")
        return df
    return df


def get_dataframe_tv(timeframe: str, symbol: str, path: str, tz='America/New_York', include_all_columns: bool = True) -> Union[pd.DataFrame, None]:
    file_path = os.path.join(path, timeframe, f"{symbol}.csv")
    logger.debug(f"{symbol}: parsing tradingview data '{file_path}'")
    default_cols = ['time', 'open', 'high', 'low', 'close', 'Volume']
    try:
        if include_all_columns:
            df = pd.read_csv(file_path, index_col='time', parse_dates=False)
        else:
            df = pd.read_csv(file_path, index_col='time', parse_dates=False, usecols=default_cols)
        if tz:
            df.index = pd.to_datetime(df.index, unit='s', utc=True).tz_convert(tz).tz_localize(None)
        else:
            df.index = pd.to_datetime(df.index, unit='s', utc=True).tz_localize(None)
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
        df.attrs = {'symbol': symbol, 'timeframe': timeframe}
        duplicates = df.index.duplicated(keep='first')
        dupe_count = duplicates.sum()
        if dupe_count > 0:
            # remove duplicate rows
            df = df[~duplicates]
        logger.debug(f"{symbol}: {len(df)} rows (start={df.index[0]}, end={df.index[-1]} dupes={dupe_count}) ")
        return df
    except Exception as e:  
        logger.warning(f"Error parsing csv '{file_path}': {e}")

    return pd.DataFrame()

def get_dataframe_ib(timeframe: str, symbol: str, path: str, tz='America/New_York') -> Optional[pd.DataFrame]:
    p = os.path.expanduser(os.path.join(path, timeframe, f"{symbol}.csv"))
    try:
        df = pd.read_csv(p, dtype={'Open': np.float32, 'High': np.float32, 'Low': np.float32, 'Close': np.float32, 'Volume': np.float32}, parse_dates=True, index_col='Date')
        # TODO need to convert to TZ America/New_York ?
        df = df.sort_index()
        df.attrs = {'symbol': symbol, 'timeframe': timeframe}
        logger.debug(f"{symbol}: {len(df)} rows (start={df.index[0]}, end={df.index[-1]})")
        return df
    except Exception as e:
        logger.warning(f"Error parsing csv '{p}': {e}")
    return pd.DataFrame()


@try_except
def load_json_data(symbol: str, path: str) -> Optional[Dict]:
    logger.debug(f"{symbol}: loading json data '{path}'")
    with open(path) as f:
        json_data = json.load(f)
        symbol_data = json_data.get(symbol)
        if not symbol_data:
            logger.warning(f"Missing symbol '{symbol}' in file '{path}'")
        else:
            return symbol_data
    return None


def json_to_dataframe(symbol: str, timeframe: str, data: Optional[Dict]) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame(columns=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df = pd.DataFrame(data, columns=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df.set_index('DateTime', inplace=True)
    # Always assume Wall Street
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('America/New_York').tz_localize(None)
    df = df.sort_index()  # TODO: Never needed before?!
    df.attrs = {'symbol': symbol, 'timeframe': timeframe}
    return df


def get_dataframe_alpaca_file(timeframe: str, symbol: str, path: str) -> Union[pd.DataFrame, None]:
    file_path = os.path.expanduser(f"{path}{os.sep}{timeframe}{os.sep}{symbol}.json")
    json_data = load_json_data(symbol, file_path)
    return json_to_dataframe(symbol, timeframe, json_data)



def get_dataframe(provider, symbol, start, end, timeframe, rth_only=False, path=None, transform='') -> pd.DataFrame:
    if path is None:
        raise Exception(f"Missing path for provider '{provider}'")

    post_process = pipe(
        lambda df: transform_timeframe(df, timeframe, (transform if transform else timeframe)),
        lambda df: filter_rth(df) if rth_only else df,
        lambda df: filter_date(df, start, end),
    )
    if provider == 'tv':
        return post_process(get_dataframe_tv(timeframe=timeframe, symbol=symbol, path=path))
    elif provider == 'alpaca-file':
        return post_process(get_dataframe_alpaca_file(timeframe=timeframe, symbol=symbol, path=path))
    elif provider == 'alpaca':
        from python_utils.alpaca import get_dataframe_alpaca
        return get_dataframe_alpaca(timeframe, symbol, start, end, rth_only)
    elif provider == 'ib':
        return post_process(get_dataframe_ib(timeframe=timeframe, symbol=symbol, path=path))
    else:
        raise Exception(f"Unknown provider '{provider}'")


def get_symbols(symbol_list):
    if symbol_list[0].startswith('/'):
        file = symbol_list[0]
        symbols = []
        with open(file) as f:
            symbols += [ticker.rstrip() for ticker in f.readlines() if not ticker.startswith('#')]
        symbols = list(set(symbols))  # NOTE: reorders elements
    else:
        symbols = list(set(symbol_list))  # NOTE: reorders elements
    return symbols


def get_dataframes(provider, symbol_list, start, end, timeframe, rth_only=False, path=None, transform='', process_workers=0) -> List[pd.DataFrame]:
    if path is None:
        raise Exception(f"Missing path for provider '{provider}'")

    symbols = get_symbols(symbol_list)
    dfs = []

    if process_workers > 0:
        with Pool(process_workers) as pool:
            dfs = pool.starmap(get_dataframe, [(provider, symbol, start, end, timeframe, rth_only, path, transform) for symbol in symbols])
            dfs = [df for df in dfs if not df.empty]
    else:
        for symbol in symbols:
            df = get_dataframe(provider, symbol, start, end, timeframe, rth_only, path, transform)
            if not df.empty:
                dfs.append(df)
    return dfs


def transform_timeframe(df: pd.DataFrame, timeframe: str, transform: str) -> pd.DataFrame:
    if timeframe == transform or df.empty:
        return df
    conversion = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    if timeframe == 'day' and transform == 'month':
        resampled = df.resample('ME').agg(conversion)
    elif timeframe == 'day' and transform == 'week':
        resampled = df.resample('W').agg(conversion)
    else:
        resampled = df.resample(f"{transform}").agg(conversion)
    resampled.attrs['timeframe'] = transform
    return resampled.dropna()
