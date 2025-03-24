import logging
import os

from typing import Dict, List, Optional, Union
from datetime import datetime
from functools import lru_cache
from alpaca.data import CryptoHistoricalDataClient, CryptoBarsRequest, StockHistoricalDataClient, StockBarsRequest, TimeFrame, TimeFrameUnit
import pandas as pd

logger = logging.getLogger(__name__)

def _parse_timeframe(tf: str) -> TimeFrame:
    if tf == 'day':
        return TimeFrame(1, TimeFrameUnit.Day)
    
    amount = int(''.join(filter(str.isdigit, tf)))
    unit = ''.join(filter(str.isalpha, tf)).lower()
    if unit == 'min' and amount == 60:
        return TimeFrame(1, TimeFrameUnit.Hour)
    return TimeFrame(amount, TimeFrameUnit(unit.capitalize()))


#@lru_cache
def get_dataframe_alpaca(timeframe: str, symbol: str, start: str, end: str, rth_only: bool) -> Union[pd.DataFrame, None]:
    if 'ALPACA_KEY_ID' not in os.environ or 'ALPACA_SECRET_KEY' not in os.environ:
        logger.warning("Missing 'ALPACA_KEY_ID' or 'ALPACA_SECRET_KEY' in environment variables")
        return pd.DataFrame(columns=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']) 
    
    client = StockHistoricalDataClient(os.environ['ALPACA_KEY_ID'], os.environ['ALPACA_SECRET_KEY'])
    
    now = datetime.now()
    request_params = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=_parse_timeframe(timeframe),
        start=start,
        end=now,
    )
    
    try:
        bars = client.get_stock_bars(request_params)
    except Exception as e:
        logger.warning(f"Error getting bars for {symbol} with request {request_params}: {e}")
        return pd.DataFrame(columns=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
    
    if bars:
        if bars.df.index.nlevels == 2:
            df = bars.df.droplevel(0)
        else:
            df = bars.df
        df = df.tz_convert('America/New_York')
        
        if timeframe != 'day' and rth_only:
            if timeframe == '60min':
                # NOTE: includes 30min of PM... should be accurate enough anyway
                df = df.between_time('09:00', '16:00', inclusive='left')
            else:
                df = df.between_time('09:30', '16:00', inclusive='left')
        #if end_dt != '':
        #    df = df[:end_dt]

        df.attrs['symbol'] = symbol
        df.attrs['timeframe'] = timeframe
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        return df
    else:
        logger.warning(f"Empty StockBarsRequest response for symbol '{symbol}'")
        return pd.DataFrame(columns=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
