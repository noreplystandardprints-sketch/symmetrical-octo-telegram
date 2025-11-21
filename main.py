import gymnasium as gym
from gymnasium import spaces
import numpy as  np
import pandas as  pd
import yfinance as  yf
import time
import sys
import os
import pandas as pd
import json
import argparse
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from curl_cffi import requests
from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.results_reporter import ResultsReporter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from tickers import ALL_TICKERS
from collections import defaultdict
import math
from typing import Protocol, Optional
from datetime import datetime

# Optional IBKR integration via ib_insync
try:
    from ib_insync import IB as _IB, Stock as _IBStock, MarketOrder as _IBMarketOrder
except Exception:
    _IB = None
    _IBStock = None
    _IBMarketOrder = None

# IBKR runtime config
IBKR_ENABLED = False
IBKR_EXECUTE = False
IBKR_HOST = "127.0.0.1"
IBKR_PORT = 7497
IBKR_CLIENT_ID = 1
_ib = None

# Trade type permissions (default: conservative for shorting)
TRADE_PERMISSIONS = {
    "BUY": True,
    "SELL": True,
    "SELL_SHORT": False,
    "BUY_TO_COVER": False,
}

def _ibkr_account_summary_dict() -> dict:
    """Return IBKR account summary as a dict {tag: value}. Requires _ib connection."""
    try:
        if _ib is None:
            return {}
        avs = _ib.accountSummary()
        out = {}
        for av in avs:
            out[str(getattr(av, 'tag', ''))] = str(getattr(av, 'value', ''))
        return out
    except Exception:
        return {}

def _ibkr_update_trade_permissions_from_account() -> None:
    """Update TRADE_PERMISSIONS based on IBKR account capabilities (e.g., shorting)."""
    summary = _ibkr_account_summary_dict()
    shorting = summary.get('ShortingEnabled', '').lower()
    if shorting in ("true", "yes", "1"):
        TRADE_PERMISSIONS["SELL_SHORT"] = True
        TRADE_PERMISSIONS["BUY_TO_COVER"] = True

def _map_action_for_permissions(side: str, action_name: Optional[str] = None) -> str:
    """Map an order side and optional semantic action to a permission key."""
    if action_name:
        an = action_name.lower()
        if an == 'sell_short':
            return 'SELL_SHORT'
        if an in ('cover_short', 'buy_to_cover'):
            return 'BUY_TO_COVER'
    side_u = side.upper()
    if side_u == 'BUY':
        return 'BUY'
    if side_u == 'SELL':
        return 'SELL'
    return side_u

def _can_execute_trade(side: str, action_name: Optional[str] = None) -> bool:
    key = _map_action_for_permissions(side, action_name)
    return bool(TRADE_PERMISSIONS.get(key, False))

#############################################
# Broker abstraction and MockBroker
#############################################

class Broker(Protocol):
    def buy(self, symbol: str, quantity: float, price: Optional[float] = None) -> dict: ...
    def sell(self, symbol: str, quantity: float, price: Optional[float] = None) -> dict: ...
    def get_price(self, symbol: str) -> float: ...
    def min_order(self, symbol: str) -> float: ...
    def get_balance(self) -> float: ...
    simulate: bool

class MockBroker:
    _order_seq = 10000
    simulate = True

    def _next_id(self) -> int:
        MockBroker._order_seq += 1
        return MockBroker._order_seq

    def buy(self, symbol: str, quantity: float, price: Optional[float] = None) -> dict:
        if not _can_execute_trade('BUY'):
            return {"status": "error", "message": "Permission denied for BUY"}
        p = float(price) if price is not None else (_get_last_price(symbol) or 100.0)
        print(f"[SIMULATION] Would buy {quantity} {symbol} at {p if price is not None else 'market price'}")
        msg = _paper_buy(symbol, float(quantity), p)
        _append_trade_log({
            "symbol": symbol,
            "action": "BUY",
            "quantity": float(quantity),
            "price": float(p),
            "status": "SIMULATED",
            "message": msg,
            "timestamp": str(datetime.now())
        })
        return {"status": "success", "order_id": self._next_id(), "message": msg}

    def sell(self, symbol: str, quantity: float, price: Optional[float] = None) -> dict:
        if not _can_execute_trade('SELL'):
            return {"status": "error", "message": "Permission denied for SELL"}
        p = float(price) if price is not None else (_get_last_price(symbol) or 100.0)
        print(f"[SIMULATION] Would sell {quantity} {symbol} at {p if price is not None else 'market price'}")
        msg = _paper_sell(symbol, float(quantity), p)
        _append_trade_log({
            "symbol": symbol,
            "action": "SELL",
            "quantity": float(quantity),
            "price": float(p),
            "status": "SIMULATED",
            "message": msg,
            "timestamp": str(datetime.now())
        })
        return {"status": "success", "order_id": self._next_id(), "message": msg}

    def get_price(self, symbol: str) -> float:
        p = _get_last_price(symbol)
        return float(p) if p is not None else 100.0

    def min_order(self, symbol: str) -> float:
        # Default minimum order size for stocks: 1 share
        return 1.0

    def get_balance(self) -> float:
        _, bal = _load_paper_account()
        return float(bal)

BROKER: Optional[Broker] = MockBroker()

class IBKRBroker:
    def __init__(self):
        _ibkr_connect_if_needed()
        self.simulate = False

    def get_price(self, symbol: str) -> float:
        p = _get_ibkr_last_price(symbol)
        if p is None or p <= 0:
            p = _get_last_price(symbol)
        return float(p) if p is not None else 0.0

    def buy(self, symbol: str, quantity: float, price: Optional[float] = None) -> dict:
        if not _can_execute_trade('BUY'):
            return {"status": "error", "message": "Permission denied for BUY"}
        if not _ibkr_connect_if_needed() or _IBStock is None or _IBMarketOrder is None:
            return {"status": "error", "message": "IBKR not connected or API unavailable"}
        p = float(price) if price is not None else (self.get_price(symbol) or 0.0)
        try:
            contract = _IBStock(symbol, 'SMART', 'USD')
            order = _IBMarketOrder('BUY', int(round(quantity)))
            trade = _ib.placeOrder(contract, order)
            _ib.sleep(2)
            status = getattr(trade.orderStatus, 'status', 'Unknown')
            _append_trade_log({
                "symbol": symbol,
                "action": "BUY",
                "quantity": float(quantity),
                "price": float(p),
                "status": status,
                "timestamp": str(datetime.now())
            })
            if status not in ('Filled', 'Submitted'):
                print(f"⚠️ Order failed or rejected: {status}")
            else:
                print(f"✅ Order successful: {status}")
            return {"status": status, "order_id": getattr(trade, 'order', None) and getattr(trade.order, 'orderId', None)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def sell(self, symbol: str, quantity: float, price: Optional[float] = None) -> dict:
        if not _can_execute_trade('SELL'):
            return {"status": "error", "message": "Permission denied for SELL"}
        if not _ibkr_connect_if_needed() or _IBStock is None or _IBMarketOrder is None:
            return {"status": "error", "message": "IBKR not connected or API unavailable"}
        p = float(price) if price is not None else (self.get_price(symbol) or 0.0)
        try:
            contract = _IBStock(symbol, 'SMART', 'USD')
            order = _IBMarketOrder('SELL', int(round(quantity)))
            trade = _ib.placeOrder(contract, order)
            _ib.sleep(2)
            status = getattr(trade.orderStatus, 'status', 'Unknown')
            _append_trade_log({
                "symbol": symbol,
                "action": "SELL",
                "quantity": float(quantity),
                "price": float(p),
                "status": status,
                "timestamp": str(datetime.now())
            })
            if status not in ('Filled', 'Submitted'):
                print(f"⚠️ Order failed or rejected: {status}")
            else:
                print(f"✅ Order successful: {status}")
            return {"status": status, "order_id": getattr(trade, 'order', None) and getattr(trade.order, 'orderId', None)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

def select_broker(args=None) -> Broker:
    """Select broker backend via CLI flag or interactive menu.
    Returns a Broker instance; defaults to MockBroker on invalid or failed selection.
    """
    global IBKR_ENABLED, IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID, IBKR_EXECUTE
    mode = None
    if args is not None and hasattr(args, "broker") and args.broker:
        mode = args.broker
    else:
        print("Select broker mode:")
        print("1. Simulation (Mock Broker)")
        print("2. Live IBKR")
        choice = input("Enter choice (1 or 2): ").strip()
        mode = "ibkr" if choice == "2" else "mock"

    if mode == "ibkr":
        IBKR_ENABLED = True
        if args is not None:
            IBKR_HOST = getattr(args, "ib_host", IBKR_HOST)
            IBKR_PORT = int(getattr(args, "ib_port", IBKR_PORT))
            IBKR_CLIENT_ID = int(getattr(args, "ib_client_id", IBKR_CLIENT_ID))
            IBKR_EXECUTE = bool(getattr(args, "ib_exec", IBKR_EXECUTE))
        broker = IBKRBroker()
        ok = _ibkr_connect_if_needed()
        print(f"IBKR pricing {'enabled' if ok else 'requested but not connected'}; execution {'on' if IBKR_EXECUTE else 'off'}.")
        print(f"✅ Connected to IBKR Paper Trading: {ok}")
        return broker if ok else MockBroker()

    # Default to simulation
    return MockBroker()

    def sell(self, symbol: str, quantity: float, price: Optional[float] = None) -> dict:
        if not _ibkr_connect_if_needed() or _IBStock is None or _IBMarketOrder is None:
            return {"status": "error", "message": "IBKR not connected or API unavailable"}
        p = float(price) if price is not None else (self.get_price(symbol) or 0.0)
        try:
            contract = _IBStock(symbol, 'SMART', 'USD')
            order = _IBMarketOrder('SELL', int(round(quantity)))
            trade = _ib.placeOrder(contract, order)
            _ib.sleep(2)
            status = getattr(trade.orderStatus, 'status', 'Unknown')
            _append_trade_log({
                "symbol": symbol,
                "action": "SELL",
                "quantity": float(quantity),
                "price": float(p),
                "status": status,
                "timestamp": str(datetime.now())
            })
            if status not in ('Filled', 'Submitted'):
                print(f"⚠️ Order failed or rejected: {status}")
            else:
                print(f"✅ Order successful: {status}")
            return {"status": status, "order_id": getattr(trade, 'order', None) and getattr(trade.order, 'orderId', None)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def min_order(self, symbol: str) -> float:
        # Basic default: 1 share
        return 1.0

    def get_balance(self) -> float:
        # Fallback to paper account cash for safety checks in AI loop
        _, bal = _load_paper_account()
        return float(bal)

#############################################
# Error handling, retry, and safe order helpers
#############################################

# Configure error logging once
logging.basicConfig(
    filename="trading_errors.log",
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s:%(message)s"
)

def retry(func, attempts: int = 3, delay: float = 2.0, *args, **kwargs):
    for i in range(attempts):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"[WARNING] Attempt {i+1} failed: {e}")
            time.sleep(delay)
    print("[ERROR] All retry attempts failed.")
    return None

def safe_buy(broker: Broker, symbol: str, quantity: float, price: Optional[float] = None):
    try:
        if not _can_execute_trade('BUY'):
            print("⚠️ Permission denied for BUY; order not submitted.")
            return None
        if quantity < broker.min_order(symbol):
            print(f"[WARNING] Quantity too small for {symbol}")
            return None
        p = float(price) if price is not None else broker.get_price(symbol)
        bal = broker.get_balance()
        if bal < p * quantity:
            print(f"[WARNING] Not enough balance to buy {quantity} of {symbol}")
            return None
        if getattr(broker, "simulate", False):
            print(f"[SIMULATION] Would buy {quantity} {symbol} at {p}")
        resp = retry(broker.buy, attempts=3, delay=2, symbol=symbol, quantity=quantity, price=p)
        if isinstance(resp, dict) and resp.get("status") == "error":
            raise Exception(resp.get("message", "Order error"))
        return resp
    except ConnectionError:
        print("[ERROR] Connection lost. Retrying in 5 seconds...")
        time.sleep(5)
        try:
            return broker.buy(symbol, quantity, price)
        except Exception as e:
            logging.error(f"Failed order after reconnect: {e}")
            print(f"[ERROR] Failed order: {e}")
            return None
    except ValueError as e:
        print(f"[ERROR] Invalid order: {e}")
        logging.error(f"Invalid order: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        logging.error(f"Failed order: {e}")
        return None

def safe_sell(broker: Broker, symbol: str, quantity: float, price: Optional[float] = None):
    try:
        if not _can_execute_trade('SELL'):
            print("⚠️ Permission denied for SELL; order not submitted.")
            return None
        if quantity < broker.min_order(symbol):
            print(f"[WARNING] Quantity too small for {symbol}")
            return None
        p = float(price) if price is not None else broker.get_price(symbol)
        if getattr(broker, "simulate", False):
            print(f"[SIMULATION] Would sell {quantity} {symbol} at {p}")
        resp = retry(broker.sell, attempts=3, delay=2, symbol=symbol, quantity=quantity, price=p)
        if isinstance(resp, dict) and resp.get("status") == "error":
            raise Exception(resp.get("message", "Order error"))
        return resp
    except ConnectionError:
        print("[ERROR] Connection lost. Retrying in 5 seconds...")
        time.sleep(5)
        try:
            return broker.sell(symbol, quantity, price)
        except Exception as e:
            logging.error(f"Failed sell after reconnect: {e}")
            print(f"[ERROR] Failed sell: {e}")
            return None
    except ValueError as e:
        print(f"[ERROR] Invalid order: {e}")
        logging.error(f"Invalid sell order: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        logging.error(f"Failed sell: {e}")
        return None

def _ibkr_connect_if_needed():
    """Connect to IBKR using ib_insync if not connected, honoring globals."""
    global _ib
    if not IBKR_ENABLED:
        return False
    if _IB is None:
        print("ib_insync not installed; run 'pip install ib_insync' to enable IBKR.")
        return False
    if _ib is None:
        _ib = _IB()
    try:
        if not _ib.isConnected():
            _ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID)
            if not _ib.isConnected():
                print(f"⚠️ Failed to connect to IBKR at {IBKR_HOST}:{IBKR_PORT}")
                return False
            _ib.reqMarketDataType(3)
            print(f"✅ Connected to IBKR at {IBKR_HOST}:{IBKR_PORT}")
        return _ib.isConnected()
    except Exception as e:
        print(f"IBKR connect failed: {e}")
        return False

def _get_ibkr_last_price(symbol: str) -> Optional[float]:
    """Fetch last price from IBKR; returns None on failure."""
    if not _ibkr_connect_if_needed() or _IBStock is None:
        return None
    try:
        contract = _IBStock(symbol, "SMART", "USD")
        ticker = _ib.reqMktData(contract, "", False, False)
        _ib.sleep(1)
        # Prefer last; fallback to close or mid
        price = ticker.last or ticker.close
        if price is None:
            if ticker.bid is not None and ticker.ask is not None:
                price = (ticker.bid + ticker.ask) / 2.0
        return float(price) if price is not None else None
    except Exception:
        return None

def _get_last_price(symbol: str) -> Optional[float]:
    """Get last price robustly using yfinance with multiple fallbacks."""
    try:
        ticker = yf.Ticker(symbol)
        # Attempt to get the most recent close price
        hist = ticker.history(period="1d")
        if not hist.empty:
            return hist["Close"].iloc[-1]
        # Fallback to current price if history is not immediately available
        info = ticker.info
        if "currentPrice" in info:
            return info["currentPrice"]
        if "regularMarketPrice" in info:
            return info["regularMarketPrice"]
    except Exception as e:
        print(f"Error fetching price for {symbol} from yfinance: {e}")
    return None

def get_price(symbol: str) -> Optional[float]:
    """Public price fetcher: use IBKR last/close when enabled; returns None on failure."""
    return _get_ibkr_last_price(symbol)

def buy_stock(symbol: str, shares: int) -> bool:
    """Place a BUY market order on IBKR paper account. Returns True if submitted."""
    if shares <= 0:
        print(f"BUY ignored: non-positive share count {shares} for {symbol}.")
        return False
    if not _can_execute_trade('BUY'):
        print("Permission denied for BUY; order not submitted.")
        return False
    if not _ibkr_connect_if_needed() or _IBStock is None or _IBMarketOrder is None:
        print("BUY skipped: IBKR not connected or ib_insync unavailable.")
        return False
    try:
        order = _IBMarketOrder('BUY', int(shares))
        _ib.placeOrder(_IBStock(symbol, 'SMART', 'USD'), order)
        print(f"✅ Placed BUY for {shares} {symbol}")
        return True
    except Exception as e:
        print(f"BUY failed for {symbol}: {e}")
        return False

def sell_stock(symbol: str, shares: int) -> bool:
    """Place a SELL market order on IBKR paper account. Returns True if submitted."""
    if shares <= 0:
        print(f"SELL ignored: non-positive share count {shares} for {symbol}.")
        return False
    if not _can_execute_trade('SELL'):
        print("Permission denied for SELL; order not submitted.")
        return False
    if not _ibkr_connect_if_needed() or _IBStock is None or _IBMarketOrder is None:
        print("SELL skipped: IBKR not connected or ib_insync unavailable.")
        return False
    try:
        order = _IBMarketOrder('SELL', int(shares))
        _ib.placeOrder(_IBStock(symbol, 'SMART', 'USD'), order)
        print(f"❌ Placed SELL for {shares} {symbol}")
        return True
    except Exception as e:
        print(f"SELL failed for {symbol}: {e}")
        return False

class TradingEnv (gym.Env):
    metadata = {"render_modes": ["human" ]}

    def __init__(self, tickers=ALL_TICKERS, start="2010-01-01", end="2019-12-31", window_size=30 ):
        super ().__init__()
        self.tickers = tickers
        self.start = start
        self.end = end
        self.window_size = window_size
        self.initial_balance = 100000
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.last_portfolio_value = self.initial_balance
        self.shares_held = {ticker: {'long': [], 'short': []} for ticker in self.tickers}
        self.trade_history = []
        self.current_step = 0
        self.data_dir = "data"

        self.data = self._download_data()
        # Ensure self.prices is a 2D array for consistency with multiple tickers
        self.prices = self.data.values
        self.dates = self.data.index

        # Action space: 0: sell long, 1: hold, 2: buy long, 3: cover short, 4: short sell
        self.action_space = spaces.Discrete(5 * len(self.tickers))  # Five actions (buy, sell, hold, short, cover) for each ticker

        # Observation space: balance, shares held (long and short), and window_size prices for each ticker
        # The observation space will be dynamic based on the number of tickers
        # 1 (balance) + 2 * len(tickers) (total long/short shares) + len(tickers) * window_size (prices)
        num_features = 1 + 2 * len(self.tickers) + len(self.tickers) * self.window_size
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(num_features,), dtype=np.float32)
        print(f"Observation space shape: {self.observation_space.shape}")

    def _download_data(self):
        os.makedirs(self.data_dir, exist_ok=True)
        dfs_to_concat = []
        successful_tickers = []

        for ticker in self.tickers:
            csv_path = os.path.join(self.data_dir, f"{ticker}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True, date_format="%Y-%m-%d")
                cols = list(df.columns)
                # Handle prior runs and various CSV header styles
                if ticker in df.columns:
                    df = df[[ticker]]
                elif 'Adj Close' in df.columns:
                    df = df[['Adj Close']].rename(columns={'Adj Close': ticker})
                elif 'Close' in df.columns:
                    df = df[['Close']].rename(columns={'Close': ticker})
                else:
                    # Fallback: try to match columns that contain expected substrings
                    price_col = None
                    for c in df.columns:
                        if isinstance(c, str):
                            if 'Adj Close' in c:
                                price_col = c
                                break
                            if 'Close' in c:
                                price_col = c
                                break
                    if price_col is not None:
                        df = df[[price_col]].rename(columns={price_col: ticker})
                    elif isinstance(df.columns, pd.MultiIndex):
                        # Attempt to extract from MultiIndex columns
                        top = df.columns.get_level_values(0)
                        if 'Adj Close' in top:
                            s = df['Adj Close']
                        elif 'Close' in top:
                            s = df['Close']
                        else:
                            print(f"Warning: No valid price column found in CSV for {ticker}. Available columns: {cols}. Skipping ticker.")
                            continue
                        if ticker in s.columns:
                            df = s[[ticker]]
                        else:
                            print(f"Warning: Price column found but ticker {ticker} missing in CSV. Skipping ticker.")
                            continue
                # Coerce to numeric and drop any non-numeric rows (e.g., malformed headers)
                df[ticker] = pd.to_numeric(df[ticker], errors='coerce')
                df = df.dropna()
                # Ensure DatetimeIndex and drop any unparseable dates
                df.index = pd.to_datetime(df.index, errors='coerce')
                df = df[~df.index.isna()].sort_index()
                print(f"Loaded {ticker} data from {csv_path}")
                dfs_to_concat.append(df)
                successful_tickers.append(ticker)
            else:
                print(f"Downloading {ticker} data from {self.start} to {self.end}")
                try:
                    session = requests.Session(impersonate="chrome")
                    session.verify = False

                    data = yf.download(
                        ticker,
                        start=self.start,
                        end=self.end,
                        session=session,
                        auto_adjust=True,
                    )

                    if data.empty:
                        print(f"Warning: No data found for {ticker}. Skipping ticker.")
                        continue
                    
                    data.to_csv(csv_path) # Save the full data to CSV
                    print(f"Downloaded and saved {ticker} data to {csv_path}")

                    print(f"Downloaded data columns for {ticker}: {data.columns.tolist()}")
                    # Normalize yfinance output across single and MultiIndex columns
                    if isinstance(data.columns, pd.MultiIndex):
                        top = data.columns.get_level_values(0)
                        if 'Adj Close' in top:
                            s = data['Adj Close']
                        elif 'Close' in top:
                            s = data['Close']
                        else:
                            print(f"Warning: No price column found for {ticker}. Skipping ticker.")
                            os.remove(csv_path)
                            continue
                        if ticker in s.columns:
                            df = s[[ticker]]
                            df.columns = [ticker]
                        else:
                            print(f"Warning: Price column found but ticker {ticker} missing in downloaded data. Skipping ticker.")
                            os.remove(csv_path)
                            continue
                    else:
                        if 'Adj Close' in data.columns:
                            df = data[['Adj Close']].rename(columns={'Adj Close': ticker})
                        elif 'Close' in data.columns:
                            df = data[['Close']].rename(columns={'Close': ticker})
                        else:
                            print(f"Warning: No price column found for {ticker}. Skipping ticker.")
                            os.remove(csv_path)
                            continue
                    # Coerce to numeric and drop any non-numeric rows
                    df[ticker] = pd.to_numeric(df[ticker], errors='coerce')
                    df = df.dropna()
                    # Ensure DatetimeIndex and drop any unparseable dates
                    df.index = pd.to_datetime(df.index, errors='coerce')
                    df = df[~df.index.isna()].sort_index()
                    dfs_to_concat.append(df)
                    successful_tickers.append(ticker)
                except Exception as e:
                    print(f"Error downloading {ticker} data: {e}")
                    continue

        if not dfs_to_concat:
            raise ValueError("No valid data downloaded for any tickers!")

        all_data = pd.concat(dfs_to_concat, axis=1)
        self.tickers = successful_tickers
        all_data = all_data.astype(np.float32)
        self.prices = all_data.values
        self.dates = all_data.index
        return all_data.dropna()

        all_data = all_data.astype(np.float32)
        self.prices = all_data.values
        self.dates = all_data.index
        return all_data.dropna()

    def _next_observation(self):
        obs = []
        for i, ticker in enumerate(self.tickers):
            end_step = min(self.current_step + self.window_size, self.prices.shape[0])
            ticker_prices = self.prices[self.current_step:end_step, i]
            if len(ticker_prices) < self.window_size:
                # pad with last known price
                ticker_prices = np.pad(ticker_prices, (0, self.window_size - len(ticker_prices)), 'edge')
            obs.extend(ticker_prices)
        
        # Flatten shares_held into a list of total shares for each ticker (long and short)
        flat_shares_held = []
        for ticker in self.tickers:
            total_long_shares = sum(purchase['shares'] for purchase in self.shares_held[ticker]['long'])
            total_short_shares = sum(sale['shares'] for sale in self.shares_held[ticker]['short'])
            flat_shares_held.append(total_long_shares)
            flat_shares_held.append(total_short_shares)
        
        return np.array([self.balance] + flat_shares_held + obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares_held = {ticker: {'long': [], 'short': []} for ticker in self.tickers}
        self.trade_history = []
        self.current_step = 0
        self.initial_portfolio_value = self.initial_balance
        self.last_month_value = self.initial_balance
        self.action_explanation = ""
        return self._next_observation(), {}

    def _get_current_prices(self):
        current_prices = {ticker: self.data[ticker].iloc[self.current_step + self.window_size - 1] for ticker in self.tickers}
        return current_prices

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.data) - self.window_size:
            self.current_step = 0  # Reset for continuous simulation or end episode

        current_prices = self._get_current_prices()
        reward = 0
        done = False
        info = {}

        # Decode action: action_idx = action // 5, trade_type = action % 5
        # trade_type: 0: sell long, 1: hold, 2: buy long, 3: cover short, 4: short sell
        ticker_idx = action // 5
        trade_type = action % 5
        current_ticker = self.tickers[ticker_idx]
        current_price = current_prices[current_ticker]

        if trade_type == 0:  # Sell Long
            if self.shares_held[current_ticker]['long']:
                shares_to_sell = self.shares_held[current_ticker]['long'][0]['shares']
                profit_from_sale = (current_price - self.shares_held[current_ticker]['long'][0]['price']) * shares_to_sell
                self.balance += shares_to_sell * current_price
                self.portfolio_value += profit_from_sale
                self.trade_history.append({
                    'step': self.current_step,
                    'ticker': current_ticker,
                    'action': 'sell_long',
                    'shares': shares_to_sell,
                    'price': current_price,
                    'balance': self.balance
                })
                self.shares_held[current_ticker]['long'].pop(0)
                self.action_explanation = f"Sold {shares_to_sell} {current_ticker} shares at {current_price:.2f}. Profit: {profit_from_sale:.2f}"
                reward += profit_from_sale
                reward += 0.1 # Small reward for selling to encourage exploration
            else:
                self.action_explanation = f"Attempted to sell long {current_ticker} but held no long shares."
                reward -= 0.05

        elif trade_type == 2:  # Buy Long
            if self.balance > current_price:
                shares_to_buy = self.balance // current_price
                if shares_to_buy > 0:
                    self.balance -= shares_to_buy * current_price
                    self.shares_held[current_ticker]['long'].append({'shares': shares_to_buy, 'price': current_price, 'step': self.current_step})
                    self.trade_history.append({
                        'step': self.current_step,
                        'ticker': current_ticker,
                        'action': 'buy_long',
                        'shares': shares_to_buy,
                        'price': current_price,
                        'balance': self.balance
                    })
                    self.action_explanation = f"Bought {shares_to_buy} {current_ticker} shares at {current_price:.2f}"
                    reward += 0.05  # Small positive reward for initiating a position
                else:
                    self.action_explanation = f"Attempted to buy {current_ticker} but not enough balance for one share."
                    reward -= 0.01
            else:
                self.action_explanation = f"Attempted to buy {current_ticker} but insufficient balance."
                reward -= 0.01

        elif trade_type == 4: # Short Sell
            # For simplicity, assume we can always short sell if not already holding long position
            # In a real scenario, margin requirements and shortable shares would be checked
            if not self.shares_held[current_ticker]['long']:
                shares_to_short = 1 # Short sell one share for now
                self.balance += shares_to_short * current_price # Receive cash from selling
                self.portfolio_value += shares_to_short * current_price
                self.shares_held[current_ticker]['short'].append({'shares': shares_to_short, 'price': current_price, 'step': self.current_step})
                self.trade_history.append({
                    'step': self.current_step,
                    'ticker': current_ticker,
                    'action': 'short_sell',
                    'shares': shares_to_short,
                    'price': current_price,
                    'balance': self.balance
                })
                self.action_explanation = f"Short sold {shares_to_short} {current_ticker} shares at {current_price:.2f}"
                reward += 0.05 # Small reward for initiating a short position
            else:
                self.action_explanation = f"Cannot short sell {current_ticker} while holding long position."
                reward -= 0.01

        elif trade_type == 3: # Cover Short
            if self.shares_held[current_ticker]['short']:
                shares_to_cover = self.shares_held[current_ticker]['short'][0]['shares']
                cost_to_cover = shares_to_cover * current_price
                profit_from_cover = (self.shares_held[current_ticker]['short'][0]['price'] - current_price) * shares_to_cover
                self.balance -= cost_to_cover
                self.portfolio_value += profit_from_cover
                self.trade_history.append({
                    'step': self.current_step,
                    'ticker': current_ticker,
                    'action': 'cover_short',
                    'shares': shares_to_cover,
                    'price': current_price,
                    'balance': self.balance
                })
                self.shares_held[current_ticker]['short'].pop(0)
                self.action_explanation = f"Covered {shares_to_cover} {current_ticker} shares at {current_price:.2f}. Profit: {profit_from_cover:.2f}"
                reward += profit_from_cover
                reward += 0.1 # Small reward for covering to encourage exploration
            else:
                self.action_explanation = f"Attempted to cover short {current_ticker} but held no short shares."
                reward -= 0.05

        elif trade_type == 1:  # Hold
            self.action_explanation = f"Held {current_ticker} shares."
            # Check for profitable long sell opportunity
            if self.shares_held[current_ticker]['long'] and current_price > self.shares_held[current_ticker]['long'][0]['price']:
                reward -= 0.05 # Penalty for holding when a profitable long sell was possible
            # Check for profitable short cover opportunity
            if self.shares_held[current_ticker]['short'] and current_price < self.shares_held[current_ticker]['short'][0]['price']:
                reward -= 0.05 # Penalty for holding when a profitable short cover was possible
            # Slight penalty for holding with no position to encourage exploration
            if not self.shares_held[current_ticker]['long'] and not self.shares_held[current_ticker]['short']:
                reward -= 0.005

        # Update portfolio_value to reflect current market value of all holdings
        self.portfolio_value = self.get_portfolio_value()

        # Reward is the change in portfolio value
        reward += (self.portfolio_value - self.last_portfolio_value)
        self.last_portfolio_value = self.portfolio_value

        # Done condition (can be modified)
        if self.current_step >= len(self.data) - self.window_size - 1:
            done = True

        obs = self._next_observation()
        return obs, reward, done, False, info

    def render(self):
        current_prices = self._get_current_prices()
        portfolio_value = self.balance
        for ticker in self.tickers:
            for purchase in self.shares_held[ticker]['long']:
                portfolio_value += purchase['shares'] * current_prices[ticker]
            for sale in self.shares_held[ticker]['short']:
                portfolio_value -= sale['shares'] * current_prices[ticker]

        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance:.2f}")
        for ticker in self.tickers:
            total_long_shares = sum(purchase['shares'] for purchase in self.shares_held[ticker]['long'])
            total_short_shares = sum(sale['shares'] for sale in self.shares_held[ticker]['short'])
            print(f"Shares of {ticker}: Long: {total_long_shares}, Short: {total_short_shares}")
        print(f"Portfolio Value: {portfolio_value:.2f}")
        print(f"Action: {self.action_explanation}")
        print("------------------------------------")

    def close(self):
        pass

    def get_trade_history(self):
        return self.trade_history

    def get_balance(self):
        return self.balance

    def get_shares_held(self):
        return self.shares_held

    def get_portfolio_value(self):
        current_prices = self._get_current_prices()
        portfolio_value = self.balance
        for ticker in self.tickers:
            for purchase in self.shares_held[ticker]['long']:
                portfolio_value += purchase['shares'] * current_prices[ticker]
            for sale in self.shares_held[ticker]['short']:
                # For short positions, the value is effectively negative, as it represents a liability
                # However, for portfolio value calculation, we consider the current market value of the shares
                # that would be needed to cover the short. This is a simplification.
                # A more precise calculation would involve tracking the profit/loss from short sales separately.
                portfolio_value -= sale['shares'] * current_prices[ticker]
        return portfolio_value

    def get_current_date(self):
        return self.dates[self.current_step + self.window_size - 1]

    def get_max_steps(self):
        return len(self.data) - self.window_size - 1

    def get_trades(self):
        return self.trade_history

    def calculate_trade_profits(self):
        profits = []
        open_long_positions = {ticker: [] for ticker in self.tickers}
        open_short_positions = {ticker: [] for ticker in self.tickers}

        for trade in self.trade_history:
            ticker = trade['ticker']
            action = trade['action']
            shares = trade['shares']
            price = trade['price']
            step = trade['step']

            if action == 'buy_long':
                open_long_positions[ticker].append({'shares': shares, 'price': price, 'step': step})
            elif action == 'sell_long':
                remaining_shares_to_cover = shares
                while remaining_shares_to_cover > 0 and open_long_positions[ticker]:
                    oldest_buy = open_long_positions[ticker][0]

                    if oldest_buy['shares'] <= remaining_shares_to_cover:
                        profit = (price - oldest_buy['price']) * oldest_buy['shares']
                        profits.append({
                            'type': 'long_trade',
                            'ticker': ticker,
                            'shares': oldest_buy['shares'],
                            'open_price': oldest_buy['price'],
                            'close_price': price,
                            'profit': profit,
                            'open_step': oldest_buy['step'],
                            'close_step': step
                        })
                        remaining_shares_to_cover -= oldest_buy['shares']
                        open_long_positions[ticker].pop(0)
                    else:
                        profit = (price - oldest_buy['price']) * remaining_shares_to_cover
                        profits.append({
                            'type': 'long_trade',
                            'ticker': ticker,
                            'shares': remaining_shares_to_cover,
                            'open_price': oldest_buy['price'],
                            'close_price': price,
                            'profit': profit,
                            'open_step': oldest_buy['step'],
                            'close_step': step
                        })
                        oldest_buy['shares'] -= remaining_shares_to_cover
                        remaining_shares_to_cover = 0
            elif action == 'short_sell':
                open_short_positions[ticker].append({'shares': shares, 'price': price, 'step': step})
            elif action == 'cover_short':
                remaining_shares_to_cover = shares
                while remaining_shares_to_cover > 0 and open_short_positions[ticker]:
                    oldest_short = open_short_positions[ticker][0]

                    if oldest_short['shares'] <= remaining_shares_to_cover:
                        profit = (oldest_short['price'] - price) * oldest_short['shares']
                        profits.append({
                            'type': 'short_trade',
                            'ticker': ticker,
                            'shares': oldest_short['shares'],
                            'open_price': oldest_short['price'],
                            'close_price': price,
                            'profit': profit,
                            'open_step': oldest_short['step'],
                            'close_step': step
                        })
                        remaining_shares_to_cover -= oldest_short['shares']
                        open_short_positions[ticker].pop(0)
                    else:
                        profit = (oldest_short['price'] - price) * remaining_shares_to_cover
                        profits.append({
                            'type': 'short_trade',
                            'ticker': ticker,
                            'shares': remaining_shares_to_cover,
                            'open_price': oldest_short['price'],
                            'close_price': price,
                            'profit': profit,
                            'open_step': oldest_short['step'],
                            'close_step': step
                        })
                        oldest_short['shares'] -= remaining_shares_to_cover
                        remaining_shares_to_cover = 0
        return profits


# Models and Logs directories
models_dir = "models"
logdir = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

def _safe_model_basename(tickers: list[str]) -> str:
    """Build a safe model base path without special characters (no hyphens)."""
    safe_tickers = [str(t).replace("-", "") for t in tickers]
    safe_name = "_".join(safe_tickers)
    return os.path.join(models_dir, f"ppo_trading_model_{safe_name}")

def _map_ppo_action_to_trade_and_shares(action: int, max_order_shares: int) -> tuple[str, int]:
    """Maps a PPO action to a trade type (buy/sell/hold) and number of shares."""
    if action == 0:
        return "hold", 0
    elif 1 <= action <= max_order_shares:
        return "buy", action
    elif max_order_shares + 1 <= action <= 2 * max_order_shares:
        return "sell", action - max_order_shares
    else:
        raise ValueError(f"Invalid PPO action: {action}")

def reset_learning():
    print("Resetting learning data...")
    for f in os.listdir(models_dir):
        if f.startswith("ppo_trading_model") and f.endswith(".zip"):
            os.remove(os.path.join(models_dir, f))
    print("Learning data reset.")

def start_training(total_timesteps=50000, tickers=ALL_TICKERS):
    print("Training started...")

    env_train = make_vec_env(lambda: TradingEnv(start="2010-01-01", end="2019-12-31", window_size=30, tickers=tickers), n_envs=1)
    print(f"Using {len(env_train.get_attr('tickers')[0])} tickers for training: {env_train.get_attr('tickers')[0]}")

    # Always use a ticker-specific safe basename to avoid mismatches
    base = _safe_model_basename(tickers)
    print(f"Model base: {base} (will save to {base}.zip)")
    zip_path = f"{base}.zip"
    if os.path.exists(zip_path):
        print("Loading existing model for incremental training...")
        try:
            model = PPO.load(base, env=env_train, verbose=1)
        except Exception as e:
            print(f"Failed to load existing model ({zip_path}): {e}. Starting fresh training.")
            logging.error(f"Model load failed at {zip_path}: {e}", exc_info=True)
            model = PPO("MlpPolicy", env_train, verbose=1, policy_kwargs=dict(net_arch=[64, 64]))
    else:
        print("No existing model found. Creating new model for training...")
        model = PPO("MlpPolicy", env_train, verbose=1, policy_kwargs=dict(net_arch=[64, 64]))

    timesteps_per_iteration = 10000
    total_timesteps_trained = 0
    
    stop_file_path = "STOP_TRAINING"
    if os.path.exists(stop_file_path):
        os.remove(stop_file_path)

    while total_timesteps_trained < total_timesteps:
        if os.path.exists(stop_file_path):
            print("STOP_TRAINING file detected. Stopping training.")
            os.remove(stop_file_path)
            break

        model.learn(total_timesteps=timesteps_per_iteration, reset_num_timesteps=False)
        total_timesteps_trained += timesteps_per_iteration
        model.save(f"{base}_{total_timesteps_trained}")
        print(f"Model saved after {total_timesteps_trained} timesteps")
        progress_percent = (total_timesteps_trained / total_timesteps) * 100
        print(f"Training Progress: {progress_percent:.2f}%")

    model.save(base)
    # Save simple metadata to help prevent mismatched loads
    try:
        obs_dim = int(env_train.observation_space.shape[0]) if getattr(env_train.observation_space, 'shape', None) else None
        meta = {"tickers": [str(t) for t in tickers], "obs_dim": obs_dim, "timestamp": str(datetime.now())}
        with open(f"{base}.meta.json", "w") as mf:
            json.dump(meta, mf)
    except Exception:
        pass
    print("Training complete and model saved.")


def simulate_trading(model, tickers=ALL_TICKERS, log_dir="logs", reset_bot=False):
    
    env = DummyVecEnv([lambda: Monitor(TradingEnv(tickers=tickers, start="2020-01-01", end="2023-12-31", window_size=30))])
    
    if reset_bot:
        print("Resetting bot learning...")
        # This will clear the replay buffer and reset the model
        model.learn(total_timesteps=1000, reset_num_timesteps=True)

    obs = env.reset()
    real_env = env.envs[0].env  # unwrap Monitor
    done = False
    episode_reward = 0
    trade_results = []
    pl_per_year = defaultdict(float)

    print("--- Detailed Trade Results Start ---")
    while not done:
        # Optional auto-reconnect to IBKR if enabled
        if IBKR_ENABLED:
            _ibkr_connect_if_needed()
        # Use stochastic actions during evaluation to encourage exploration and surface trades
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        
        episode_reward += reward[0]

        # Get current date for trade logging
        current_date = real_env.get_current_date()

        # Log trades if any occurred in this step
        for trade in real_env.get_trades():
            if trade['step'] == real_env.current_step:
                trade_results.append({
                    'date': str(current_date),
                    'ticker': trade['ticker'],
                    'action': trade['action'],
                    'shares': trade['shares'],
                    'price': trade['price'],
                    'balance': real_env.get_balance()
                })
                # Execute IBKR paper trade when enabled
                if IBKR_EXECUTE and _ibkr_connect_if_needed() and _IBStock is not None and _IBMarketOrder is not None:
                    try:
                        symbol = trade['ticker']
                        shares = int(trade['shares'])
                        action = trade['action']
                        side = 'BUY' if action in ('buy_long', 'cover_short') else 'SELL'
                        order = _IBMarketOrder(side, shares)
                        _ib.placeOrder(_IBStock(symbol, 'SMART', 'USD'), order)
                        print(f"IBKR order placed: {side} {shares} {symbol}")
                    except Exception as e:
                        print(f"IBKR order failed for {trade['ticker']}: {e}")
                print(f"Date: {current_date}, Ticker: {trade['ticker']}, Action: {trade['action']}, Shares: {trade['shares']}, Price: {trade['price']:.2f}, Balance: {real_env.get_balance():.2f}")

        # Calculate P/L per year
        current_year = current_date.year
        # Update P/L for the current year based on the portfolio value at the current step
        pl_per_year[current_year] = real_env.get_portfolio_value() - real_env.initial_balance

    print("--- Detailed Trade Results End ---")

    final_portfolio_value = real_env.get_portfolio_value()
    print(f"Episode finished. Final Portfolio Value: {final_portfolio_value:.2f}")
    print(f"Total Episode Reward: {episode_reward:.2f}")

    # Calculate detailed trade profits
    detailed_profits = real_env.calculate_trade_profits()

    # Assign all trades from the environment's trade_history to trade_results
    trade_results = real_env.get_trades()

    # Display P/L per year
    print("\nProfit/Loss per year:")
    for year, pl in pl_per_year.items():
        print(f"Year {year}: {pl:.2f}")

    return trade_results, pl_per_year, detailed_profits

def save_simulation_data(trade_results, pl_per_year, detailed_profits, results_dir):
    print("Saving simulation data...")

    # Save trade results
    results_dir = "simulation_results"
    os.makedirs(results_dir, exist_ok=True)

    # Convert NumPy types to standard Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(elem) for elem in obj]
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float16, np.float32, np.float64)):
            return float(obj)
        else:
            return obj

    trade_results_serializable = convert_numpy_types(trade_results)
    detailed_profits_serializable = convert_numpy_types(detailed_profits)

    # Save detailed trade results as JSON
    with open(f"{results_dir}/trade_results.json", "w") as f:
        import json
        json.dump(trade_results_serializable, f, indent=4)
    print(f"Detailed trade results saved to {results_dir}/trade_results.json")

    # Save profit/loss per year as a text file
    with open(f"{results_dir}/pl_per_year.txt", "w") as f:
        f.write("Profit/Loss per year:\n")
        f.write(str(pl_per_year))
    print(f"Profit/Loss per year saved to {results_dir}/pl_per_year.txt")

    # Save detailed trade profits
    with open(os.path.join(results_dir, "detailed_trade_profits.json"), "w") as f:
        json.dump(detailed_profits_serializable, f, indent=4)
    print(f"Detailed trade profits saved to {results_dir}/detailed_trade_profits.json")

    print("Simulation data saved.")

def _parse_tickers_arg(tickers_arg: str | None):
    if not tickers_arg:
        return ["SPY"]
    return [t.strip().upper() for t in tickers_arg.split(',') if t.strip()]


def _prompt_str(prompt: str, default: str):
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default


def _prompt_int(prompt: str, default: int):
    val = input(f"{prompt} [{default}]: ").strip()
    try:
        return int(val) if val else default
    except ValueError:
        return default


def _prompt_choice(prompt: str, choices: list[str], default: str):
    cs = '/'.join(choices)
    val = input(f"{prompt} ({cs}) [{default}]: ").strip().lower()
    return val if val in choices else default


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Trading Bot UI and actions")
    parser.add_argument("--menu", action="store_true", help="Launch interactive menu UI")
    parser.add_argument("--action", choices=["save_knowledge", "research", "resimulate", "live", "paper"], help="Run a single action non-interactively")
    parser.add_argument("--tickers", type=str, help="Comma-separated tickers, e.g., SPY,AAPL")
    parser.add_argument("--start", type=str, default="2010-01-01", help="Start date for research/simulation")
    parser.add_argument("--end", type=str, default="2019-12-31", help="End date for research/simulation")
    parser.add_argument("--window-size", type=int, default=30, help="Window size for env/research")
    parser.add_argument("--format", choices=["csv", "parquet"], default="parquet", help="Big knowledge file format")
    parser.add_argument("--minutes", type=int, default=30, help="Minutes for wall-clock research")
    parser.add_argument("--save-every-secs", type=int, default=60, help="Periodic save interval in seconds for wall-clock research")
    # Live dashboard options
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Polling interval in seconds for live dashboard")
    parser.add_argument("--positions", type=str, help="Positions as comma-separated TICKER:QTY:AVGCOST entries")
    parser.add_argument("--compact", action="store_true", help="Compact single-line output for live dashboard")
    # Paper trading account options
    parser.add_argument("--paper-op", choices=["init", "view", "buy", "sell", "set-balance", "live", "ai"], help="Paper account operation")
    parser.add_argument("--symbol", type=str, help="Symbol for paper buy/sell")
    parser.add_argument("--shares", type=float, default=0.0, help="Share quantity for buy/sell (positive for buy, positive value used for sell)")
    parser.add_argument("--price", type=float, help="Override price for buy/sell; if omitted, live price is used")
    parser.add_argument("--balance", type=float, help="Set cash balance for paper account")
    parser.add_argument("--trade-shares", type=float, default=1.0, help="Share quantity per AI trade in paper control")
    # IBKR integration options
    parser.add_argument("--use-ibkr", action="store_true", help="Use IBKR for live prices and optional execution")
    parser.add_argument("--ib-host", type=str, default="127.0.0.1", help="IBKR host (default 127.0.0.1)")
    parser.add_argument("--ib-port", type=int, default=7497, help="IBKR port (7497 TWS, 4001 Gateway)")
    parser.add_argument("--ib-client-id", type=int, default=1, help="IBKR client ID")
    parser.add_argument("--ib-exec", action="store_true", help="Execute orders on IBKR paper account for buys/sells")
    parser.add_argument("--broker", choices=["mock", "ibkr"], default="mock", help="Select broker backend: mock (simulation) or ibkr (live)")
    return parser.parse_args()


def do_save_knowledge(tickers, start, end, window_size, big_format):
    knowledge_dir = "knowledge"
    os.makedirs(knowledge_dir, exist_ok=True)
    basename = f"market_knowledge_{'_'.join(tickers)}_{start[:4]}_{end[:4]}"
    knowledge_path = os.path.join(knowledge_dir, f"{basename}.json")
    knowledge = run_market_research_session(tickers=tickers, start=start, end=end, window_size=window_size, knowledge_path=knowledge_path)
    big_ext = "parquet" if big_format == "parquet" else "csv"
    big_path = os.path.join(knowledge_dir, f"market_knowledge_big_{'_'.join(tickers)}_{start[:4]}_{end[:4]}.{big_ext}")
    save_knowledge_big(knowledge, big_path, format=big_format)
    print(f"Saved knowledge JSON at {knowledge_path}\nSaved big file at {big_path}")


def do_wall_clock_research(tickers, start, end, window_size, minutes, save_every_secs, big_format):
    knowledge_dir = "knowledge"
    os.makedirs(knowledge_dir, exist_ok=True)
    big_ext = "parquet" if big_format == "parquet" else "csv"
    wall_json = os.path.join(knowledge_dir, f"market_knowledge_{'_'.join(tickers)}_wallclock.json")
    wall_big = os.path.join(knowledge_dir, f"market_knowledge_big_{'_'.join(tickers)}_wallclock.{big_ext}")
    run_market_research_wall_clock(
        tickers=tickers,
        start=start,
        end=end,
        window_size=window_size,
        minutes=minutes,
        knowledge_path=wall_json,
        big_file_path=wall_big,
        save_every_secs=save_every_secs,
        format=big_format,
    )
    print(f"Saved wall-clock knowledge JSON at {wall_json}\nSaved wall-clock big file at {wall_big}")


def do_resimulate(tickers):
    env_train = make_vec_env(lambda: TradingEnv(tickers=tickers, start="2010-01-01", end="2019-12-31", window_size=30), n_envs=1)
    base = _safe_model_basename(tickers)
    zip_path = f"{base}.zip"
    if not os.path.exists(zip_path):
        print(f"No model found at {zip_path}. Training a new model for {tickers}...")
        start_training(total_timesteps=50000, tickers=tickers)
    try:
        model = PPO.load(base, env=env_train, verbose=1)
    except Exception as e:
        print(f"Error loading model from {zip_path}: {e}. Training a fresh model and retrying.")
        logging.error(f"Resimulate: model load failed at {zip_path}: {e}", exc_info=True)
        start_training(total_timesteps=50000, tickers=tickers)
        model = PPO.load(base, env=env_train, verbose=1)
    trade_results, pl_per_year, detailed_profits = simulate_trading(model, tickers=tickers, log_dir=logdir, reset_bot=False)
    save_simulation_data(trade_results, pl_per_year, detailed_profits, "simulation_results")
    print("Resimulation complete and results saved.")


def _parse_positions_arg(positions_arg: str | None, tickers: list[str]) -> dict:
    """Parse positions input like 'AAPL:10:180,META:5:300' into dict {ticker: (qty, avg_cost)}.
    Missing tickers default to (0.0, 0.0).
    """
    positions: dict[str, tuple[float, float]] = {}
    if positions_arg:
        try:
            parts = [p.strip() for p in positions_arg.split(',') if p.strip()]
            for part in parts:
                toks = [t.strip() for t in part.split(':')]
                if len(toks) != 3:
                    print(f"Warning: invalid positions entry '{part}', expected TICKER:QTY:AVGCOST. Skipping.")
                    continue
                t, qty, avg = toks
                try:
                    positions[t] = (float(qty), float(avg))
                except ValueError:
                    print(f"Warning: non-numeric qty/avg-cost in '{part}'. Skipping.")
        except Exception as e:
            print(f"Warning: failed to parse positions '{positions_arg}': {e}")
    for t in tickers:
        positions.setdefault(t, (0.0, 0.0))
    return positions


def _get_last_price(t: str) -> Optional[float]:
    """Get last price robustly using yfinance with multiple fallbacks.
    Tries fast_info, then progressively longer periods/intervals, then yf.download.
    """
    # Prefer IBKR source if enabled
    if IBKR_ENABLED:
        ib_price = _get_ibkr_last_price(t)
        if ib_price is not None:
            return ib_price
    try:
        tk = yf.Ticker(t)
        # Try fast_info first
        try:
            fi = tk.fast_info
            if fi is not None:
                val = None
                try:
                    val = fi.get('last_price')  # dict-like FastInfo
                except Exception:
                    # Some versions expose mapping via __getitem__ only
                    try:
                        val = fi['last_price']
                    except Exception:
                        val = None
                if val is not None:
                    return float(val)
        except Exception:
            pass

        # Prepare an insecure curl session to bypass local cert issues
        curl_sess = None
        try:
            curl_sess = requests.Session(impersonate="chrome")
            curl_sess.verify = False
        except Exception:
            curl_sess = None

        # Fallback to history with increasing coverage
        history_attempts = [
            ('1d', '1m'),
            ('1d', '5m'),
            ('5d', '1m'),
            ('5d', '5m'),
            ('1mo', '1d'),
        ]
        for period, interval in history_attempts:
            try:
                if curl_sess is not None:
                    hist = tk.history(period=period, interval=interval, auto_adjust=False, session=curl_sess)
                else:
                    hist = tk.history(period=period, interval=interval, auto_adjust=False)
                if isinstance(hist, pd.DataFrame) and not hist.empty:
                    col = 'Close' if 'Close' in hist.columns else ('Adj Close' if 'Adj Close' in hist.columns else None)
                    if col:
                        return float(hist[col].iloc[-1])
            except Exception:
                continue

        # Final fallback via yf.download
        download_attempts = [
            {'period': '1d', 'interval': '1m', 'auto_adjust': False},
            {'period': '5d', 'interval': '1m', 'auto_adjust': False},
            {'period': '1mo', 'interval': '1d', 'auto_adjust': False},
        ]
        for kwargs in download_attempts:
            try:
                if curl_sess is not None:
                    df = yf.download(t, progress=False, session=curl_sess, **kwargs)
                else:
                    df = yf.download(t, progress=False, **kwargs)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    col = 'Close' if 'Close' in df.columns else ('Adj Close' if 'Adj Close' in df.columns else None)
                    if col:
                        return float(df[col].iloc[-1])
            except Exception:
                continue
        # Ultimate fallback via Yahoo Finance quote API (insecure verify disabled)
        try:
            sess = requests.Session(impersonate="chrome")
            sess.verify = False
            url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={t}"
            resp = sess.get(url, timeout=5)
            if resp and resp.status_code == 200:
                js = resp.json()
                results = js.get('quoteResponse', {}).get('result', [])
                if results:
                    rm = results[0].get('regularMarketPrice')
                    if rm is not None and not (isinstance(rm, float) and (math.isnan(rm) or math.isinf(rm))):
                        return float(rm)
        except Exception:
            pass
        return None
    except Exception:
        return None


def do_live_dashboard(tickers: list[str], poll_interval: float = 1.0, positions_arg: str | None = None, compact: bool = False, duration_secs: Optional[float] = None):
    """Render a live dashboard that updates price, position, and P&L every poll_interval seconds.
    - positions_arg: 'TICKER:QTY:AVGCOST' entries separated by commas.
    - compact: if True, prints a single-line summary; otherwise prints one line per ticker.
    - duration_secs: optional limit; otherwise runs until Ctrl+C.
    """
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    positions = _parse_positions_arg(positions_arg, tickers)
    start_ts = time.time()
    try:
        while True:
            # Duration check
            if duration_secs is not None and (time.time() - start_ts) >= duration_secs:
                break
            total_pnl = 0.0
            lines = []
            for t in tickers:
                price = _get_last_price(t)
                qty, avg = positions.get(t, (0.0, 0.0))
                pnl = (price - avg) * qty if (price is not None and qty != 0) else 0.0
                total_pnl += pnl
                status = "Profit" if pnl > 0 else "Loss" if pnl < 0 else "Break-even"
                price_str = f"{price:.2f}" if price is not None else "N/A"
                lines.append(f"{t}: Price {price_str} | Qty {qty:.2f} | Avg {avg:.2f} | P&L {pnl:.2f} | {status}")
            if compact:
                sys.stdout.write("\r" + ("; ".join(lines) + f" || Total P&L {total_pnl:.2f}"))
                sys.stdout.flush()
            else:
                # Clear screen and print multi-line
                sys.stdout.write("\x1b[2J\x1b[H")  # ANSI clear and home
                for ln in lines:
                    print(ln)
                print(f"Total P&L: {total_pnl:.2f}")
            time.sleep(max(0.1, float(poll_interval)))
    except KeyboardInterrupt:
        print("\nLive dashboard stopped by user.")


# (moved earlier)

def _json_load_account(path: str) -> dict:
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                # Normalize structure
                bal = float(data.get('balance', 100000.0))
                pos = data.get('positions', {}) or {}
                # Ensure each position has qty and avg_price
                norm_pos = {}
                for sym, val in pos.items():
                    if isinstance(val, dict):
                        qty = float(val.get('qty', 0.0))
                        avg = float(val.get('avg_price', 0.0))
                    else:
                        qty = float(val or 0.0)
                        avg = 0.0
                    norm_pos[sym.upper()] = {'qty': qty, 'avg_price': avg}
                return {'balance': bal, 'positions': norm_pos}
    except Exception:
        pass
    return {'balance': 100000.0, 'positions': {}}


def _json_save_account(path: str, data: dict):
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Save error: {e}")


def do_ibkr_ai_bot(watchlist: list[str], save_file: str, buy_threshold: float, sell_threshold: float, order_shares: float, poll_interval: float = 1.0):
    # Ensure IBKR pricing/execution context
    if not IBKR_ENABLED:
        print("IBKR is not enabled. Enable it in 'IBKR settings' first.")
    ok = _ibkr_connect_if_needed()
    if not ok:
        print("IBKR not connected. Will attempt reconnects; using limited data until then.")

    account = _json_load_account(save_file)
    # Prepare contracts and subscribe to market data
    contracts = []
    tickers = []
    for sym in [s.strip().upper() for s in watchlist if s.strip()]:
        if _IBStock is not None and ok:
            try:
                c = _IBStock(sym, 'SMART', 'USD')
                contracts.append(c)
                t = _ib.reqMktData(c, '', False, False)
                tickers.append(t)
            except Exception as e:
                print(f"Subscription error for {sym}: {e}")
        else:
            contracts.append(None)
            tickers.append(None)

    print("Watching:", ", ".join([s for s in watchlist]))
    try:
        while True:
            # Trading loop
            for idx, sym in enumerate(watchlist):
                sym = sym.upper()
                t = tickers[idx]
                price = None
                if t is not None:
                    # Derive best-available price from stream
                    try:
                        price = t.last or t.close
                        if price is None and t.bid is not None and t.ask is not None:
                            price = (t.bid + t.ask) / 2.0
                    except Exception:
                        price = None
                if price is None:
                    # Fallback to existing price fetcher
                    price = _get_last_price(sym)
                if price is None:
                    continue

                pos = account['positions'].get(sym, {'qty': 0.0, 'avg_price': 0.0})
                qty = float(pos.get('qty', 0.0))
                avg = float(pos.get('avg_price', 0.0))

                # Simple threshold strategy
                if price <= buy_threshold and account['balance'] >= price * order_shares:
                    # Optional IBKR execution
                    if IBKR_EXECUTE and ok and _IBMarketOrder is not None and contracts[idx] is not None:
                        try:
                            order = _IBMarketOrder('BUY', int(round(order_shares)))
                            trade = _ib.placeOrder(contracts[idx], order)
                            _ib.sleep(0.2)
                            status = getattr(trade.orderStatus, 'status', 'Unknown')
                            print(f"IBKR BUY {sym} {order_shares} status: {status}")
                        except Exception as e:
                            print(f"IBKR BUY error: {e}")
                    # Update local account
                    cost = price * order_shares
                    new_qty = qty + order_shares
                    new_avg = ((qty * avg) + (order_shares * price)) / new_qty if new_qty > 0 else 0.0
                    account['balance'] -= cost
                    account['positions'][sym] = {'qty': new_qty, 'avg_price': new_avg}
                    print(f"✅ Bought {order_shares} {sym} at ${price:.2f}")
                elif price >= sell_threshold and qty >= order_shares:
                    # Optional IBKR execution
                    if IBKR_EXECUTE and ok and _IBMarketOrder is not None and contracts[idx] is not None:
                        try:
                            order = _IBMarketOrder('SELL', int(round(order_shares)))
                            trade = _ib.placeOrder(contracts[idx], order)
                            _ib.sleep(0.2)
                            status = getattr(trade.orderStatus, 'status', 'Unknown')
                            print(f"IBKR SELL {sym} {order_shares} status: {status}")
                        except Exception as e:
                            print(f"IBKR SELL error: {e}")
                    proceeds = price * order_shares
                    account['balance'] += proceeds
                    new_qty = qty - order_shares
                    # If fully closed, remove; otherwise keep avg
                    if new_qty <= 0:
                        account['positions'].pop(sym, None)
                    else:
                        account['positions'][sym] = {'qty': new_qty, 'avg_price': avg}
                    realized = order_shares * (price - avg)
                    print(f"✅ Sold {order_shares} {sym} at ${price:.2f} | Realized ${realized:.2f}")

            # Save account each loop
            _json_save_account(save_file, account)
            # Compute account value and print summary line
            total_value = 0.0
            for sym, pos in account['positions'].items():
                lp = _get_last_price(sym)
                if lp is not None:
                    total_value += lp * float(pos.get('qty', 0.0))
            total_balance = float(account['balance']) + total_value
            print(f"💵 Total Account Value: ${total_balance:,.2f} | Cash ${float(account['balance']):,.2f} | Positions {len(account['positions'])}")
            # Poll interval
            try:
                if ok:
                    _ib.sleep(max(0.1, float(poll_interval)))
                else:
                    time.sleep(max(0.1, float(poll_interval)))
            except Exception:
                time.sleep(max(0.1, float(poll_interval)))
            # Try reconnect occasionally if disconnected
            if not ok:
                ok = _ibkr_connect_if_needed()
    except KeyboardInterrupt:
        _json_save_account(save_file, account)
        print("Session saved & stopped.")


def _load_holdings(save_file: str) -> dict:
    """Load saved positions from JSON file; returns empty dict if missing."""
    try:
        if os.path.exists(save_file):
            with open(save_file, "r") as f:
                return json.load(f)
    except Exception as e:
        logging.warning(f"Failed to load holdings from {save_file}: {e}")
    return {}


def _save_holdings(save_file: str, holdings: dict) -> None:
    """Save current positions to JSON file."""
    try:
        with open(save_file, "w") as f:
            json.dump(holdings, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save holdings to {save_file}: {e}", exc_info=True)


def _compute_ibkr_balance(holdings):
    """
    Compute current IBKR balance (paper or sim) from holdings.
    """
    balance = 0.0
    for symbol, position in holdings.items():
        qty = position.get('quantity', 0)
        price = position.get('price', 0)
        balance += qty * price
    return balance

def do_ibkr_ai_bot_ppo(watchlist: list[str], save_file: str, max_order_shares: int, poll_interval: int = 60) -> None:
    """PPO-driven IBKR AI bot to execute model-predicted actions live with safety checks"""
    import os
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    try:
        # Normalize watchlist symbols
        watchlist = [s.strip().upper() for s in watchlist if s and s.strip()]

        # Load per-symbol PPO models (train if missing) with intact ticker names and single .zip
        models = {}
        models_dir = "/Users/ashtonphillips/Desktop/Trading/models"
        os.makedirs(models_dir, exist_ok=True)
        for symbol in watchlist:
            filename = f"ppo_trading_model_{symbol}.zip"
            model_path = os.path.join(models_dir, filename)
            print(f"Loading PPO model for {symbol} from {model_path}...")
            if os.path.exists(model_path):
                try:
                    model = PPO.load(model_path)
                    models[symbol] = model
                    print(f"Loaded PPO model for {symbol}")
                except Exception as e:
                    print(f"Warning: Failed to load PPO model for {symbol} ({e}). Will retrain.")
            if symbol not in models:
                # Train a new model if it doesn't exist or failed to load
                print(f"No trained PPO model found for {symbol}. Training new model...")
                env_train = Monitor(TradingEnv(tickers=[symbol], start="2010-01-01", end="2019-12-31", window_size=30))
                model = PPO('MlpPolicy', env_train, verbose=0)
                model.learn(total_timesteps=50000)
                model.save(model_path)
                models[symbol] = model
                print(f"Training complete for {symbol} and saved at {model_path}")

        # Connect to IBKR; if not connected, run in simulation mode instead of exiting
        ibkr_connected = _ibkr_connect_if_needed()
        if not ibkr_connected:
            print("IBKR not connected. Running PPO in simulation mode...")

        # Initialize holdings and balance via broker/paper account
        holdings = _load_holdings(save_file)
        last_balance = BROKER.get_balance() if BROKER is not None else _compute_ibkr_balance(holdings)

        # Cache per-symbol simulation environments to avoid repeated CSV loads
        envs_sim: dict[str, TradingEnv] = {}
        for symbol in models.keys():
            try:
                envs_sim[symbol] = TradingEnv(tickers=[symbol], start="2020-01-01", end="2023-12-31", window_size=30)
            except Exception as e:
                logging.warning(f"Failed to initialize env for {symbol}: {e}. Will create on demand.")

        print(f"PPO IBKR bot started - watching {watchlist}")
        print("Starting main PPO trading loop...")
        while True:
            for symbol, model in models.items():
                logging.info(f"PPO loop cycle started for {symbol}")
                # Use cached single-symbol env to generate an observation
                env_sim = envs_sim.get(symbol)
                if env_sim is None:
                    try:
                        envs_sim[symbol] = TradingEnv(tickers=[symbol], start="2020-01-01", end="2023-12-31", window_size=30)
                        env_sim = envs_sim[symbol]
                    except Exception as e:
                        logging.error(f"Env creation failed for {symbol}: {e}")
                        continue
                obs = env_sim.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]

                # Predict action from model
                action, _ = model.predict(obs, deterministic=True)

                # Map discrete action (0..4 for single ticker) to trade type
                # 0: sell long, 1: hold, 2: buy long, 3: cover short, 4: short sell
                trade = "hold"
                if isinstance(action, (int, np.integer)):
                    if action == 2:
                        trade = "buy"
                    elif action in (0, 4):
                        trade = "sell"

                # Get price from broker (fallback to last price fetch)
                price = BROKER.get_price(symbol) if BROKER is not None else _get_last_price(symbol)
                if not price:
                    logging.warning(f"Failed to retrieve price for {symbol} - skipping cycle")
                    continue

                # Determine shares dynamically based on balance and max allowed
                # Buy up to ~5% of balance, bounded by max_order_shares; sell up to holdings
                if trade == "buy":
                    budget = float(last_balance) * 0.05
                    calc_qty = int(budget / float(price)) if price else 0
                    shares_to_trade = max(1, min(int(max_order_shares), calc_qty))
                elif trade == "sell":
                    shares_to_trade = min(int(max_order_shares), int(holdings.get(symbol, 0)))
                else:
                    shares_to_trade = 0

                # Execute action with safety checks (simulate if not connected)
                if trade == "buy" and shares_to_trade > 0 and validate_order(symbol, price, shares_to_trade, last_balance):
                    if ibkr_connected and BROKER is not None:
                        safe_buy(BROKER, symbol, float(shares_to_trade), price)
                        logging.info(f"Live BUY {shares_to_trade} {symbol} at {price}")
                    else:
                        logging.info(f"Simulated BUY {shares_to_trade} {symbol} at {price}")
                    holdings[symbol] = holdings.get(symbol, 0) + shares_to_trade
                elif trade == "sell" and shares_to_trade > 0 and validate_order(symbol, price, shares_to_trade, last_balance):
                    qty = shares_to_trade
                    if ibkr_connected and BROKER is not None:
                        safe_sell(BROKER, symbol, float(qty), price)
                        logging.info(f"Live SELL {qty} {symbol} at {price}")
                    else:
                        logging.info(f"Simulated SELL {qty} {symbol} at {price}")
                    holdings[symbol] = max(0, holdings.get(symbol, 0) - qty)

                # Update balance and persist holdings
                if ibkr_connected and BROKER is not None:
                    last_balance = BROKER.get_balance()
                else:
                    last_balance = _compute_ibkr_balance(holdings)
                _save_holdings(save_file, holdings)

                logging.info(f"Cycle complete for {symbol} - Action: {trade}, Price: {price}, Balance: {last_balance}, Connected: {ibkr_connected}")

            # Attempt reconnect if previously disconnected
            if not ibkr_connected:
                ibkr_connected = _ibkr_connect_if_needed()
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\nStopping PPO AI bot gracefully. Saving positions and returning to menu...")
        try:
            _save_holdings(save_file, holdings)
            print(f"Saved holdings to {save_file}")
        except Exception as e2:
            logging.warning(f"Failed to save holdings on shutdown: {e2}")
        return
    except Exception as e:
        logging.error(f"PPO IBKR bot failed: {str(e)}", exc_info=True)
        raise

def do_ibkr_ai_bot_auto(watchlist: list[str], save_file: str, buy_pct: float, sell_pct: float, order_shares: float, poll_interval: float = 1.0):
    """IBKR auto-threshold bot.

    - Anchors thresholds per symbol to the first observed price
    - Buy when price drops by `buy_pct` below anchor
    - Sell when price rises by `sell_pct` above anchor
    - Resets anchor to current price after each trade to avoid repeated triggers
    """
    if not IBKR_ENABLED:
        print("IBKR is not enabled. Enable it in 'IBKR settings' first.")
    ok = _ibkr_connect_if_needed()
    if not ok:
        print("IBKR not connected. Will attempt reconnects; using limited data until then.")

    account = _json_load_account(save_file)
    anchors: dict[str, float] = {}

    # Prepare contracts and subscribe to market data
    contracts = []
    tickers = []
    for sym in [s.strip().upper() for s in watchlist if s.strip()]:
        if _IBStock is not None and ok:
            try:
                c = _IBStock(sym, 'SMART', 'USD')
                contracts.append(c)
                t = _ib.reqMktData(c, '', False, False)
                tickers.append(t)
            except Exception as e:
                print(f"Subscription error for {sym}: {e}")
                logging.error(f"IBKR subscription error for {sym}: {e}", exc_info=True)
                contracts.append(None)
                tickers.append(None)
        else:
            contracts.append(None)
            tickers.append(None)

    print("Watching (auto thresholds):", ", ".join([s for s in watchlist]))
    print(f"Auto thresholds: buy {buy_pct*100:.2f}% below anchor, sell {sell_pct*100:.2f}% above anchor")
    try:
        while True:
            for idx, sym in enumerate(watchlist):
                sym = sym.upper()
                t = tickers[idx]
                price = None
                if t is not None:
                    try:
                        price = t.last or t.close
                        if price is None and t.bid is not None and t.ask is not None:
                            price = (t.bid + t.ask) / 2.0
                    except Exception:
                        price = None
                if price is None:
                    price = _get_last_price(sym)
                if price is None:
                    continue

                # Initialize anchor on first valid price
                if sym not in anchors:
                    anchors[sym] = float(price)
                    print(f"Anchor set for {sym}: ${anchors[sym]:.2f}")

                anchor = anchors[sym]
                buy_thr = anchor * (1.0 - float(buy_pct))
                sell_thr = anchor * (1.0 + float(sell_pct))

                pos = account['positions'].get(sym, {'qty': 0.0, 'avg_price': 0.0})
                qty = float(pos.get('qty', 0.0))
                avg = float(pos.get('avg_price', 0.0))

                # Buy if drop below anchored threshold
                if price <= buy_thr and account['balance'] >= price * order_shares:
                    if IBKR_EXECUTE and ok and _IBMarketOrder is not None and contracts[idx] is not None:
                        try:
                            order = _IBMarketOrder('BUY', int(round(order_shares)))
                            trade = _ib.placeOrder(contracts[idx], order)
                            _ib.sleep(0.2)
                            status = getattr(trade.orderStatus, 'status', 'Unknown')
                            print(f"IBKR BUY {sym} {order_shares} status: {status}")
                        except Exception as e:
                            print(f"IBKR BUY error: {e}")
                            logging.error(f"IBKR BUY error for {sym}: {e}", exc_info=True)
                    cost = price * order_shares
                    new_qty = qty + order_shares
                    new_avg = ((qty * avg) + (order_shares * price)) / new_qty if new_qty > 0 else 0.0
                    account['balance'] -= cost
                    account['positions'][sym] = {'qty': new_qty, 'avg_price': new_avg}
                    anchors[sym] = float(price)  # reset anchor after trade
                    print(f"✅ Bought {order_shares} {sym} at ${price:.2f} | New anchor ${anchors[sym]:.2f}")
                # Sell if rise above anchored threshold
                elif price >= sell_thr and qty >= order_shares:
                    if IBKR_EXECUTE and ok and _IBMarketOrder is not None and contracts[idx] is not None:
                        try:
                            order = _IBMarketOrder('SELL', int(round(order_shares)))
                            trade = _ib.placeOrder(contracts[idx], order)
                            _ib.sleep(0.2)
                            status = getattr(trade.orderStatus, 'status', 'Unknown')
                            print(f"IBKR SELL {sym} {order_shares} status: {status}")
                        except Exception as e:
                            print(f"IBKR SELL error: {e}")
                            logging.error(f"IBKR SELL error for {sym}: {e}", exc_info=True)
                    proceeds = price * order_shares
                    account['balance'] += proceeds
                    new_qty = qty - order_shares
                    if new_qty <= 0:
                        account['positions'].pop(sym, None)
                    else:
                        account['positions'][sym] = {'qty': new_qty, 'avg_price': avg}
                    anchors[sym] = float(price)  # reset anchor after trade
                    realized = order_shares * (price - avg)
                    print(f"✅ Sold {order_shares} {sym} at ${price:.2f} | Realized ${realized:.2f} | New anchor ${anchors[sym]:.2f}")

            # Save account each loop
            _json_save_account(save_file, account)
            # Compute account value and print summary line
            total_value = 0.0
            for sym, pos in account['positions'].items():
                lp = _get_last_price(sym)
                if lp is not None:
                    total_value += lp * float(pos.get('qty', 0.0))
            total_balance = float(account['balance']) + total_value
            print(f"💵 Total Account Value: ${total_balance:,.2f} | Cash ${float(account['balance']):,.2f} | Positions {len(account['positions'])}")
            # Poll interval
            try:
                if ok:
                    _ib.sleep(max(0.1, float(poll_interval)))
                else:
                    time.sleep(max(0.1, float(poll_interval)))
            except Exception:
                time.sleep(max(0.1, float(poll_interval)))
            # Try reconnect occasionally if disconnected
            if not ok:
                ok = _ibkr_connect_if_needed()
    except KeyboardInterrupt:
        _json_save_account(save_file, account)
        print("Session saved & stopped.")


def run_cli_menu():
    global IBKR_ENABLED, IBKR_EXECUTE, IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID
    while True:
        print("\n=== Trading Bot Menu ===")
        print("1) Save market knowledge")
        print("2) Run customizable research session")
        print("3) Resimulate trading")
        print("4) Live dashboard (prices & P&L)")
        print("5) Paper trading account")
        print("6) Trade permissions")
        print("7) IBKR settings")
        print("8) IBKR AI bot")
        print("9) Exit")
        choice = input("Select an option [1-9]: ").strip()
        if choice == "1":
            tickers_in = _prompt_str("Tickers (comma-separated)", "SPY")
            tickers = _parse_tickers_arg(tickers_in)
            start = _prompt_str("Start date", "2010-01-01")
            end = _prompt_str("End date", "2019-12-31")
            window_size = _prompt_int("Window size", 30)
            fmt = _prompt_choice("Big file format", ["csv", "parquet"], "parquet")
            do_save_knowledge(tickers, start, end, window_size, fmt)
        elif choice == "2":
            tickers_in = _prompt_str("Tickers (comma-separated)", "SPY")
            tickers = _parse_tickers_arg(tickers_in)
            start = _prompt_str("Start date", "2010-01-01")
            end = _prompt_str("End date", "2019-12-31")
            window_size = _prompt_int("Window size", 30)
            minutes = _prompt_int("Session duration (minutes)", 30)
            interval = _prompt_int("Save interval (seconds)", 60)
            fmt = _prompt_choice("Big file format", ["csv", "parquet"], "parquet")
            do_wall_clock_research(tickers, start, end, window_size, minutes, interval, fmt)
        elif choice == "3":
            tickers_in = _prompt_str("Tickers (comma-separated)", "SPY")
            tickers = _parse_tickers_arg(tickers_in)
            do_resimulate(tickers)
        elif choice == "4":
            tickers_in = _prompt_str("Tickers (comma-separated)", "AAPL")
            tickers = _parse_tickers_arg(tickers_in)
            positions_in = _prompt_str("Positions (TICKER:QTY:AVGCOST, optional)", ",".join([f"{t}:0:0" for t in tickers]))
            poll = _prompt_int("Poll interval (seconds)", 1)
            compact_choice = _prompt_choice("Compact output?", ["yes", "no"], "yes")
            compact = compact_choice == "yes"
            print("Starting live dashboard. Press Ctrl+C to stop.")
            do_live_dashboard(tickers, poll_interval=float(poll), positions_arg=positions_in, compact=compact)
        elif choice == "5":
            # Paper trading sub-menu
            print("\n--- Paper Trading Account ---")
            # Only show allowed trade operations (BUY/SELL) based on permissions
            ops = ["view", "live", "init"]
            if TRADE_PERMISSIONS.get("BUY", False):
                ops.append("buy")
            if TRADE_PERMISSIONS.get("SELL", False):
                ops.append("sell")
            ops += ["set-balance", "ai"]
            op = _prompt_choice("Operation", ops, "view")
            if op == "init":
                bal = _prompt_int("Starting cash balance", 100000)
                _save_paper_account(pd.DataFrame(columns=["symbol", "shares", "avg_price"]), float(bal))
                print(f"Initialized paper account with balance {float(bal):.2f}.")
            elif op == "set-balance":
                bal = _prompt_int("New cash balance", 100000)
                print(_paper_set_balance(float(bal)))
            elif op == "buy":
                symbol = _prompt_str("Symbol", "AAPL").upper()
                shares = _prompt_int("Shares", 1)
                price_in = _prompt_str("Price (blank for live)", "")
                price = float(price_in) if price_in.strip() else None
                print(_paper_buy(symbol, float(shares), price))
            elif op == "sell":
                symbol = _prompt_str("Symbol", "AAPL").upper()
                shares = _prompt_int("Shares", 1)
                price_in = _prompt_str("Price (blank for live)", "")
                price = float(price_in) if price_in.strip() else None
                print(_paper_sell(symbol, float(shares), price))
            elif op == "live":
                poll = _prompt_int("Poll interval (seconds)", 1)
                print("Starting paper account live view. Press Ctrl+C to stop.")
                do_paper_live(poll_interval=float(poll))
            elif op == "ai":
                tickers_in = _prompt_str("Tickers (comma-separated)", "AAPL")
                tickers = _parse_tickers_arg(tickers_in)
                poll = _prompt_int("Poll interval (seconds)", 5)
                trade_shares = _prompt_int("Trade shares per action", 1)
                print("Starting AI control of paper account. Press Ctrl+C to stop.")
                do_paper_ai_control(tickers=tickers, poll_interval=float(poll), trade_shares=float(trade_shares))
            else:  # view
                print(_paper_view_once())
        elif choice == "6":
            # Trade permissions sub-menu
            print("\n--- Trade Permissions ---")
            print(f"Current permissions: {json.dumps(TRADE_PERMISSIONS)}")
            if IBKR_ENABLED:
                ok = _ibkr_connect_if_needed()
                summary = _ibkr_account_summary_dict() if ok else {}
                acct_type = summary.get('AccountType', 'Unknown')
                shorting_enabled = summary.get('ShortingEnabled', 'Unknown')
                print(f"IBKR connected: {ok} | AccountType: {acct_type} | ShortingEnabled: {shorting_enabled}")
            op = _prompt_choice("Operation", ["view", "update-from-ibkr", "toggle", "back"], "view")
            if op == "view":
                print("Allowed trade types:")
                for k in ["BUY", "SELL", "SELL_SHORT", "BUY_TO_COVER"]:
                    print(f"- {k}: {'allowed' if TRADE_PERMISSIONS.get(k, False) else 'denied'}")
            elif op == "update-from-ibkr":
                _ibkr_update_trade_permissions_from_account()
                print("Updated from IBKR account summary.")
                print(f"Current permissions: {json.dumps(TRADE_PERMISSIONS)}")
            elif op == "toggle":
                which = _prompt_choice("Trade type", ["BUY", "SELL", "SELL_SHORT", "BUY_TO_COVER"], "BUY")
                allowed = _prompt_choice("Allow?", ["yes", "no"], "yes") == "yes"
                TRADE_PERMISSIONS[which] = bool(allowed)
                print(f"Set {which} to {'allowed' if allowed else 'denied'}.")
            else:
                pass
        elif choice == "7":
            # IBKR settings sub-menu
            print("\n--- IBKR Settings ---")
            print(f"Current: enabled={IBKR_ENABLED}, exec={IBKR_EXECUTE}, host={IBKR_HOST}, port={IBKR_PORT}, clientId={IBKR_CLIENT_ID}")
            enabled_choice = _prompt_choice("Enable IBKR pricing?", ["yes", "no"], "yes" if IBKR_ENABLED else "no")
            IBKR_ENABLED = (enabled_choice == "yes")
            exec_choice = _prompt_choice("Enable IBKR execution?", ["yes", "no"], "no" if not IBKR_EXECUTE else "yes")
            IBKR_EXECUTE = (exec_choice == "yes")
            IBKR_HOST = _prompt_str("IBKR host", IBKR_HOST)
            IBKR_PORT = _prompt_int("IBKR port (7497 TWS, 4001 Gateway)", IBKR_PORT)
            IBKR_CLIENT_ID = _prompt_int("IBKR client ID", IBKR_CLIENT_ID)
            ok = _ibkr_connect_if_needed()
            print(f"IBKR pricing {'enabled' if ok else 'requested but not connected'}; execution {'on' if IBKR_EXECUTE else 'off'}.")
            print(f"✅ Connected to IBKR Paper Trading: {ok}")
        elif choice == "8":
            # IBKR AI bot sub-menu
            print("\n--- IBKR AI Bot ---")
            ai_choice = _prompt_choice("Select AI Bot Mode", ["auto-threshold", "absolute-threshold", "ppo-policy"], "auto-threshold")
            if ai_choice == "auto-threshold":
                watchlist_in = _prompt_str("Watchlist (comma-separated)", "SPY")
                watchlist = _parse_tickers_arg(watchlist_in)
                save_file = _prompt_str("Save file (e.g., ibkr_auto_account.json)", "ibkr_auto_account.json")
                buy_pct = _prompt_float("Buy threshold percentage (e.g., 0.01 for 1% drop)", 0.01)
                sell_pct = _prompt_float("Sell threshold percentage (e.g., 0.01 for 1% rise)", 0.01)
                order_shares = _prompt_int("Order shares", 1)
                poll_interval = _prompt_int("Poll interval (seconds)", 1)
                print("Starting IBKR auto-threshold AI bot. Press Ctrl+C to stop.")
                do_ibkr_ai_bot_auto(watchlist, save_file, buy_pct, sell_pct, order_shares, float(poll_interval))
            elif ai_choice == "absolute-threshold":
                watchlist_in = _prompt_str("Watchlist (comma-separated)", "SPY")
                watchlist = _parse_tickers_arg(watchlist_in)
                save_file = _prompt_str("Save file (e.g., ibkr_account.json)", "ibkr_account.json")
                buy_threshold = _prompt_float("Buy threshold (e.g., 100.0)", 100.0)
                sell_threshold = _prompt_float("Sell threshold (e.g., 101.0)", 101.0)
                order_shares = _prompt_int("Order shares", 1)
                poll_interval = _prompt_int("Poll interval (seconds)", 1)
                print("Starting IBKR AI bot. Press Ctrl+C to stop.")
                do_ibkr_ai_bot(watchlist, save_file, buy_threshold, sell_threshold, order_shares, float(poll_interval))
            elif ai_choice == "ppo-policy":
                watchlist_in = _prompt_str("Watchlist (comma-separated)", "SPY")
                watchlist = _parse_tickers_arg(watchlist_in)
                save_file = _prompt_str("Save file (e.g., ibkr_ppo_account.json)", "ibkr_ppo_account.json")
                max_order_shares = _prompt_int("Max order shares", 1)
                poll_interval = _prompt_int("Poll interval (seconds)", 60)
                print("Starting IBKR PPO AI bot. Press Ctrl+C to stop.")
                do_ibkr_ai_bot_ppo(watchlist, save_file, max_order_shares, poll_interval)
        elif choice == "9":
            print("Goodbye.")
            break
        else:
            print("Invalid choice. Please select 1-9.")


# --- Market Research & Knowledge ---
def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def _compute_indicators(price_series: pd.Series) -> pd.DataFrame:
    price = price_series.astype(float)
    returns = price.pct_change()
    sma_20 = price.rolling(20).mean()
    ema_50 = price.ewm(span=50, adjust=False).mean()
    ema_200 = price.ewm(span=200, adjust=False).mean()
    rsi_14 = _compute_rsi(price, period=14)
    ema_12 = price.ewm(span=12, adjust=False).mean()
    ema_26 = price.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    vol_30 = returns.rolling(30).std()

    df = pd.DataFrame({
        'price': pd.Series(price),
        'returns': returns,
        'sma_20': sma_20,
        'ema_50': ema_50,
        'ema_200': ema_200,
        'rsi_14': rsi_14,
        'macd': macd,
        'macd_signal': macd_signal,
        'vol_30': vol_30
    }, index=price_series.index)
    return df

def _classify_trend(ema_50: float, ema_200: float, rsi_14: float, macd: float, macd_signal: float) -> str:
    if pd.isna(ema_50) or pd.isna(ema_200) or pd.isna(rsi_14) or pd.isna(macd) or pd.isna(macd_signal):
        return 'unknown'
    if ema_50 > ema_200 and rsi_14 >= 55 and macd > macd_signal:
        return 'bullish'
    if ema_50 < ema_200 and rsi_14 <= 45 and macd < macd_signal:
        return 'bearish'
    return 'sideways'

def run_market_research_session(tickers=ALL_TICKERS, start="2010-01-01", end="2019-12-31", window_size=30, duration_days=None, knowledge_path="knowledge/market_knowledge.json"):
    # Determine analysis end based on duration
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    if duration_days is not None:
        end_dt = start_dt + pd.Timedelta(days=duration_days)
    end_str = end_dt.strftime('%Y-%m-%d')

    # Build environment to leverage existing data loading
    env = TradingEnv(tickers=tickers, start=start, end=end_str, window_size=window_size)
    df_prices = env.data  # DataFrame with columns per ticker, index as dates

    knowledge = {
        'metadata': {
            'tickers': list(env.tickers),
            'start': start,
            'end': end_str,
            'window_size': window_size
        },
        'features': {},
        'summary': {}
    }

    for ticker in env.tickers:
        series = df_prices[ticker].astype(float)
        indicators = _compute_indicators(series)

        # Volatility regime thresholds
        q = indicators['vol_30'].quantile([0.33, 0.66]).to_list()
        low_th, high_th = q[0], q[1]
        def _vol_state(v: float) -> str:
            if pd.isna(v):
                return 'unknown'
            if v <= low_th:
                return 'low'
            if v >= high_th:
                return 'high'
            return 'medium'

        indicators['trend_state'] = indicators.apply(lambda r: _classify_trend(r['ema_50'], r['ema_200'], r['rsi_14'], r['macd'], r['macd_signal']), axis=1)
        indicators['volatility_state'] = indicators['vol_30'].apply(_vol_state)

        # Convert to serializable list
        feats = []
        for idx, row in indicators.iterrows():
            feats.append({
                'date': str(idx),
                'price': float(row['price']) if not pd.isna(row['price']) else None,
                'returns': float(row['returns']) if not pd.isna(row['returns']) else None,
                'sma_20': float(row['sma_20']) if not pd.isna(row['sma_20']) else None,
                'ema_50': float(row['ema_50']) if not pd.isna(row['ema_50']) else None,
                'ema_200': float(row['ema_200']) if not pd.isna(row['ema_200']) else None,
                'rsi_14': float(row['rsi_14']) if not pd.isna(row['rsi_14']) else None,
                'macd': float(row['macd']) if not pd.isna(row['macd']) else None,
                'macd_signal': float(row['macd_signal']) if not pd.isna(row['macd_signal']) else None,
                'vol_30': float(row['vol_30']) if not pd.isna(row['vol_30']) else None,
                'trend_state': row['trend_state'],
                'volatility_state': row['volatility_state']
            })
        knowledge['features'][ticker] = feats

        summary = {
            'avg_return': float(indicators['returns'].mean(skipna=True)),
            'avg_vol_30': float(indicators['vol_30'].mean(skipna=True)),
            'bullish_days': int((indicators['trend_state'] == 'bullish').sum()),
            'bearish_days': int((indicators['trend_state'] == 'bearish').sum()),
            'sideways_days': int((indicators['trend_state'] == 'sideways').sum()),
            'volatility_thresholds': {'low': float(low_th) if not pd.isna(low_th) else None, 'high': float(high_th) if not pd.isna(high_th) else None}
        }
        knowledge['summary'][ticker] = summary

    # Save knowledge
    os.makedirs(os.path.dirname(knowledge_path) or '.', exist_ok=True)
    with open(knowledge_path, 'w') as f:
        json.dump(knowledge, f, indent=4)
    print(f"Market knowledge saved to {knowledge_path}")
    return knowledge

def load_market_knowledge(path: str):
    with open(path, 'r') as f:
        return json.load(f)

def clone_market_knowledge(src_path: str, dst_path: str):
    os.makedirs(os.path.dirname(dst_path) or '.', exist_ok=True)
    import shutil
    shutil.copyfile(src_path, dst_path)
    print(f"Cloned market knowledge to {dst_path}")

def _knowledge_to_dataframe(knowledge: dict) -> pd.DataFrame:
    rows = []
    features = knowledge.get('features', {})
    for ticker, feats in features.items():
        for item in feats:
            rows.append({
                'ticker': ticker,
                'date': item.get('date'),
                'price': item.get('price'),
                'returns': item.get('returns'),
                'sma_20': item.get('sma_20'),
                'ema_50': item.get('ema_50'),
                'ema_200': item.get('ema_200'),
                'rsi_14': item.get('rsi_14'),
                'macd': item.get('macd'),
                'macd_signal': item.get('macd_signal'),
                'vol_30': item.get('vol_30'),
                'trend_state': item.get('trend_state'),
                'volatility_state': item.get('volatility_state'),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['ticker', 'date'])
    return df

def save_knowledge_big(knowledge: dict, big_file_path: str, format: str = 'parquet') -> str:
    df = _knowledge_to_dataframe(knowledge)
    os.makedirs(os.path.dirname(big_file_path) or '.', exist_ok=True)
    if format == 'parquet':
        try:
            df.to_parquet(big_file_path, index=False)
        except Exception:
            # Fallback to CSV if parquet engine unavailable
            csv_path = big_file_path if big_file_path.endswith('.csv') else big_file_path.replace('.parquet', '.csv')
            df.to_csv(csv_path, index=False)
            big_file_path = csv_path
    elif format == 'csv':
        df.to_csv(big_file_path, index=False)
    else:
        with open(big_file_path, 'w') as f:
            json.dump(knowledge, f, indent=4)
    print(f"Big knowledge saved to {big_file_path}")
    return big_file_path

def run_market_research_wall_clock(tickers=ALL_TICKERS, start="2010-01-01", end="2019-12-31", window_size=30, minutes: int = 30, knowledge_path="knowledge/market_knowledge_wallclock.json", big_file_path: str | None = "knowledge/market_knowledge_big.parquet", save_every_secs: int = 60, format: str = 'parquet'):
    deadline = time.time() + minutes * 60
    aggregate_df = None
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        print(f"[Research Session] Iteration {iteration} running...")
        knowledge = run_market_research_session(tickers=tickers, start=start, end=end, window_size=window_size, duration_days=None, knowledge_path=knowledge_path)
        df = _knowledge_to_dataframe(knowledge)
        if aggregate_df is None:
            aggregate_df = df
        else:
            if not df.empty:
                aggregate_df = pd.concat([aggregate_df, df], ignore_index=True)
                aggregate_df.drop_duplicates(subset=['ticker', 'date'], inplace=True)
        if big_file_path:
            save_knowledge_big(knowledge, big_file_path, format=format)
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        time.sleep(min(save_every_secs, max(1, remaining)))
    print("[Research Session] Completed.")
    return aggregate_df if aggregate_df is not None else pd.DataFrame()


#############################################
# Paper trading account utilities (moved before __main__)
#############################################

# Files to persist paper account state
ACCOUNT_CSV = os.path.join(os.path.dirname(__file__), 'account.csv')
BALANCE_TXT = os.path.join(os.path.dirname(__file__), 'balance.txt')
TRADES_JSON = os.path.join(os.path.dirname(__file__), 'paper_account.json')

def _load_trade_log() -> dict:
    try:
        if os.path.exists(TRADES_JSON):
            with open(TRADES_JSON, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'trades' in data:
                return data
    except Exception:
        pass
    return {"trades": []}

def _append_trade_log(record: dict) -> None:
    try:
        data = _load_trade_log()
        data.setdefault('trades', []).append(record)
        with open(TRADES_JSON, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def validate_order(symbol: str, price: float, quantity: float, balance: float) -> bool:
    try:
        MIN_ORDER_VALUE = 1.00
        MAX_ORDER_VALUE = balance * 0.05
        order_value = float(price) * float(quantity)
        if order_value < MIN_ORDER_VALUE:
            print(f"⚠️ Skipping {symbol}: below minimum order value (${order_value:.2f})")
            return False
        if order_value > MAX_ORDER_VALUE:
            print(f"⚠️ Skipping {symbol}: exceeds allowed position size (${order_value:.2f})")
            return False
        if quantity <= 0:
            print(f"⚠️ Skipping {symbol}: non-positive quantity {quantity}")
            return False
        return True
    except Exception:
        return False

def _load_paper_account() -> tuple[pd.DataFrame, float]:
    """Load positions and cash balance from disk, with sensible defaults."""
    # Load positions CSV
    if os.path.exists(ACCOUNT_CSV):
        try:
            positions = pd.read_csv(ACCOUNT_CSV)
        except Exception:
            positions = pd.DataFrame(columns=["symbol", "shares", "avg_price"])
    else:
        positions = pd.DataFrame(columns=["symbol", "shares", "avg_price"])

    # Normalize columns to expected schema
    rename_map = {}
    if "buy_price" in positions.columns and "avg_price" not in positions.columns:
        rename_map["buy_price"] = "avg_price"
    if rename_map:
        positions = positions.rename(columns=rename_map)

    for col in ["symbol", "shares", "avg_price"]:
        if col not in positions.columns:
            positions[col] = [] if col == "symbol" else 0.0

    # Ensure dtypes are correct
    if not positions.empty:
        positions["symbol"] = positions["symbol"].astype(str)
        positions["shares"] = pd.to_numeric(positions["shares"], errors="coerce").fillna(0).astype(float)
        positions["avg_price"] = pd.to_numeric(positions["avg_price"], errors="coerce").fillna(0).astype(float)

    # Load balance
    if os.path.exists(BALANCE_TXT):
        try:
            with open(BALANCE_TXT, "r") as f:
                cash_balance = float(f.read().strip())
        except Exception:
            cash_balance = 100000.0
    else:
        cash_balance = 100000.0
    return positions, cash_balance

def _save_paper_account(positions: pd.DataFrame, cash_balance: float) -> None:
    """Persist positions and cash balance to disk."""
    try:
        # Use expected columns and order
        out = positions.copy()
        if "avg_price" in out.columns:
            out = out[["symbol", "shares", "avg_price"]]
        out.to_csv(ACCOUNT_CSV, index=False)
    except Exception:
        pass
    try:
        with open(BALANCE_TXT, "w") as f:
            f.write(str(float(cash_balance)))
    except Exception:
        pass

def _paper_buy(symbol: str, shares: float, price: Optional[float] = None) -> str:
    if not _can_execute_trade('BUY'):
        return "Permission denied for BUY."
    positions, cash_balance = _load_paper_account()
    # Resolve price if not provided
    if price is None:
        price = _get_last_price(symbol) or 0.0
    if price <= 0:
        return f"Price unavailable for {symbol}; buy aborted."
    cost = price * shares
    if cash_balance < cost:
        return f"Insufficient cash to buy {shares} {symbol} at {price:.2f}. Cash {cash_balance:.2f}."
    # IBKR-style order validation
    if not validate_order(symbol, price, shares, cash_balance):
        return f"Validation blocked buy {shares} {symbol} at {price:.2f}."

    # Optionally execute via IBKR paper account
    if IBKR_EXECUTE:
        if _ibkr_connect_if_needed() and _IBStock is not None and _IBMarketOrder is not None:
            try:
                contract = _IBStock(symbol, 'SMART', 'USD')
                order = _IBMarketOrder('BUY', int(round(shares)))
                trade = _ib.placeOrder(contract, order)
                _ib.sleep(2)
                status = getattr(trade.orderStatus, 'status', 'Unknown')
                if status not in ('Filled', 'Submitted'):
                    print(f"⚠️ Order failed or rejected: {status}")
                else:
                    print(f"✅ Order successful: {status}")
                _append_trade_log({
                    "symbol": symbol,
                    "action": "BUY",
                    "quantity": float(shares),
                    "price": float(price),
                    "status": status,
                    "timestamp": str(datetime.now())
                })
            except Exception as e:
                print(f"IBKR BUY error: {e}")

    # Optionally execute via IBKR paper account
    if IBKR_EXECUTE:
        if _ibkr_connect_if_needed() and _IBStock is not None and _IBMarketOrder is not None:
            try:
                contract = _IBStock(symbol, 'SMART', 'USD')
                order = _IBMarketOrder('BUY', int(round(shares)))
                trade = _ib.placeOrder(contract, order)
                _ib.sleep(2)
                status = getattr(trade.orderStatus, 'status', 'Unknown')
                if status not in ('Filled', 'Submitted'):
                    print(f"⚠️ Order failed or rejected: {status}")
                else:
                    print(f"✅ Order successful: {status}")
                _append_trade_log({
                    "symbol": symbol,
                    "action": "BUY",
                    "quantity": float(shares),
                    "price": float(price),
                    "status": status,
                    "timestamp": str(datetime.now())
                })
            except Exception as e:
                print(f"IBKR BUY error: {e}")

    if symbol in positions["symbol"].values:
        idx = positions.index[positions["symbol"] == symbol][0]
        old_shares = float(positions.at[idx, "shares"]) or 0.0
        old_avg = float(positions.at[idx, "avg_price"]) or 0.0
        new_shares = old_shares + shares
        new_avg = ((old_shares * old_avg) + (shares * price)) / new_shares if new_shares > 0 else 0.0
        positions.at[idx, "shares"] = new_shares
        positions.at[idx, "avg_price"] = new_avg
    else:
        positions = pd.concat([
            positions,
            pd.DataFrame([{"symbol": symbol, "shares": float(shares), "avg_price": float(price)}])
        ], ignore_index=True)

    cash_balance -= cost
    _save_paper_account(positions, cash_balance)
    return f"Bought {shares} {symbol} at {price:.2f}. Cash {cash_balance:.2f}."

def _paper_sell(symbol: str, shares: float, price: Optional[float] = None) -> str:
    if not _can_execute_trade('SELL'):
        return "Permission denied for SELL."
    positions, cash_balance = _load_paper_account()
    if symbol not in positions["symbol"].values:
        return f"No position in {symbol} to sell."
    idx = positions.index[positions["symbol"] == symbol][0]
    have = float(positions.at[idx, "shares"]) or 0.0
    avg = float(positions.at[idx, "avg_price"]) or 0.0
    if have < shares:
        return f"Not enough shares to sell {shares} {symbol}. Have {have}."
    if price is None:
        price = _get_last_price(symbol) or 0.0
    if price <= 0:
        return f"Price unavailable for {symbol}; sell aborted."
    # IBKR-style order validation (use proceeds limit similar to buy)
    if not validate_order(symbol, price, min(shares, have), cash_balance + have * price):
        return f"Validation blocked sell {shares} {symbol} at {price:.2f}."

    # Optionally execute via IBKR paper account
    if IBKR_EXECUTE:
        if _ibkr_connect_if_needed() and _IBStock is not None and _IBMarketOrder is not None:
            try:
                contract = _IBStock(symbol, 'SMART', 'USD')
                order = _IBMarketOrder('SELL', int(round(shares)))
                trade = _ib.placeOrder(contract, order)
                _ib.sleep(2)
                status = getattr(trade.orderStatus, 'status', 'Unknown')
                if status not in ('Filled', 'Submitted'):
                    print(f"⚠️ Order failed or rejected: {status}")
                else:
                    print(f"✅ Order successful: {status}")
                _append_trade_log({
                    "symbol": symbol,
                    "action": "SELL",
                    "quantity": float(shares),
                    "price": float(price),
                    "status": status,
                    "timestamp": str(datetime.now())
                })
            except Exception as e:
                print(f"IBKR SELL error: {e}")

    # Optionally execute via IBKR paper account
    if IBKR_EXECUTE:
        if _ibkr_connect_if_needed() and _IBStock is not None and _IBMarketOrder is not None:
            try:
                contract = _IBStock(symbol, 'SMART', 'USD')
                order = _IBMarketOrder('SELL', int(round(shares)))
                trade = _ib.placeOrder(contract, order)
                _ib.sleep(2)
                status = getattr(trade.orderStatus, 'status', 'Unknown')
                if status not in ('Filled', 'Submitted'):
                    print(f"⚠️ Order failed or rejected: {status}")
                else:
                    print(f"✅ Order successful: {status}")
                _append_trade_log({
                    "symbol": symbol,
                    "action": "SELL",
                    "quantity": float(shares),
                    "price": float(price),
                    "status": status,
                    "timestamp": str(datetime.now())
                })
            except Exception as e:
                print(f"IBKR SELL error: {e}")

    proceeds = price * shares
    cash_balance += proceeds
    positions.at[idx, "shares"] = have - shares
    # Remove position if zeroed
    if float(positions.at[idx, "shares"]) <= 0:
        positions = positions.drop(index=idx).reset_index(drop=True)
    _save_paper_account(positions, cash_balance)
    realized = shares * (price - avg)
    return f"Sold {shares} {symbol} at {price:.2f} (avg {avg:.2f}). Realized P&L {realized:.2f}. Cash {cash_balance:.2f}."

def _paper_set_balance(amount: float) -> str:
    positions, _ = _load_paper_account()
    _save_paper_account(positions, amount)
    return f"Set cash balance to {amount:.2f}."

def _paper_view_once() -> str:
    positions, cash_balance = _load_paper_account()
    total_value = cash_balance
    lines = []
    if positions.empty:
        lines.append("No positions.")
    else:
        for _, row in positions.iterrows():
            symbol = str(row['symbol'])
            sh = float(row['shares'])
            avg = float(row['avg_price'])
            price = _get_last_price(symbol)
            if price is None or price <= 0:
                lines.append(f"{symbol} - Shares: {sh:.2f}, Live Price: N/A, Avg: {avg:.2f}")
                continue
            pos_val = sh * price
            unreal = sh * (price - avg)
            status = "Profit" if unreal > 0 else ("Loss" if unreal < 0 else "Break-even")
            total_value += pos_val
            lines.append(f"{symbol} - Shares: {sh:.2f}, Live Price: {price:.2f}, Position Value: {pos_val:.2f}, Unrealized P&L: {unreal:.2f} ({status})")
    lines.append(f"Cash: {cash_balance:.2f} | Total Account Value: {total_value:.2f}")
    # Persist after recompute
    _save_paper_account(positions, cash_balance)
    return "\n".join(lines)

def do_paper_live(poll_interval: float = 1.0) -> None:
    """Continuously show paper account view with live prices until Ctrl+C."""
    try:
        while True:
            out = _paper_view_once()
            print(out)
            print("")
            time.sleep(max(0.2, float(poll_interval)))
    except KeyboardInterrupt:
        print("Stopped paper account live view.")

def do_paper_ai_control(tickers: list[str], poll_interval: float = 5.0, trade_shares: float = 1.0) -> None:
    """Let the trained PPO model take control of the paper account.

    - Loads PPO model for the provided tickers
    - Runs the model in a loop to decide actions
    - Maps model actions to paper buy/sell using live prices
    - Ignores short/cover actions (paper account models long-only)
    """
    if not tickers:
        tickers = ["AAPL"]
    env = make_vec_env(lambda: TradingEnv(tickers=tickers, start="2010-01-01", end="2019-12-31", window_size=30), n_envs=1)
    base = _safe_model_basename(tickers)
    zip_path = f"{base}.zip"
    if not os.path.exists(zip_path):
        print(f"No model found at {zip_path}. Training a new model for {tickers}...")
        start_training(total_timesteps=50000, tickers=tickers)
    try:
        model = PPO.load(base, env=env, verbose=0)
    except Exception as e:
        print(f"Error loading model from {zip_path}: {e}. Training a fresh model and retrying.")
        logging.error(f"Paper AI control: model load failed at {zip_path}: {e}", exc_info=True)
        start_training(total_timesteps=50000, tickers=tickers)
        model = PPO.load(base, env=env, verbose=0)

    # Reset env and start control loop
    try:
        obs = env.reset()
    except Exception:
        # Gymnasium vs SB3 reset API differences
        obs, _ = env.reset()
    # Normalize obs if reset returned a tuple
    if isinstance(obs, tuple):
        obs = obs[0]

    print("Starting AI control of paper account. Press Ctrl+C to stop.")
    try:
        MAX_TRADES_PER_MINUTE = 10
        trade_counter = 0
        window_start = time.time()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            # For VecEnv, action may be array; extract scalar
            if isinstance(action, (list, tuple, np.ndarray)):
                act = int(action[0])
            else:
                act = int(action)
            ticker_idx = act // 5
            trade_type = act % 5
            ticker = tickers[ticker_idx]

            # Reset limiter window after 60 seconds
            now = time.time()
            if now - window_start >= 60:
                window_start = now
                trade_counter = 0

            if trade_type == 2:  # Buy Long
                price = BROKER.get_price(ticker) if BROKER is not None else _get_last_price(ticker)
                positions, cash_balance = _load_paper_account()
                if validate_order(ticker, price or 0.0, float(trade_shares), cash_balance):
                    if trade_counter >= MAX_TRADES_PER_MINUTE:
                        print("🚨 Too many trades too quickly — halting AI")
                        break
                    trade_counter += 1
                    if BROKER is not None:
                        resp = safe_buy(BROKER, ticker.upper(), float(trade_shares), price)
                        if isinstance(resp, dict):
                            print(resp.get("message", f"BUY {ticker} {trade_shares} at {price}"))
                    else:
                        print(_paper_buy(ticker.upper(), float(trade_shares), price))
                else:
                    print(f"Skipped BUY {ticker} due to validation.")
            elif trade_type == 0:  # Sell Long
                positions, _ = _load_paper_account()
                have = 0.0
                if not positions.empty and ticker in positions["symbol"].astype(str).values:
                    idxs = positions.index[positions["symbol"].astype(str) == ticker]
                    if len(idxs) > 0:
                        have = float(positions.at[idxs[0], "shares"]) or 0.0
                if have > 0:
                    sell_qty = min(have, float(trade_shares))
                    price = BROKER.get_price(ticker) if BROKER is not None else _get_last_price(ticker)
                    if trade_counter >= MAX_TRADES_PER_MINUTE:
                        print("🚨 Too many trades too quickly — halting AI")
                        break
                    trade_counter += 1
                    if BROKER is not None:
                        resp = safe_sell(BROKER, ticker.upper(), float(sell_qty), price)
                        if isinstance(resp, dict):
                            print(resp.get("message", f"SELL {ticker} {sell_qty} at {price}"))
                    else:
                        print(_paper_sell(ticker.upper(), float(sell_qty), price))
                else:
                    print(f"AI signaled sell, but no long position in {ticker}.")
            else:
                # Hold, short sell, cover short -> no-op for paper long-only account
                action_name = {0: "sell_long", 1: "hold", 2: "buy_long", 3: "cover_short", 4: "short_sell"}.get(trade_type, "unknown")
                print(f"AI action: {action_name} on {ticker}; no paper trade executed.")

            # Step environment to advance policy state (VecEnv returns 4 values)
            try:
                obs, _, dones, _ = env.step(np.array([act]))
            except Exception:
                # Fallback for non-vec envs
                step_out = env.step(act)
                if isinstance(step_out, tuple) and len(step_out) >= 4:
                    obs, _, done_flag, _ = step_out[:4]
                    dones = np.array([bool(done_flag)])
                else:
                    obs = step_out
                    dones = np.array([False])
            # Normalize obs if step returned a tuple
            if isinstance(obs, tuple):
                obs = obs[0]
            if np.any(dones):
                try:
                    obs = env.reset()
                except Exception:
                    obs, _ = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
            time.sleep(max(0.2, float(poll_interval)))
    except KeyboardInterrupt:
        print("Stopped AI control.")



if __name__ == "__main__":
    args = parse_cli_args()
    # Choose broker according to --broker flag or interactive prompt
    BROKER = select_broker(args)
    if getattr(args, "menu", False):
        run_cli_menu()
    if getattr(args, "action", None):
        act_tickers = _parse_tickers_arg(getattr(args, "tickers", None))
        if args.action == "save_knowledge":
            do_save_knowledge(act_tickers, args.start, args.end, args.window_size, args.format)
        elif args.action == "research":
            do_wall_clock_research(act_tickers, args.start, args.end, args.window_size, args.minutes, args.save_every_secs, args.format)
        elif args.action == "resimulate":
            do_resimulate(act_tickers)
        elif args.action == "live":
            do_live_dashboard(
                act_tickers,
                poll_interval=getattr(args, "poll_interval", 1.0),
                positions_arg=getattr(args, "positions", None),
                compact=getattr(args, "compact", False)
            )
        elif args.action == "paper":
            op = getattr(args, "paper_op", None)
            if op == "init":
                bal = getattr(args, "balance", 100000.0) or 100000.0
                _save_paper_account(pd.DataFrame(columns=["symbol", "shares", "avg_price"]), float(bal))
                print(f"Initialized paper account with balance {float(bal):.2f}.")
            elif op == "set-balance":
                bal = getattr(args, "balance", None)
                if bal is None:
                    print("--balance required for set-balance")
                else:
                    print(_paper_set_balance(float(bal)))
            elif op == "buy":
                sym = getattr(args, "symbol", None)
                sh = float(getattr(args, "shares", 0.0))
                pr = getattr(args, "price", None)
                if not sym or sh <= 0:
                    print("--symbol and --shares>0 required for buy")
                else:
                    print(_paper_buy(sym.upper(), sh, pr))
            elif op == "sell":
                sym = getattr(args, "symbol", None)
                sh = float(getattr(args, "shares", 0.0))
                pr = getattr(args, "price", None)
                if not sym or sh <= 0:
                    print("--symbol and --shares>0 required for sell")
                else:
                    print(_paper_sell(sym.upper(), sh, pr))
            elif op == "live":
                poll = float(getattr(args, "poll_interval", 1.0))
                print("Starting paper account live view. Press Ctrl+C to stop.")
                do_paper_live(poll_interval=poll)
            elif op == "ai":
                poll = float(getattr(args, "poll_interval", 5.0))
                trade_shares = float(getattr(args, "trade_shares", 1.0))
                do_paper_ai_control(tickers=act_tickers or ["AAPL"], poll_interval=poll, trade_shares=trade_shares)
            else:  # view or unspecified
                print(_paper_view_once())
        # Early exit after executing a non-interactive action to avoid running training/simulation defaults
        sys.exit(0)
    # Create directories if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("simulation_results", exist_ok=True)

    # Define tickers for training and simulation
    my_tickers = ["SPY"]

    # Create env_train here so it's accessible for model loading
    # The `my_tickers` variable is already correctly passed to TradingEnv here.
    env_train = make_vec_env(lambda: TradingEnv(tickers=my_tickers, start="2010-01-01", end="2019-12-31", window_size=30), n_envs=1)
    print(f"Using {len(env_train.get_attr('tickers')[0])} tickers for training: {env_train.get_attr('tickers')[0]}")

    # Check if the user wants to reset the bot's learning (hardcoded to True for debugging)
    reset_bot_learning = False

    # Optional: run a market research session to build transferable knowledge
    RUN_RESEARCH_SESSION = False
    if RUN_RESEARCH_SESSION:
        knowledge_dir = "knowledge"
        os.makedirs(knowledge_dir, exist_ok=True)
        knowledge_path = os.path.join(knowledge_dir, f"market_knowledge_{'_'.join(my_tickers)}_2010_2019.json")
        run_market_research_session(tickers=my_tickers, start="2010-01-01", end="2019-12-31", window_size=30, duration_days=None, knowledge_path=knowledge_path)

    # Option to save one big knowledge file (Parquet/CSV) for next bot
    SAVE_BIG_KNOWLEDGE_FILE = False
    if SAVE_BIG_KNOWLEDGE_FILE:
        knowledge_path = os.path.join("knowledge", f"market_knowledge_{'_'.join(my_tickers)}_2010_2019.json")
        knowledge = load_market_knowledge(knowledge_path)
        big_file_path = os.path.join("knowledge", f"market_knowledge_big_{'_'.join(my_tickers)}_2010_2019.parquet")
        save_knowledge_big(knowledge, big_file_path, format='parquet')

    # Option to run a longer wall-clock research session (e.g., 30 minutes)
    RUN_LONG_RESEARCH_MINUTES = False  # set to e.g., 30 for a 30-minute session
    if RUN_LONG_RESEARCH_MINUTES and RUN_LONG_RESEARCH_MINUTES > 0:
        long_big_path = os.path.join("knowledge", f"market_knowledge_big_{'_'.join(my_tickers)}_wallclock.parquet")
        run_market_research_wall_clock(
            tickers=my_tickers,
            start="2010-01-01",
            end="2019-12-31",
            window_size=30,
            minutes=RUN_LONG_RESEARCH_MINUTES,
            knowledge_path=os.path.join("knowledge", f"market_knowledge_{'_'.join(my_tickers)}_wallclock.json"),
            big_file_path=long_big_path,
            save_every_secs=60,
            format='parquet'
        )

    if reset_bot_learning:
        reset_learning()
        # Start training with multiple tickers
        start_training(total_timesteps=50000, tickers=my_tickers)

    # Load the trained model
    base = _safe_model_basename(my_tickers)
    model_path_with_tickers = base
    zip_path = f"{base}.zip"
    if not os.path.exists(zip_path):
        print(f"No model found at {zip_path}. Training a new model for {my_tickers}...")
        start_training(total_timesteps=50000, tickers=my_tickers)
    try:
        model = PPO.load(base, env=env_train, verbose=1)
    except Exception as e:
        print(f"Error loading model from {zip_path}: {e}. Training a fresh model and retrying.")
        logging.error(f"Bottom block: model load failed at {zip_path}: {e}", exc_info=True)
        # If model loading fails, train a new one
        start_training(total_timesteps=50000, tickers=my_tickers)
        model = PPO.load(base, env=env_train, verbose=1)

    # Simulate trading with the trained model and multiple tickers
    trade_results, pl_per_year, detailed_profits = simulate_trading(model, tickers=my_tickers, log_dir=logdir, reset_bot=False)

    # Save simulation results
    save_simulation_data(trade_results, pl_per_year, detailed_profits, "simulation_results")
    print("Simulation data saved.")
#############################################
# Paper trading account utilities
#############################################

# Files to persist paper account state
ACCOUNT_CSV = os.path.join(os.path.dirname(__file__), 'account.csv')
BALANCE_TXT = os.path.join(os.path.dirname(__file__), 'balance.txt')

def _load_paper_account() -> tuple[pd.DataFrame, float]:
    """Load positions and cash balance from disk, with sensible defaults."""
    # Load positions CSV
    if os.path.exists(ACCOUNT_CSV):
        try:
            positions = pd.read_csv(ACCOUNT_CSV)
        except Exception:
            positions = pd.DataFrame(columns=["symbol", "shares", "avg_price"])
    else:
        positions = pd.DataFrame(columns=["symbol", "shares", "avg_price"])

    # Normalize columns to expected schema
    rename_map = {}
    if "buy_price" in positions.columns and "avg_price" not in positions.columns:
        rename_map["buy_price"] = "avg_price"
    if rename_map:
        positions = positions.rename(columns=rename_map)

    for col in ["symbol", "shares", "avg_price"]:
        if col not in positions.columns:
            positions[col] = [] if col == "symbol" else 0.0

    # Ensure dtypes are correct
    if not positions.empty:
        positions["symbol"] = positions["symbol"].astype(str)
        positions["shares"] = pd.to_numeric(positions["shares"], errors="coerce").fillna(0).astype(float)
        positions["avg_price"] = pd.to_numeric(positions["avg_price"], errors="coerce").fillna(0).astype(float)

    # Load balance
    if os.path.exists(BALANCE_TXT):
        try:
            with open(BALANCE_TXT, "r") as f:
                cash_balance = float(f.read().strip())
        except Exception:
            cash_balance = 100000.0
    else:
        cash_balance = 100000.0
    return positions, cash_balance

def _save_paper_account(positions: pd.DataFrame, cash_balance: float) -> None:
    """Persist positions and cash balance to disk."""
    try:
        # Use expected columns and order
        out = positions.copy()
        if "avg_price" in out.columns:
            out = out[["symbol", "shares", "avg_price"]]
        out.to_csv(ACCOUNT_CSV, index=False)
    except Exception:
        pass
    try:
        with open(BALANCE_TXT, "w") as f:
            f.write(str(float(cash_balance)))
    except Exception:
        pass

def _paper_buy(symbol: str, shares: float, price: Optional[float] = None) -> str:
    if not _can_execute_trade('BUY'):
        return "Permission denied for BUY; paper trade not executed."
    positions, cash_balance = _load_paper_account()
    # Resolve price if not provided
    if price is None:
        price = _get_last_price(symbol) or 0.0
    if price <= 0:
        return f"Price unavailable for {symbol}; buy aborted."
    cost = price * shares
    if cash_balance < cost:
        return f"Insufficient cash to buy {shares} {symbol} at {price:.2f}. Cash {cash_balance:.2f}."

    if symbol in positions["symbol"].values:
        idx = positions.index[positions["symbol"] == symbol][0]
        old_shares = float(positions.at[idx, "shares"]) or 0.0
        old_avg = float(positions.at[idx, "avg_price"]) or 0.0
        new_shares = old_shares + shares
        new_avg = ((old_shares * old_avg) + (shares * price)) / new_shares if new_shares > 0 else 0.0
        positions.at[idx, "shares"] = new_shares
        positions.at[idx, "avg_price"] = new_avg
    else:
        positions = pd.concat([
            positions,
            pd.DataFrame([{"symbol": symbol, "shares": float(shares), "avg_price": float(price)}])
        ], ignore_index=True)

    cash_balance -= cost
    _save_paper_account(positions, cash_balance)
    return f"Bought {shares} {symbol} at {price:.2f}. Cash {cash_balance:.2f}."

def _paper_sell(symbol: str, shares: float, price: Optional[float] = None) -> str:
    if not _can_execute_trade('SELL'):
        return "Permission denied for SELL; paper trade not executed."
    positions, cash_balance = _load_paper_account()
    if symbol not in positions["symbol"].values:
        return f"No position in {symbol} to sell."
    idx = positions.index[positions["symbol"] == symbol][0]
    have = float(positions.at[idx, "shares"]) or 0.0
    avg = float(positions.at[idx, "avg_price"]) or 0.0
    if have < shares:
        return f"Not enough shares to sell {shares} {symbol}. Have {have}."
    if price is None:
        price = _get_last_price(symbol) or 0.0
    if price <= 0:
        return f"Price unavailable for {symbol}; sell aborted."

    proceeds = price * shares
    cash_balance += proceeds
    positions.at[idx, "shares"] = have - shares
    # Remove position if zeroed
    if float(positions.at[idx, "shares"]) <= 0:
        positions = positions.drop(index=idx).reset_index(drop=True)
    _save_paper_account(positions, cash_balance)
    realized = shares * (price - avg)
    return f"Sold {shares} {symbol} at {price:.2f} (avg {avg:.2f}). Realized P&L {realized:.2f}. Cash {cash_balance:.2f}."

def _paper_set_balance(amount: float) -> str:
    positions, _ = _load_paper_account()
    _save_paper_account(positions, amount)
    return f"Set cash balance to {amount:.2f}."

def _paper_view_once() -> str:
    positions, cash_balance = _load_paper_account()
    total_value = cash_balance
    lines = []
    if positions.empty:
        lines.append("No positions.")
    else:
        for _, row in positions.iterrows():
            symbol = str(row['symbol'])
            sh = float(row['shares'])
            avg = float(row['avg_price'])
            price = _get_last_price(symbol)
            if price is None or price <= 0:
                lines.append(f"{symbol} - Shares: {sh:.2f}, Live Price: N/A, Avg: {avg:.2f}")
                continue
            pos_val = sh * price
            unreal = sh * (price - avg)
            status = "Profit" if unreal > 0 else ("Loss" if unreal < 0 else "Break-even")
            total_value += pos_val
            lines.append(f"{symbol} - Shares: {sh:.2f}, Live Price: {price:.2f}, Position Value: {pos_val:.2f}, Unrealized P&L: {unreal:.2f} ({status})")
    lines.append(f"Cash: {cash_balance:.2f} | Total Account Value: {total_value:.2f}")
    # Persist after recompute
    _save_paper_account(positions, cash_balance)
    return "\n".join(lines)

def do_paper_live(poll_interval: float = 1.0) -> None:
    """Continuously show paper account view with live prices until Ctrl+C."""
    try:
        while True:
            out = _paper_view_once()
            print(out)
            print("")
            time.sleep(max(0.2, float(poll_interval)))
    except KeyboardInterrupt:
        print("Stopped paper account live view.")