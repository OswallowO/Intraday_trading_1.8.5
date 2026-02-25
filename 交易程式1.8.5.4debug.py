# ==============================================================================
# 交易程式 1.8.5.4 - 當沖量化終端 (PyQt5 旗艦專業版)
# ==============================================================================
import json
import os
import math
import subprocess
import sys
import time as time_module
import warnings
import traceback
import shioaji_logic
import importlib
import csv
import threading
import re
import builtins
from datetime import datetime, time, timedelta, date
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 確保 PyQt5 等套件已安裝 ---
REQUIRED = [
    ("fugle_marketdata", "fugle-marketdata"),
    ("pandas",           "pandas"),
    ("yaml",             "pyyaml"),
    ("numpy",            "numpy"),
    ("colorama",         "colorama"),
    ("tabulate",         "tabulate"),
    ("openpyxl",         "openpyxl"),
    ("dateutil",         "python-dateutil"),
    ("matplotlib",       "matplotlib"),
    ("PyQt5",            "PyQt5"),
    ("scipy",            "scipy"),
    ("fastdtw",          "fastdtw")
]

def ensure_packages(pkgs):
    missing = []
    for mod, pkg in pkgs:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("首次執行偵測到以下套件尚未安裝：", ", ".join(missing))
        for pkg in missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        for mod, pkg in pkgs:
            globals()[mod] = importlib.import_module(mod)

ensure_packages(REQUIRED)

import pandas as pd
import yaml
import numpy as np
import colorama
import shioaji as sj
import touchprice as tp
import requests, bs4
import orjson
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from fugle_marketdata import RestClient
from colorama import init, Fore, Style

# --- PyQt5 Imports ---
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QTextEdit, 
                             QInputDialog, QMessageBox, QDialog, QLineEdit, 
                             QComboBox, QFormLayout, QRadioButton, QScrollArea, 
                             QFrame, QButtonGroup, QDialogButtonBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, pyqtSlot, QTimer
from PyQt5.QtGui import QFont, QColor, QTextCursor, QPalette

plt.rcParams['axes.unicode_minus'] = False
colorama.init(autoreset=True)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="urllib3.connection")

# -------------------- 全域變數與鎖 --------------------
data_lock = threading.Lock()
in_memory_intraday_data = {}

RED = Fore.RED; GREEN = Fore.GREEN; YELLOW = Fore.YELLOW; BLUE = Fore.BLUE; RESET = Style.RESET_ALL
pd.set_option('future.no_silent_downcasting', True)

capital_per_stock = 0
transaction_fee = 0
transaction_discount = 0
trading_tax = 0
below_50 = 0
price_gap_50_to_100 = 0
price_gap_100_to_500 = 0
price_gap_500_to_1000 = 0
price_gap_above_1000 = 0
allow_reentry_after_stop_loss = False

previous_stop_loss_codes = set()
open_positions: dict[str, dict] = {} 
triggered_limit_up_stocks: set[str] = set()

from PyQt5.QtCore import pyqtSignal

# ==============================================================================
# 🟢 全新相似度引擎：基於 DTW 絕對距離的 0~1 評分演算法
# ==============================================================================
def calculate_dtw_pearson(df_lead, df_follow, window_start, window_end):
    import numpy as np
    try:
        from fastdtw import fastdtw
    except ImportError: 
        print("⚠️ 缺少 fastdtw 套件，請確認已安裝！")
        return 0
    
    sub_lead = df_lead[(df_lead['time'] >= window_start) & (df_lead['time'] <= window_end)]
    sub_fol = df_follow[(df_follow['time'] >= window_start) & (df_follow['time'] <= window_end)]
    
    if len(sub_lead) < 2 or len(sub_fol) < 2: 
        return 0
        
    s1 = sub_lead['rise'].values
    s2 = sub_fol['rise'].values
    
    s1_std = np.std(s1)
    s2_std = np.std(s2)
    
    if s1_std < 1e-5 or s2_std < 1e-5:
        return 0
        
    s1_norm = (s1 - np.mean(s1)) / s1_std
    s2_norm = (s2 - np.mean(s2)) / s2_std
    
    try:
        dist, path = fastdtw(s1_norm, s2_norm)
        
        # 🟢 數學轉換：計算平均每個對齊點的「距離誤差」
        avg_dist = dist / len(path)
        
        # 🟢 分數映射：將距離轉換為 0 ~ 1 的絕對相似度
        # 誤差為0 -> 相似度1 / 誤差超過1 -> 相似度0
        similarity = max(0, 1 - avg_dist)
        
        return similarity
        
    except Exception as e:
        print(f"⚠️ DTW 計算發生異常: {e}")
        return 0

# =========================================================
# 補回 1.8.0.8 遺漏的輔助與相容性函數 (修復 NameError)
# =========================================================
# 1. 相容舊版盤中邏輯的退出變數與空函數 (PyQt5 已改用實體按鈕)
quit_flag = {"quit": False}

def check_quit_flag_loop():
    pass  # PyQt5 已改用實體緊急按鈕，不再需要背景掃描迴圈
    
def show_exit_menu():
    print("💡 提示：在 PyQt5 介面中，請直接點選左側面板的【🛑 緊急/手動平倉】按鈕")

# 2. 數據儲存與轉換輔助函數
def load_nb_matrix_dict():
    if os.path.exists('nb_matrix_dict.json'):
        with open('nb_matrix_dict.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_nb_matrix_dict(nb_matrix_dict):
    with open('nb_matrix_dict.json', 'w', encoding='utf-8') as f:
        json.dump(nb_matrix_dict, f, indent=4, ensure_ascii=False, default=str)

def consolidate_and_save_stock_symbols():
    matrix_dict_analysis = load_matrix_dict_analysis()
    if not matrix_dict_analysis:
        print("matrix_dict_analysis.json 檔案不存在或為空，無法進行統整")
        return
    nb_matrix_dict = {"consolidated_symbols": matrix_dict_analysis}
    save_nb_matrix_dict(nb_matrix_dict)

def convert_datetime_to_str(obj):
    if isinstance(obj, dict):
        return {k: convert_datetime_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_datetime_to_str(element) for element in obj]
    elif isinstance(obj, (datetime, pd.Timestamp, time)):
        return obj.isoformat()
    return obj
# =========================================================

# ==================== PyQt5 終端機重導向 (Signals & Slots) ====================
ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

class EmittingStream(QObject):
    textWritten = pyqtSignal(str)
    def write(self, text):
        self.textWritten.emit(text)
    def flush(self):
        pass

# 用於將盤中數據傳遞給 UI 表格的訊號發射器
class SignalDispatcher(QObject):
    portfolio_updated = pyqtSignal(list)
    progress_updated = pyqtSignal(int, str)  # 🟢 新增：負責傳遞進度 % 數與文字
    progress_visible = pyqtSignal(bool)      # 🟢 新增：負責控制進度條顯示/隱藏

ui_dispatcher = SignalDispatcher()

# 用來緩存最新一分鐘的持倉與損益資料，讓 UI 隨開即看
cached_portfolio_data = []

# ==================== 基礎資料與爬蟲函數 ====================
def _crawl_tw_isin_table(mode: str):
    url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    r.encoding = "big5"
    soup = bs4.BeautifulSoup(r.text, "lxml")
    rows = soup.select("table tr")[1:]
    pairs = []
    for tr in rows:
        tds = tr.find_all("td")
        if not tds: continue
        raw = tds[0].text.strip()
        if raw[:4].isdigit():
            code = raw[:4]
            name = raw.split("\u3000", 1)[1] if "\u3000" in raw else raw[4:]
            pairs.append((code, name))
    return pairs

STOCK_NAME_MAP = {}
def load_twse_name_map(json_path="twse_stocks_by_market.json"):
    global STOCK_NAME_MAP
    if STOCK_NAME_MAP: return
    try:
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                STOCK_NAME_MAP = json.load(f)
            return
        tse_map = {c: n for c, n in _crawl_tw_isin_table("2")}
        otc_map = {c: n for c, n in _crawl_tw_isin_table("4")}
        STOCK_NAME_MAP = {"TSE": tse_map, "OTC": otc_map}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(STOCK_NAME_MAP, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"載入股票中文名稱失敗：{e}")
        STOCK_NAME_MAP = {}

def get_stock_name(code):
    for market in ["TSE", "OTC"]:
        if code in STOCK_NAME_MAP.get(market, {}): return STOCK_NAME_MAP[market][code]
    return ""

def init_fugle_client():
    try:
        config = load_config("config.yaml")
        client = RestClient(api_key=config['api_key'])
        return client, config['api_key']
    except Exception as e:
        print(f"初始化富果API客戶端時發生錯誤：{e}")
        sys.exit(1)

def load_config(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# ------------------ 工具與防護函數 ------------------
def view_kline_data(json_path, symbol_to_group):
    """
    查看盤中K線數據，依族群分類並繪製標準化close走勢圖
    - 使用Z-score標準化
    - 自動處理中文顯示
    - 指定時間格式避免警告
    """
    # ✅ 修正 1：關閉所有之前開過的圖表，釋放記憶體
    plt.close('all')

    # ✅ 修正 2：強制設定 Matplotlib 的字型為微軟正黑體，解決方塊字問題
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解決負號變成方塊的問題
    plt.rcParams['figure.max_open_warning'] = 0

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"找不到檔案：{json_path}")
        
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    stock_data = {}
    for symbol, records in raw_data.items():
        df = pd.DataFrame(records)
        if 'time' in df.columns and 'close' in df.columns and 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format="%Y-%m-%d %H:%M:%S")
            df = df.sort_values(by='datetime')
            stock_data[symbol] = df
        else:
            print(f"股票 {symbol} 缺少必要欄位，略過。")
    
    group_to_stocks = {}
    for symbol, group in symbol_to_group.items():
        if symbol in stock_data:
            group_to_stocks.setdefault(group, []).append(symbol)
    
    for group, symbols in group_to_stocks.items():
        fig, ax = plt.subplots(figsize=(12, 6))
        for symbol in symbols:
            df = stock_data[symbol]
            close = df['close']
            close_z = (close - close.mean()) / close.std() if close.std() != 0 else close - close.mean()
            ax.plot(df['datetime'], close_z, label=symbol)
        
        ax.set_title(f"{group} 族群標準化收盤價走勢")
        ax.set_xlabel("時間")
        ax.set_ylabel("標準化收盤價 (Z-score)")
        ax.legend()
        ax.grid(True)

    plt.show()

def safe_fugle_api_call(api_func, max_retries=3, **kwargs):
    for attempt in range(max_retries + 1):
        try: return api_func(**kwargs)
        except Exception as e:
            error_str = str(e)
            if any(x in error_str for x in ["429", "Too Many Requests", "Rate Limit"]):
                if attempt < max_retries:
                    sleep_time = 2 ** attempt
                    print(f"{YELLOW}⚠️ [Fugle] 觸發限流，等待 {sleep_time} 秒後重試...{RESET}")
                    time_module.sleep(sleep_time)
                else: return None
            elif any(x in error_str for x in ["502", "503", "504"]):
                if attempt < max_retries:
                    time_module.sleep(2 ** attempt)
                else: return None
            else: return None
    return None

def _reconnect_shioaji_if_needed():
    global api, to  # ✅ 加入 to
    print(f"{YELLOW}⚠️ 偵測到 Shioaji 異常，啟動重連機制...{RESET}")
    try:
        api.login(api_key=shioaji_logic.TEST_API_KEY, secret_key=shioaji_logic.TEST_API_SECRET)
        api.activate_ca(ca_path=shioaji_logic.CA_CERT_PATH, ca_passwd=shioaji_logic.CA_PASSWORD)
        time_module.sleep(2)
        to = tp.TouchOrderExecutor(api)  # ✅ 重新綁定觸價單
        print(f"{GREEN}✅ Shioaji 重新登入成功！{RESET}")
    except Exception as e:
        print(f"{RED}❌ Shioaji 重連失敗: {e}{RESET}")

def safe_place_order(api_instance, contract, order, max_retries=1):
    for attempt in range(max_retries + 1):
        try: return api_instance.place_order(contract, order)
        except Exception as e:
            if attempt < max_retries: _reconnect_shioaji_if_needed()
            else: raise e

def safe_add_touch_condition(to_instance, tcond, max_retries=1):
    for attempt in range(max_retries + 1):
        try:
            to_instance.add_condition(tcond)
            return
        except Exception as e:
            if attempt < max_retries: _reconnect_shioaji_if_needed()
            else: raise e

def safe_delete_touch_condition(to_instance, cond, max_retries=1):
    for attempt in range(max_retries + 1):
        try:
            to_instance.delete_condition(cond)
            return
        except Exception as e:
            if attempt < max_retries: _reconnect_shioaji_if_needed()

def write_trade_log(message: str):
    log_folder = "trade_logs"
    os.makedirs(log_folder, exist_ok=True)
    log_path = os.path.join(log_folder, f"{datetime.now().strftime('%Y-%m-%d')}.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")

# ------------------ K線處理與計算 ------------------
def calculate_2min_pct_increase_and_highest(new_candle, existing_candles):
    new_candle['2min_pct_increase'] = 0.0
    new_candle['highest'] = new_candle.get('high', 0)
    if not existing_candles: return new_candle
    all_candles = existing_candles + [new_candle]
    relevant_candles = all_candles if len(existing_candles) < 2 else existing_candles[-1:] + [new_candle]
    rise_values = [float(c.get('rise', 0.0)) for c in relevant_candles if c.get('rise') is not None]
    if len(rise_values) >= 2:
        pct_increase = max(rise_values) - min(rise_values) if rise_values[-1] >= rise_values[0] else min(rise_values) - max(rise_values)
        new_candle['2min_pct_increase'] = round(pct_increase, 2)
    new_candle['highest'] = max(max(c.get('highest', 0) for c in existing_candles), new_candle.get('high', 0))
    return new_candle

def fetch_intraday_data(client, symbol, trading_day, yesterday_close_price, start_time=None, end_time=None):
    try:
        _from = datetime.strptime(f"{trading_day} {start_time or '09:00'}", "%Y-%m-%d %H:%M")
        to = datetime.strptime(f"{trading_day} {end_time or '13:30'}", "%Y-%m-%d %H:%M")
        candles_rsp = safe_fugle_api_call(client.stock.intraday.candles, symbol=symbol, timeframe='1', _from=_from.isoformat(), to=to.isoformat())
        if not candles_rsp or 'data' not in candles_rsp: return pd.DataFrame()
        candles_df = pd.DataFrame(candles_rsp['data'])
        if 'volume' not in candles_df.columns: return pd.DataFrame()
        candles_df['volume'] = pd.to_numeric(candles_df['volume'], errors='coerce')
        candles_df['datetime'] = pd.to_datetime(candles_df['date'], errors='coerce').dt.tz_localize(None).dt.floor('min')
        candles_df.set_index('datetime', inplace=True)
        original_df = candles_df.reset_index()[['datetime', 'volume']].rename(columns={'volume': 'orig_volume'})
        candles_df = candles_df.reindex(pd.date_range(start=_from, end=to, freq='1min'))
        candles_df.reset_index(inplace=True)
        candles_df.rename(columns={'index': 'datetime'}, inplace=True)
        candles_df['date'] = candles_df['datetime'].dt.strftime('%Y-%m-%d')
        candles_df['time'] = candles_df['datetime'].dt.strftime('%H:%M:%S')
        candles_df = pd.merge(candles_df, original_df, how='left', on='datetime')
        for col in ['open', 'high', 'low', 'close']:
            vals, last_v = candles_df[col].to_numpy(), yesterday_close_price
            for i in range(len(vals)):
                v, c = candles_df.at[i, 'volume'], candles_df.at[i, 'close']
                if v > 0 and not pd.isna(c): last_v = c
                if pd.isna(vals[i]) or v == 0: vals[i] = last_v
            candles_df[col] = vals
        candles_df['volume'] = candles_df['orig_volume'].fillna(0)
        candles_df['symbol'] = symbol
        candles_df['昨日收盤價'] = yesterday_close_price
        candles_df['漲停價'] = truncate_to_two_decimals(calculate_limit_up_price(yesterday_close_price))
        candles_df[['symbol', '昨日收盤價', '漲停價']] = candles_df[['symbol', '昨日收盤價', '漲停價']].ffill().bfill()
        candles_df['rise'] = (candles_df['close'] - candles_df['昨日收盤價']) / candles_df['昨日收盤價'] * 100
        candles_df['highest'] = candles_df['high'].cummax().fillna(yesterday_close_price)
        return candles_df[['symbol', 'date', 'time', 'open', 'high', 'low', 'close', 'volume', '昨日收盤價', '漲停價', 'rise', 'highest']]
    except Exception: return pd.DataFrame()

def fetch_realtime_intraday_data(client, symbol, trading_day, yesterday_close_price, start_time=None, end_time=None):
    return fetch_intraday_data(client, symbol, trading_day, yesterday_close_price, start_time, end_time)

def fetch_daily_kline_data(client, symbol, days=2):
    end_date = get_recent_trading_day()
    start_date = end_date - timedelta(days=days)
    try:
        data = safe_fugle_api_call(client.stock.historical.candles, symbol=symbol, from_=start_date.strftime('%Y-%m-%d'), to=end_date.strftime('%Y-%m-%d'))
        if data and 'data' in data and data['data']: return pd.DataFrame(data['data'])
    except Exception: pass
    return pd.DataFrame()

def get_recent_trading_day():
    today, now_time = datetime.now().date(), datetime.now().time()
    def last_friday(d):
        while d.weekday() != 4: d -= timedelta(days=1)
        return d
    w = today.weekday()
    if w in [5, 6]: return last_friday(today)
    if w == 0 and now_time < time(13, 30): return last_friday(today)
    if w > 0 and now_time < time(13, 30): return today - timedelta(days=1)
    return today

def save_settings():
    with open('settings.json', 'w', encoding='utf-8') as f:
        json.dump({
            'capital_per_stock': capital_per_stock, 'transaction_fee': transaction_fee,
            'transaction_discount': transaction_discount, 'trading_tax': trading_tax,
            'below_50': below_50, 'price_gap_50_to_100': price_gap_50_to_100,
            'price_gap_100_to_500': price_gap_100_to_500, 'price_gap_500_to_1000': price_gap_500_to_1000,
            'price_gap_above_1000': price_gap_above_1000, 'allow_reentry_after_stop_loss': allow_reentry_after_stop_loss
        }, f, indent=4)

def load_settings():
    global capital_per_stock, transaction_fee, transaction_discount, trading_tax
    global below_50, price_gap_50_to_100, price_gap_100_to_500, price_gap_500_to_1000, price_gap_above_1000, allow_reentry_after_stop_loss
    if os.path.exists('settings.json'):
        with open('settings.json', 'r', encoding='utf-8') as f:
            s = json.load(f)
            capital_per_stock = s.get('capital_per_stock', 1000)
            transaction_fee = s.get('transaction_fee', 0.1425)
            transaction_discount = s.get('transaction_discount', 20.0)
            trading_tax = s.get('trading_tax', 0.15)
            below_50, price_gap_50_to_100 = s.get('below_50', 500), s.get('price_gap_50_to_100', 1000)
            price_gap_100_to_500, price_gap_500_to_1000 = s.get('price_gap_100_to_500', 2000), s.get('price_gap_500_to_1000', 3000)
            price_gap_above_1000 = s.get('price_gap_above_1000', 5000)
            allow_reentry_after_stop_loss = s.get('allow_reentry_after_stop_loss', False)

# --- 行情與儲存輔助 ---
def calculate_limit_up_price(close_price):
    lu = close_price * 1.10
    unit = 0.01 if lu < 10 else 0.05 if lu < 50 else 0.1 if lu < 100 else 0.5 if lu < 500 else 1 if lu < 1000 else 5
    return (lu // unit) * unit

def truncate_to_two_decimals(v): return math.floor(v * 100) / 100 if isinstance(v, float) else v

def load_matrix_dict_analysis():
    return json.load(open('matrix_dict_analysis.json', 'r', encoding='utf-8')) if os.path.exists('matrix_dict_analysis.json') else {}

def save_matrix_dict(d):
    with open('matrix_dict_analysis.json', 'w', encoding='utf-8') as f: json.dump(d, f, indent=4, ensure_ascii=False)

def save_auto_intraday_data(data):
    global in_memory_intraday_data, data_lock
    with data_lock: in_memory_intraday_data = data.copy()
    try: b = orjson.dumps(data, option=orjson.OPT_NON_STR_KEYS)
    except: return
    threading.Thread(target=lambda: open('auto_intraday.json', 'wb').write(b), daemon=True).start()

def load_disposition_stocks():
    try: return json.load(open('Disposition.json', 'r', encoding='utf-8'))
    except: return []

def fetch_disposition_stocks(client, matrix_dict):
    dispo = []
    for g, stocks in matrix_dict.items():
        for s in stocks:
            try:
                if safe_fugle_api_call(client.stock.intraday.ticker, symbol=s).get('isDisposition', False): dispo.append(s)
            except: pass
    with open('Disposition.json', 'w', encoding='utf-8') as f: json.dump(dispo, f, ensure_ascii=False, indent=4)

def load_kline_data():
    daily = json.load(open('daily_kline_data.json', 'r', encoding='utf-8')) if os.path.exists('daily_kline_data.json') else {}
    intra = json.load(open('intraday_kline_data.json', 'r', encoding='utf-8')) if os.path.exists('intraday_kline_data.json') else {}
    return daily, intra

def ensure_continuous_time_series(df):
    df['date'] = pd.to_datetime(df['date'])
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
    idx = pd.MultiIndex.from_product([df['date'].unique(), pd.date_range('09:00', '13:30', freq='1min').time], names=['date', 'time'])
    df.set_index(['date', 'time'], inplace=True)
    df = df.reindex(idx)
    df[['symbol', '昨日收盤價', '漲停價']] = df[['symbol', '昨日收盤價', '漲停價']].ffill().bfill()
    if 'high' not in df.columns: df['high'] = df['close']
    df['close'] = df['close'].ffill().fillna(df['昨日收盤價'])
    for c in ['open', 'high', 'low']: df[c] = df[c].ffill().fillna(df['close'])
    df['volume'] = df['volume'].fillna(0)
    df['2min_pct_increase'] = df['2min_pct_increase'].fillna(0.0) if '2min_pct_increase' in df.columns else 0.0
    return df.reset_index()

def initialize_stock_data(symbols, daily, intra):
    res = {}
    for s in symbols:
        if s in intra and not pd.DataFrame(intra[s]).empty:
            res[s] = ensure_continuous_time_series(pd.DataFrame(intra[s])).drop(columns=['average'], errors='ignore')
    return res

def purge_disposition_from_nb(disposition_list, nb_path='nb_matrix_dict.json'):
    if not os.path.exists(nb_path): return
    try: nb_dict = json.load(open(nb_path, 'r', encoding='utf-8'))
    except: return
    if 'consolidated_symbols' not in nb_dict or not isinstance(nb_dict['consolidated_symbols'], dict): return
    changed = False
    for grp, syms in nb_dict['consolidated_symbols'].items():
        filtered = [s for s in dict.fromkeys(syms) if s not in disposition_list]
        if len(filtered) != len(syms):
            nb_dict['consolidated_symbols'][grp] = filtered
            changed = True
    if changed:
        with open(nb_path, 'w', encoding='utf-8') as f: json.dump(nb_dict, f, ensure_ascii=False, indent=4)

def load_symbols_to_analyze():
    nb = load_matrix_dict_analysis()
    syms = [s for g in nb.values() for s in g]
    disp = load_disposition_stocks()
    return [s for s in syms if s not in disp]

def load_group_symbols():
    return json.load(open('nb_matrix_dict.json', 'r', encoding='utf-8')) if os.path.exists('nb_matrix_dict.json') else {}

def exit_trade(selected_stock_df, shares, entry_price, sell_cost, entry_fee, tax, message_log, current_time, hold_time, entry_time, use_f_exit=False):
    global transaction_fee, transaction_discount, trading_tax, in_position, has_exited, current_position
    current_time_str = current_time if isinstance(current_time, str) else current_time.strftime('%H:%M:%S')
    selected_stock_df['time'] = pd.to_datetime(selected_stock_df['time'], format='%H:%M:%S').dt.time
    entry_time_obj = datetime.strptime(entry_time, '%H:%M:%S').time() if isinstance(entry_time, str) else entry_time

    if use_f_exit:
        end_time = datetime.strptime('13:30', '%H:%M').time()
        end_price_series = selected_stock_df[selected_stock_df['time'] == end_time]['close']
        if not end_price_series.empty: end_price = end_price_series.values[0]
        else: return None, None
    else:
        entry_index_series = selected_stock_df[selected_stock_df['time'] == entry_time_obj].index
        if not entry_index_series.empty:
            exit_index = entry_index_series[0] + hold_time
            if exit_index >= len(selected_stock_df): return None, None
            end_price = selected_stock_df.iloc[exit_index]['close']
        else: return None, None

    buy_cost = shares * end_price * 1000
    exit_fee = int(buy_cost * (transaction_fee * 0.01) * (transaction_discount * 0.01))
    profit = sell_cost - buy_cost - entry_fee - exit_fee - tax
    return_rate = (profit * 100) / (buy_cost - exit_fee) if (buy_cost - exit_fee) != 0 else 0.0
    message_log.append((current_time_str, f"{RED}出場！利潤：{int(profit)} 元，報酬率：{return_rate:.2f}%{RESET}"))
    in_position = False
    has_exited = True
    return profit, return_rate

# ------------------ Shioaji API & 平倉邏輯 ------------------
api = sj.Shioaji(simulation=True)

# ✅ 修正：必須先登入獲取 Contracts，才能初始化 TouchPrice
try:
    print(f"{YELLOW}⏳ 正在初始化 Shioaji API 並自動登入預設帳戶...{RESET}")
    api.login(api_key=shioaji_logic.TEST_API_KEY, secret_key=shioaji_logic.TEST_API_SECRET)
    api.activate_ca(ca_path=shioaji_logic.CA_CERT_PATH, ca_passwd=shioaji_logic.CA_PASSWORD)
    print(f"{GREEN}✅ Shioaji 登入成功！合約資料已就緒。{RESET}")
except Exception as e:
    print(f"{RED}⚠️ Shioaji 初始登入失敗: {e}{RESET}")

try:
    to = tp.TouchOrderExecutor(api)
except Exception as e:
    print(f"{RED}⚠️ 觸價單模組初始化失敗，請稍後在介面中重新登入。{RESET}")
    to = None

def exit_trade_live():
    global open_positions, data_lock, api
    with data_lock: conditions_dict = dict(to.conditions)
    exit_data = {code: sum(int(getattr(c.order, 'quantity', 0)) for c in conds) for code, conds in conditions_dict.items() if sum(int(getattr(c.order, 'quantity', 0)) for c in conds) > 0}
    for stock_code, shares in exit_data.items():
        try:
            contract = getattr(api.Contracts.Stocks.TSE, f"TSE{stock_code}")
            order = api.Order(action=sj.constant.Action.Buy, price=contract.limit_up, quantity=shares, price_type=sj.constant.StockPriceType.LMT, order_type=sj.constant.OrderType.ROC, order_lot=sj.constant.StockOrderLot.Common, account=api.stock_account)
            safe_place_order(api, contract, order)
            with data_lock: open_positions.pop(stock_code, None)
            print(f"{RED}✅ {stock_code} {shares}張 已送出市價平倉{RESET}")
        except Exception as e: print(f"平倉 {stock_code} 錯誤: {e}")
    with data_lock:
        for conds in conditions_dict.values():
            for c in conds: safe_delete_touch_condition(to, c)

def close_one_stock(code: str):
    global data_lock, api
    with data_lock:
        conds = to.conditions.get(code, [])
        qty = sum(getattr(c.order, 'quantity', 0) for c in conds)
    if qty == 0: return print(f"⚠️ {code} 無委託或持倉")
    try:
        contract = getattr(api.Contracts.Stocks.TSE, f"TSE{code}")
        order = api.Order(action=sj.constant.Action.Buy, price=contract.limit_up, quantity=qty, price_type=sj.constant.StockPriceType.LMT, order_type=sj.constant.OrderType.ROC, order_lot=sj.constant.StockOrderLot.Common, account=api.stock_account)
        safe_place_order(api, contract, order)
        print(f"{GREEN}✅ 已平倉 {code} 共 {qty} 張{RESET}")
    except Exception as e: print(f"平倉 {code} 錯誤: {e}")
    with data_lock:
        for c in conds: safe_delete_touch_condition(to, c)
        to.conditions.pop(code, None)
        open_positions.pop(code, None)

def update_variable(file_path, var_name, new_value, is_raw=False):
    lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.lstrip().startswith(var_name + " ="):
                new_line = f'{var_name} = r"{new_value}"\n' if is_raw else f'{var_name} = "{new_value}"\n'
                lines.append(new_line)
            else: lines.append(line)
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    importlib.reload(shioaji_logic)

def monitor_stop_loss_orders():
    global to, group_positions, previous_stop_loss_codes, allow_reentry_after_stop_loss, data_lock
    with data_lock:
        current_codes = set(to.conditions.keys()) if isinstance(to.conditions, dict) else set()
        if not current_codes and not isinstance(to.conditions, dict):
            for cond in to.conditions:
                try: current_codes.add(cond.order_contract.code)
                except: pass
        removed_codes = previous_stop_loss_codes - current_codes
        if removed_codes and allow_reentry_after_stop_loss:
            nb = load_nb_matrix_dict().get("consolidated_symbols", {})
            for code in removed_codes:
                for group, symbols in nb.items():
                    if code in symbols and group in group_positions and group_positions[group] == "已進場":
                        group_positions[group] = False
                        print(f"停損出場：股票 {code}。")
        previous_stop_loss_codes = current_codes.copy()

def initialize_triggered_limit_up(auto_intraday_data: dict):
    for sym, kbars in auto_intraday_data.items():
        for i in range(1, len(kbars)):
            prev, curr = kbars[i-1], kbars[i]
            if curr["high"] == curr["漲停價"] and prev["high"] < curr["漲停價"]:
                triggered_limit_up_stocks.add(sym)
                break

# ---------------- 回測程式：單一族群分析 ----------------
def calculate_average_over_high(group_name=None, progress_callback=None):
    daily_kline_data, intraday_kline_data = load_kline_data()

    matrix_dict_analysis = load_matrix_dict_analysis()
    
    if group_name is None:
        group_name = input("請輸入要分析的族群名稱：")
    
    if group_name not in matrix_dict_analysis:
        print("沒有此族群資料")
        return None

    symbols_to_analyze = matrix_dict_analysis[group_name]
    disposition_stocks = load_disposition_stocks()
    symbols_to_analyze = [symbol for symbol in symbols_to_analyze if symbol not in disposition_stocks]

    if not symbols_to_analyze:
        print(f"{group_name} 中沒有可供分析的股票。")
        return None

    print(f"開始分析族群 {group_name} 中的股票...")
    any_condition_one_triggered = False 
    group_over_high_averages = []
    
    total_symbols = len(symbols_to_analyze) # 🟢 取得總數計算進度
    
    for i, symbol in enumerate(symbols_to_analyze):
        # 🟢 發送進度更新
        if progress_callback:
            progress_callback(int((i / total_symbols) * 100), f"正在分析: {symbol}")
            
        print(f"\n正在分析股票：{symbol}")
        
        if symbol not in daily_kline_data or symbol not in intraday_kline_data:
            print(f"無法取得 {symbol} 的日 K 線或一分 K 線數據，跳過。")
            continue
        
        daily_kline_df = pd.DataFrame(daily_kline_data[symbol])
        intraday_data = pd.DataFrame(intraday_kline_data[symbol])

        condition_one_triggered = False
        condition_two_triggered = False
        previous_high = None
        condition_two_time = None
        over_high_intervals = []

        for idx, row in intraday_data.iterrows():
            current_time = pd.to_datetime(row['time']).time()
            if previous_high is None:
                previous_high = row['high']
                continue

            if not condition_one_triggered:
                if row['2min_pct_increase'] >= 2:
                    condition_one_triggered = True
                    condition_two_triggered = False
                    any_condition_one_triggered = True

                    print(f"{symbol} 觸發條件一，開始監測兩分鐘漲幅，兩分鐘漲幅: {row['2min_pct_increase']:.2f}%")

            if condition_one_triggered and not condition_two_triggered:
                if row['high'] <= previous_high:
                    current_time_str = current_time.strftime('%H:%M:%S')
                    print(f"{symbol} 觸發條件二！時間：{current_time_str}")

                    condition_two_time = current_time
                    condition_two_triggered = True

            elif condition_two_triggered:
                if row['highest'] > previous_high:
                    condition_three_time_str = current_time.strftime('%H:%M:%S')
                    print(f"{symbol} 觸發條件三！時間：{condition_three_time_str}")
                    if condition_two_time:
                        today = datetime.today().date()
                        condition_two_datetime = datetime.combine(today, condition_two_time)
                        condition_three_datetime = datetime.combine(today, current_time)
                        interval = (condition_three_datetime - condition_two_datetime).total_seconds() / 60
                        print(f"{symbol} 過高間隔：{interval:.2f} 分鐘")
                        over_high_intervals.append(interval)

                    condition_one_triggered = False
                    condition_two_triggered = False
                    condition_two_time = None

            previous_high = row['high']

        if over_high_intervals:
            q1 = np.percentile(over_high_intervals, 25)
            q3 = np.percentile(over_high_intervals, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            filtered_intervals = [interval for interval in over_high_intervals if lower_bound <= interval <= upper_bound]
            if filtered_intervals:
                average_interval = sum(filtered_intervals) / len(filtered_intervals)
                print(f"{symbol} 平均過高間隔：{average_interval:.2f} 分鐘")
                group_over_high_averages.append(average_interval)
            else:
                print(f"{symbol} 沒有有效的過高間隔數據")
        else:
            print(f"{symbol} 沒有觸發過高間隔的情形")

    if group_over_high_averages:
        group_average_over_high = sum(group_over_high_averages) / len(group_over_high_averages)
        print(f"{group_name} 平均過高間隔：{group_average_over_high:.2f} 分鐘")
        return group_average_over_high
    else:
        print(f"{group_name} 沒有有效的過高間隔數據")
        return None


# ------------------ 更新K線數據：更新數據 ------------------
def update_kline_data():
    client, api_key = init_fugle_client()
    matrix_dict_analysis = load_matrix_dict_analysis()
    if not matrix_dict_analysis:
        print("沒有任何族群資料，請先管理族群。")
        return

    print("正在更新處置股清單...")
    fetch_disposition_stocks(client, matrix_dict_analysis)
    print("處置股清單已更新。")

    disposition_stocks = load_disposition_stocks()
    symbols_to_analyze = [sym for group in matrix_dict_analysis.values() for sym in group if sym not in disposition_stocks]

    # ===== ① 更新日 K 線資料 =====
    print("✅ 開始更新日K線數據至 daily_kline_data.json...")
    existing_daily_kline_data = {}
    if os.path.exists('daily_kline_data.json'):
        with open('daily_kline_data.json', 'r', encoding='utf-8') as f:
            try: existing_daily_kline_data = json.load(f)
            except json.JSONDecodeError: existing_daily_kline_data = {}
    else:
        print("⚠️ daily_kline_data.json 不存在，將建立新檔案。")

    initial_api_count = 0

    # 💡 修正 1：移除前20檔的「省略更新」邏輯，確保所有股票（含新加入）都會被強制更新
    for symbol in symbols_to_analyze:
        if initial_api_count >= 55:
            print("已達到55次API請求，休息1分鐘...")
            time_module.sleep(60)
            initial_api_count = 0

        daily_kline_df = fetch_daily_kline_data(client, symbol, days=5)
        initial_api_count += 1

        if daily_kline_df.empty:
            print(f"❌ 無法取得 {symbol} 的日K數據，跳過。")
            continue

        daily_kline_data = daily_kline_df.to_dict(orient='records')
        existing_daily_kline_data[symbol] = daily_kline_data

    # 🟢 --- 新增清理邏輯：踢除不在本次名單中的幽靈股票 ---
    # 這裡使用 symbols_to_analyze (即本次更新的所有目標) 作為白名單
    current_active_symbols = set(symbols_to_analyze)
    existing_daily_kline_data = {
        s: d for s, d in existing_daily_kline_data.items() if s in current_active_symbols
    }
    # --------------------------------------------------

    with open('daily_kline_data.json', 'w', encoding='utf-8') as f:
        json.dump(existing_daily_kline_data, f, indent=4, ensure_ascii=False)

    print(f"✅ 日K線數據已更新並清理(僅保留名單內股票)，寫入 daily_kline_data.json。")

    # ===== ② 更新一分 K 線資料 =====
    print("✅ 開始更新一分K線資料至 intraday_kline_data.json...")

    def get_recent_trading_day():
        today = datetime.now().date()
        now_time = datetime.now().time()
        market_open = datetime.strptime("09:00", "%H:%M").time()
        market_close = datetime.strptime("13:30", "%H:%M").time()

        def last_friday(date):
            while date.weekday() != 4:
                date -= timedelta(days=1)
            return date

        weekday = today.weekday()

        if weekday == 5:  return last_friday(today)
        elif weekday == 6: return last_friday(today)
        elif weekday == 0:
            if now_time < market_open: return last_friday(today)
            elif market_open <= now_time <= market_close: return last_friday(today)
            else: return today
        else:
            if now_time < market_open: return today - timedelta(days=1)
            elif market_open <= now_time <= market_close: return today - timedelta(days=1)
            else: return today

    intraday_kline_data = {}
    count = 0

    trading_day = get_recent_trading_day().strftime('%Y-%m-%d')
    print(f"📅 本次一分K更新使用交易日: {trading_day}")

    for symbol in symbols_to_analyze:
        if count >= 55:
            print("已達到55次API請求，休息1分鐘...")
            time_module.sleep(60)
            count = 0

        daily_data = existing_daily_kline_data.get(symbol, [])
        if not daily_data:
            print(f"{symbol} 日K資料不足，無法判斷昨收，跳過。")
            continue
        
        # 💡 修正 3：套用與盤中交易相同的精確「昨收價」判斷邏輯
        sorted_daily_data = sorted(daily_data, key=lambda x: x['date'], reverse=True)
        if len(sorted_daily_data) > 1:
            now2 = datetime.now()
            weekday = now2.weekday()
            if 0 <= weekday <= 4 and 8 <= now2.hour < 15:
                yesterday_close_price = sorted_daily_data[0].get('close', 0)
            else:
                yesterday_close_price = sorted_daily_data[1].get('close', 0)
        else:
            yesterday_close_price = sorted_daily_data[0].get('close', 0)

        intraday_df = fetch_intraday_data(
            client=client,
            symbol=symbol,
            trading_day=trading_day,
            yesterday_close_price=yesterday_close_price,
            start_time="09:00",
            end_time="13:30"
        )
        count += 1

        if intraday_df.empty:
            print(f"無法取得 {symbol} 的一分K數據，跳過。")
            continue
        
        updated_records = []
        records = intraday_df.to_dict(orient='records')
        for i, candle in enumerate(records): 
            updated_candle = calculate_2min_pct_increase_and_highest(candle, records[:i])
            updated_records.append(updated_candle)
        intraday_df = pd.DataFrame(updated_records)
        intraday_kline_data[symbol] = intraday_df.to_dict(orient='records')
        print(f"{symbol} 的一分K資料已加入。")

    intraday_kline_data_str = convert_datetime_to_str(intraday_kline_data)
    with open('intraday_kline_data.json', 'w', encoding='utf-8') as f:
        json.dump(intraday_kline_data_str, f, indent=4, ensure_ascii=False, default=str)
    print("✅ 一分K線資料已寫入 intraday_kline_data.json。")

    consolidate_and_save_stock_symbols()
    print("✅ 股票代號已統整並儲存至 nb_matrix_dict.json。")

# ------------------ 回測程式主程式 ------------------
def process_group_data(stock_data_collection, wait_minutes, hold_minutes,
                       matrix_dict_analysis, verbose=True, progress_callback=None):
    # ────────── 0-A. 本地旗標初始化 ────────── #
    in_position         = False
    has_exited          = False
    current_position    = None
    stop_loss_triggered = False
    hold_time           = 0

    # ────────── 0-B. 需要的全域設定 ────────── #
    global capital_per_stock, transaction_fee, transaction_discount, trading_tax
    global price_gap_below_50, price_gap_50_to_100, price_gap_100_to_500
    global price_gap_500_to_1000, price_gap_above_1000
    global allow_reentry_after_stop_loss

    # ---------- 0-C. 開盤前三分鐘平均量 ---------- #
    FIRST3_AVG_VOL: dict[str, float] = {}
    for sym, df in stock_data_collection.items():
        first3 = df[df['time'].astype(str).isin(['09:00:00', '09:01:00', '09:02:00'])]
        FIRST3_AVG_VOL[sym] = first3['volume'].mean() if not first3.empty else 0

    # ---------- 0-D. 其他狀態變數 ---------- #
    message_log: list[tuple[str, str]] = []
    tracking_stocks: set[str] = set()
    leader                      = None
    leader_peak_rise            = None
    leader_rise_before_decline  = None
    in_waiting_period           = False
    waiting_time                = 0
    pull_up_entry               = False
    limit_up_entry              = False
    first_condition_one_time    = None

    # ---------- 0-E. 組 merge DataFrame ---------- #
    merged_df = None
    req_cols = ['time', 'rise', 'high', '漲停價', 'close', '2min_pct_increase', 'volume']
    for sym, df in stock_data_collection.items():
        if not all(c in df.columns for c in req_cols): continue
        tmp = df[req_cols].copy()
        tmp = tmp.rename(columns={
            'rise': f'rise_{sym}', 'high': f'high_{sym}', '漲停價': f'limit_up_price_{sym}',
            'close': f'close_{sym}', '2min_pct_increase': f'2min_pct_increase_{sym}', 'volume': f'volume_{sym}'
        })
        merged_df = tmp if merged_df is None else pd.merge(merged_df, tmp, on='time', how='outer')

    if merged_df is None or merged_df.empty: return None, None
    merged_df.sort_values('time', inplace=True, ignore_index=True)

    # ═══════════ 1. 逐分鐘主迴圈 ═══════════ #
    total_profit = total_profit_rate = total_trades = 0
    total_rows = len(merged_df)

    for i, row in merged_df.iterrows():
        # 🟢 進度條更新
        if progress_callback and i % 5 == 0: 
            percent = int(((i + 1) / total_rows) * 100)
            progress_callback(percent, f"回測進行中: {row['time'].strftime('%H:%M')}")
        
        current_time     = row['time']
        current_time_str = current_time.strftime('%H:%M:%S')

        # ── 1-1. 持倉期間：強制 / 時間平倉 / 條件停損 ── #
        if in_position and not has_exited:
            hold_time += 1
            if current_time_str == '13:30:00':
                profit, rate = exit_trade(
                    stock_data_collection[current_position['symbol']], current_position['shares'], current_position['entry_price'],
                    current_position['sell_cost'], current_position['entry_fee'], current_position['tax'],
                    message_log, current_time, hold_time, current_position['entry_time'], use_f_exit=True
                )
                if profit is not None: total_trades += 1; total_profit += profit; total_profit_rate += rate
                in_position = False; has_exited  = True; current_position = None
                continue

            if current_position.get('actual_hold_minutes') is not None and hold_time >= current_position['actual_hold_minutes']:
                profit, rate = exit_trade(
                    stock_data_collection[current_position['symbol']], current_position['shares'], current_position['entry_price'],
                    current_position['sell_cost'], current_position['entry_fee'], current_position['tax'],
                    message_log, current_time, hold_time, current_position['entry_time']
                )
                if profit is not None: total_trades += 1; total_profit += profit; total_profit_rate += rate
                in_position = False; has_exited  = True
                continue

            sel_df  = stock_data_collection[current_position['symbol']]
            now_row = sel_df[sel_df['time'] == current_time]
            if not now_row.empty:
                h_now = truncate_to_two_decimals(now_row.iloc[0]['high'])
                thresh = truncate_to_two_decimals(current_position['stop_loss_threshold'])
                if h_now >= thresh:
                    exit_price = thresh
                    exit_cost  = current_position['shares'] * exit_price * 1000
                    exit_fee   = int(exit_cost * (transaction_fee*0.01) * (transaction_discount*0.01))
                    profit = current_position['sell_cost'] - exit_cost - current_position['entry_fee'] - exit_fee - current_position['tax']
                    rate = (profit * 100) / (current_position['sell_cost'] - current_position['entry_fee'] - exit_fee)
                    message_log.append((current_time_str, f"{Fore.RED}停損觸發，利潤 {int(profit)} 元 ({rate:.2f}%){Style.RESET_ALL}"))
                    total_trades += 1; total_profit += profit; total_profit_rate += rate
                    in_position = False; has_exited  = True; current_position = None; stop_loss_triggered = True
                    if not allow_reentry_after_stop_loss: break
            continue  

        # ── 1-2. 檢查觸發 (拉高/漲停) ── #
        trigger_list = []
        for sym in stock_data_collection.keys():
            pct, vol, high, lup = row.get(f'2min_pct_increase_{sym}'), row.get(f'volume_{sym}'), row.get(f'high_{sym}'), row.get(f'limit_up_price_{sym}')
            avgv = FIRST3_AVG_VOL.get(sym, 0)

            hit_limit = False
            if high is not None and lup is not None and high == lup:
                if current_time_str == '09:00:00': hit_limit = True
                else:
                    prev_time = (datetime.combine(date.today(), current_time) - timedelta(minutes=1)).time()
                    prev_high = stock_data_collection[sym].loc[stock_data_collection[sym]['time'] == prev_time, 'high']
                    if prev_high.empty or prev_high.iloc[0] < lup: hit_limit = True
            
            if hit_limit: trigger_list.append({'symbol': sym, 'condition': 'limit_up'}); continue
            if pct is not None and pct >= 2 and vol is not None and avgv and vol > 1.3*avgv: trigger_list.append({'symbol': sym, 'condition': 'pull_up'})

        # ── 1-3. 處理觸發結果 ── #
        for item in trigger_list:
            sym, cond = item['symbol'], item['condition']
            if cond == 'limit_up':
                tracking_stocks.add(sym)
                leader = sym; in_waiting_period = True; waiting_time = 0
                
                # 🟢 修正：無縫升級！如果本來就在拉高進場，保留原有的 first_condition_one_time，不洗掉歷史！
                if not (pull_up_entry or limit_up_entry):
                    first_condition_one_time = datetime.combine(date.today(), current_time)
                
                pull_up_entry = False; limit_up_entry = True
                if verbose: message_log.append((current_time_str, f"{YELLOW}{sym} 漲停觸發 (無縫升級)，保留發動起點！{RESET}"))
            else:
                # 🟢 修正：避免拉高觸發洗掉漲停的狀態
                if not pull_up_entry and not limit_up_entry: 
                    pull_up_entry = True; limit_up_entry = False
                    tracking_stocks.clear()
                    first_condition_one_time = datetime.combine(date.today(), current_time)
                tracking_stocks.add(sym)
                if verbose: message_log.append((current_time_str, f"{YELLOW}{sym} 拉高觸發，加入追蹤{RESET}"))

        # 無論漲停或拉高，全面擴充追蹤清單 (>1.5%)
        if pull_up_entry or limit_up_entry:
            for sym in stock_data_collection.keys():
                if sym in tracking_stocks: continue
                pct = row.get(f'2min_pct_increase_{sym}')
                if pct is not None and pct >= 1.5: tracking_stocks.add(sym)

        # ── 1-4. 領漲選擇與反轉偵測 ── #
        if tracking_stocks:
            max_sym, max_rise = None, None
            for sym in tracking_stocks:
                r = row.get(f'rise_{sym}')
                if r is not None and (max_rise is None or r > max_rise): max_rise, max_sym = r, sym
            
            if leader != max_sym:
                if leader and verbose: message_log.append((current_time_str, f"{Fore.CYAN}領漲替換：{leader} → {max_sym}{Style.RESET_ALL}"))
                leader = max_sym; leader_peak_rise = max_rise; leader_rise_before_decline = max_rise
                in_waiting_period = False; waiting_time = 0  # 領漲換人，重置等待
                
                # 🟢 修正 2：領漲換人時，將時間基準點重置到現在！破解 DTW 雙峰陷阱
                first_condition_one_time = datetime.combine(date.today(), current_time)
                
                if verbose: message_log.append((current_time_str, f"{Fore.MAGENTA}🚀 領漲替換觸發，時間窗重置，重新監控新領漲{Style.RESET_ALL}"))
            
            if leader:
                h_now = row.get(f'high_{leader}')
                prev_time = (datetime.combine(date.today(), current_time) - timedelta(minutes=1)).time()
                prev_row = stock_data_collection[leader][stock_data_collection[leader]['time'] == prev_time]
                if not prev_row.empty:
                    h_prev = prev_row.iloc[0]['high']
                    if h_now <= h_prev and not in_waiting_period:
                        in_waiting_period = True; waiting_time = 0; leader_rise_before_decline = max_rise
                        if verbose: message_log.append((current_time_str, f"領漲 {leader} 反轉，開始等待"))

        # ── 1-5. 等待時間計數 & 最終篩選進場 ── #
        if in_waiting_period:
            # 🟢 滾動相似度檢查 (直接剔除不合格者)
            # 💡 修正：將 15 分鐘的歷史包袱改為 2 分鐘。領漲換人時，只專注比對「換人後」的波型！
            window_start_t = max((datetime.combine(date.today(), first_condition_one_time.time()) - timedelta(minutes=2)).time(), time(9,0))
            to_remove = []
            for sym in list(tracking_stocks):
                if sym == leader: continue
                corr = calculate_dtw_pearson(stock_data_collection[leader], stock_data_collection[sym], window_start_t, current_time)
                if corr < 0.4:
                    to_remove.append(sym)
                    if progress_callback: progress_callback(int(((i+1)/total_rows)*100), f"❌ 相似度剔除: {sym} ({corr:.2f})")
            for sym in to_remove:
                tracking_stocks.remove(sym)
                if verbose: message_log.append((current_time_str, f"{Fore.RED}[滾動剔除] {sym} 相似度 {corr:.2f} < 0.4{Style.RESET_ALL}"))

            if waiting_time >= wait_minutes:
                in_waiting_period = False; waiting_time = 0
                filtered_stocks = set(tracking_stocks)

                # 🟢 修正：與實戰完全統一的爆量記憶機制
                def _vol_break(sym, join_time):
                    df = stock_data_collection[sym]
                    avgv = FIRST3_AVG_VOL.get(sym, 0)
                    if avgv == 0: return False
                    later = df[df['time'] >= join_time.time()]
                    return (later['volume'] >= 1.5 * avgv).any()

                def _rise_peak_flat(sym: str, join_time: datetime) -> bool:
                    df = stock_data_collection[sym]
                    sub = df[df['time'] >= join_time.time()]
                    pkidx = sub['rise'].idxmax()
                    pkval = sub.loc[pkidx, 'rise']
                    later = sub.loc[pkidx+1:]
                    later_max = later['rise'].max() if not later.empty else None
                    return (later_max is None) or (later_max <= pkval + 0.5)

                # 🟢 修正：剔除所有冗餘代碼，一次性乾淨篩選
                # 🟢 修正：剔除所有冗餘代碼，一次性乾淨篩選 (加入深度除錯訊息)
                eligible = []
                for sym in filtered_stocks:
                    if sym == leader: continue
                    
                    # 🔍 檢查關卡 1：爆量條件
                    if not _vol_break(sym, first_condition_one_time):
                        if verbose: message_log.append((current_time_str, f"🔍 [除錯] {sym} 剔除：等待期間未出現爆量 1.5 倍的 K 棒"))
                        continue
                        
                    # 🔍 檢查關卡 2：不過高條件
                    if not _rise_peak_flat(sym, first_condition_one_time):
                        if verbose: message_log.append((current_time_str, f"🔍 [除錯] {sym} 剔除：等待期間突破了前高 (破壞作頭型態)"))
                        continue
                    
                    rise_now = row.get(f'rise_{sym}')
                    # 🔍 檢查關卡 3：漲幅限制
                    if rise_now is None or not (-1 <= rise_now <= 6):
                        if verbose: message_log.append((current_time_str, f"🔍 [除錯] {sym} 剔除：當前漲幅 {rise_now}% 不在 -1% ~ 6% 之間"))
                        continue
                        
                    price_now = row.get(f'close_{sym}')
                    # 🔍 檢查關卡 4：資金上限
                    if price_now is None or price_now > capital_per_stock*1.5:
                        if verbose: message_log.append((current_time_str, f"🔍 [除錯] {sym} 剔除：股價超出單筆資金上限"))
                        continue
                    
                    row_sym = stock_data_collection[sym].loc[stock_data_collection[sym]['time'] == current_time].iloc[0]
                    eligible.append({'symbol': sym, 'rise': rise_now, 'row': row_sym})
                    if verbose: message_log.append((current_time_str, f"🎯 [除錯] {sym} 成功通過所有濾網，加入進場候選名單！"))

                if not eligible:
                    pull_up_entry = limit_up_entry = False; tracking_stocks.clear()
                    if verbose: message_log.append((current_time_str, "等待結束無符合股票，流程重置"))
                else:
                    eligible.sort(key=lambda x: x['rise'], reverse=True)
                    
                    # =============== 🟢 修改：奇數進場中位數，偶數進場中位數後一位 ===============
                    total_eligible = len(eligible)
                    if total_eligible == 1:
                        target_idx = 0
                    elif total_eligible % 2 == 1:
                        # 奇數 (如 3, 5, 7)：取正中間
                        target_idx = total_eligible // 2
                    else:
                        # 偶數 (如 2, 4, 6)：取中位數後一位
                        # 例如 2 檔取 Index 1 (第2名)；4 檔取 Index 2 (第3名)
                        target_idx = total_eligible // 2
                    # =====================================================================

                    chosen = eligible[target_idx]
                    
                    if verbose: 
                        message_log.append((current_time_str, f"🎯 [選股策略] 共有 {total_eligible} 檔候選，採奇數中位數/偶數中位數後一，選擇第 {target_idx + 1} 名進場！"))
                    # =====================================================================

                    rowch   = chosen['row']
                    entry_p = rowch['close']
                    shares  = round((capital_per_stock*10000)/(entry_p*1000))
                    sell_cost = shares * entry_p * 1000
                    entry_fee = int(sell_cost * (transaction_fee*0.01) * (transaction_discount*0.01))
                    tax   = int(sell_cost * (trading_tax*0.01))
                    
                    if entry_p < 10: gap, tick = below_50, 0.01
                    elif entry_p < 50: gap, tick = below_50, 0.05
                    elif entry_p < 100: gap, tick = price_gap_50_to_100, 0.1
                    elif entry_p < 500: gap, tick = price_gap_100_to_500, 0.5
                    elif entry_p < 1000: gap, tick = price_gap_500_to_1000, 1
                    else: gap, tick = price_gap_above_1000, 5

                    highest_on_entry = rowch['highest'] or entry_p
                    if (highest_on_entry-entry_p)*1000 < gap: stop_thr = entry_p + gap/1000
                    else: stop_thr = highest_on_entry + tick

                    actual_hold_minutes = hold_minutes
                    if actual_hold_minutes is not None:
                        expected_exit = datetime.combine(date.today(), current_time) + timedelta(minutes=actual_hold_minutes)
                        if expected_exit.time() >= time(13, 26):
                            actual_hold_minutes = None
                            if verbose: message_log.append((current_time_str, f"{YELLOW}預計出場時間 {expected_exit.strftime('%H:%M:%S')} 超過 13:26，轉為 F 尾盤平倉{RESET}"))
                    
                    current_position = {
                        'symbol': chosen['symbol'], 'shares': shares, 'entry_price': entry_p, 'sell_cost': sell_cost,
                        'entry_fee': entry_fee, 'tax': tax, 'entry_time': current_time_str, 'current_price_gap': gap,
                        'tick_unit': tick, 'highest_on_entry': highest_on_entry, 'stop_loss_threshold': stop_thr,
                        'actual_hold_minutes': actual_hold_minutes
                    }
                    in_position = True; has_exited = False; hold_time = 0
                    pull_up_entry = limit_up_entry = False; tracking_stocks.clear()
                    if verbose: message_log.append((current_time_str, f"{Fore.GREEN}進場！{chosen['symbol']} {shares}張 價 {entry_p:.2f} 停損 {stop_thr:.2f}{Style.RESET_ALL}"))
            else:
                if leader:
                    rise_now = row.get(f"rise_{leader}")
                    if leader_rise_before_decline is not None and rise_now is not None and rise_now > leader_rise_before_decline:
                        if verbose: message_log.append((current_time_str, f"{Fore.YELLOW}🚀 領漲股 {leader} 再創新高 {rise_now:.2f}%，觸發自我替換{Style.RESET_ALL}"))
                        leader_rise_before_decline = rise_now
                        in_waiting_period = False; waiting_time = 0
                        continue  

                waiting_time += 1
                if verbose: message_log.append((current_time_str, f"等待中，第 {waiting_time} 分鐘"))

    # ═══════════ 2. 回測結果輸出 ═══════════ #
    message_log.sort(key=lambda x: x[0])
    for t, msg in message_log: print(f"[{t}] {msg}")

    if total_trades:
        avg_rate = total_profit_rate / total_trades
        c = GREEN if total_profit < 0 else (RED if total_profit > 0 else "")
        print(f"\n{c}模擬完成，總利潤：{int(total_profit)} 元，平均報酬率：{avg_rate:.2f}%{RESET}\n")
        return total_profit, avg_rate
    else:
        print("無交易，無法計算利潤")
        return None, None

def process_live_trading_logic(
    symbols_to_analyze, current_time_str, wait_minutes, hold_minutes, message_log,
    in_position, has_exited, current_position, hold_time, already_entered_stocks,
    stop_loss_triggered, final_check_active, final_check_count, in_waiting_period,
    waiting_time, leader, tracking_stocks, previous_rise_values, leader_peak_rise,
    leader_rise_before_decline, first_condition_one_time, can_trade, group_positions,
    nb_matrix_path="nb_matrix_dict.json"
):
    monitor_stop_loss_orders()

    global capital_per_stock, transaction_fee, transaction_discount, trading_tax
    global below_50, price_gap_50_to_100, price_gap_100_to_500
    global price_gap_500_to_1000, price_gap_above_1000, triggered_limit_up_stocks
    global in_memory_intraday_data, data_lock
    
    price_gap_below_50 = below_50 
    if quit_flag['quit']: threading.Thread(target=show_exit_menu, daemon=True).start(); quit_flag['quit'] = False

    try: current_dt = datetime.strptime(current_time_str, "%H:%M")
    except ValueError: return
    trading_time = current_dt.time()
    trading_txt  = current_dt.strftime("%H:%M:%S")

    if not os.path.exists(nb_matrix_path): return
    with open(nb_matrix_path, "r", encoding="utf-8") as f: nb_dict = json.load(f)
    consolidated_symbols = nb_dict.get("consolidated_symbols", {})
    if not isinstance(consolidated_symbols, dict): return

    if in_memory_intraday_data:
        with data_lock: auto_intraday_data = in_memory_intraday_data.copy()
    else:
        if not os.path.exists("auto_intraday.json"): return
        with open("auto_intraday.json", "r", encoding="utf-8") as f: auto_intraday_data = json.load(f)

    stock_df = {}
    for sym in symbols_to_analyze:
        df = pd.DataFrame(auto_intraday_data.get(sym, [])).copy()
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S").dt.time
            df.sort_values("time", inplace=True); df.reset_index(drop=True, inplace=True)
        stock_df[sym] = df

    FIRST3_AVG_VOL: dict[str, float] = {}
    for sym, df in stock_df.items():
        if df.empty or "time" not in df.columns: FIRST3_AVG_VOL[sym] = 0; continue
        first3 = df[df["time"].astype(str).isin(["09:00:00", "09:01:00", "09:02:00"])]
        FIRST3_AVG_VOL[sym] = first3["volume"].mean() if not first3.empty else 0

    # ------------------------- 1. 觸發檢查 ------------------------------- #
    trigger_list = []
    if trading_time >= time(13, 0): print(f"⏰ {trading_txt} 已超過13:00，停止觸發。")
    else:
        for grp, syms in consolidated_symbols.items():
            if grp in group_positions and group_positions[grp]: continue
            for sym in syms:
                if sym not in symbols_to_analyze: continue
                df = stock_df[sym]
                if df.empty: continue
                row_now = df[df["time"] == trading_time]
                if row_now.empty: continue
                row_now = row_now.iloc[0]

                hit_limit = False
                if sym not in triggered_limit_up_stocks and row_now["high"] == row_now["漲停價"]:
                    prev_t = (datetime.combine(date.today(), trading_time) - timedelta(minutes=1)).time()
                    prev = df[df["time"] == prev_t]
                    prev_high = prev.iloc[0]["high"] if not prev.empty else None
                    if prev.empty or (prev_high is not None and prev_high < row_now["漲停價"]):
                        hit_limit = True; triggered_limit_up_stocks.add(sym)
                        for g2, gstat in group_positions.items():
                            if isinstance(gstat, dict) and gstat.get("trigger") == "拉高進場" and sym in consolidated_symbols.get(g2, []):
                                # 🟢 修正：無縫升級為漲停進場！保留原本的 start_time 與追蹤清單
                                gstat["trigger"] = "漲停進場"
                                gstat["wait_start"] = datetime.combine(date.today(), trading_time)
                                gstat["wait_counter"] = 0
                                gstat["leader"] = sym
                                msg = f"🚀 {sym} 衝上漲停，{g2} 族群從拉高無縫升級為漲停進場，保留發動起點！"
                                print(msg); message_log.append((trading_txt, msg))
                                hit_limit = False # 已經升級處理完畢，不需要當作新事件加入 trigger_list

                pull_up = False
                if row_now["2min_pct_increase"] >= 2:
                    avgv = FIRST3_AVG_VOL.get(sym, 0)
                    if avgv and row_now["volume"] > 1.3 * avgv: pull_up = True

                if hit_limit or pull_up: trigger_list.append({"symbol": sym, "group": grp, "condition": "limit_up" if hit_limit else "pull_up"})

    trigger_list.sort(key=lambda x: 0 if x["condition"] == "limit_up" else 1)
    for item in trigger_list:
        grp, cond_txt = item["group"], "漲停進場" if item["condition"] == "limit_up" else "拉高進場"
        if grp not in group_positions or not group_positions[grp]:
            group_positions[grp] = {"status": "觀察中", "trigger": cond_txt, "start_time": datetime.combine(date.today(), trading_time), "tracking": {}, "leader": None}
            print(f"族群 {grp} 進入觀察中（{cond_txt}）")
            if cond_txt == "漲停進場":
                group_positions[grp]["wait_start"] = datetime.combine(date.today(), trading_time)
                group_positions[grp]["wait_counter"] = 0
                group_positions[grp]["leader"] = item["symbol"]

    # ------------------------- 2. 更新追蹤清單 --------------------------- #
    for grp, gstat in group_positions.items():
        if not (isinstance(gstat, dict) and gstat["status"] == "觀察中"): continue
        track = gstat.setdefault("tracking", {})
        for sym in consolidated_symbols[grp]:
            df = stock_df[sym]
            if df.empty: continue
            row_now = df[df["time"] == trading_time]
            if row_now.empty: continue
            if row_now.iloc[0]["2min_pct_increase"] >= 1.5 and sym not in track:
                track[sym] = {"join_time": datetime.combine(date.today(), trading_time), "base_vol": row_now.iloc[0]["volume"], "base_rise": row_now.iloc[0]["rise"]}
                print(f"{YELLOW}{sym} 加入 {grp} 追蹤清單（2min↑1.5%）{RESET}")

    # ----------------------- 3. 領漲處理 ------------------------ #
    for grp, gstat in group_positions.items():
        if not (isinstance(gstat, dict) and gstat["status"] == "觀察中"): continue
        track = gstat.get("tracking", {})
        if not track: continue

        max_sym, max_rise = None, None
        for sym in track:
            df = stock_df[sym]
            row_now = df[df["time"] == trading_time]
            if row_now.empty: continue
            rise_now = row_now.iloc[0]["rise"]
            if max_rise is None or rise_now > max_rise: max_rise, max_sym = rise_now, sym

        if gstat.get("leader") is None:
            gstat["leader"] = max_sym
            print(f"{gstat.get('trigger')} {grp} 確立領漲：{max_sym}")
        else:
            if max_sym and max_sym != gstat["leader"]:
                print(f"領漲替換：{gstat['leader']} → {max_sym}")
                gstat["leader"] = max_sym; gstat["leader_peak"] = max_rise; gstat["leader_reversal_rise"] = max_rise
                gstat["status"] = "觀察中"; gstat.pop("wait_start", None); gstat["wait_counter"] = 0
                
                # 🟢 修正 2：實戰中領漲換人時，一併重置起始時間，破解 DTW 雙峰陷阱！
                gstat["start_time"] = datetime.combine(date.today(), trading_time)
                print(f"🚀 領漲替換觸發，時間窗重置，重新監控新領漲")
                
        lead_sym = gstat["leader"]
        if not lead_sym: continue
        df_lead = stock_df[lead_sym]
        idx_now = df_lead[df_lead["time"] == trading_time].index
        if idx_now.empty: continue
        idx_now = idx_now[0]
        
        if "wait_start" not in gstat:
            if idx_now - 1 >= 0 and df_lead.loc[idx_now, "high"] <= df_lead.loc[idx_now - 1, "high"]:
                gstat["wait_start"] = datetime.combine(date.today(), trading_time)
                gstat["wait_counter"] = 0
                gstat["leader_reversal_rise"] = df_lead.loc[idx_now, "rise"]
                print(f"{gstat.get('trigger')} {grp} 領漲 {lead_sym} 反轉，開始等待")

    for grp, gstat in group_positions.items():
        if not (isinstance(gstat, dict) and gstat["status"] == "觀察中"): continue
        if "wait_start" not in gstat: continue

        lead = gstat.get("leader")
        if lead and gstat.get("leader_reversal_rise") is not None:
            df_lead = stock_df.get(lead, pd.DataFrame())
            row_now = df_lead[df_lead["time"] == trading_time]
            if not row_now.empty and row_now.iloc[0]["rise"] > gstat["leader_reversal_rise"]:
                print(f"🚀 領漲股 {lead} 再創新高，觸發自我替換")
                gstat["leader_reversal_rise"] = row_now.iloc[0]["rise"]
                gstat["status"] = "觀察中"; gstat.pop("wait_start", None); gstat["wait_counter"] = 0
                continue

        gstat["wait_counter"] += 1
        print(f"{gstat.get('trigger')} {grp} 等待第 {gstat['wait_counter']} 分鐘")
        
        leader_sym = gstat.get("leader") 
        if leader_sym and "tracking" in gstat:
            # 💡 修正：實戰版同步將 15 分鐘改為 2 分鐘，斬斷舊領漲的波型包袱！
            window_start_live = max(time(9,0), (gstat["start_time"] - timedelta(minutes=2)).time())
            to_remove_live = []
            for s_sym in list(gstat["tracking"].keys()):
                if s_sym == leader_sym: continue
                c_corr = calculate_dtw_pearson(stock_df[leader_sym], stock_df[s_sym], window_start_live, trading_time)
                if c_corr < 0.4: to_remove_live.append(s_sym)
            for s_sym in to_remove_live:
                gstat["tracking"].pop(s_sym, None)
                print(f"{RED}[滾動剔除] {s_sym} 相似度降至 {c_corr:.2f} < 0.4{RESET}")

    # ---------------- 4. 等待完成 → 篩選股票進場 ---------------- #
    def _vol_break(sym: str, join_time: datetime) -> bool:
        # 🟢 統一使用安全、輕量的 .any() 判定
        df = stock_df[sym]
        if df.empty: return False
        avgv = FIRST3_AVG_VOL.get(sym, 0)
        if avgv == 0: return False
        later = df[df["time"] >= join_time.time()]
        return (later["volume"] >= 1.5 * avgv).any()

    def _rise_peak_flat(sym: str, join_time: datetime) -> bool:
        df = stock_df[sym]
        if df.empty: return False
        sub = df[(df["time"] >= join_time.time()) & (df["time"] <= trading_time)]
        if sub.empty: return False
        pk_idx = sub["rise"].idxmax()
        pk_v = sub.loc[pk_idx, "rise"]
        ltr = sub[sub.index > pk_idx]
        return (ltr["rise"] <= pk_v + 0.5).all()

    groups_ready = []
    now_f = datetime.combine(date.today(), trading_time)
    for grp, gstat in group_positions.items():
        if not (isinstance(gstat, dict) and gstat["status"] == "觀察中"): continue
        if "wait_start" not in gstat: continue
        if (now_f - gstat["wait_start"]).total_seconds() / 60 >= wait_minutes - 1:
            groups_ready.append(grp)

    for grp in groups_ready:
        gstat = group_positions[grp]
        filtered_track = gstat.get("tracking", {}).copy()
        leader_sym = gstat.get("leader")
        
        if not filtered_track:
            print(f"{grp} 相似度篩選後無候選 → 取消觀察"); group_positions[grp] = False; continue

        eligible = []
        for sym, info in filtered_track.items():
            if sym == leader_sym: continue
            
            # 🔍 檢查關卡 1：爆量條件
            if not _vol_break(sym, gstat["start_time"]):
                msg = f"🔍 [除錯] {sym} 剔除：等待期間未出現爆量 1.5 倍的 K 棒"
                print(msg); message_log.append((trading_txt, msg))
                continue
                
            # 🔍 檢查關卡 2：不過高條件
            if not _rise_peak_flat(sym, gstat["start_time"]):
                msg = f"🔍 [除錯] {sym} 剔除：等待期間突破了前高 (破壞作頭型態)"
                print(msg); message_log.append((trading_txt, msg))
                continue
            
            df = stock_df[sym]
            row_now = df[df["time"] == trading_time]
            if row_now.empty: continue
            rise_now = row_now.iloc[0]["rise"]
            
            # 🔍 檢查關卡 3：漲幅限制
            if not (-1 <= rise_now <= 6):
                msg = f"🔍 [除錯] {sym} 剔除：當前漲幅 {rise_now:.2f}% 不在 -1% ~ 6% 之間"
                print(msg); message_log.append((trading_txt, msg))
                continue
                
            entry_price = row_now.iloc[0]["close"]
            # 🔍 檢查關卡 4：資金上限
            if entry_price > capital_per_stock * 1.5:
                msg = f"🔍 [除錯] {sym} 剔除：股價 {entry_price:.2f} 超出資金上限"
                print(msg); message_log.append((trading_txt, msg))
                continue

            try:
                contract = api.Contracts.Stocks.TSE.get(sym) or api.Contracts.Stocks.OTC.get(sym)
                if not contract: continue
                is_day_trade_yes = False
                if hasattr(contract, 'day_trade'):
                    val = contract.day_trade
                    if (isinstance(val, str) and val == "Yes") or (hasattr(val, 'value') and val.value == "Yes") or val == sj.constant.DayTrade.Yes:
                        is_day_trade_yes = True
                if not is_day_trade_yes:
                    msg = f"🔍 [除錯] {sym} 剔除：今日不可當沖"
                    print(msg); message_log.append((trading_txt, msg))
                    continue
            except: continue

            eligible.append({"symbol": sym, "rise": rise_now, "row": row_now.iloc[0]})
            msg = f"🎯 [除錯] {sym} 成功通過所有濾網，加入進場候選名單！"
            print(msg); message_log.append((trading_txt, msg))

        if not eligible:
            print(f"{grp} 等待完成，但無符合條件股票 → 取消觀察"); group_positions[grp] = False; continue

        eligible.sort(key=lambda x: x["rise"], reverse=True)
        
        # =============== 🟢 修改：奇數進場中位數，偶數進場中位數後一位 (實戰版) ===============
        total_eligible = len(eligible)
        if total_eligible == 1:
            target_idx = 0
        elif total_eligible % 2 == 1:
            target_idx = total_eligible // 2
        else:
            target_idx = total_eligible // 2
        # =====================================================================

        chosen = eligible[target_idx]
        msg = f"🎯 [選股策略] 共有 {total_eligible} 檔候選，採奇數中位數/偶數中位數後一，選擇第 {target_idx + 1} 名進場！"
        print(msg); message_log.append((trading_txt, msg))
        # =====================================================================

        row = chosen["row"]
        entry_px = row["close"]
        shares = round((capital_per_stock * 10000) / (entry_px * 1000))
        sell_amt = shares * entry_px * 1000
        fee = int(sell_amt * (transaction_fee * 0.01) * (transaction_discount * 0.01))
        tax = int(sell_amt * (trading_tax * 0.01))

        if entry_px < 10: gap, tick = price_gap_below_50, 0.01
        elif entry_px < 50: gap, tick = price_gap_below_50, 0.05
        elif entry_px < 100: gap, tick = price_gap_50_to_100, 0.1
        elif entry_px < 500: gap, tick = price_gap_100_to_500, 0.5
        elif entry_px < 1000: gap, tick = price_gap_500_to_1000, 1
        else: gap, tick = price_gap_above_1000, 5

        highest_on_entry = row["highest"] or entry_px
        if (highest_on_entry - entry_px) * 1000 < gap: stop_type, stop_thr = "price_difference", entry_px + gap / 1000
        else: stop_type, stop_thr = "over_high", highest_on_entry + tick

        limit_up = row["漲停價"]
        if limit_up < 10: tick_for_limit = 0.01
        elif limit_up < 50: tick_for_limit = 0.05
        elif limit_up < 100: tick_for_limit = 0.1
        elif limit_up < 500: tick_for_limit = 0.5
        elif limit_up < 1000: tick_for_limit = 1
        else: tick_for_limit = 5

        ceiling = limit_up - 2 * tick_for_limit
        if stop_thr > ceiling: stop_thr, stop_type = ceiling, "ceiling_limit"

        planned_exit = None
        if hold_minutes is not None:
            expected_exit = datetime.combine(date.today(), trading_time) + timedelta(minutes=hold_minutes)
            if expected_exit.time() >= time(13, 26): message_log.append((trading_txt, f"{YELLOW}預計出場時間超過 13:26，持有時間自動轉為 F{RESET}"))
            else: planned_exit = expected_exit

        with data_lock:
            open_positions[chosen['symbol']] = {
                'entry_price': entry_px, 'shares': shares, 'sell_cost': sell_amt,
                'entry_fee': fee, 'stop_loss': stop_thr, 'planned_exit': planned_exit
            }

        stock_code_str = chosen["symbol"]
        with open("twse_stocks_by_market.json", "r", encoding="utf-8") as f: stock_market_map = json.load(f)
        contract = getattr(api.Contracts.Stocks.TSE, "TSE" + stock_code_str) if stock_code_str in stock_market_map.get("TSE", {}) else getattr(api.Contracts.Stocks.OTC, "OTC" + stock_code_str)

        order = api.Order(price=0, quantity=shares, action=sj.constant.Action.Sell, price_type=sj.constant.StockPriceType.MKT, order_type=sj.constant.OrderType.IOC, order_lot=sj.constant.StockOrderLot.Common, daytrade_short=True, account=api.stock_account)
        trade = safe_place_order(api, contract, order)

        t_cmd = tp.TouchCmd(code=f"{stock_code_str}", close=tp.Price(price=stop_thr, trend="Equal"))
        o_cmd = tp.OrderCmd(code=f"{stock_code_str}", order=sj.Order(price=0, quantity=shares, action="Buy", order_type="ROD", price_type="MKT"))
        tcond = tp.TouchOrderCond(t_cmd, o_cmd)
        
        with data_lock:
            if stock_code_str not in to.contracts: to.contracts[stock_code_str] = contract
            safe_add_touch_condition(to, tcond)
            group_positions[grp] = "已進場"

        msg = f"{GREEN}進場！{stock_code_str} {shares}張 成交價 {entry_px:.2f} 停損價 {stop_thr:.2f}{RESET}"
        write_trade_log(f"進場！股票：{stock_code_str}，張數：{shares}，成交價：{entry_px:.2f}，停損價：{stop_thr:.2f}")
        print(msg); message_log.append((trading_txt, msg))

    message_log.sort(key=lambda x: x[0])
    for t, m in message_log: print(f"[{t}] {m}")
    message_log.clear()
# ------------------ 交易程式：開始交易 ------------------
def start_trading(mode='full', wait_minutes=None, hold_minutes=None):
    """
    mode:
        'full' – 第一次執行：正常詢問等待/持有分鐘。
        'post' – 盤後遞迴呼叫：沿用上一輪 wait_minutes / hold_minutes，不再詢問。
    """
    client, api_key = init_fugle_client()

    # ===== 處置股過濾=====
    matrix_dict_analysis = load_matrix_dict_analysis()
    fetch_disposition_stocks(client, matrix_dict_analysis)   # ① 先更新 Disposition.json
    disposition_stocks = load_disposition_stocks()           # ② 讀最新處置股
    purge_disposition_from_nb(disposition_stocks)           # ③ 刪 nb_matrix_dict 中的處置股
    # ====================

    symbols_to_analyze = load_symbols_to_analyze()
    stop_trading = False
    max_symbols_to_fetch = 20

    group_symbols = load_group_symbols()
    if not group_symbols:
        print("沒有加載到任何族群資料，請確認 nb_matrix_dict.json 的存在與內容。")
        return
    consolidated_symbols = group_symbols.get('consolidated_symbols', {})
    if not consolidated_symbols:
        print("沒有找到 'consolidated_symbols'，請確認資料結構。")
        return
    group_positions = {group: False for group in consolidated_symbols.keys()}

    # 🟢 修正：嚴格定義時間段 (8:30, 9:00, 13:26, 13:30)
    now = datetime.now()
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')
    pre_market_start = now.replace(hour=8, minute=30, second=0, microsecond=0)
    market_start     = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_exit      = now.replace(hour=13, minute=26, second=0, microsecond=0)
    market_end       = now.replace(hour=13, minute=30, second=0, microsecond=0)

    print("開始進行盤中交易狀態判定...")

    # ==================== 狀態 1：【凌晨盤後】( < 08:30 ) ====================
    if now < pre_market_start:
        wait_sec = (pre_market_start - now).total_seconds()
        print(f"目前為 {now_str}，尚未到達盤前更新時間。將休眠 {wait_sec:.0f} 秒至 08:30...")
        time_module.sleep(wait_sec)
        start_trading(mode, wait_minutes, hold_minutes) # 睡醒重啟
        return

    # ==================== 狀態 2：【下午盤後】( >= 13:30 ) ====================
    elif now >= market_end:
        tomorrow_pre_market = (now + timedelta(days=1)).replace(hour=8, minute=30, second=0, microsecond=0)
        wait_sec = (tomorrow_pre_market - now).total_seconds()
        print(f"目前為 {now_str}，今日已收盤。系統將休眠 {wait_sec:.0f} 秒至明日 08:30...")
        time_module.sleep(wait_sec)
        start_trading(mode, wait_minutes, hold_minutes) # 睡醒重啟
        return

    # ==================== 狀態 3：【盤前更新】( 08:30 ~ 08:59:59 ) ====================
    elif pre_market_start <= now < market_start:
        print(f"目前為 {now_str}，進入盤前時間，開始更新日K線資料...")
        
        # ---------- 取得 / 比對日 K（僅在盤前更新） ----------
        existing_auto_daily_data = {}
        if os.path.exists('auto_daily.json'):
            with open('auto_daily.json', 'r', encoding='utf-8') as f:
                try:
                    existing_auto_daily_data = json.load(f)
                except json.JSONDecodeError:
                    existing_auto_daily_data = {}
        else:
            print("auto_daily.json 不存在，將建立新的。")

        print("開始取得日K線數據並與現有資料比對...")
        auto_daily_data = {}
        data_is_same = True
        initial_api_count = 0
        symbols_fetched = 0

        for symbol in symbols_to_analyze[:max_symbols_to_fetch]:
            if initial_api_count >= 55:
                print("已達到55次API請求，休息1分鐘...")
                time_module.sleep(60)
                initial_api_count = 0
            daily_kline_df = fetch_daily_kline_data(client, symbol, days=2)
            initial_api_count += 1
            if daily_kline_df.empty:
                print(f"無法取得 {symbol} 的日K數據，跳過。")
                continue
            daily_kline_data = daily_kline_df.to_dict(orient='records')
            auto_daily_data[symbol] = daily_kline_data
            existing_data = existing_auto_daily_data.get(symbol)
            if existing_data != daily_kline_data:
                data_is_same = False
                print(f"{symbol} 的日K數據與現有資料不同，將更新資料。")
                existing_auto_daily_data[symbol] = daily_kline_data
            else:
                print(f"{symbol} 的日K數據與現有資料相同，跳過更新。")
            symbols_fetched += 1

        if not data_is_same:
            remaining_symbols = symbols_to_analyze[max_symbols_to_fetch:]
            print(f"發現前 {max_symbols_to_fetch} 支股票的日K數據有更新，開始取得剩餘股票的日K數據並更新。")
            for symbol in remaining_symbols:
                if initial_api_count >= 55:
                    print("已達到55次API請求，休息1分鐘...")
                    time_module.sleep(60)
                    initial_api_count = 0
                daily_kline_df = fetch_daily_kline_data(client, symbol, days=2)
                initial_api_count += 1
                if daily_kline_df.empty:
                    print(f"無法取得 {symbol} 的日K數據，跳過。")
                    continue
                daily_kline_data = daily_kline_df.to_dict(orient='records')
                auto_daily_data[symbol] = daily_kline_data
                existing_data = existing_auto_daily_data.get(symbol)
                if existing_data != daily_kline_data:
                    print(f"{symbol} 的日K數據與現有資料不同，將更新資料。")
                    existing_auto_daily_data[symbol] = daily_kline_data
                else:
                    print(f"{symbol} 的日K數據與現有資料相同，跳過更新。")

        if symbols_fetched < max_symbols_to_fetch:
            print(f"注意：僅取得了 {symbols_fetched} 支股票的日K數據。")

        with open('auto_daily.json', 'w', encoding='utf-8') as f:
            json.dump(existing_auto_daily_data, f, ensure_ascii=False, indent=4)

        print(f"{YELLOW}已更新 auto_daily.json。{RESET}")
        print(f"{YELLOW}盤前更新完成。{RESET}")

        # 更新完畢後，睡到 09:00
        now = datetime.now()
        wait_seconds = (market_start - now).total_seconds()
        if wait_seconds > 0:
            print(f"等待 {wait_seconds/60:.1f} 分鐘直到開盤開始盤中交易...")
            time_module.sleep(wait_seconds)

        print("開盤！自動切換到盤中交易模式…")
        start_trading(mode='post', wait_minutes=wait_minutes, hold_minutes=hold_minutes)
        return

    # ==================== 狀態 4：【盤中監控】( 09:00 ~ 13:29:59 ) ====================
    elif market_start <= now < market_end:
        print(f"目前為 {now_str}，盤中交易時間，直接載入歷史資料。")

        # 🟢 修正：盤中直接讀取 auto_daily.json，不再發送日 K 的 API 浪費時間
        existing_auto_daily_data = {}
        if os.path.exists('auto_daily.json'):
            with open('auto_daily.json', 'r', encoding='utf-8') as f:
                try:
                    existing_auto_daily_data = json.load(f)
                except json.JSONDecodeError:
                    existing_auto_daily_data = {}
        else:
            print("⚠️ 找不到 auto_daily.json，今日昨收價可能為 0。")

        fetch_time = datetime.now() - timedelta(minutes=1)
        trading_day = fetch_time.strftime('%Y-%m-%d')
        
        # 整理昨收價
        yesterday_close_prices = {}
        for symbol in symbols_to_analyze:
            daily_data = existing_auto_daily_data.get(symbol, [])
            if not daily_data:
                yesterday_close_prices[symbol] = 0
            else:
                sorted_daily_data = sorted(daily_data, key=lambda x: x['date'], reverse=True)
                if len(sorted_daily_data) > 1:
                    now2 = datetime.now()
                    weekday = now2.weekday()
                    if 0 <= weekday <= 4 and 8 <= now2.hour < 15:
                        yesterday_close = sorted_daily_data[0].get('close', 0)
                    else:
                        yesterday_close = sorted_daily_data[1].get('close', 0)
                else:
                    yesterday_close = sorted_daily_data[0].get('close', 0)
                yesterday_close_prices[symbol] = yesterday_close

        # ---------- 一分K初次補齊 ----------
        t_fetch_hist = time_module.perf_counter()
        print("🔁 [歷史] 開始補齊今日 09:00 到目前為止的一分K資料...")
        
        market_real_end = now.replace(hour=13, minute=30, second=0, microsecond=0)
        if now < market_real_end:
            full_intraday_end = (now - timedelta(minutes=1)).strftime('%H:%M')
        else:
            full_intraday_end = "13:30"

        auto_intraday_data = {}
        initial_api_count = 0
        with ThreadPoolExecutor(max_workers=200) as executor:
            future_to_symbol = {}
            for symbol in symbols_to_analyze:
                if initial_api_count >= 200:
                    time_module.sleep(60)
                    initial_api_count = 0
                yc = yesterday_close_prices.get(symbol, 0)
                if yc == 0:
                    continue
                future = executor.submit(
                    fetch_intraday_data,
                    client=client,
                    symbol=symbol,
                    trading_day=trading_day,
                    yesterday_close_price=yc,
                    start_time="09:00",
                    end_time=full_intraday_end
                )
                future_to_symbol[future] = symbol
                initial_api_count += 1
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                df = future.result()
                if df.empty:
                    continue
                updated_records = []
                records = df.to_dict(orient='records')
                for i, candle in enumerate(records):
                    updated_candle = calculate_2min_pct_increase_and_highest(candle, records[:i])
                    updated_records.append(updated_candle)
                df = pd.DataFrame(updated_records)
                auto_intraday_data[symbol] = df.to_dict(orient='records')

        print(f"✅ [歷史] 補齊完成，耗時：{time_module.perf_counter() - t_fetch_hist:.2f} 秒")
        t_save_json = time_module.perf_counter()
        save_auto_intraday_data(auto_intraday_data)
        initialize_triggered_limit_up(auto_intraday_data)

        # ---------- 盤中主迴圈 ----------
        threading.Thread(target=check_quit_flag_loop, daemon=True).start()

        # 初始化盤中狀態
        has_exited = False
        current_position = None
        hold_time = 0
        message_log = []
        already_entered_stocks = []
        stop_loss_triggered = False
        final_check_active = False
        final_check_count = 0
        in_waiting_period = False
        waiting_time = 0
        leader = None
        tracking_stocks = set()
        previous_rise_values = {}
        leader_peak_rise = None
        leader_rise_before_decline = None
        first_condition_one_time = None
        can_trade = True
        exit_live_done = False

        while not stop_trading:
            now_loop = datetime.now()

            # 🟢 修正：13:26 尾盤強制出場
            if now_loop >= market_exit and not exit_live_done:
                print(f"🔍 13:26 觸發：檢查觸價委託單，目前尚有 {len(to.conditions)} 檔股票在觸價委託中。")
                exit_trade_live()
                exit_live_done = True

            # 🟢 修正：13:30 收盤，準時結束今日迴圈
            if now_loop >= market_end:
                print(f"\n⏰ 時間已達 13:30，今日盤中交易結束。")
                break
            # 🟢 新增：檢查持倉是否達到設定的「持有時間」
            with data_lock:
                for sym, pos_info in list(open_positions.items()):
                    planned_exit = pos_info.get('planned_exit')
                    if planned_exit and now_loop >= planned_exit:
                        print(f"{RED}⏰ {sym} 已達設定持有時間，執行自動平倉！{RESET}")
                        write_trade_log(f"⏰ {sym} 已達持有時間，自動平倉。")
                        pos_info['planned_exit'] = None  # 防止在執行期間重複觸發
                        # 呼叫現成的平倉函數 (撤銷觸價單 + 市價賣出)
                        threading.Thread(target=close_one_stock, args=(sym,), daemon=True).start()

            now_sec = datetime.now().second
            time_module.sleep(60 - now_sec)

            fetch_time = datetime.now() - timedelta(minutes=1)
            trading_day = fetch_time.strftime('%Y-%m-%d')
            fetch_time_str = fetch_time.strftime('%H:%M')
            if fetch_time.time() > market_end.time():
                fetch_time_str = "13:30"
            
            t_fetch_realtime = time_module.perf_counter()
            print(f"{YELLOW}⏱ [即時] 開始取得 {fetch_time_str} 的一分K資料...{RESET}")
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("\n" + "=" * 50)
            print(f"\n{timestamp} 市場開盤中，取得 {fetch_time_str} 分的即時一分K數據。")

            updated_intraday_data = {}
            with ThreadPoolExecutor(max_workers=200) as executor:
                future_to_symbol = {}
                for symbol in symbols_to_analyze:
                    yc = yesterday_close_prices.get(symbol, 0)
                    if yc == 0:
                        continue
                    fut = executor.submit(
                        fetch_realtime_intraday_data,
                        client=client,
                        symbol=symbol,
                        trading_day=trading_day,
                        yesterday_close_price=yc,
                        start_time=fetch_time_str,
                        end_time=fetch_time_str
                    )
                    future_to_symbol[fut] = symbol
                for fut in as_completed(future_to_symbol):
                    sym = future_to_symbol[fut]
                    df = fut.result()
                    if df.empty:
                        continue
                    candle = df.to_dict(orient='records')[0]
                    candle = calculate_2min_pct_increase_and_highest(candle, auto_intraday_data.get(sym, []))
                    if '漲停價' in candle:
                        candle['漲停價'] = truncate_to_two_decimals(candle['漲停價'])
                    updated_intraday_data.setdefault(sym, []).append(candle)

            for sym, lst in updated_intraday_data.items():
                auto_intraday_data.setdefault(sym, []).extend(lst)
                auto_intraday_data[sym] = auto_intraday_data[sym][-1000:]

            print(f"✅ [即時] 一分K取得完成，耗時：{time_module.perf_counter() - t_fetch_realtime:.2f} 秒")
            save_auto_intraday_data(auto_intraday_data)

            process_live_trading_logic(
                symbols_to_analyze,
                fetch_time_str,
                wait_minutes,
                hold_minutes,
                message_log,
                False,
                has_exited,
                current_position,
                hold_time,
                already_entered_stocks,
                stop_loss_triggered,
                final_check_active,
                final_check_count,
                in_waiting_period,
                waiting_time,
                leader,
                tracking_stocks,
                previous_rise_values,
                leader_peak_rise,
                leader_rise_before_decline,
                first_condition_one_time,
                can_trade,
                group_positions
            )
            # 每分鐘計算即時損益，並發送訊號給 UI 持倉監控表
            with data_lock:
                ui_data = []
                for sym, pos_info in open_positions.items():
                    current_price = pos_info['entry_price']
                    if sym in auto_intraday_data and not pd.DataFrame(auto_intraday_data[sym]).empty:
                        current_price = pd.DataFrame(auto_intraday_data[sym]).iloc[-1]['close']
                    
                    buy_cost = pos_info['shares'] * current_price * 1000
                    profit = pos_info.get('sell_cost', 0) - buy_cost - pos_info.get('entry_fee', 0)
                    
                    ui_data.append({
                        "symbol": sym,
                        "entry_price": pos_info['entry_price'],
                        "current_price": current_price,
                        "profit": profit,
                        "stop_loss": pos_info.get('stop_loss', '未設定')
                    })
            global cached_portfolio_data
            cached_portfolio_data = ui_data  # 存入緩存

            try:
                ui_dispatcher.portfolio_updated.emit(ui_data)
            except Exception:
                pass

        # 🟢 修正：跳出 13:30 迴圈後，直接準備隔日重新喚醒
        now = datetime.now()
        tomorrow_pre_market = (now + timedelta(days=1)).replace(hour=8, minute=30, second=0, microsecond=0)
        wait_sec = (tomorrow_pre_market - now).total_seconds()
        print(f"今日交易已完成。系統將自動休眠 {wait_sec:.0f} 秒至明日 08:30 進行盤前更新...")
        time_module.sleep(wait_sec)
        start_trading(mode, wait_minutes, hold_minutes)
        return

# ==============================================================================
# 🟢 新增：族群連動分析引擎 (支援宏觀與微觀動態時間窗)
# ==============================================================================
class CorrelationAnalysisThread(QThread):
    finished_signal = pyqtSignal(list)

    def __init__(self, mode, wait_mins):
        super().__init__()
        self.mode = mode
        self.wait_mins = wait_mins

    def run(self):
        result_data = []
        try:
            # 🟢 修正 1：使用內建函數讀取，避開 JSON 巢狀結構陷阱
            _, history_data = load_kline_data()
            groups = load_matrix_dict_analysis()
            
            # 🟢 修正 2：同步過濾處置股，避免無效運算
            dispo = load_disposition_stocks() 
            
            for grp_name, stocks in groups.items():
                stock_dfs = {}
                valid_stocks = [s for s in stocks if s not in dispo]
                
                for s in valid_stocks:
                    if s in history_data and history_data[s]:
                        import pandas as pd
                        df = pd.DataFrame(history_data[s])
                        if not df.empty and 'time' in df.columns:
                            df['time'] = pd.to_datetime(df['time'], format="%H:%M:%S").dt.time
                            stock_dfs[s] = df
                
                if len(stock_dfs) < 2: continue
                
                if self.mode == "macro":
                    # --- [A] 宏觀模式 ---
                    leader = None; max_rise = -999
                    for s, df in stock_dfs.items():
                        s_max = df['rise'].max()
                        if s_max > max_rise: max_rise = s_max; leader = s
                    
                    if not leader: continue
                    w_start = time(9,0); w_end = time(13,30)
                    for s in stock_dfs.keys():
                        if s == leader: continue
                        sim = calculate_dtw_pearson(stock_dfs[leader], stock_dfs[s], w_start, w_end)
                        result_data.append({'group': grp_name, 'leader': leader, 'follower': s, 
                                            'window': '09:00~13:30 (全天)', 'similarity': sim})
                                            
                elif self.mode == "micro":
                    # --- [B] 微觀模式：模擬實戰動態時間窗 ---
                    leader = None; start_time = None; wait_counter = 0; in_waiting = False
                    leader_peak_rise = -999
                    intercept_w_start = None; intercept_w_end = None
                    tracking_stocks = set(stock_dfs.keys())
                    
                    time_range = [ (datetime.combine(date.today(), time(9,0)) + timedelta(minutes=i)).time() for i in range(271) ]
                    
                    for current_t in time_range:
                        cur_max_sym = None; cur_max_rise = -999
                        for s in tracking_stocks:
                            df_s = stock_dfs[s]
                            row = df_s[df_s['time'] == current_t]
                            if not row.empty:
                                r = row.iloc[0]['rise']
                                if r > cur_max_rise: cur_max_rise = r; cur_max_sym = s
                        
                        if not cur_max_sym: continue
                        
                        # 領漲替換邏輯
                        if leader != cur_max_sym:
                            leader = cur_max_sym
                            start_time = current_t
                            leader_peak_rise = cur_max_rise
                            in_waiting = False
                            wait_counter = 0
                        else:
                            if cur_max_rise < leader_peak_rise and not in_waiting:
                                in_waiting = True
                                wait_counter = 0
                            elif cur_max_rise > leader_peak_rise:
                                leader_peak_rise = cur_max_rise
                                in_waiting = False 
                        
                        if in_waiting:
                            wait_counter += 1
                            if wait_counter >= self.wait_mins:
                                intercept_w_end = current_t
                                intercept_w_start = max(time(9,0), (datetime.combine(date.today(), start_time) - timedelta(minutes=2)).time())
                                break
                    
                    if leader and intercept_w_start and intercept_w_end:
                        window_str = f"{intercept_w_start.strftime('%H:%M')}~{intercept_w_end.strftime('%H:%M')}"
                        for s in tracking_stocks:
                            if s == leader: continue
                            sim = calculate_dtw_pearson(stock_dfs[leader], stock_dfs[s], intercept_w_start, intercept_w_end)
                            result_data.append({'group': grp_name, 'leader': leader, 'follower': s, 
                                                'window': window_str, 'similarity': sim})

        except Exception as e:
            print(f"分析失敗: {e}")
            
        self.finished_signal.emit(result_data)


# ==================================================================================
# ==================== PyQt5 專業圖形介面 (GUI) 類別定義 ===========================
# ==================================================================================
class BaseDialog(QDialog):
    """自訂深色彈出視窗基底"""
    def __init__(self, title, size=(400, 300)):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(*size)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.setStyleSheet("""
            QDialog { background-color: #1E1E1E; color: white; }
            QLabel { font-size: 14px; font-weight: bold; color: #E0E0E0; }
            QLineEdit, QComboBox { background-color: #2C2C2C; color: white; border: 1px solid #555; padding: 5px; border-radius: 4px;}
            QPushButton { font-size: 14px; border-radius: 5px; }
            /* 🟢 修正：強制下拉選單的列表為白底黑字，選中時為藍底白字 */
            QComboBox QAbstractItemView {
                background-color: white;
                color: black;
                selection-background-color: #2980B9;
                selection-color: white;
            }
        """)

class LoginDialog(BaseDialog):
    def __init__(self):
        super().__init__("登入/修改帳戶", (450, 350))
        from PyQt5.QtWidgets import QHBoxLayout, QFileDialog # 🟢 引入所需元件
        
        layout = QFormLayout(self)
        self.e_api = QLineEdit(shioaji_logic.TEST_API_KEY)
        self.e_sec = QLineEdit(shioaji_logic.TEST_API_SECRET)
        self.e_ca = QLineEdit(shioaji_logic.CA_CERT_PATH)
        self.e_pw = QLineEdit(shioaji_logic.CA_PASSWORD)
        
        layout.addRow("API Key:", self.e_api)
        layout.addRow("API Secret:", self.e_sec)
        
        # 🟢 新增：包含「瀏覽...」按鈕的水平佈局
        ca_layout = QHBoxLayout()
        ca_layout.addWidget(self.e_ca)
        btn_browse = QPushButton("📁 瀏覽...")
        btn_browse.setStyleSheet("background-color: #34495E; color: white; padding: 4px 10px; border-radius: 4px;")
        btn_browse.clicked.connect(self.browse_cert)
        ca_layout.addWidget(btn_browse)
        
        layout.addRow("憑證路徑:", ca_layout)
        layout.addRow("憑證密碼:", self.e_pw)

        btn = QPushButton("💾 儲存修改")
        btn.setStyleSheet("background-color: #27AE60; color: white; padding: 10px; margin-top: 15px;")
        btn.clicked.connect(self.save)
        layout.addRow(btn)

    # 🟢 新增：開啟檔案選擇對話框
    def browse_cert(self):
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "選擇憑證檔案", "", "Certificate Files (*.p12 *.pfx);;All Files (*)")
        if path:
            self.e_ca.setText(path)

    def save(self):
        update_variable("shioaji_logic.py", "TEST_API_KEY", self.e_api.text())
        update_variable("shioaji_logic.py", "TEST_API_SECRET", self.e_sec.text())
        update_variable("shioaji_logic.py", "CA_CERT_PATH", self.e_ca.text(), is_raw=True)
        update_variable("shioaji_logic.py", "CA_PASSWORD", self.e_pw.text())
        
        global api, to
        try:
            print(f"{YELLOW}⏳ 正在套用新帳戶重新登入...{RESET}")
            api.login(api_key=self.e_api.text(), secret_key=self.e_sec.text())
            api.activate_ca(ca_path=self.e_ca.text(), ca_passwd=self.e_pw.text())
            to = tp.TouchOrderExecutor(api)
            print(f"{GREEN}✅ 帳戶資料已更新並登入成功！{RESET}")
        except Exception as e:
            print(f"{RED}❌ 登入失敗，請檢查憑證是否正確: {e}{RESET}")
            
        self.accept()

class TradeDialog(BaseDialog):
    def __init__(self):
        super().__init__("啟動盤中交易", (350, 250))
        layout = QFormLayout(self)
        self.w_wait = QLineEdit("5")
        self.w_hold = QLineEdit("F")
        layout.addRow("等待時間 (分鐘):", self.w_wait)
        layout.addRow("持有時間 (分鐘, F=強制):", self.w_hold)
        
        btn = QPushButton("▶ 啟動監控")
        btn.setStyleSheet("background-color: #C0392B; color: white; padding: 10px; margin-top: 10px;")
        btn.clicked.connect(self.run_trade)
        layout.addRow(btn)

        btn_login = QPushButton("🔑 登入/修改帳戶")
        btn_login.setStyleSheet("background-color: #2980B9; color: white; padding: 10px;")
        btn_login.clicked.connect(lambda: LoginDialog().exec_())
        layout.addRow(btn_login)

    def run_trade(self):
        try: w = int(self.w_wait.text())
        except: return QMessageBox.critical(self, "錯誤", "等待時間需為整數")
        h_str = self.w_hold.text().strip().upper()
        try: 
            h = None if h_str == 'F' else int(h_str)
            if h is not None and h < 1:
                return QMessageBox.critical(self, "錯誤", "持有時間最少需為 1 分鐘 (或輸入 F)")
        except: return QMessageBox.critical(self, "錯誤", "持有時間格式錯誤")
        
        self.accept()
        threading.Thread(target=start_trading, args=('full', w, h), daemon=True).start()

# ==============================================================================
# 🟢 新增：盤後數據與連動分析模組 (繼承 BaseDialog - 修復版)
# ==============================================================================
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView, QComboBox

class CorrelationResultDialog(BaseDialog):
    def __init__(self, result_data, parent=None):
        super().__init__("🧬 族群連動分析掃描結果", (850, 600))
        layout = QVBoxLayout(self)
        self.result_data = result_data  # 🟢 將資料存起來供匯出使用
        
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["族群", "最終領漲股", "跟漲股", "結算時間窗", "DTW相似度", "評估結果"])
        self.table.setStyleSheet("QTableWidget { background-color: #1e1e1e; color: #d4d4d4; gridline-color: #444; }"
                                 "QHeaderView::section { background-color: #2C3E50; color: white; font-weight: bold; padding: 5px; }")
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        self.table.setRowCount(len(result_data))
        for i, row_data in enumerate(result_data):
            self.table.setItem(i, 0, QTableWidgetItem(str(row_data['group'])))
            self.table.setItem(i, 1, QTableWidgetItem(str(row_data['leader'])))
            self.table.setItem(i, 2, QTableWidgetItem(str(row_data['follower'])))
            self.table.setItem(i, 3, QTableWidgetItem(str(row_data['window'])))
            
            sim_val = row_data['similarity']
            sim_item = QTableWidgetItem(f"{sim_val:.3f}")
            if sim_val >= 0.4:
                sim_item.setForeground(QColor("#2ECC40")) # 綠色
                eval_text = "✅ 合格 (連動)"
            else:
                sim_item.setForeground(QColor("#FF4136")) # 紅色
                eval_text = "❌ 剔除 (背離)"
                
            self.table.setItem(i, 4, sim_item)
            self.table.setItem(i, 5, QTableWidgetItem(eval_text))
            
        layout.addWidget(self.table)

        # 🟢 新增：匯出 CSV 按鈕
        btn_export = QPushButton("📥 匯出為 CSV 檔")
        btn_export.setStyleSheet("background-color: #27AE60; color: white; font-size: 14px; font-weight: bold; padding: 10px; border-radius: 5px;")
        btn_export.clicked.connect(self.export_to_csv)
        layout.addWidget(btn_export)

    # 🟢 新增：CSV 匯出邏輯
    def export_to_csv(self):
        from PyQt5.QtWidgets import QFileDialog
        import csv
        path, _ = QFileDialog.getSaveFileName(self, "儲存 CSV 檔案", "族群連動分析結果.csv", "CSV 檔案 (*.csv)")
        if path:
            try:
                # 使用 utf-8-sig 編碼，確保用 Excel 打開不會出現中文亂碼
                with open(path, 'w', encoding='utf-8-sig', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["族群", "最終領漲股", "跟漲股", "結算時間窗", "DTW相似度", "評估結果"])
                    for r in self.result_data:
                        sim = r['similarity']
                        eval_text = "合格 (連動)" if sim >= 0.4 else "剔除 (背離)"
                        writer.writerow([r['group'], r['leader'], r['follower'], r['window'], f"{sim:.3f}", eval_text])
                QMessageBox.information(self, "匯出成功", f"資料已成功儲存至：\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "匯出失敗", f"寫入檔案時發生錯誤：\n{e}")

class CorrelationConfigDialog(BaseDialog):
    def __init__(self, parent=None):
        super().__init__("設定連動分析參數", (400, 200))
        
        layout = QVBoxLayout(self)

        self.mode_combo = QComboBox()
        # 🟢 修正：順序對調，先顯示 [A] 宏觀連動，再顯示 [B] 微觀模擬
        self.mode_combo.addItems(["[A] 一整天宏觀連動 (09:00~13:30)", "[B] 實戰微觀模擬 (動態攔截結算窗)"])
        self.mode_combo.setStyleSheet("background-color: #2d2d2d; color: white; padding: 5px;")
        
        self.wait_spin = QLineEdit("5")
        self.wait_spin.setStyleSheet("background-color: #2d2d2d; color: white; padding: 5px;")

        form_layout = QFormLayout()
        form_layout.addRow(QLabel("分析模式："), self.mode_combo)
        form_layout.addRow(QLabel("微觀等待時間 (分鐘)："), self.wait_spin)
        layout.addLayout(form_layout)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.setStyleSheet("QPushButton { background-color: #34495E; color: white; padding: 6px 15px; border-radius: 4px; }")
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_settings(self):
        # 🟢 修正：由於選單順序互換，index 0 現在對應 "macro"，index 1 對應 "micro"
        mode = "macro" if self.mode_combo.currentIndex() == 0 else "micro"
        try: wait_time = int(self.wait_spin.text())
        except: wait_time = 5
        return mode, wait_time

class AnalysisMenuDialog(BaseDialog):
    def __init__(self, parent=None):
        # 🟢 修正：正確呼叫 BaseDialog 的 title 與 size
        super().__init__("盤後數據與分析中心", (320, 200))
        
        # 🟢 修正：宣告專屬的 Layout
        layout = QVBoxLayout(self)
        
        self.choice = None
        
        btn_avg_high = QPushButton("📈 計算平均過高 (原有功能)")
        btn_avg_high.setStyleSheet("QPushButton { background-color: #34495E; color: white; font-size: 14px; padding: 12px; border-radius: 6px; font-weight: bold;}")
        btn_avg_high.clicked.connect(self.choose_avg_high)
        
        btn_correlation = QPushButton("🧬 族群連動分析掃描 (新功能)")
        btn_correlation.setStyleSheet("QPushButton { background-color: #8E44AD; color: white; font-size: 14px; padding: 12px; border-radius: 6px; font-weight: bold;}")
        btn_correlation.clicked.connect(self.choose_correlation)

        layout.addWidget(btn_avg_high)
        layout.addSpacing(10)
        layout.addWidget(btn_correlation)

    def choose_avg_high(self): self.choice = 'avg_high'; self.accept()
    def choose_correlation(self): self.choice = 'correlation'; self.accept()

from PyQt5.QtWidgets import QListView
class SimulateDialog(BaseDialog):
    def __init__(self):
        super().__init__("自選進場模式 (回測)", (400, 250))
        layout = QFormLayout(self)
        self.w_grp = QComboBox()
        
        # 🟢 終極解法：強制給它一個獨立的 QListView 並直接塞入死樣式
        view = QListView()
        view.setStyleSheet("""
            QListView {
                background-color: white;
                color: black;
                font-weight: bold;
                font-size: 14px;
            }
            QListView::item:selected {
                background-color: #2980B9;
                color: white;
            }
        """)
        self.w_grp.setView(view)
        
        self.w_grp.addItem("所有族群")
        self.w_grp.addItems(list(load_matrix_dict_analysis().keys()))
        self.w_wait = QLineEdit("5")
        self.w_hold = QLineEdit("F")
        
        layout.addRow("分析族群:", self.w_grp)
        layout.addRow("等待時間 (分鐘):", self.w_wait)
        layout.addRow("持有時間 (分鐘, F=尾盤):", self.w_hold)
        
        btn = QPushButton("▶ 開始分析")
        btn.setStyleSheet("background-color: #E67E22; color: white; padding: 10px;")
        btn.clicked.connect(self.run_sim)
        layout.addRow(btn)

    def run_sim(self):
        grp = self.w_grp.currentText()
        try: w = int(self.w_wait.text())
        except: return QMessageBox.critical(self, "錯誤", "等待時間需為整數")
        h_str = self.w_hold.text().strip().upper()
        try: 
            h = None if h_str == 'F' else int(h_str)
            if h is not None and h < 1:
                return QMessageBox.critical(self, "錯誤", "持有時間最少需為 1 分鐘 (或輸入 F)")
        except: return QMessageBox.critical(self, "錯誤", "持有時間格式錯誤")
        self.accept()

        def _logic():
            ui_dispatcher.progress_visible.emit(True) # 顯示進度條
            mat = load_matrix_dict_analysis()
            d_kline, i_kline = load_kline_data()
            dispo = load_disposition_stocks()
            
            if grp != "所有族群": 
                if grp not in mat: 
                    ui_dispatcher.progress_visible.emit(False)
                    return print(f"❌ 找不到族群: {grp}")
                print(f"\n🎯 正在分析單一族群：{grp}")
                syms = [s for s in mat[grp] if s not in dispo]
                data = initialize_stock_data(syms, d_kline, i_kline)
                
                # 🟢 綁定進度回傳
                def cb(p, msg): ui_dispatcher.progress_updated.emit(p, msg)
                process_group_data(data, w, h, mat, verbose=True, progress_callback=cb)
            else: 
                print("\n🌐 啟動全市場族群掃描...")
                tp_sum, rate_list = 0, []
                total = len(mat)
                for i, (g, s) in enumerate(mat.items()):
                    print(f"\n正在分析族群：{g}")
                    data = initialize_stock_data([x for x in s if x not in dispo], d_kline, i_kline)
                    
                    # 🟢 計算全市場綜合進度
                    def cb(p, msg): 
                        overall = int((i/total)*100 + (p/total))
                        ui_dispatcher.progress_updated.emit(overall, f"[{g}] {msg}")
                        
                    tp, ap = process_group_data(data, w, h, mat, verbose=True, progress_callback=cb)
                    if tp is not None: tp_sum += tp; rate_list.append(ap)
                if rate_list: 
                    avg_rate = sum(rate_list)/len(rate_list)
                    c = GREEN if tp_sum < 0 else (RED if tp_sum > 0 else "")
                    print(f"\n{c}================================")
                    print(f"{c}💰 當日總利潤：{int(tp_sum)} 元")
                    print(f"{c}📈 平均報酬率：{avg_rate:.2f}%")
                    print(f"{c}================================{RESET}")
                else: 
                    print("\n⚠️ 當日無任何交易產生。")
                    
            ui_dispatcher.progress_visible.emit(False) # 隱藏進度條

        threading.Thread(target=_logic, daemon=True).start()

class MaximizeDialog(BaseDialog):
    def __init__(self):
        super().__init__("極大化利潤模式", (400, 350))
        layout = QFormLayout(self)
        self.e_grp = QComboBox()
        
        # 🟢 終極解法：強制給它一個獨立的 QListView 並直接塞入死樣式 (解決白底白字)
        view = QListView()
        view.setStyleSheet("""
            QListView {
                background-color: white;
                color: black;
                font-weight: bold;
                font-size: 14px;
            }
            QListView::item:selected {
                background-color: #2980B9;
                color: white;
            }
        """)
        self.e_grp.setView(view)
        
        self.e_grp.addItems(list(load_matrix_dict_analysis().keys()))
        self.e_ws = QLineEdit("3"); self.e_we = QLineEdit("5")
        self.e_hs = QLineEdit("10"); self.e_he = QLineEdit("20")

        layout.addRow("族群名稱:", self.e_grp)
        layout.addRow("等待時間起始 (分):", self.e_ws)
        layout.addRow("等待時間結束 (分):", self.e_we)
        layout.addRow("持有時間起始 (0代表F):", self.e_hs)
        layout.addRow("持有時間結束 (0代表F):", self.e_he)

        btn = QPushButton("▶ 執行暴力破解")
        btn.setStyleSheet("background-color: #8E44AD; color: white; padding: 10px;")
        btn.clicked.connect(self.run_max)
        layout.addRow(btn)

    def run_max(self):
        grp = self.e_grp.currentText()
        try: 
            ws, we, hs, he = int(self.e_ws.text()), int(self.e_we.text()), int(self.e_hs.text()), int(self.e_he.text())
            if (hs != 0 and hs < 1) or (he != 0 and he < 1):
                return QMessageBox.critical(self, "錯誤", "持有時間最少需為 1 分鐘 (0 代表 F)")
        except: return QMessageBox.critical(self, "錯誤", "時間參數必須是整數")
        self.accept()

        def _logic():
            ui_dispatcher.progress_visible.emit(True) # 顯示進度條
            mat = load_matrix_dict_analysis()
            data = initialize_stock_data([s for s in mat[grp] if s not in load_disposition_stocks()], *load_kline_data())
            results_df = pd.DataFrame(columns=['等待時間', '持有時間', '總利潤', '平均報酬率'])
            
            total_steps = (we - ws + 1) * (he - hs + 1)
            step = 0
            for w in range(ws, we + 1):
                for h in range(hs, he + 1):
                    h_val = None if h == 0 else h
                    print(f"分析中：等待 {w} 分鐘、持有 {'F' if h_val is None else h_val} 分鐘")
                    
                    # 🟢 計算多重迴圈綜合進度
                    def cb(p, msg):
                        overall = int((step/total_steps)*100 + (p/total_steps))
                        ui_dispatcher.progress_updated.emit(overall, f"(測數:{step+1}/{total_steps}) {msg}")
                        
                    tp, ap = process_group_data(data, w, h_val, mat, verbose=False, progress_callback=cb)
                    new_row = pd.DataFrame([{'等待時間': w, '持有時間': 'F' if h_val is None else h_val, '總利潤': float(tp or 0), '平均報酬率': float(ap or 0)}])
                    results_df = pd.concat([results_df, new_row], ignore_index=True)
                    step += 1
                    
            if not results_df.empty:
                best = results_df.loc[results_df['總利潤'].idxmax()]
                print(f"\n🏆 最佳組合：等待 {best['等待時間']} 分 / 持有 {best['持有時間']} 分 / 利潤：{int(best['總利潤'])} 元\n")
                
            ui_dispatcher.progress_visible.emit(False) # 隱藏進度條
        threading.Thread(target=_logic, daemon=True).start()

class AverageHighDialog(BaseDialog):
    def __init__(self):
        super().__init__("計算平均過高", (350, 200))
        layout = QVBoxLayout(self)
        
        b1 = QPushButton("單一族群分析")
        b1.setStyleSheet("background-color: #2980B9; color: white; padding: 10px;")
        b1.clicked.connect(self.run_single)

        b2 = QPushButton("全部族群分析")
        b2.setStyleSheet("background-color: #16A085; color: white; padding: 10px;")
        b2.clicked.connect(self.run_all)

        layout.addWidget(b1); layout.addWidget(b2)

    # 修改單一族群與全市場掃描邏輯
    def run_single(self):
        grp, ok = QInputDialog.getItem(self, "選擇", "選擇族群:", list(load_matrix_dict_analysis().keys()), 0, False)
        if ok and grp:
            self.accept()
            def _logic():
                ui_dispatcher.progress_visible.emit(True)
                def cb(p, msg): ui_dispatcher.progress_updated.emit(p, msg)
                calculate_average_over_high(grp, progress_callback=cb)
                ui_dispatcher.progress_visible.emit(False)
            threading.Thread(target=_logic, daemon=True).start()

    def run_all(self):
        self.accept()
        def _logic():
            ui_dispatcher.progress_visible.emit(True)
            groups = load_matrix_dict_analysis()
            avgs = []
            total = len(groups)
            for i, g in enumerate(groups.keys()):
                def cb(p, msg): 
                    overall = int((i/total)*100 + (p/total))
                    ui_dispatcher.progress_updated.emit(overall, f"[{g}] {msg}")
                avg = calculate_average_over_high(g, progress_callback=cb)
                if avg: avgs.append(avg)
            if avgs: print(f"\n全部族群的平均過高間隔：{sum(avgs)/len(avgs):.2f} 分鐘")
            ui_dispatcher.progress_visible.emit(False)
        threading.Thread(target=_logic, daemon=True).start()

class SettingsDialog(BaseDialog):
    def __init__(self):
        super().__init__("系統參數設定", (450, 600))
        self.setStyleSheet("""
            QDialog, QWidget, QScrollArea { background-color: #F5F5F5; color: black; }
            QLabel { font-size: 14px; font-weight: bold; color: black; }
            QLineEdit, QComboBox { background-color: white; color: black; border: 1px solid #999; padding: 5px; border-radius: 4px;}
            QPushButton { font-size: 14px; border-radius: 5px; color: white; background-color: #27AE60; }
        """)
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        form = QFormLayout(w)
        
        self.e_cap = QLineEdit(str(capital_per_stock)); form.addRow("投入資本額 (萬元):", self.e_cap)
        self.e_fee = QLineEdit(str(transaction_fee)); form.addRow("手續費 (%):", self.e_fee)
        self.e_dis = QLineEdit(str(transaction_discount)); form.addRow("手續費折數 (%):", self.e_dis)
        self.e_tax = QLineEdit(str(trading_tax)); form.addRow("證交稅 (%):", self.e_tax)
        
        form.addRow(QLabel("--- 停損價差 ---"))
        self.e_50 = QLineEdit(str(below_50)); form.addRow("50元以下 (元):", self.e_50)
        self.e_100 = QLineEdit(str(price_gap_50_to_100)); form.addRow("50~100元 (元):", self.e_100)
        self.e_500 = QLineEdit(str(price_gap_100_to_500)); form.addRow("100~500元 (元):", self.e_500)
        self.e_1000 = QLineEdit(str(price_gap_500_to_1000)); form.addRow("500~1000元 (元):", self.e_1000)
        self.e_above = QLineEdit(str(price_gap_above_1000)); form.addRow("1000元以上 (元):", self.e_above)
        
        self.reentry = QComboBox()
        self.reentry.addItems(["關閉", "開啟"])
        self.reentry.setCurrentIndex(1 if allow_reentry_after_stop_loss else 0)
        form.addRow("停損再進場:", self.reentry)

        scroll.setWidget(w)
        layout.addWidget(scroll)

        btn = QPushButton("💾 儲存所有設定")
        btn.setStyleSheet("background-color: #27AE60; color: white; padding: 10px;")
        btn.clicked.connect(self.save)
        layout.addWidget(btn)

    def save(self):
        global capital_per_stock, transaction_fee, transaction_discount, trading_tax
        global below_50, price_gap_50_to_100, price_gap_100_to_500, price_gap_500_to_1000, price_gap_above_1000, allow_reentry_after_stop_loss
        try:
            capital_per_stock = int(self.e_cap.text())
            transaction_fee, transaction_discount, trading_tax = float(self.e_fee.text()), float(self.e_dis.text()), float(self.e_tax.text())
            below_50, price_gap_50_to_100 = float(self.e_50.text()), float(self.e_100.text())
            price_gap_100_to_500, price_gap_500_to_1000, price_gap_above_1000 = float(self.e_500.text()), float(self.e_1000.text()), float(self.e_above.text())
            allow_reentry_after_stop_loss = (self.reentry.currentIndex() == 1)
            save_settings()
            print("✅ 系統參數已儲存！")
            self.accept()
        except: QMessageBox.critical(self, "錯誤", "數字格式不正確")

class GroupManagerDialog(BaseDialog):
    def __init__(self):
        super().__init__("管理股票族群", (600, 500))
        layout = QVBoxLayout(self)
        self.text = QTextEdit(); self.text.setReadOnly(True)
        self.text.setStyleSheet("font-family: Consolas; font-size: 14px;")
        layout.addWidget(self.text)
        
        btn_layout = QHBoxLayout()
        # ✅ 修正：強制設定按鈕背景色為深色，字體為白色
        b1 = QPushButton("➕ 新增族群")
        b1.setStyleSheet("background-color: #2C3E50; color: white;")
        b1.clicked.connect(self.add_grp)
        
        b2 = QPushButton("➕ 新增個股")
        b2.setStyleSheet("background-color: #2C3E50; color: white;")
        b2.clicked.connect(self.add_stk)
        b3 = QPushButton("🗑️ 刪除族群"); b3.setStyleSheet("background-color:#C0392B;"); b3.clicked.connect(self.del_grp)
        b4 = QPushButton("🗑️ 刪除個股"); b4.setStyleSheet("background-color:#C0392B;"); b4.clicked.connect(self.del_stk)
        for b in [b1, b2, b3, b4]: btn_layout.addWidget(b)
        layout.addLayout(btn_layout)
        self.refresh()

    def refresh(self):
        self.text.clear()
        groups = load_matrix_dict_analysis()
        load_twse_name_map()
        for g, s in groups.items():
            self.text.append(f"📁 族群: {g}")
            for code in s: self.text.append(f"   - {code} {get_stock_name(code)}")
            self.text.append("-" * 40)

    def add_grp(self):
        grp, ok = QInputDialog.getText(self, "新增", "輸入新族群名稱:")
        if ok and grp:
            g = load_matrix_dict_analysis()
            if grp not in g: g[grp] = []
            save_matrix_dict(g); self.refresh(); print(f"已新增族群: {grp}")

    def add_stk(self):
        g = load_matrix_dict_analysis()
        grp, ok = QInputDialog.getItem(self, "新增", "要加入哪個族群？", list(g.keys()), 0, False)
        if ok and grp:
            code, ok2 = QInputDialog.getText(self, "新增", "輸入股票代號:")
            if ok2 and code and code not in g[grp]:
                g[grp].append(code); save_matrix_dict(g); self.refresh(); print(f"個股 {code} 已加入 {grp}")

    def del_grp(self):
        g = load_matrix_dict_analysis()
        grp, ok = QInputDialog.getItem(self, "刪除", "選擇要刪除的族群:", list(g.keys()), 0, False)
        if ok and grp:
            del g[grp]; save_matrix_dict(g); self.refresh(); print(f"已刪除族群: {grp}")

    def del_stk(self):
        g = load_matrix_dict_analysis()
        grp, ok = QInputDialog.getItem(self, "刪除", "從哪個族群刪除？", list(g.keys()), 0, False)
        if ok and grp and g[grp]:
            code, ok2 = QInputDialog.getItem(self, "刪除", "選擇股票:", g[grp], 0, False)
            if ok2 and code:
                g[grp].remove(code); save_matrix_dict(g); self.refresh(); print(f"已移除個股: {code}")

class DispositionDialog(BaseDialog):
    def __init__(self):
        super().__init__("處置股清單", (300, 400))
        layout = QVBoxLayout(self)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setStyleSheet("font-family: Consolas; font-size: 14px;")
        layout.addWidget(self.text)
        try:
            with open('Disposition.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                stocks = data if isinstance(data, list) else data.get("stock_codes", [])
                if stocks:
                    load_twse_name_map()
                    for i, code in enumerate(stocks, 1): 
                        name = get_stock_name(code)
                        self.text.append(f"{i}. {code} {name}")
                else:
                    self.text.append("目前無處置股。")
        except: self.text.append("無法讀取處置股檔案。")

# 補回 1.8.0.8 的畫圖函數轉接器 (若你原本有 view_kline_data 函數，請確保它在上方已被定義)
def trigger_matplotlib_chart():
    try:
        symbol_to_group = {s: g for g, syms in load_matrix_dict_analysis().items() for s in syms}
        print("📈 正在開啟走勢圖...")
        # ✅ 修正：Matplotlib 必須在主執行緒執行，直接呼叫函數，不再使用 threading.Thread
        view_kline_data('./intraday_kline_data.json', symbol_to_group)
    except Exception as e:
        print(f"畫圖發生錯誤: {e}")

class EmergencyDialog(BaseDialog):
    def __init__(self):
        super().__init__("緊急平倉中心", (350, 200))
        layout = QVBoxLayout(self)
        
        b1 = QPushButton("💥 一鍵全部平倉 (市價)")
        b1.setStyleSheet("background-color: #E74C3C; font-size: 14px; font-weight: bold; color: white; padding: 10px;")
        b1.clicked.connect(lambda: [self.accept(), threading.Thread(target=exit_trade_live, daemon=True).start()])
        
        # 🟢 修正：強制加上 background-color 與 color: white
        b2 = QPushButton("🎯 指定單一股票平倉")
        b2.setStyleSheet("background-color: #2980B9; color: white; padding: 10px; font-size: 14px;")
        b2.clicked.connect(self.single_close)

        b3 = QPushButton("❌ 強制關閉程式 (不平倉)")
        b3.setStyleSheet("background-color: #7F8C8D; color: white; padding: 10px; font-size: 14px;")
        b3.clicked.connect(lambda: os._exit(0))

        for b in [b1, b2, b3]: layout.addWidget(b)

    def single_close(self):
        code, ok = QInputDialog.getText(self, "單一平倉", "請輸入股票代號:")
        if ok and code:
            self.accept()
            threading.Thread(target=close_one_stock, args=(code,), daemon=True).start()

from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView

class PortfolioMonitorDialog(BaseDialog):
    def __init__(self):
            super().__init__("📊 即時持倉監控面板", (650, 300))
            layout = QVBoxLayout(self)
            
            self.table = QTableWidget(0, 5)
            self.table.setHorizontalHeaderLabels(["股票代號", "進場價", "即時現價", "未實現損益", "停損價"])
            
            # 🟢 修正 1：隱藏左側空白的垂直行號標題欄
            self.table.verticalHeader().setVisible(False)
            
            # 🟢 修正 2：設定欄寬為等比例自動填滿，並且「鎖死不可拖拉」
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            # 如果你希望每一欄等寬且使用者無法用滑鼠去拉動分隔線，加上這行：
            self.table.horizontalHeader().setSectionsClickable(False)
            for i in range(5):
                self.table.horizontalHeader().setSectionResizeMode(i, QHeaderView.Fixed)
                self.table.setColumnWidth(i, 120)  # 給定固定寬度 (650寬度/5)
                
            # 🟢 修正 3：關閉整個表格的編輯功能（只能看不能改）
            self.table.setEditTriggers(QTableWidget.NoEditTriggers)

            self.table.setStyleSheet("""
                QTableWidget { background-color: #1E1E1E; color: white; gridline-color: #444444; font-size: 15px; }
                QHeaderView::section { background-color: #2C3E50; color: white; font-weight: bold; padding: 8px; }
            """)
            layout.addWidget(self.table)
            
            ui_dispatcher.portfolio_updated.connect(self.update_table)

            # 🟢 新增：視窗建立的瞬間，立刻用背景已經算好的快取資料畫圖
            if cached_portfolio_data:
                self.update_table(cached_portfolio_data)

    @pyqtSlot(list)
    def update_table(self, data_list):
        self.table.setRowCount(len(data_list))
        for row, data in enumerate(data_list):
            self.table.setItem(row, 0, QTableWidgetItem(str(data['symbol'])))
            self.table.setItem(row, 1, QTableWidgetItem(f"{data['entry_price']:.2f}"))
            self.table.setItem(row, 2, QTableWidgetItem(f"{data['current_price']:.2f}"))
            
            # 損益顯示顏色
            profit_item = QTableWidgetItem(f"{int(data['profit'])} 元")
            if data['profit'] > 0: profit_item.setForeground(QColor("#FF4136"))
            elif data['profit'] < 0: profit_item.setForeground(QColor("#2ECC40"))
            self.table.setItem(row, 3, profit_item)
            
            self.table.setItem(row, 4, QTableWidgetItem(f"{data['stop_loss']:.2f}" if isinstance(data['stop_loss'], float) else str(data['stop_loss'])))


# ==================== 主視窗 (MainWindow) ====================
class QuantMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("交易程式 1.8.5.4 - 當沖量化終端")
        self.resize(1100, 700)
        self.setStyleSheet("background-color: #121212;")

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # ─── 左側導航欄 ───
        sidebar = QFrame()
        sidebar.setFixedWidth(230)
        sidebar.setStyleSheet("background-color: #1E1E1E; border-radius: 10px;")
        vbox = QVBoxLayout(sidebar)
        vbox.setSpacing(15)

        title = QLabel("日內cat 量化終端")
        title.setStyleSheet("color: #FFFFFF; font-size: 20px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        vbox.addWidget(title)

        def make_btn(text, callback, color="#2C3E50"):
            btn = QPushButton(text)
            # 🟢 修正：動態決定反白顏色。如果底色剛好撞衫，就給它一個更亮的鋼鐵藍色
            hover_color = "#4B6584" if color.upper() == "#34495E" else "#34495E"
            
            btn.setStyleSheet(f"""
                QPushButton {{ 
                    background-color: {color}; 
                    color: white; 
                    font-size: 15px; 
                    padding: 12px; 
                    border-radius: 6px; 
                    font-weight: bold;
                }} 
                QPushButton:hover {{ 
                    background-color: {hover_color}; 
                }}
            """)
            btn.clicked.connect(callback)
            return btn

        vbox.addWidget(make_btn("▶ 啟動盤中交易", lambda: TradeDialog().exec_()))
        # 動態監控面板按鈕
        # 改為呼叫專屬的非阻塞顯示方法
        vbox.addWidget(make_btn("📊 即時持倉監控", self.show_portfolio_monitor, "#8E44AD"))
        
        # 回測子選單 (直接做成多個按鈕)
        lbl_bt = QLabel("── 回測分析 ──")
        lbl_bt.setStyleSheet("color: #888888; font-size: 12px; margin-top: 10px;")
        vbox.addWidget(lbl_bt)
        
        # 🟢 替換：將原本的計算平均過高，升級為綜合的「盤後數據與分析中心」
        vbox.addWidget(make_btn("📊 盤後數據與分析", self.open_analysis_menu, "#34495E"))
        
        vbox.addWidget(make_btn("🎯 自選進場模式", lambda: SimulateDialog().exec_(), "#34495E"))
        vbox.addWidget(make_btn("💰 極大化利潤", lambda: MaximizeDialog().exec_(), "#34495E"))

        # ── 系統與數據管理 ──
        lbl_sys = QLabel("── 系統管理 ──")
        lbl_sys.setStyleSheet("color: #888888; font-size: 12px; margin-top: 10px;")
        vbox.addWidget(lbl_sys)
        
        vbox.addWidget(make_btn("📁 管理股票族群", lambda: GroupManagerDialog().exec_()))
        vbox.addWidget(make_btn("🔄 更新 K 線數據", lambda: threading.Thread(target=update_kline_data, daemon=True).start()))
        
        # 🆕 補回的兩個按鈕！
        vbox.addWidget(make_btn("📄 查看處置股", lambda: DispositionDialog().exec_(), "#27AE60"))
        vbox.addWidget(make_btn("📈 畫圖查看走勢", trigger_matplotlib_chart, "#27AE60"))
        
        vbox.addWidget(make_btn("⚙️ 參數設定", lambda: SettingsDialog().exec_()))
        
        vbox.addStretch()
        vbox.addWidget(make_btn("🛑 緊急/手動平倉", lambda: EmergencyDialog().exec_(), "#C0392B"))

        # ─── 右側終端機與進度條 (垂直佈局) ───
        right_vbox = QVBoxLayout()
        right_vbox.setContentsMargins(0, 0, 0, 0)
        
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: #000000; color: #FFFFFF; font-family: Consolas; font-size: 14px; border: 1px solid #333333; padding: 10px;")

        # 🟢 1.8.5 進度條：已修改佈局至終端機正下方，並美化樣式
        from PyQt5.QtWidgets import QProgressBar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                background-color: #1E1E1E;
                color: white;
                text-align: center;
                height: 22px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #2980B9;
            }
        """)
        self.progress_bar.hide() # 預設隱藏，等待觸發

        right_vbox.addWidget(self.console, stretch=1)
        right_vbox.addWidget(self.progress_bar)

        layout.addWidget(sidebar)
        layout.addLayout(right_vbox, stretch=1)

        # 啟動輸出重導向
        self.stream = EmittingStream()
        self.stream.textWritten.connect(self.normal_output)
        sys.stdout = self.stream
        sys.stderr = self.stream

        # 🟢 綁定全域訊號，以接收背景執行緒的進度更新
        ui_dispatcher.progress_updated.connect(self.update_progress)
        ui_dispatcher.progress_visible.connect(self.progress_bar.setVisible)

    @pyqtSlot(int, str)
    def update_progress(self, percent, msg):
        self.progress_bar.setValue(percent)
        self.progress_bar.setFormat(f"{msg}  %p%" if msg else "%p%")

    @pyqtSlot(str)
    def normal_output(self, text):
        html_text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        html_text = html_text.replace(' ', '&nbsp;').replace('\n', '<br>')
        import re
        color_map = [
            (r'\x1b\[(?:31|91)m|\033\[(?:31|91)m', '#FF4136'), # 紅 (停損/出場/負利潤)
            (r'\x1b\[(?:32|92)m|\033\[(?:32|92)m', '#2ECC40'), # 綠 (進場/正利潤)
            (r'\x1b\[(?:33|93)m|\033\[(?:33|93)m', '#FFDC00'), # 黃 (觸發進場)
            (r'\x1b\[(?:34|94)m|\033\[(?:34|94)m', '#0074D9'), # 藍
        ]
        for pattern, color in color_map:
            html_text = re.sub(pattern, f'<span style="color: {color}; font-weight: bold;">', html_text)
        html_text = re.sub(r'\x1b\[0m|\x1b\[39m|\033\[0m', '</span>', html_text)
        
        self.console.moveCursor(QTextCursor.End)
        self.console.insertHtml(html_text)
        self.console.moveCursor(QTextCursor.End)
    
    # 🟢 新增：以非阻塞 (Modeless) 方式開啟監控面板
    def show_portfolio_monitor(self):
        # 檢查是否已經有開啟的面板，避免重複開好幾個視窗
        if not hasattr(self, 'monitor_dialog') or not self.monitor_dialog.isVisible():
            self.monitor_dialog = PortfolioMonitorDialog()
            self.monitor_dialog.show()  # 使用 show() 取代 exec_() 就不會鎖死主視窗
        else:
            # 如果已經開啟了，就把視窗拉到最上層
            self.monitor_dialog.raise_()
            self.monitor_dialog.activateWindow()

    # =========================================================================
    # 🟢 新增：盤後數據與連動分析選單控制
    # =========================================================================
    def open_analysis_menu(self):
        dialog = AnalysisMenuDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            if dialog.choice == 'avg_high':
                # 呼叫您原本的 AverageHighDialog
                AverageHighDialog().exec_() 
            elif dialog.choice == 'correlation':
                self.open_correlation_config()

    def open_correlation_config(self):
        if not os.path.exists("intraday_kline_data.json"):
            # 您也可以在這裡印出警告到 console，或是用 QMessageBox
            print("\x1b[31m⚠️ 錯誤：找不到 intraday_kline_data.json 歷史資料！\x1b[0m")
            return
            
        config_dialog = CorrelationConfigDialog(self)
        if config_dialog.exec_() == QDialog.Accepted:
            mode, wait_mins = config_dialog.get_settings()
            mode_text = "微觀實戰模擬" if mode == "micro" else "全天宏觀連動"
            
            # 將執行訊息印在終端機上
            print(f"\x1b[35m🧬 啟動族群連動分析 ({mode_text}, 等待: {wait_mins}分)...\x1b[0m")
            
            # 啟動背景執行緒，避免介面卡死
            self.corr_thread = CorrelationAnalysisThread(mode, wait_mins)
            self.corr_thread.finished_signal.connect(self.show_correlation_results)
            self.corr_thread.start()

    def show_correlation_results(self, result_data):
        print(f"\x1b[32m✅ 族群連動分析完成，共產出 {len(result_data)} 筆數據。\x1b[0m")
        # 顯示非阻塞的結果表格
        self.corr_dialog = CorrelationResultDialog(result_data, self)
        self.corr_dialog.show()

# ==================== 程式進入點 ====================
def main():
    try:
        load_settings()
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        
        print("=" * 60)
        print("✅ 系統核心模組載入完成 (PyQt5 專業版)")
        print("✅ 安全鎖、非同步I/O、斷線重連機制 已全面啟動")
        print("👉 請點擊左側面板按鈕進行操作")
        print("=" * 60)

        window = QuantMainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        # ✅ 如果發生未處理異常，彈出一個訊息框而不是直接閃退
        from PyQt5.QtWidgets import QMessageBox
        error_msg = traceback.format_exc()
        print(error_msg) # 同時印在控制台
        # 如果 app 已經建立就彈窗，沒建立就印出來
        if 'app' in locals():
            QMessageBox.critical(None, "系統崩潰", f"發生致命錯誤：\n{error_msg}")
        else:
            print(f"致命錯誤：{error_msg}")

if __name__ == "__main__":
    main()