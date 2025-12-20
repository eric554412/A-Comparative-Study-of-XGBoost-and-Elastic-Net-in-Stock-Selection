import os
import io
import time
import json
from datetime import datetime
from dateutil.rrule import rrule, DAILY
import requests
import pandas as pd
from tqdm import tqdm

START_DATE = '2020-01-01'
END_DATE = '2024-12-31'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(SAVE_DIR, 'raw_csv')
FINAL_PARQUET = os.path.join(SAVE_DIR, 'twse_miindex_allbut0999_daily.parquet')

# 用來取得 Cookie 的頁面 URL 以及 API 端點
PAGE_URL = "https://www.twse.com.tw/zh/trading/historical/mi-index.html"
RWD_JSON = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date={date}&type={typ}&response=json"

MI_TYPE = "ALLBUT0999"
SLEEP_SEC = 3.0          # 每抓完一天休息 3 秒
RETRY = 3                # 端點重試次數
RETRY_SLEEP = 2.0        # 單次重試間隔

HEADERS_PAGE = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}

HEADERS_API = {
    "User-Agent": HEADERS_PAGE["User-Agent"],
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": HEADERS_PAGE["Accept-Language"],
    "Referer": PAGE_URL,
    "Connection": "keep-alive",
    "X-Requested-With": "XMLHttpRequest",
}

COLS = [
    "date","證券代號","證券名稱",
    "成交股數","成交筆數","成交金額",
    "開盤價","最高價","最低價","收盤價",
    "漲跌(+/-)","漲跌價差",
    "最後揭示買價","最後揭示買量",
    "最後揭示賣價","最後揭示賣量",
    "本益比"
]


# 資料處理函式
def ensure_dir():
    '''確保資料夾存在'''
    os.makedirs(SAVE_DIR, exist_ok = True)
    os.makedirs(RAW_DIR, exist_ok = True)


def dt_iter(start, end):
    s = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()
    for d in rrule(DAILY, dtstart = s, until = e):
        yield d.date()


def clean_numeric(s):
    '''清理數值欄位'''
    if pd.isna(s):
        return pd.NA
    s = str(s).replace(',', '').strip()
    if s in ["", "--", "—", "NAN", "nan"]:
        return pd.NA
    if s in ["+", "-"]:
        return s
    try:
        if '.' in s:
            return float(s)
        return int(s)
    except Exception:
        return s


def normalize_columns(df: pd.DataFrame):
    mapping = {
        "最後買價": "最後揭示買價",
        "最後買量": "最後揭示買量",
        "最後賣價": "最後揭示賣價",
        "最後賣量": "最後揭示賣量",
    }
    return df.rename(columns = mapping)


def force_cols(df: pd.DataFrame, trade_date_str: str):
    '''強制欄位存在並排序'''
    df.insert(0, "date", trade_date_str)
    for c in COLS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[COLS]
    for c in ["成交股數","成交筆數","成交金額",
              "開盤價","最高價","最低價","收盤價",
              "漲跌價差","最後揭示買價","最後揭示買量",
              "最後揭示賣價","最後揭示賣量","本益比"]:
        df[c] = df[c].map(clean_numeric)    
    return df


def parse_rwd_json(payload: dict, trade_date_str: str):
    tables = payload.get("tables") or []
    if not tables:
        return None
    for t in tables:
        fields = t.get("fields") or []
        data = t.get("data") or []
        # 找到有我們要的欄位
        if data and {"證券代號", "證券名稱"}.issubset(set(fields)):
            df = pd.DataFrame(data, columns = fields)
            df = normalize_columns(df)
            df = force_cols(df, trade_date_str)
            return df
    return None


# 爬蟲工具函式
def session_get_with_retry(session: requests.Session, url: str, headers: dict, time_out = 20, expect = None):
    '''遇到 307/403/429 或 Content-Type 不符就重試，期間重新熱身一次。'''
    for attempt in range(1, RETRY + 1):
        r = session.get(url = url, headers = headers, timeout = time_out, allow_redirects = True)
        ct = r.headers.get("Content-Type", "")
        bad_status = (
            r.status_code in (403, 429) or # 被擋 / 頻率太快
            r.status_code >= 500 or        # 伺服器錯誤
            r.status_code == 307           # 重導到錯誤頁
        )
        bad_type = (expect is not None) and (expect not in ct)
        if not bad_status and not bad_type:
            return r
        # 重試前回首頁熱身 + 等待
        try:
            session.get(PAGE_URL, headers = HEADERS_PAGE, timeout = 20)
        except Exception:
            pass
        if attempt < RETRY:
            time.sleep(RETRY_SLEEP)
    return r


def fetch_one_day(session: requests.Session, trade_date):
    '''抓取單日資料'''
    dstr = trade_date.strftime("%Y%m%d")    # 用在 URL 查詢參數
    trade_date_str = trade_date.isoformat() # 用在 DataFrame 的 date 欄位
    # 熱身拿 Cookie
    try:
        session.get(PAGE_URL, headers = HEADERS_PAGE, timeout = 20)
        time.sleep(0.5)
    except Exception:
        pass
    
    # 1.RWD_JSON
    url = RWD_JSON.format(date = dstr, typ = MI_TYPE)
    r = session_get_with_retry(session = session, url = url, headers = HEADERS_API, time_out = 25, expect = "application/json")
    if r is not None and r.status_code == 200 and 'application/json' in r.headers.get("Content-Type", ""):
        try:
            js = r.json()
            df = parse_rwd_json(js, trade_date_str)
            if df is not None and len(df) > 0:
                return df, js
        except Exception as e:
            print(f'Error parsing RWD JSON for {dstr}: {e}')
    return None, None


def main():
    ensure_dir()
    # 讓舊的 parquet（可續跑）
    if os.path.exists(FINAL_PARQUET):
        df_all = pd.read_parquet(FINAL_PARQUET)
        done = set(pd.to_datetime(df_all["date"]).dt.date)
    else:
        df_all = None
        done = set()
    all_dates = list(dt_iter(START_DATE, END_DATE))
    pbar = tqdm(all_dates, desc = 'Crawling TWSE MI_INDEX (ALLBUT0999)')
    
    with requests.Session() as session:
        for d in pbar:
            if d in done:
                pbar.write(f'SKIP {d} (already done)')
                continue
            
            df, raw = fetch_one_day(session = session, trade_date = d)
            if df is None or df.empty:
                print(f"NO_DATA {d}")
                time.sleep(SLEEP_SEC)
                continue
            # 儲存每日原始 CSV
            day_csv = os.path.join(RAW_DIR, f"{d.strftime('%Y%m%d')}_{MI_TYPE}.csv")
            df.to_csv(day_csv, index = False, encoding = 'utf-8-sig')
            # 存原始json
            if isinstance(raw, dict):
                json_path = os.path.join(RAW_DIR, f"{d.strftime('%Y%m%d')}_{MI_TYPE}.json")
                with open(json_path, "w", encoding = 'utf-8') as f:
                    json.dump(raw, f, ensure_ascii = False)
            # 合併大表到 parquet
            if df_all is None:
                df_all = df
            else:
                df_all = pd.concat([df_all, df], ignore_index = True)
            df_all = df_all.drop_duplicates(subset=['date', '證券代號'])
            df_all['date'] = pd.to_datetime(df_all['date'])
            df_all = df_all.sort_values(by = ['date', '證券代號']).reset_index(drop = True)
            df_all.to_parquet(FINAL_PARQUET, index = False)
            done.add(d)
            time.sleep(SLEEP_SEC)
    
    print("資料爬取完成：", os.path.abspath(FINAL_PARQUET))
    print("每日 CSV/JSON:", os.path.abspath(RAW_DIR))


if __name__ == "__main__":
    main()