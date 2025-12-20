from pathlib import Path
from typing import List, Optional
import logging
import unicodedata
import pandas as pd
import numpy as np

TWSE_PATH  = Path('data/twse_miindex_allbut0999_daily.parquet')
TEJ_PATH   = Path('data/tej_apiprcd_long_2020-01-01_2025-11-04.parquet')
PRICE_OUT  = Path('data/twse_miindex_stock_only_adj.parquet')  # 將會被建立

CHIPS_DAILY_PATH   = Path('data/20251104011454.csv')         # UTF-16, TSV
FINANCE_MONTHLY_PATH = Path('data/20250305084429_close.csv') # CSV/TSV
FACTOR_MONTHLY_PATH= Path('data/20250305064046_factor.csv')  # CSV

FINAL_CSV = Path('merged_monthly.csv')


def _normalize_columns(df: pd.DataFrame):
    '''欄名正規化'''
    df = df.copy()
    df.columns = [
        (unicodedata.normalize("NFKC", str(c))
        .replace("\ufeff", "")
        .replace("\u200b", "")
        .replace("\u3000", " ")
        .strip())
        for c in df.columns
    ]
    return df


def _norm_ym(s):
    '''取前 6 碼作為 YYYYMM'''
    return (pd.Series(s).astype(str)
            .str.replace(r"[^0-9]", "", regex=True)
            .str.slice(0, 6))
    

# 1.讀 TWSE、TEJ 並建「調整後日價」
def load_twse(path: Path):
    '''讀取 twse 並取普通股'''
    df = pd.read_parquet(path)
    mask_numeric = df['證券代號'].astype(str).str.fullmatch(r"[1-9]\d{3}")
    df = df.loc[mask_numeric].copy()
    df['date'] = pd.to_datetime(df["date"], errors = 'coerce')
    df = (df.dropna(subset = ['date'])
            .drop_duplicates(subset = ['date', '證券代號'])
            .sort_values(['date', '證券代號'], kind = 'mergesort')
            .reset_index(drop = True))
    return df


def load_tej_adjfac(path: Path):
    '''讀取 tej 資料，拿取股價調整因子'''
    use_cols = ['coid', 'mdate', 'adjfac']
    df = pd.read_parquet(path, columns = use_cols).copy()
    mask_stock = df['coid'].astype(str).str.fullmatch(r"[1-9]\d{3}")
    df = df.loc[mask_stock].copy()
    df['mdate'] = pd.to_datetime(df['mdate'], errors = 'coerce')
    df = df.dropna(subset = ['mdate'])
    df['adjfac'] = pd.to_numeric(df['adjfac'], errors = 'coerce').fillna(1)
    df['adjfac'] = df['adjfac'].where(df['adjfac'] > 0, 1.0)
    df = (df.drop_duplicates(subset = ['coid', 'mdate'], keep = 'last')
            .sort_values(['coid', 'mdate'], kind = 'mergesort')
            .reset_index(drop = True))
    return df


def build_adjusted_daily_price(twse_path: Path, tej_path: Path, out_path: Path):
    '''調整開高收低'''
    logging.info("Loading TWSE daily")
    df_twse = load_twse(twse_path)
    
    logging.info("Loading TEJ adjfac")
    df_tej = load_tej_adjfac(tej_path)
    
    logging.info("Merging & adjusting prices")
    df = df_twse.merge(df_tej, left_on = ['證券代號', 'date'],
                       right_on = ['coid', 'mdate'], how = 'inner')
    
    for col in ['開盤價', '最高價', '最低價', '收盤價']:
        if col in df.columns:
            df[col + '_adj'] = pd.to_numeric(df[col], errors = 'coerce') * df["adjfac"]
    
    keep_cols = [
        'date', '證券代號',
        *[c + '_adj' for c in ("開盤價","最高價","最低價","收盤價") if c in df.columns],
        'adjfac'
    ]
    keep_cols = [c for c in keep_cols if c is not None]
    out = (df[keep_cols]
           .drop_duplicates(subset = ['date', '證券代號'])
           .sort_values(['date', '證券代號'], kind = 'mergesort')
           .reset_index(drop = True))
    out_path.parent.mkdir(parents = True, exist_ok = True)
    out.to_parquet(out_path, index = False)
    logging.info("Adjusted daily saved %s ,rows=%s, stocks=%s",
                 out_path, f"{len(out):,}", f"{out['證券代號'].nunique():,}")
    return out


# 2.聚合日到月，月報酬
def to_monthly_from_price_frame(df_price_daily: pd.DataFrame):
    df = df_price_daily.copy()
    df['date'] = pd.to_datetime(df['date'], errors = 'coerce')
    df = df.dropna(subset = ['date'])
    df['年月'] = df['date'].dt.strftime("%Y%m")
    adj_cols = {
        "開盤價_adj": "開盤價",
        "最高價_adj": "最高價",
        "最低價_adj": "最低價",
        "收盤價_adj": "收盤價",
    }
    have_adj = [c for c in adj_cols if c in df.columns]
    if "收盤價_adj" not in have_adj:
        raise KeyError('缺少 收盤價_adj')
    
    keep = ['date', '證券代號', '年月'] + have_adj
    df = df[keep].sort_values(['證券代號', 'date'], kind = 'mergesort')
    
    agg_dict = {c: 'last' for c in have_adj}
    price_m = (df.groupby(['證券代號', '年月'], as_index = False)
                 .agg(agg_dict)
                 .rename(columns = adj_cols))
    price_m['收盤價'] = pd.to_numeric(price_m['收盤價'], errors = 'coerce')
    price_m = price_m.sort_values(['證券代號', '年月']).reset_index(drop = True)
    
    def _safe_grp_ret(g):
        g = g.copy()
        prev = g['收盤價'].shift(1)
        ret = (g['收盤價'] / prev) - 1.0
        ret = ret.where((prev > 0) & np.isfinite(prev))
        ret = ret.replace([np.inf, -np.inf], np.nan)
        g['月報酬'] = ret
        return g
    price_m = price_m.groupby('證券代號', as_index = False, group_keys = False).apply(_safe_grp_ret)
    return price_m


# 3.籌碼面日到月
def aggregate_chips_monthly(df_chips_daily: pd.DataFrame) -> pd.DataFrame:
    '''籌碼面日資料聚合成月資料'''
    df = _normalize_columns(df_chips_daily)

    # 日期欄重新標準化，產生「年月」
    if "年月日" not in df.columns:
        for a in ["年 月 日","年  月  日"]:
            if a in df.columns:
                df = df.rename(columns={a: "年月日"})
                break
        if "年月日" not in df.columns and "年月" not in df.columns:
            raise KeyError(f"缺少日期欄位")

    if "年月" not in df.columns:
        raw = df["年月日"].astype(str).str.replace(r"\D", "", regex=True).str[:8]
        dt  = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
        if dt.isna().all():
            raise ValueError("籌碼資料，日期解析失敗")
        df["年月"] = dt.dt.strftime("%Y%m")
    else:
        df["年月"] = _norm_ym(df["年月"])

    # 證券代碼換成證券代號
    if "證券代號" not in df.columns and "證券代碼" in df.columns:
        df = df.rename(columns={"證券代碼": "證券代號"})
    if "證券代號" in df.columns:
        df["證券代號"] = (
            df["證券代號"].astype(str).str.strip()
              .str.extract(r"([1-9]\d{3})", expand=False)
        )

    # 只保留 202001 以後
    df = df[_norm_ym(df["年月"]).astype(int) >= 202001]

    if "三大法人買賣超(張)" in df.columns and "合計買賣超(張)" not in df.columns:
        df = df.rename(columns={"三大法人買賣超(張)": "合計買賣超(張)"})
    
    # 開始聚合
    wanted_sum = ["外資買賣超(張)", "投信買賣超(張)", "自營買賣超(張)", "合計買賣超(張)"]
    wanted_last = ["外資連續累計買賣超(張)", "投信連續累計買賣超(張)", "自營連續累計買賣超(張)"]
    have_sum = [c for c in wanted_sum if c in df.columns]
    have_last = [c for c in wanted_last if c in df.columns]
    
    keep_cols = ['證券代號', '年月'] + have_sum + have_last
    df = df[keep_cols].copy()
    
    num_cols = df.select_dtypes(include = [np.number, "float64", "int64"]).columns.tolist()
    if num_cols:
        df[num_cols] = df[num_cols].fillna(0)
        
    agg_dict = {c: 'sum' for c in have_sum}
    agg_dict.update({c: 'last' for c in have_last})
    
    chips_m = (df.groupby(['證券代號', '年月'], as_index = False)
                 .agg(agg_dict))
    
    need_last = [c for c in wanted_last if c in chips_m.columns]
    if len(need_last) > 1:
        chips_m["三大法人連續累計買賣超(張)"] = chips_m[need_last].sum(axis = 1)
    else:
        chips_m["三大法人連續累計買賣超(張)"] = 0
    
    final_cols = [
        "證券代號", "年月",
        "外資買賣超(張)", "投信買賣超(張)", "自營買賣超(張)", "合計買賣超(張)",
        "外資連續累計買賣超(張)", "投信連續累計買賣超(張)", "自營連續累計買賣超(張)",
        "三大法人連續累計買賣超(張)"
    ]
    final_cols = [c for c in final_cols if c in chips_m.columns]
    chips_m = chips_m[final_cols].copy()
    return chips_m


# 4. 讀取財務、fama-french因子
def load_finance_monthly(path: Path):
    '''載入財務因子'''
    df = pd.read_csv(path, encoding = 'utf-16', sep = '\t')
    drop_candidates = ["收盤價(元)_月", "報酬率％_月"]
    df = df.drop(columns = [c for c in drop_candidates if c in df.columns], errors = 'ignore')
    df = _normalize_columns(df)
    if '證券代號' not in df.columns and '證券代碼' in df.columns:
        df = df.rename(columns = {'證券代碼': '證券代號'})
    if '證券代號' not in df.columns:
        raise KeyError('財務因子檔缺少證券代號欄位')
    df['證券代號'] = (df['證券代號'].astype(str).str.strip()
                  .str.extract(r"([1-9]\d{3})", expand=False))
    if '年月' not in df.columns:
        raise KeyError('財務因子檔缺少年月欄')
    df['年月'] = _norm_ym(df['年月'])
    return df 


def load_factor_monthly(path: Path):
    '''載入 fama-french 因子'''
    df = pd.read_csv(path, encoding = 'utf-16', sep = '\t')
    df = _normalize_columns(df)
    df['年月'] = _norm_ym(df['年月'])
    return df.drop_duplicates(subset = ['年月'])


def merge_all(price_monthly: pd.DataFrame,
              chips_monthly: pd.DataFrame,
              finance_monthly: pd.DataFrame,
              factor_monthly: pd.DataFrame):
    '''合併調整後價格、籌碼、財務、fama-french'''
    m = price_monthly.merge(chips_monthly, on = ['證券代號', '年月'], how = 'inner')
    m = m.merge(finance_monthly, on = ['證券代號', '年月'], how = 'left')
    m = m.merge(factor_monthly, on = '年月', how = 'left')
    return m


def run():
    logging.info('1. Building adjusted daily price parquet')
    adj_daily = build_adjusted_daily_price(TWSE_PATH, TEJ_PATH, PRICE_OUT)
    
    logging.info('2. Daily to monthly (adjusted, month-end last + 月報酬)')
    price_monthly = to_monthly_from_price_frame(adj_daily)
    
    logging.info('3. Loading chips daily and aggregating to monthly')
    chips_daily = pd.read_csv('/Users/huyiming/Downloads/interview/20251104011454.csv', encoding = 'utf-16', sep = '\t')
    chips_monthly = aggregate_chips_monthly(chips_daily)
    
    logging.info('4. Loading close monthly and factor monthly')
    finance_m = load_finance_monthly(FINANCE_MONTHLY_PATH)
    factor_m = load_factor_monthly(FACTOR_MONTHLY_PATH)
    
    logging.info('5. Merging all pieces')
    merged = merge_all(price_monthly, chips_monthly, finance_m, factor_m)
    
    # 嚴格檢查財務因子必須存在
    strict_cols = [
        "週轉率％_月","本益比-TEJ","股價淨值比-TEJ",
        "股價營收比-TEJ","股利殖利率-TSE","現金股利率"
    ]
    have_strict = [c for c in strict_cols if c in merged.columns]
    
    num_cols = merged.select_dtypes(include = ['number']).columns
    merged[num_cols] = merged[num_cols].replace([np.inf, -np.inf], np.nan)
    
    if have_strict:
        bad_tics = (
            merged.groupby('證券代號')[have_strict]
                  .apply(lambda g: g.isna().any().any())
        )
        drop_tics = bad_tics[bad_tics].index.to_list()
        if drop_tics:
            print(f"合併後檢查，因 {have_strict} 出現缺值，整檔剔除 {len(drop_tics)} 檔股票")
            merged = merged[~merged['證券代號'].isin(drop_tics)].copy()
    
    # 只取整個時間序列都有資訊的股票
    all_months = np.sort(merged['年月'].unique())
    n_all = len(all_months)
    cnt = (merged.drop_duplicates(subset = ['證券代號', '年月'])
                 .groupby('證券代號')['年月']
                 .nunique())
    bad_tics_len = cnt[cnt < n_all].index.tolist()
    if bad_tics_len:
        print(f'因時間序列月份不齊（需要{n_all}個月）：整檔剔除 {len(bad_tics_len)} 檔股票')
        merged = merged[~merged['證券代號'].isin(bad_tics_len)].copy()
    
    logging.info("Done, rows = %s, stocks = %s",
                 f"{len(merged):,}", f"{merged['證券代號'].nunique():,}")
    
    return merged


if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO, format = "%(asctime)s %(levelname)s %(message)s")
    df_merged_all = run()
    
    if "證券代碼" in df_merged_all.columns:
        df_merged_all = df_merged_all.drop(columns = ['證券代碼'])
    
    df_merged_all = df_merged_all.drop(columns = ['投信買賣超(張)' , '自營買賣超(張)', '投信連續累計買賣超(張)','自營連續累計買賣超(張)'])
    df_merged_all.to_csv(FINAL_CSV, index = False, encoding = 'utf-8-sig')
    
    print(f"輸出最終檔案: {FINAL_CSV}, 筆數 {len(df_merged_all):,}, 股票 {df_merged_all['證券代號'].nunique():,} 檔")