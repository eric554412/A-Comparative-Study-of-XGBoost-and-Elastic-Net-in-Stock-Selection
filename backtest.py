import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Union, Optional
from matplotlib.dates import MonthLocator, DateFormatter
import matplotlib.pyplot as plt


@dataclass
class CostConfig:
    commission_buy: float = 0.001425
    commission_sell: float = 0.001425
    tax_sell: float = 0.0030
    slippage_buy: float = 0.0
    slippage_sell: float = 0.0

@dataclass
class BacktestConfig:
    n_quantiles: int = 4
    initial_capital_per_bucket: float = 10000000
    ann: int = 252


def load_px_daily_from_parquet_twse(
    parquet_path: str,
    tic_col: str = '證券代號',
    date_col: str = 'date',         
    open_adj_col: str = '開盤價_adj',
    close_adj_col: str = '收盤價_adj'):
    '''讀日價並計算日報酬'''
    df = pd.read_parquet(parquet_path)

    need = [tic_col, date_col, open_adj_col, close_adj_col]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f'Parquet 缺少欄位: {miss}')

    out = df[[tic_col, date_col, open_adj_col, close_adj_col]].copy()
    out = out.rename(columns={
        tic_col: 'tic',
        date_col: 'date',
        close_adj_col: 'px'        
    })
    out['tic'] = out['tic'].astype(str).str.strip()
    out['date'] = pd.to_datetime(out['date']).dt.tz_localize(None)
    out = out.sort_values(['tic', 'date'])

    out['ret_d'] = out.groupby('tic')['px'].pct_change()

    out = out.rename(columns={
        open_adj_col: '開盤價_adj',
        'px': 'px'})
    out['收盤價_adj'] = out['px']
    return out[['tic', 'date', 'px', 'ret_d', '開盤價_adj', '收盤價_adj']]



def _build_first_trading_day_map(
    trading_day: pd.DatetimeIndex,
    target_month_starts: List[pd.Timestamp]):
    '''把 target_date(月第一天)映射到這個月的「第一個交易日」'''
    td = pd.DatetimeIndex(sorted(pd.to_datetime(trading_day).tz_localize(None).unique()))
    f_map = {}
    for m in sorted(pd.to_datetime(target_month_starts).tz_localize(None).unique()):
        i = td.searchsorted(m, side = 'left')
        if i < len(td):
            f_map[pd.Timestamp(m)] = pd.Timestamp(td[i])
    return f_map


def _get_ret_series(px_daily: pd.DataFrame, tickers: List[str], d: pd.Timestamp):
    '''抓某一天這批股票的日報酬；缺值補0'''
    if len(tickers) == 0:
        return pd.Series(dtype = float)
    sub = px_daily[(px_daily['date'] == d) & (px_daily['tic'].isin(tickers))]
    s = sub.set_index('tic')['ret_d']
    return s.reindex(tickers).fillna(0)


def _simple_average_benchmark_daily_ret(px_daily: pd.DataFrame, universe: List[str], d: pd.Timestamp):
    '''簡單加權平均（指定股票集合的當日平均報酬）'''
    if len(universe) == 0:
        return 0
    s = _get_ret_series(px_daily, universe, d)
    return float(s.mean())


def backtest_daily_from_monthly_picks(
    df_result_monthly: pd.DataFrame, # 必含: target_date(YYYY-MM-01), tic, pred_return
    px_daily: pd.DataFrame,          # 由 load_px_daily_from_parquet_tw 載入：tic, date, px, ret_d
    cfg: BacktestConfig = BacktestConfig(),
    cost: CostConfig = CostConfig()
):
    '''
    每個月月初「開盤」等權再平衡；
    再平衡日:先吃隔夜 (close to open) 開盤調倉扣成本, 吃盤中(open to close)。
    非再平衡日：不交易，用 close to close。
    最後輸出每日資金曲線、報酬率、與換股成本記錄。
    '''
    def _get_ret_oc(px_daily: pd.DataFrame, tickers: list, d: pd.Timestamp):
        # open to close return
        sub = px_daily.loc[px_daily['date'] == d, ['tic', '開盤價_adj', '收盤價_adj']].copy()
        if sub.empty:
            return pd.Series(0.0, index=tickers, dtype=float)
        s = (sub.set_index('tic')['收盤價_adj'] / sub.set_index('tic')['開盤價_adj'] - 1.0)
        return s.reindex(tickers).fillna(0.0)

    def _get_prev_close(px_daily: pd.DataFrame, tickers: list, d: pd.Timestamp, trading_day_idx: pd.DatetimeIndex):
        # 找 d 的前一個交易日收盤價（adj）
        pos = trading_day_idx.get_indexer([d])[0]
        if pos <= 0:
            return pd.Series(index=tickers, dtype=float)  # 找不到前一日，之後當成 0 報酬
        d_prev = trading_day_idx[pos - 1]
        sub = px_daily.loc[px_daily['date'] == d_prev, ['tic', '收盤價_adj']].copy()
        return sub.set_index('tic')['收盤價_adj'].reindex(tickers)

    def _get_ret_co(px_daily: pd.DataFrame, tickers: list, d: pd.Timestamp, trading_day_idx: pd.DatetimeIndex):
        # close to open return
        if len(tickers) == 0:
            return pd.Series(dtype=float)
        px_prev = _get_prev_close(px_daily, tickers, d, trading_day_idx)
        px_open = px_daily.loc[px_daily['date'] == d, ['tic', '開盤價_adj']].set_index('tic')['開盤價_adj']
        r = (px_open.reindex(tickers) / px_prev - 1.0)
        return r.fillna(0.0)

    dfp = df_result_monthly.copy()

    need_cols = {'target_date', 'tic', 'pred_return'}
    miss = need_cols - set(dfp.columns)
    if miss:
        raise KeyError(f'df_result_monthly 缺欄位: {miss}')
    dfp['tic'] = dfp['tic'].astype(str).str.strip()
    dfp['target_date'] = pd.to_datetime(dfp['target_date']).dt.tz_localize(None)

    # 分 decile
    dfp['rank'] = (dfp.groupby('target_date')['pred_return']
                      .transform(lambda s: pd.qcut(s, cfg.n_quantiles, labels=False, duplicates='drop')))

    trading_day = pd.DatetimeIndex(px_daily['date'].sort_values().unique())
    first_day_map = _build_first_trading_day_map(trading_day, dfp['target_date'].unique())
    if len(first_day_map) == 0:
        raise RuntimeError('無法建立在平衡日')

    start_day = min(first_day_map.values())
    end_day = trading_day.max()
    day_range = trading_day[(trading_day >= start_day) & (trading_day <= end_day)]

    # 建立nav表
    ranks = list(range(cfg.n_quantiles))
    cols = ranks + ['benchmark']
    daily_nav = pd.DataFrame(index=day_range, columns=cols, dtype=float)
    daily_nav.iloc[0] = cfg.initial_capital_per_bucket
    w_prev = {r: pd.Series(dtype=float) for r in ranks}

    monthly_lists = {(t, r): dfp[(dfp['target_date'] == t) & (dfp['rank'] == r)]['tic'].astype(str).unique().tolist()
                     for t in dfp['target_date'].unique() for r in ranks}



    diag_rows = []

    for i, d in enumerate(day_range):
        # 取得「當天全市場」做 Benchmark 的集合 
        today_univ = px_daily.loc[px_daily['date'] == d, 'tic'].unique().tolist()  # CHANGED

        # 每天檢查今天是不是每個月的第一天交易日
        rb_months = [m for m, fd in first_day_map.items() if fd == d]
        is_rebalance = len(rb_months) > 0

        if is_rebalance:
            # 再平衡日:先吃隔夜, 開盤調倉扣成本, 吃盤中
            m = rb_months[0]
            for r in ranks:
                curr_list = monthly_lists.get((m, r), [])
                N = len(curr_list)
                prev_nav = daily_nav.iloc[i - 1, r] if i > 0 else cfg.initial_capital_per_bucket

                if N == 0:
                    daily_nav.iloc[i, r] = prev_nav
                    w_prev[r] = pd.Series(dtype=float)
                    diag_rows.append({'date': d, 'month': m, 'rank': r,
                                      'buy_turnover': 0.0, 'sell_turnover': 0.0,
                                      'trade_cost': 0.0, 'n_names': 0})
                    continue

                # 目標等權
                w_target = pd.Series(1.0 / N, index=curr_list, dtype=float)

                # 隔夜：昨收權重 × close to open, 得到 nav_open
                if i > 0 and not w_prev[r].empty:
                    r_co = _get_ret_co(px_daily, w_prev[r].index.tolist(), d, trading_day)
                    nav_open = prev_nav * (1.0 + float((w_prev[r] * r_co).sum()))
                    # 開盤前的舊權重自然漂移到開盤（用於 turnover 比較）
                    w_old_open = w_prev[r] * (1.0 + r_co)
                    s = w_old_open.sum()
                    w_old_open = (w_old_open / s) if s > 0 else w_old_open
                else:
                    nav_open = prev_nav
                    w_old_open = pd.Series(dtype=float)

                # 開盤調倉
                w_all = w_target.index.union(w_old_open.index)
                w_old = w_old_open.reindex(w_all).fillna(0.0)
                w_new = w_target.reindex(w_all).fillna(0.0)
                delta = w_new - w_old
                buy_turnover  = float(delta.clip(lower=0).sum())
                sell_turnover = float((-delta).clip(lower=0).sum())
                trade_cost = buy_turnover * (cost.commission_buy + cost.slippage_buy) \
                           + sell_turnover * (cost.commission_sell + cost.tax_sell + cost.slippage_sell)
                nav_after_cost = nav_open * (1.0 - trade_cost)

                # 盤中：新權重 × open to close
                r_oc = _get_ret_oc(px_daily, w_target.index.tolist(), d)
                port_ret_oc = float((w_target * r_oc).sum())
                daily_nav.iloc[i, r] = nav_after_cost * (1.0 + port_ret_oc)

                # 收盤後權重（給隔天）
                w_next = w_target * (1.0 + r_oc)
                s = w_next.sum()
                w_prev[r] = (w_next / s) if s > 0 else w_next

    
                diag_rows.append({
                    'date': d, 'month': m, 'rank': r,
                    'buy_turnover': buy_turnover,
                    'sell_turnover': sell_turnover,
                    'trade_cost': trade_cost,
                    'n_names': int(N)
                })

            # benchmark（用全市場等權平均）   
            prev_b = daily_nav.iloc[i - 1, daily_nav.columns.get_loc('benchmark')] if i > 0 else cfg.initial_capital_per_bucket
            rb = _simple_average_benchmark_daily_ret(px_daily, today_univ, d)  
            daily_nav.iloc[i, daily_nav.columns.get_loc('benchmark')] = prev_b * (1.0 + rb)

        else:
            # 非再平衡日：close to close
            if i == 0:
                continue
            d_prev = day_range[i - 1]
            for r in ranks:
                prev_nav = daily_nav.loc[d_prev, r]
                if pd.isna(prev_nav):
                    prev_nav = cfg.initial_capital_per_bucket
                if w_prev[r].empty:
                    daily_nav.iloc[i, r] = prev_nav
                    continue

                sub = _get_ret_series(px_daily, w_prev[r].index.tolist(), d) 
                ret_p = float((w_prev[r] * sub).sum())
                daily_nav.iloc[i, r] = prev_nav * (1.0 + ret_p)

                # 權重自然漂移
                w_new = w_prev[r] * (1.0 + sub)
                s = w_new.sum()
                w_prev[r] = (w_new / s) if s > 0 else w_new

            # benchmark（用全市場等權平均）      
            prev_b = daily_nav.loc[d_prev, 'benchmark']
            rb = _simple_average_benchmark_daily_ret(px_daily, today_univ, d)   
            daily_nav.iloc[i, daily_nav.columns.get_loc('benchmark')] = prev_b * (1.0 + rb)

    daily_ret = daily_nav.pct_change().fillna(0.0)
    diag = pd.DataFrame(diag_rows).sort_values(['date','rank'])
    return daily_nav, daily_ret, diag


def _mdd_pct(nav: pd.Series):
    '''計算最大回撤'''
    nav = nav.dropna()
    if nav.empty:
        return float('nan')
    roll_max = nav.cummax()
    drawdown = nav / roll_max - 1.0
    return float(drawdown.min())


def summarize_daily_performance(
    daily_nav: pd.DataFrame,
    daily_ret: pd.DataFrame,
    cols: List[Union[int, str]] = (0, 1, 2, 3, 'benchmark'),
    ann: int = 252):
    '''
    mean / std / Sharpe / annual_return / total_return / CAGR / MDD(%) / Calmar
    '''
    cols = [c for c in cols if c in daily_nav.columns and c in daily_ret.columns]
    stats = daily_ret[cols].agg(['mean', 'std']).T
    stats['sharpe'] = (stats['mean'] / stats['std']) * np.sqrt(ann)
    stats['annual_return'] = np.power(1.0 + stats['mean'], ann) - 1.0
    # 計算 CAGR
    rows = []
    n_days = len(daily_ret)
    years = max(n_days / ann, 1e-9)
    for c in cols:
        nav = daily_nav[c].dropna()
        if len(nav) >= 2 and nav.iloc[0] > 0:
            total_ratio = float(nav.iloc[-1] / nav.iloc[0])
            total_return = total_ratio - 1.0
            cagr = float(total_ratio ** (1 / years) - 1)
        else:
            total_return = float('nan')
            cagr = float('nan')
        mdd = _mdd_pct(nav)
        calmar = float(cagr / abs(mdd)) if (isinstance(mdd, float) and mdd < 0) else float('nan')
        
        rows.append({
            'portfolio': c,
            'total_return': total_return,
            'CAGR': cagr,
            'MDD(%)': mdd,
            'Calmar': calmar
        })
    extra = pd.DataFrame(rows).set_index('portfolio')
    out = stats.join(extra, how = 'left')
    return out


def plot_selected_nav(
    nav: pd.DataFrame,
    cols = [0, 1, 2 , 3, 'benchmark'],
    title: str = 'Daily NAV — Quantiles vs Benchmark',
    save_path: str = None,
    month_interval:int = 2):
    '''畫出績效圖'''
    data = nav.copy()
    idx = pd.to_datetime(data.index).tz_localize(None)
    data = data.set_index(idx)
    
    fig, ax = plt.subplots(figsize = (12, 6))
    for c in cols:
        if c in data.columns:
            ax.plot(data.index, data[c], label = str(c))
    ax.set_title(title)
    ax.set_ylabel("NAV")
    ax.grid(True, alpha = 0.3)
    ax.legend()
    ax.xaxis.set_major_locator(MonthLocator(interval=month_interval))  
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))  
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()        

    
             
                    
                                 