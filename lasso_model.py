import pandas as pd
import numpy as np
import joblib
import warnings 
from typing import Dict
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance, partial_dependence
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# 匯入回測模組
from backtest import (
    load_px_daily_from_parquet_twse,
    backtest_daily_from_monthly_picks,
    summarize_daily_performance,
    plot_selected_nav,
    BacktestConfig, CostConfig
)
# 設定字型
font_path = "/Users/huyiming/Library/Fonts/NotoSansCJKtc-Regular.otf"
try:
    fm.fontManager.addfont(font_path)
    family_name = fm.FontProperties(fname = font_path).get_name()
    plt.rcParams["font.family"] = family_name
except Exception:
    pass

plt.rcParams["axes.unicode_minus"] = False
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

DATA_PATH = "merged_monthly.csv"
PARQUET_DAILY_PATH = "data/twse_miindex_stock_only_adj.parquet"
OUT_DIR = "lasso"


def process_for_lasso(df: pd.DataFrame):
    '''進行簡單資料前處理'''
    df = df.copy()
    df = df.rename(columns = {'證券代號': 'tic', '年月': 'ym'})
    if '月報酬' in df.columns:
        df['return'] = pd.to_numeric(df['月報酬'], errors = 'coerce')
    else:
        raise KeyError('找不到月報酬欄位')
    ym_str = df['ym'].astype(str).str.replace(r"[^0-9]", "", regex=True).str[:6]
    df['date'] = pd.to_datetime(ym_str + '01', format = '%Y%m%d', errors = 'coerce')
    df = df.sort_values(by = ['tic', 'date']).reset_index(drop = True)
    df['next_return'] = df.groupby("tic")['return'].shift(-1)
    df = df.dropna(subset = ['next_return']).reset_index(drop = True)
    price_cols_block = {
        "開盤價", "最高價", "最低價", "收盤價",
        "開盤價_adj", "最高價_adj", "最低價_adj", "收盤價_adj",
        "未調整收盤價(元)"
    }
    exclude_col = {"ym", "tic", "date", "return", "next_return", "月報酬", "報酬率％_月"} | price_cols_block
    numeric_cols = df.select_dtypes(include = ['number']).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_col]
    return df, feature_cols


def train_lasso_rolling(df: pd.DataFrame, feature_cols,
                              start_date = '2020-01',
                              alphas = np.logspace(-3, -0.5, 15),
                              max_iter = 20000):
    '''擴充式訓練模型，每個月產生股票清單'''
    df = df.copy()
    dates = sorted(df['date'].unique())
    if pd.to_datetime(start_date) not in dates:
        dates_arr = np.array(dates)
        start_idx = int(np.searchsorted(dates_arr, pd.to_datetime(start_date), side = 'left'))
    else:
        start_idx = dates.index(pd.to_datetime(start_date))
    results = []
    model_dict: Dict[pd.Timestamp, dict] = {}
    for i in tqdm(range(start_idx, len(dates) - 1)):
        train_end = dates[i]
        predict_month = dates[i + 1]
        df_train_all = df[df['date'] < train_end]
        df_test = df[df['date'] == train_end]
        if len(df_train_all) < 100 or df_train_all.empty:
            continue
        # 訓練/驗證切分, 把 df_train_all 依列順序切 2/3 與 1/3 做 train/val
        split_idx = int(len(df_train_all) * 2 / 3)
        df_train = df_train_all.iloc[:split_idx]
        df_val = df_train_all.iloc[split_idx:]
        # 標準化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(df_train[feature_cols])
        y_train = df_train['next_return'].values
        X_val = scaler.transform(df_val[feature_cols])
        y_val = df_val['next_return'].values
        X_test = scaler.transform(df_test[feature_cols])
        y_test = df_test['next_return'].values
        # Grid search
        best_mse = float('inf')
        best_model = None
        best_alpha = None
        best_l1 = None
        # grid_search
        for a in alphas:
            model = Lasso(alpha = a, max_iter = max_iter)
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, val_pred)
            if mse < best_mse:
                best_mse = mse
                best_model = model
                best_alpha = a
        # Test 用當月特徵去預測下個月
        y_pred = best_model.predict(X_test)
        df_result = df_test[['date', 'tic']].copy()
        df_result['target_date'] = predict_month
        df_result['pred_return'] = y_pred
        df_result['real_return'] = y_test
        df_result['alpha'] = best_alpha
        df_result['l1_ratio'] = best_l1 
        results.append(df_result)
        model_dict[predict_month] = {
            "model": best_model,
            "scaler": scaler,
            "alpha": best_alpha,
            "train_end": train_end
        }
    
    if not results:
        raise RuntimeError("No expanding window produced results")
    df_all_result = pd.concat(results, ignore_index = True)
    joblib.dump(model_dict, f"{OUT_DIR}/lasso_models.pkl") 
    return df_all_result, model_dict


def analyze_best_portfolio_months(
    df_all_results: pd.DataFrame,
    models_dict: dict,
    df_cleaned: pd.DataFrame,
    feature_cols: list[str],
    n_quantiles: int = 4,
    top_n: int = 12,
    importance: str = 'permutation',
    out_prefix: str = 'lasso_best_decile',
    out_dir: str = '.'):
    '''
    先挑整段期間報酬最高的 decile，
    再在該 decile 裡找「最佳月份」與「最差月份」，各自計算特徵重要性並畫圖
    '''
    tmp = df_all_results.copy()
    tmp['rank'] = tmp.groupby('target_date')['pred_return']\
                     .transform(lambda x: pd.qcut(x, n_quantiles, labels = False, duplicates = 'drop'))
    bars = (tmp.groupby(['target_date', 'rank'])['real_return']
                .mean()
                .unstack())
    bars = bars.reindex(columns = range(n_quantiles))
    # 找整段期間「平均報酬最高」的 decile，並抓出該 decile 的最佳/最差月份
    perf_by_decile = bars.mean(axis = 0, skipna = True)
    best_decile = int(perf_by_decile.idxmax())
    series_best = bars[best_decile].dropna()
    if series_best.empty:
        raise RuntimeError('best decile 沒有可用的月份資料。')
    t_best_max = series_best.idxmax()
    t_best_min = series_best.idxmin()
    print(f"[Best decile] = {best_decile}")
    print(f"[Best month]  {t_best_max:%Y-%m}  ret={series_best.loc[t_best_max]:.2%}")
    print(f"[Worst month] {t_best_min:%Y-%m}  ret={series_best.loc[t_best_min]:.2%}")
    
    def _importance_at(t: pd.Timestamp):
        '''回傳該月份的特徵重要性 df(feature, importance)，是對應上個月的特徵'''
        if t not in models_dict:
            raise KeyError(f'models_dict 找不到 {t} 的模型。')
        mobj = models_dict[t]
        model, scaler = mobj['model'], mobj['scaler']
        test_month = mobj['train_end']
        df_te = df_cleaned[df_cleaned['date'] == test_month]
        if df_te.empty:
            return pd.DataFrame({"feature": feature_cols, "importance": 0.0})
        X_te = scaler.transform(df_te[feature_cols])
        y_te = df_te['next_return'].values
        if importance == 'permutation':
            imp = permutation_importance(
                model, X_te, y_te,
                n_repeats = 50, random_state = 42,
                scoring = 'neg_mean_squared_error')
            vals = np.abs(imp.importances_mean)
        return (pd.DataFrame({"feature": feature_cols, "importance": vals})
                  .sort_values("importance", ascending=False)
                  .reset_index(drop=True))
    # 取得「最佳月」與「最差月」的重要性表
    imp_best = _importance_at(t_best_max).rename(columns = {"importance": "importance_best"})
    imp_worst = _importance_at(t_best_min).rename(columns = {"importance": "importance_worst"})
    
    merged = (pd.merge(imp_best, imp_worst, how = 'outer', on = 'feature')
                .fillna(0)
                .assign(total = lambda d: d['importance_best'] + d['importance_worst'])
                .sort_values('total', ascending = False)
                .drop(columns = 'total')
                .head(top_n)
                .reset_index(drop = True))
    def _plot_two_side(df_merged: pd.DataFrame, png_path: str):
        '''畫出長條圖： 最好的月份 vs. 最差的月份重要性'''
        fig, ax = plt.subplots(figsize = (9, 5))
        (df_merged.set_index('feature')[['importance_best', 'importance_worst']]
         .sort_values('importance_best')
         .plot.barh(ax = ax))
        ax.set_xlabel('Importance')
        ax.set_title(f"Best-Decile Feature Importance\n"
                     f"Best {t_best_max:%Y-%m} vs. Worst {t_best_min:%Y-%m}")
        ax.legend([f"Best {t_best_max:%Y-%m}", f"Worst {t_best_min:%Y-%m}"], loc="lower right")
        plt.tight_layout()
        plt.savefig(png_path, dpi = 300)
        plt.show()
        plt.close()
    _plot_two_side(merged, f"{out_dir}/{out_prefix}_best_vs_worst_importance.png")
    
    def _plot_pdp(t: pd.Timestamp, tag: str, feat: str):
        '''對單一月份 + 單一特徵畫部分依賴曲線'''
        mobj = models_dict[t]
        model, scaler = mobj['model'], mobj['scaler']
        test_month = mobj['train_end']
        
        df_tr = df_cleaned[df_cleaned['date'] < test_month]
        df_te = df_cleaned[df_cleaned['date'] == test_month]
        if df_te.empty:
            print(f"[WARN] test month {test_month:%Y-%m} 沒有樣本，跳過 PDP：{tag}")
            return
        X_tr = scaler.transform(df_tr[feature_cols]) if not df_tr.empty else None
        X_te = scaler.transform(df_te[feature_cols])
        idx = feature_cols.index(feat)
        fig, ax = plt.subplots(figsize = (5.5, 4))
        if X_tr is not None:
            pdp_tr = partial_dependence(model, X_tr, [idx], kind = 'average')
            ax.plot(pdp_tr["grid_values"][0], pdp_tr["average"][0], label="Train", linewidth=2)
        pdp_te = partial_dependence(model, X_te, [idx], kind = 'average')
        ax.plot(pdp_te["grid_values"][0], pdp_te["average"][0], label="Test", linewidth=2, linestyle="--")
        ax.legend()
        ax.set_title(f"{tag} PDP @ {t:%Y-%m} — {feat}")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{out_prefix}_{tag}_pdp_{feat}.png", dpi=300)
        plt.show()
        plt.close()
        
    best_feat_best = imp_best.sort_values('importance_best', ascending = False).iloc[0]['feature'] \
                     if not imp_best.empty else feature_cols[0]
    best_feat_worst = imp_worst.sort_values('importance_worst', ascending = False).iloc[0]['feature'] \
                     if not imp_worst.empty else best_feat_best
    _plot_pdp(t_best_max, "best", best_feat_best)
    _plot_pdp(t_best_min, "worst", best_feat_worst)
   
    return {
        "best_decile": best_decile,
        "t_best": t_best_max,
        "t_worst": t_best_min,
        "ret_best": float(series_best.loc[t_best_max]),
        "ret_worst": float(series_best.loc[t_best_min]),
        "feat_best": best_feat_best,
        "feat_worst": best_feat_worst,
        "importance_table": merged,           # 前 top_n 的比較表（用來畫圖/檢視）
        "importance_best_full": imp_best,     # 最佳月完整重要性表
        "importance_worst_full": imp_worst,   # 最差月完整重要性表
    }


if __name__ == '__main__':
    df0 = pd.read_csv(DATA_PATH)
    df_cleaned, feature_cols = process_for_lasso(df0)
    print(f"特徵數：{len(feature_cols)}；樣本筆數：{len(df_cleaned):,}")
    # 開始訓練
    df_all_results, models_dict = train_lasso_rolling(
        df_cleaned, feature_cols,
        start_date = '2020-01',
        alphas = np.logspace(-4, -0.5, 40),
        max_iter = 20000
    )
    joblib.dump(models_dict, f"{OUT_DIR}/lasso_models.pkl")
    df_all_results.to_csv(f"{OUT_DIR}/monthly_picks_with_scores.csv", index=False)
    summary_bw = analyze_best_portfolio_months(
        df_all_results = df_all_results,
        models_dict = models_dict,
        df_cleaned = df_cleaned, 
        feature_cols = feature_cols,
        n_quantiles = 4, 
        top_n = 12,
        importance = 'permutation',
        out_prefix="lasso_best_decile",
        out_dir = OUT_DIR
    )
    print(summary_bw)
    # 開始回測
    px_daily = load_px_daily_from_parquet_twse(PARQUET_DAILY_PATH)
    cfg = BacktestConfig(n_quantiles=4, initial_capital_per_bucket=10_000_000.0, ann=252)
    cost = CostConfig(commission_buy=0.001425, commission_sell=0.001425, tax_sell=0.0030,
                  slippage_buy=0.0, slippage_sell=0.0)
    daily_nav, daily_ret, diag = backtest_daily_from_monthly_picks(
        df_all_results, px_daily, cfg = cfg, cost = cost
    )
    daily_nav.to_csv(f"{OUT_DIR}/daily_nav_quantiles_benchmark.csv")
    daily_ret.to_csv(f"{OUT_DIR}/daily_ret_quantiles_benchmark.csv")
    diag.to_csv(f"{OUT_DIR}/daily_turnover_cost_diag.csv", index=False)
    
    stats_full = summarize_daily_performance(
        daily_nav, daily_ret, cols = [0, 1, 2, 3, 'benchmark'], ann = cfg.ann
    )
    print("===日頻績效===")
    print(stats_full.round(6))
    stats_full.round(6).to_csv(f"{OUT_DIR}/daily_stats_full_with_mdd_cagr.csv")
    plot_selected_nav(daily_nav, cols = [0, 1, 2, 3, 'benchmark'],
                      month_interval = 4,
                      title = "Daily NAV (monthly ticks)",
                      save_path = f"{OUT_DIR}/daily_nav_monthly_ticks.png")
    
