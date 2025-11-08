"""
V9_auto_adaptive_stable_fixed.py
--------------------------------
版本：V9 稳定修正版（兼容旧版 sklearn，无 squared 参数）
自动自适应智能建模脚本。
"""

import os
import argparse
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import KNNImputer, SimpleImputer
import lightgbm as lgb
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def impute_data(df, method="knn"):
    """数据填补函数"""
    numeric_df = df.select_dtypes(include=[np.number])
    if method == "knn":
        print(f"🔧 使用 KNN 填补 ({numeric_df.shape[1]} 列)...")
        imputer = KNNImputer(n_neighbors=5)
        imputed = imputer.fit_transform(numeric_df)
    elif method == "mean":
        print(f"🔧 使用均值填补 ({numeric_df.shape[1]} 列)...")
        imputer = SimpleImputer(strategy="mean")
        imputed = imputer.fit_transform(numeric_df)
    elif method == "ffill_bfill":
        print("🔧 使用前后值填补...")
        return df.ffill().bfill()
    else:
        raise ValueError(f"未知填补方法: {method}")
    df[numeric_df.columns] = imputed
    return df


def build_and_train(df, target_col, outdir, random_state=42):
    """构建并训练模型"""
    print(f"🎯 正在训练目标: {target_col}")
    feature_cols = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
    df_train = df.dropna(subset=[target_col])
    if len(df_train) < 10:
        print(f"⚠️ 跳过 {target_col}, 样本太少 ({len(df_train)})")
        return None

    X = df_train[feature_cols]
    y = df_train[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))  # ✅ 修复兼容
    print(f"✅ {target_col}: R²={r2:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}")

    # 保存模型
    os.makedirs(outdir, exist_ok=True)
    model_path = os.path.join(outdir, f"model_{target_col}.joblib")
    joblib.dump(model, model_path)
    print(f"💾 模型已保存: {model_path}")
    return {"target": target_col, "r2": r2, "mae": mae, "rmse": rmse}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--resample", type=int, default=10)
    parser.add_argument("--impute", choices=["knn", "ffill_bfill", "mean"], default="knn")
    parser.add_argument("--lag_hours", type=int, default=3)
    parser.add_argument("--max_rows", type=int, default=20000)
    parser.add_argument("--time_budget", type=int, default=600)
    parser.add_argument("--subsample_frac", type=float, default=1.0)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    print(f"📂 加载数据: {args.input}")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"✅ 原始样本数: {len(df)}")

    # 时间列处理
    time_cols = [c for c in df.columns if "time" in c.lower()]
    if time_cols:
        df[time_cols[0]] = pd.to_datetime(df[time_cols[0]], errors="coerce")
        df = df.set_index(time_cols[0]).sort_index()
        print(f"🕒 使用时间列: {time_cols[0]}")
    else:
        raise ValueError("未找到时间列。")

    # personality_mean
    behavior_cols = [c for c in df.columns if "behavior_" in c.lower()]
    if len(behavior_cols) > 0:
        print(f"🧠 计算 personality_mean，使用 {len(behavior_cols)} 个行为列。")
        df[behavior_cols] = df[behavior_cols].apply(pd.to_numeric, errors="coerce")
        df["personality_mean"] = df[behavior_cols].mean(axis=1)
    else:
        df["personality_mean"] = np.nan
        print("⚠️ 未发现行为列，personality_mean 填 NaN。")

    # 重采样
    df = df.resample(f"{args.resample}T").mean(numeric_only=True)
    print(f"✅ 成功重采样为 {args.resample} 分钟间隔，共 {len(df)} 条记录。")

    # 滞后特征
    for h in range(1, args.lag_hours + 1):
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[f"{col}_lag{h}h"] = df[col].shift(h)
    print(f"🔁 已生成滞后特征 (1~{args.lag_hours} 小时)。")

    # 填补缺失值
    df = impute_data(df, method=args.impute)

    # 限制行数
    if len(df) > args.max_rows:
        df = df.tail(args.max_rows)
        print(f"📏 限制最大行数为 {args.max_rows}")

    # 确定目标列
    target_candidates = [c for c in df.columns if any(x in c.lower() for x in ["stress", "mood", "pam", "sleep"])]
    print(f"🎯 训练目标: {target_candidates}")

    results = []
    for t in target_candidates:
        res = build_and_train(df, t, args.outdir, random_state=args.random_state)
        if res:
            results.append(res)

    print("\n✅ 智能建模 V9 自适应稳定版 完成 🚀")
    summary_path = os.path.join(args.outdir, "summary_results.csv")
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(f"📊 汇总结果已保存: {summary_path}")


if __name__ == "__main__":
    main()
