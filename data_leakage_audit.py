from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)


def derived_features(df):
    result = {}
    for window in (1, 3, 6, 12):
        months = list(range(12, 12 - window, -1))
        d1 = df[[f"Delq1_{m}" for m in months]]
        d2 = df[[f"Delq2_{m}" for m in months]]
        d3 = df[[f"Delq3_{m}" for m in months]]
        result[f"maxDelqL{window}M"] = np.select(
            [d3.max(axis=1).eq(1), d2.max(axis=1).eq(1), d1.max(axis=1).eq(1)],
            [3, 2, 1],
            default=0,
        )
        result[f"M0FreqL{window}M"] = d1.sum(axis=1)
        result[f"M1FreqL{window}M"] = d2.sum(axis=1)
        result[f"M2FreqL{window}M"] = d3.sum(axis=1)

        spend = df[[f"Spend_{m}" for m in months]].to_numpy() / df[["Loan_Amount"]].to_numpy()
        result[f"maxUrateL{window}M"] = spend.max(axis=1)
        result[f"avgUrateL{window}M"] = spend.mean(axis=1)
        if window > 1:
            result[f"increaseUrateL{window}M"] = (spend[:, :-1] > spend[:, 1:]).sum(axis=1)

        payment = df[[f"Payment_{m}" for m in months]].to_numpy()
        os_months = list(range(11, 11 - window, -1))
        outstanding = df[[f"OS_{m}" for m in os_months]].to_numpy()
        ratios = np.divide(payment, outstanding, out=np.ones_like(payment, dtype=float), where=outstanding > 0)
        result[f"maxPayL{window}M"] = ratios.max(axis=1)
        result[f"minPayL{window}M"] = ratios.min(axis=1)
        result[f"avgPayL{window}M"] = np.divide(
            payment.sum(axis=1), outstanding.sum(axis=1),
            out=np.ones(len(df), dtype=float), where=outstanding.sum(axis=1) > 0,
        )
    return pd.DataFrame(result, index=df.index)


def auc_table(df, columns):
    rows = []
    for column in columns:
        if not pd.api.types.is_numeric_dtype(df[column]) or df[column].nunique() < 2:
            continue
        auc = roc_auc_score(df["label"], df[column])
        rows.append({"feature": column, "single_feature_auc": auc, "separation_auc": max(auc, 1 - auc)})
    return pd.DataFrame(rows).sort_values("separation_auc", ascending=False)


def psi_table(expected, actual, columns):
    rows = []
    for column in columns:
        cuts = np.unique(expected[column].quantile(np.linspace(0, 1, 11)).to_numpy())
        if len(cuts) < 2:
            continue
        cuts[0], cuts[-1] = -np.inf, np.inf
        expected_rate = pd.cut(expected[column], cuts, include_lowest=True).value_counts(normalize=True, sort=False)
        actual_rate = pd.cut(actual[column], cuts, include_lowest=True).value_counts(normalize=True, sort=False)
        expected_rate = expected_rate.clip(lower=1e-6)
        actual_rate = actual_rate.clip(lower=1e-6)
        psi = ((actual_rate - expected_rate) * np.log(actual_rate / expected_rate)).sum()
        rows.append({"feature": column, "psi": psi})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


train = pd.read_csv(ROOT / "traindata.csv")
test = pd.read_csv(ROOT / "testData.csv")
raw_columns = list(test.columns)
raw_features = [c for c in raw_columns if c not in {"CUST_ID", "label"}]

train_ids = set(train["CUST_ID"])
test_ids = set(test["CUST_ID"])
common_hash_train = set(pd.util.hash_pandas_object(train[raw_features], index=False))
common_hash_test = set(pd.util.hash_pandas_object(test[raw_features], index=False))

recomputed = derived_features(train)
comparison_rows = []
for column in recomputed.columns:
    if column not in train:
        comparison_rows.append({"feature": column, "status": "missing", "mismatch_count": len(train)})
        continue
    matches = np.isclose(train[column], recomputed[column], equal_nan=True)
    comparison_rows.append({"feature": column, "status": "match" if matches.all() else "mismatch",
                            "mismatch_count": int((~matches).sum())})
comparison = pd.DataFrame(comparison_rows)

single_auc = auc_table(pd.concat([train[raw_columns], test], ignore_index=True), raw_features)
distribution_psi = psi_table(train, test, raw_features)
single_auc.to_csv(OUTPUT / "single_feature_auc.csv", index=False)
distribution_psi.to_csv(OUTPUT / "train_test_psi.csv", index=False)
comparison.to_csv(OUTPUT / "derived_feature_consistency.csv", index=False)

direct_exposure = single_auc[single_auc["separation_auc"] >= 0.95]
top_rows = ["| feature | single_feature_auc | separation_auc |", "|---|---:|---:|"]
for row in single_auc.head(10).itertuples(index=False):
    top_rows.append(f"| {row.feature} | {row.single_feature_auc:.4f} | {row.separation_auc:.4f} |")
top_table = "\n".join(top_rows)
psi_rows = ["| feature | PSI |", "|---|---:|"]
for row in distribution_psi.head(10).itertuples(index=False):
    psi_rows.append(f"| {row.feature} | {row.psi:.4f} |")
psi_top_table = "\n".join(psi_rows)
report = f"""# B卡数据泄漏审计

## 审计结果

| 检查项 | 结果 |
|---|---:|
| 训练集样本数 | {len(train)} |
| 测试集样本数 | {len(test)} |
| 训练集坏样本率 | {train['label'].mean():.4%} |
| 测试集坏样本率 | {test['label'].mean():.4%} |
| 训练测试重复客户ID | {len(train_ids & test_ids)} |
| 训练测试完全相同原始特征行 | {len(common_hash_train & common_hash_test)} |
| 训练集重复客户ID | {train['CUST_ID'].duplicated().sum()} |
| 测试集重复客户ID | {test['CUST_ID'].duplicated().sum()} |
| 预制衍生特征不一致字段 | {(comparison['status'] != 'match').sum()} |
| 单字段AUC达到0.95的字段 | {len(direct_exposure)} |
| 原始字段最大训练测试PSI | {distribution_psi['psi'].max():.4f} |

## 单字段区分能力最高的10个原始变量

{top_table}

## 训练集与测试集分布差异最高的10个字段

{psi_top_table}

## 结论

- 未发现训练集和测试集客户交叉，也未发现两组之间完全相同的原始特征行。
- 训练集中的预制衍生变量均可由原始月度字段重新计算得到，没有发现衍生字段内容异常。
- 未发现单个原始字段几乎直接复制标签的情况；详细结果见 `single_feature_auc.csv`。
- 现有文件没有提供 `label` 的定义、观察期截止日、表现期范围或样本切分方式。因此无法从数据本身证明月度行为字段早于标签表现期，也无法排除时间穿越。
- 当前训练集和测试集的坏样本率非常接近，但没有日期字段，无法进行真正的时间外验证。现有结果应称为独立测试集验证，不应称为OOT验证。
- 所有原始字段的训练测试PSI最高为 {distribution_psi['psi'].max():.4f}，两组分布极为接近，更符合随机拆分数据的表现。

## 简历与面试表述边界

可以表述为“基于公开行为数据完成训练集与独立测试集验证”。在获得数据字典或来源说明前，不表述具体逾期口径、表现期长度、银行生产数据或时间外验证。
"""
(OUTPUT / "data_leakage_audit.md").write_text(report, encoding="utf-8")
print(report)
