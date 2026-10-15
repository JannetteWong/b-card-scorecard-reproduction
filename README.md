# B卡行为评分卡复现项目

本项目基于公开行为数据，复现客户行为评分卡的特征构造、分箱与WOE编码、变量筛选、逻辑回归、独立测试集验证和评分转换流程。

## 快速运行

推荐使用 Python 3.11：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe B_card.py
```

主程序会将指标、模型规则和逐笔评分写入 `outputs/`。如需重新生成审计报告和图表：

```powershell
.\.venv\Scripts\python.exe data_leakage_audit.py
.\.venv\Scripts\python.exe generate_evaluation_charts.py
```

## 文件结构

```text
├── B_card.py                       # 主建模流程
├── scorecard_function.py           # 分箱、WOE、KS及评分转换函数
├── data_leakage_audit.py           # 数据一致性与泄漏审计
├── generate_evaluation_charts.py   # 评估图表生成
├── traindata.csv                   # 训练数据
├── testData.csv                    # 独立测试数据
├── requirements.txt                # Python依赖
├── 项目报告.md                      # 方法、结果和限制
└── outputs/                        # 指标、规则、逐笔评分、审计结果及图片
```

## 基线结果

| 数据集 | 样本数 | 坏样本率 | KS | AUC |
|---|---:|---:|---:|---:|
| 训练集 | 28,099 | 13.99% | 0.6371 | 0.8364 |
| 测试集 | 12,043 | 13.83% | 0.6495 | 0.8443 |

## 结果边界

数据没有提供 `label` 的具体业务口径、观察期、表现期和原始拆分方式。因此，本项目使用“公开行为数据”和“独立测试集验证”的表述，不将结果描述为真实银行生产模型或时间外验证。
