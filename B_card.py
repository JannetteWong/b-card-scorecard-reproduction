import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn import ensemble
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import scorecard_function

def DelqFeatures(event,window,type):
    current = 12
    start = 12 - window + 1
    delq1 = [event[a] for a in ['Delq1_' + str(t) for t in range(current, start - 1, -1)]]
    delq2 = [event[a] for a in ['Delq2_' + str(t) for t in range(current, start - 1, -1)]]
    delq3 = [event[a] for a in ['Delq3_' + str(t) for t in range(current, start - 1, -1)]]
    if type == 'max delq':
        if max(delq3) == 1:
            return 3
        elif max(delq2) == 1:
            return 2
        elif max(delq1) == 1:
            return 1
        else:
            return 0
    if type in ['M0 times','M1 times', 'M2 times']:
        if type.find('M0')>-1:
            return sum(delq1)
        elif type.find('M1')>-1:
            return sum(delq2)
        else:
            return sum(delq3)

def UrateFeatures(event, window, type):
    current = 12
    start = 12 - window + 1
    monthlySpend = [event[a] for a in ['Spend_' + str(t) for t in range(current, start - 1, -1)]]
    limit = event['Loan_Amount']
    monthlyUrate = [x / limit for x in monthlySpend]
    if type == 'mean utilization rate':
        return np.mean(monthlyUrate)
    if type == 'max utilization rate':
        return max(monthlyUrate)
    if type == 'increase utilization rate':
        currentUrate = monthlyUrate[0:-1]
        previousUrate = monthlyUrate[1:]
        compareUrate = [int(x[0]>x[1]) for x in zip(currentUrate,previousUrate)]
        return sum(compareUrate)

def PaymentFeatures(event, window, type):
    current = 12
    start = 12 - window + 1
    currentPayment = [event[a] for a in ['Payment_' + str(t) for t in range(current, start - 1, -1)]]
    previousOS = [event[a] for a in ['OS_' + str(t) for t in range(current-1, start - 2, -1)]]
    monthlyPayRatio = []
    for Pay_OS in zip(currentPayment,previousOS):
        if Pay_OS[1]>0:
            payRatio = Pay_OS[0]*1.0 / Pay_OS[1]
            monthlyPayRatio.append(payRatio)
        else:
            monthlyPayRatio.append(1)
    if type == 'min payment ratio':
        return min(monthlyPayRatio)
    if type == 'max payment ratio':
        return max(monthlyPayRatio)
    if type == 'mean payment ratio':
        total_payment = sum(currentPayment)
        total_OS = sum(previousOS)
        if total_OS > 0:
            return total_payment / total_OS
        else:
            return 1

#################################
#   1, 读取数据，衍生初始变量   #
#################################
print("=" * 60)
print("第1阶段：读取数据，衍生初始变量")
print("=" * 60)

folderOfData = Path(__file__).resolve().parent
output_dir = folderOfData / 'outputs'
output_dir.mkdir(exist_ok=True)

trainData = pd.read_csv(folderOfData / 'traindata.csv', header=0)
testData = pd.read_csv(folderOfData / 'testData.csv', header=0)

print(f"训练集样本数: {trainData.shape[0]}, 特征数: {trainData.shape[1]}")
print(f"测试集样本数: {testData.shape[0]}, 特征数: {testData.shape[1]}")
print(f"训练集违约率: {trainData['label'].mean():.4f}")
print(f"测试集违约率: {testData['label'].mean():.4f}")
print()


allFeatures = []
print("开始衍生逾期类型特征...")
'''
逾期类型的特征在行为评分卡（预测违约行为）中，一般是非常显著的变量。
通过设定时间窗口，可以衍生以下类型的逾期变量：
'''
# 考虑过去1个月，3个月，6个月，12个月
for t in [1,3,6,12]:
    print(f"  - 正在生成{t}个月窗口的逾期特征...")
    # 1，过去t时间窗口内的最大逾期状态
    allFeatures.append('maxDelqL'+str(t)+"M")
    trainData['maxDelqL'+str(t)+"M"] = trainData.apply(lambda x: DelqFeatures(x,t,'max delq'),axis=1)

    # 2，过去t时间窗口内的，M0,M1,M2的次数
    allFeatures.append('M0FreqL' + str(t) + "M")
    trainData['M0FreqL' + str(t) + "M"] = trainData.apply(lambda x: DelqFeatures(x,t,'M0 times'),axis=1)

    allFeatures.append('M1FreqL' + str(t) + "M")
    trainData['M1FreqL' + str(t) + "M"] = trainData.apply(lambda x: DelqFeatures(x, t, 'M1 times'), axis=1)

    allFeatures.append('M2FreqL' + str(t) + "M")
    trainData['M2FreqL' + str(t) + "M"] = trainData.apply(lambda x: DelqFeatures(x, t, 'M2 times'), axis=1)



print("开始衍生额度使用率类型特征...")
'''
额度使用率类型特征在行为评分卡模型中，通常是与违约高度相关的
'''
# 考虑过去1个月，3个月，6个月，12个月
for t in [1,3,6,12]:
    print(f"  - 正在生成{t}个月窗口的额度使用率特征...")
    # 1，过去t时间窗口内的最大月额度使用率
    allFeatures.append('maxUrateL' + str(t) + "M")
    trainData['maxUrateL' + str(t) + "M"] = trainData.apply(lambda x: UrateFeatures(x,t,'max utilization rate'),axis = 1)

    # 2，过去t时间窗口内的平均月额度使用率
    allFeatures.append('avgUrateL' + str(t) + "M")
    trainData['avgUrateL' + str(t) + "M"] = trainData.apply(lambda x: UrateFeatures(x, t, 'mean utilization rate'),
                                                            axis=1)

    # 3，过去t时间窗口内，月额度使用率增加的月份。该变量要求t>1
    if t > 1:
        allFeatures.append('increaseUrateL' + str(t) + "M")
        trainData['increaseUrateL' + str(t) + "M"] = trainData.apply(lambda x: UrateFeatures(x, t, 'increase utilization rate'),
                                                                axis=1)

print("开始衍生还款类型特征...")
'''
还款类型特征也是行为评分卡模型中常用的特征
'''
# 考虑过去1个月，3个月，6个月，12个月
for t in [1,3,6,12]:
    print(f"  - 正在生成{t}个月窗口的还款特征...")
    # 1，过去t时间窗口内的最大月还款率
    allFeatures.append('maxPayL' + str(t) + "M")
    trainData['maxPayL' + str(t) + "M"] = trainData.apply(lambda x: PaymentFeatures(x, t, 'max payment ratio'),
                                                            axis=1)

    # 2，过去t时间窗口内的最小月还款率
    allFeatures.append('minPayL' + str(t) + "M")
    trainData['minPayL' + str(t) + "M"] = trainData.apply(lambda x: PaymentFeatures(x, t, 'min payment ratio'),
                                                          axis=1)

    # 3，过去t时间窗口内的平均月还款率
    allFeatures.append('avgPayL' + str(t) + "M")
    trainData['avgPayL' + str(t) + "M"] = trainData.apply(lambda x: PaymentFeatures(x, t, 'mean payment ratio'),
                                                          axis=1)


print(f"\n特征工程完成！总共生成了 {len(allFeatures)} 个特征:")
print(f"特征列表: {allFeatures[:10]}..." if len(allFeatures) > 10 else f"特征列表: {allFeatures}")

print("\n检查类别型变量与bad的相关度:")
'''
类别型变量：过去t时间内最大的逾期状态
需要检查与bad的相关度
'''
print("maxDelqL1M 各值的违约率:")
print(trainData.groupby(['maxDelqL1M'])['label'].mean())
print("\nmaxDelqL3M 各值的违约率:")
print(trainData.groupby(['maxDelqL3M'])['label'].mean())
print("\nmaxDelqL6M 各值的违约率:")
print(trainData.groupby(['maxDelqL6M'])['label'].mean())
print("\nmaxDelqL12M 各值的违约率:")
print(trainData.groupby(['maxDelqL12M'])['label'].mean())





############################
#   2, 分箱，计算WOE并编码   #
############################
print("\n" + "=" * 60)
print("第2阶段：分箱，计算WOE并编码")
print("=" * 60)

'''
对类别型变量的分箱和WOE计算
可以通过计算取值个数的方式判断是否是类别型变量
'''
categoricalFeatures = []
numericalFeatures = []
WOE_IV_dict = {}
print("正在区分类别型变量和数值型变量...")
for var in allFeatures:
    if len(set(trainData[var])) > 5:
        numericalFeatures.append(var)
    else:
        categoricalFeatures.append(var)

print(f"类别型变量数量: {len(categoricalFeatures)}")
print(f"数值型变量数量: {len(numericalFeatures)}")
print(f"类别型变量: {categoricalFeatures}")

print("\n检查类别型变量的单调性...")
not_monotone =[]
for var in categoricalFeatures:
    #检查bad rate在箱中的单调性
    if not scorecard_function.BadRateMonotone(trainData, var, 'label'):
        not_monotone.append(var)
print(f"不满足单调性的变量: {not_monotone}")


print("\n处理M2FreqL3M变量...")
print("M2FreqL3M 各值的样本数量:")
print(trainData.groupby(['M2FreqL3M'])['label'].count())   #其中，M2FreqL3M＝3总共只有3个样本，因此要进行合并

print("\n将 M2FreqL3M>=1的合并为一组...")
# 将 M2FreqL3M>=1的合并为一组，计算WOE和IV
trainData['M2FreqL3M_Bin'] = trainData['M2FreqL3M'].apply(lambda x: int(x>=1))
print("M2FreqL3M_Bin 各值的违约率:")
print(trainData.groupby(['M2FreqL3M_Bin'])['label'].mean())
WOE_IV_dict['M2FreqL3M_Bin'] = scorecard_function.CalcWOE(trainData, 'M2FreqL3M_Bin', 'label')

print("\n处理M1FreqL3M变量...")
#检查M1FreqL3M单调性
print("M1FreqL3M 各值的违约率:")
print(trainData.groupby(['M1FreqL3M'])['label'].mean())  
print("M1FreqL3M 各值的样本数量:")
print(trainData.groupby(['M1FreqL3M'])['label'].count())

print("\n将 M1FreqL3M>=1的合并为一组...")
trainData['M1FreqL3M_Bin'] = trainData['M1FreqL3M'].apply(lambda x: int(x>=1))
print("M1FreqL3M_Bin 各值的违约率:")
print(trainData.groupby(['M1FreqL3M_Bin'])['label'].mean())
WOE_IV_dict['M1FreqL3M_Bin'] = scorecard_function.CalcWOE(trainData, 'M1FreqL3M_Bin', 'label')


print("\n检查类别型变量的箱占比...")
'''
对其他单调的类别型变量，检查是否有一箱的占比低于5%。 如果有，将该变量进行合并
'''
small_bin_var = []
large_bin_var = []
N = trainData.shape[0]
for var in categoricalFeatures:
    if var not in not_monotone:
        total = trainData.groupby([var])[var].count()
        pcnt = total * 1.0 / N
        if min(pcnt)<0.05:
            small_bin_var.append({var:pcnt.to_dict()})
        else:
            large_bin_var.append(var)

print(f"有小箱的变量数量: {len(small_bin_var)}")
print(f"无小箱的变量数量: {len(large_bin_var)}")
print("有小箱的变量及其分布:")
for i in small_bin_var:
    print (i)


print("\n删除部分M2频次变量（因为箱分布不合理）...")
#对于M2FreqL1M、M2FreqL6M和M2FreqL12M，由于有部分箱占了很大比例，故删除
allFeatures.remove('M2FreqL1M')
allFeatures.remove('M2FreqL6M')
allFeatures.remove('M2FreqL12M')
print("已删除: M2FreqL1M, M2FreqL6M, M2FreqL12M")

print("\n对小箱变量进行重新分箱...")
#对于small_bin_var中的其他变量，将最小的箱和相邻的箱进行合并并计算WOE
trainData['maxDelqL1M_Bin'] = trainData['maxDelqL1M'].apply(lambda x: scorecard_function.MergeByCondition(x, ['==0', '==1', '>=2']))
trainData['maxDelqL3M_Bin'] = trainData['maxDelqL3M'].apply(lambda x: scorecard_function.MergeByCondition(x, ['==0', '==1', '>=2']))
trainData['maxDelqL6M_Bin'] = trainData['maxDelqL6M'].apply(lambda x: scorecard_function.MergeByCondition(x, ['==0', '==1', '>=2']))
print("正在计算 maxDelq 变量的WOE...")
for var in ['maxDelqL1M_Bin','maxDelqL3M_Bin','maxDelqL6M_Bin']:
    WOE_IV_dict[var] = scorecard_function.CalcWOE(trainData, var, 'label')
    print(f"  - {var} WOE计算完成")

print("\n对满足条件的类别型变量直接计算WOE...")    
'''
对于不需要合并、原始箱的bad rate单调的特征，直接计算WOE和IV
'''
for var in large_bin_var:
    WOE_IV_dict[var] = scorecard_function.CalcWOE(trainData, var, 'label')
    print(f"  - {var} WOE计算完成")


print("\n开始处理数值型变量分箱...")
'''
对于数值型变量，需要先分箱，再计算WOE、IV
分箱的结果需要满足：
1，箱数不超过5
2，bad rate单调
3，每箱占比不低于5%
'''
bin_dict = []
for var in numericalFeatures:
    print(f"  - 正在处理 {var}...")
    binNum = 5
    newBin = var + '_Bin'
    bin = scorecard_function.ChiMerge(trainData, var, 'label', max_interval=binNum, minBinPcnt = 0.05)
    trainData[newBin] = trainData[var].apply(lambda x: scorecard_function.AssignBin(x, bin))
    # 如果不满足单调性，就降低分箱个数
    original_binNum = binNum
    while not scorecard_function.BadRateMonotone(trainData, newBin, 'label'):
        binNum -= 1
        bin = scorecard_function.ChiMerge(trainData, var, 'label', max_interval=binNum, minBinPcnt=0.05)
        trainData[newBin] = trainData[var].apply(lambda x: scorecard_function.AssignBin(x, bin))
    
    if binNum < original_binNum:
        print(f"    调整分箱数: {original_binNum} -> {binNum}")
    else:
        print(f"    分箱数: {binNum}")
        
    WOE_IV_dict[newBin] = scorecard_function.CalcWOE(trainData, newBin, 'label')
    bin_dict.append({var:bin})
    print(f"    {var} 分箱和WOE计算完成")

##############################
#   3, 单变量分析和多变量分析   #
##############################
print("\n" + "=" * 60)
print("第3阶段：单变量分析和多变量分析")
print("=" * 60)

print(f"总共计算了 {len(WOE_IV_dict)} 个变量的WOE和IV")

#  选取IV高于0.02的变量
high_IV = [(k,v['IV']) for k,v in WOE_IV_dict.items() if v['IV'] >= 0.02]
high_IV_sorted = sorted(high_IV, key=lambda k: k[1],reverse=True)

print(f"\nIV大于0.02的变量有 {len(high_IV)} 个:")
for (var,iv) in high_IV_sorted:
    print(f"  - {var}: {iv:.4f}")

print("\n生成WOE编码变量...")
for (var,iv) in high_IV:
    newVar = var+"_WOE"
    trainData[newVar] = trainData[var].map(lambda x: WOE_IV_dict[var]['WOE'][x])
print("\n多变量分析：检查变量间相关性...")
'''
多变量分析：比较两两线性相关性。如果相关系数的绝对值高于阈值，剔除IV较低的一个
'''
deleted_index = []
cnt_vars = len(high_IV_sorted)
high_corr_pairs = []

for i in range(cnt_vars):
    if i in deleted_index:
        continue
    x1 = high_IV_sorted[i][0]+"_WOE"
    for j in range(cnt_vars):
        if i == j or j in deleted_index:
            continue
        y1 = high_IV_sorted[j][0]+"_WOE"
        roh = np.corrcoef(trainData[x1],trainData[y1])[0,1]
        if abs(roh)>0.7:
            high_corr_pairs.append((high_IV_sorted[i][0], high_IV_sorted[j][0], roh))
            x1_IV = high_IV_sorted[i][1]
            y1_IV = high_IV_sorted[j][1]
            if x1_IV > y1_IV:
                deleted_index.append(j)
            else:
                deleted_index.append(i)

if high_corr_pairs:
    print("发现高相关性变量对:")
    for pair in high_corr_pairs:
        print(f"  - {pair[0]} vs {pair[1]}: 相关系数 = {pair[2]:.4f}")
else:
    print("未发现高相关性变量对")

single_analysis_vars = [high_IV_sorted[i][0]+"_WOE" for i in range(cnt_vars) if i not in deleted_index]
print(f"\n经相关性筛选后，剩余 {len(single_analysis_vars)} 个变量:")
for var in single_analysis_vars:
    print(f"  - {var.replace('_WOE', '')}")


print("\n绘制相关性热力图...")
X = trainData[single_analysis_vars]
f, ax = plt.subplots(figsize=(10, 8))
corr = X.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax)
print("相关性热力图已生成")

print("\n多变量分析：计算VIF...")
'''
多变量分析：VIF
'''
X = np.matrix(trainData[single_analysis_vars])
VIF_list = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
max_vif = max(VIF_list)
print(f"最大VIF值: {max_vif:.3f}")
if max_vif < 10:
    print("所有变量的VIF值都小于10，无多重共线性问题")
else:
    print("存在多重共线性问题，需要进一步处理")
# 最大的VIF是 3.429，小于10，因此这一步认为没有多重共线性
multi_analysis = single_analysis_vars



################################
#   4, 建立逻辑回归模型预测违约   #
################################
print("\n" + "=" * 60)
print("第4阶段：建立逻辑回归模型预测违约")
print("=" * 60)

print(f"使用 {len(multi_analysis)} 个变量建立逻辑回归模型...")
X = trainData[multi_analysis].copy()
X['intercept'] = [1] * X.shape[0]
y = trainData['label']
logit = sm.Logit(y, X)
logit_result = logit.fit()
pvalues = logit_result.pvalues
params = logit_result.params
fit_result = pd.concat([params,pvalues],axis=1)
fit_result.columns = ['coef','p-value']
fit_result = fit_result.sort_values(by = 'coef')
print("\n逻辑回归结果:")
print(fit_result)


print("\n检查个别变量的单独回归系数:")
print(f"increaseUrateL3M 系数: {sm.Logit(y, trainData['increaseUrateL3M_WOE']).fit().params[0]:.4f}")  
print(f"minPayL6M_Bin 系数: {sm.Logit(y, trainData['minPayL6M_Bin_WOE']).fit().params[0]:.4f}")  
print(f"avgUrateL12M_Bin 系数: {sm.Logit(y, trainData['avgUrateL12M_Bin_WOE']).fit().params[0]:.4f}") 
print(f"minPayL1M_Bin 系数: {sm.Logit(y, trainData['minPayL1M_Bin_WOE']).fit().params[0]:.4f}")  
print(f"M0FreqL6M_Bin 系数: {sm.Logit(y, trainData['M0FreqL6M_Bin_WOE']).fit().params[0]:.4f}")  
print(f"minPayL3M_Bin 系数: {sm.Logit(y, trainData['minPayL3M_Bin_WOE']).fit().params[0]:.4f}") 

print("\n使用GBDT计算特征重要性...")
clf = ensemble.GradientBoostingClassifier(random_state=42)
gbdt_model = clf.fit(X, y)
importace = gbdt_model.feature_importances_.tolist()
featureImportance = zip(multi_analysis,importace)
featureImportanceSorted = sorted(featureImportance, key=lambda k: k[1],reverse=True)

print("GBDT特征重要性排序:")
for i, (feature, importance) in enumerate(featureImportanceSorted):
    print(f"  {i+1}. {feature.replace('_WOE', '')}: {importance:.4f}")

pd.DataFrame(featureImportanceSorted).plot(kind='bar')
print("特征重要性柱状图已生成")


print("\n逐步回归特征选择...")
# 先假定模型可以容纳4个特征，再逐步增加特征个数，直到有特征的系数为正，或者p值超过0.1
n = 4
featureSelected = [i[0] for i in featureImportanceSorted[:n]]
print(f"初始选择前{n}个特征: {[f.replace('_WOE', '') for f in featureSelected]}")

X_train = X[featureSelected+['intercept']]
logit = sm.Logit(y, X_train)
logit_result = logit.fit()
pvalues = logit_result.pvalues
params = logit_result.params
fit_result = pd.concat([params,pvalues],axis=1)
fit_result.columns = ['coef','p-value']
print(f"\n前{n}个特征的回归结果:")
print(fit_result)
print("\n逐步添加特征...")
while(n<len(featureImportanceSorted)):
    nextVar = featureImportanceSorted[n][0]
    featureSelected = featureSelected + [nextVar]
    X_train = X[featureSelected+['intercept']]
    logit = sm.Logit(y, X_train)
    logit_result = logit.fit()
    params = logit_result.params
    print(f"尝试添加变量: {nextVar.replace('_WOE', '')} - 系数: {params[nextVar]:.6f}")
    if max(params) < 0:
        print(f"  保留变量 {nextVar.replace('_WOE', '')}")
        n += 1
    else:
        print(f"  移除变量 {nextVar.replace('_WOE', '')} (系数为正)")
        featureSelected.remove(nextVar)
        n += 1

print(f"\n最终选择的特征数量: {len(featureSelected)}")
print(f"最终特征列表: {[f.replace('_WOE', '') for f in featureSelected]}")

X_train = X[featureSelected+['intercept']]
logit = sm.Logit(y, X_train)
logit_result = logit.fit()
pvalues = logit_result.pvalues
params = logit_result.params
fit_result = pd.concat([params,pvalues],axis=1)
fit_result.columns = ['coef','p-value']
fit_result = fit_result.sort_values(by  = 'p-value')
print("\n逐步回归最终结果:")
print(fit_result)


print("\n使用L1正则化进行进一步特征选择...")
X2 = X[featureSelected+['intercept']]
for alpha in range(100,0,-1):
    l1_logit = sm.Logit.fit_regularized(sm.Logit(y, X2), start_params=None, method='l1', alpha=alpha)
    pvalues = l1_logit.pvalues
    params = l1_logit.params
    if max(pvalues)>=0.1 or max(params)>0:
        break

bestAlpha = alpha + 1
print(f"最佳正则化参数 alpha: {bestAlpha}")

l1_logit = sm.Logit.fit_regularized(sm.Logit(y, X2), start_params=None, method='l1', alpha=bestAlpha)
params = l1_logit.params
params2 = params.to_dict()
featuresInModel = [k for k, v in params2.items() if k!='intercept' and v < -0.0000001]
print(f"L1正则化后保留的特征: {[f.replace('_WOE', '') for f in featuresInModel]}")
print(f"特征数量: {len(featuresInModel)}")

print("\n建立最终逻辑回归模型...")
X_train = X[featuresInModel + ['intercept']]
logit = sm.Logit(y, X_train)
logit_result = logit.fit()
trainData['pred'] = logit_result.predict(X_train)

print("\n计算训练集模型效果...")
train_ks = scorecard_function.KS(trainData, 'pred', 'label')
train_auc = roc_auc_score(trainData['label'],trainData['pred'])
print(f"训练集 KS: {train_ks:.4f}")
print(f"训练集 AUC: {train_auc:.4f}")

##################################
#   5，在测试集上测试逻辑回归的结果   #
###################################
print("\n" + "=" * 60)
print("第5阶段：在测试集上测试逻辑回归的结果")
print("=" * 60)

# 准备WOE编码后的变量
modelFeatures = [i.replace('_Bin','').replace('_WOE','') for i in featuresInModel]

numFeatures = [i for i in modelFeatures if i in numericalFeatures]
charFeatures = [i for i in modelFeatures if i in categoricalFeatures]

print(f"模型中的原始特征: {modelFeatures}")
print(f"数值型特征: {numFeatures}")
print(f"类别型特征: {charFeatures}")

print("\n为测试集生成特征...")
testData['maxDelqL1M'] = testData.apply(lambda x: DelqFeatures(x,1,'max delq'),axis=1)
testData['maxDelqL3M'] = testData.apply(lambda x: DelqFeatures(x,3,'max delq'),axis=1)
testData['M0FreqL3M'] = testData.apply(lambda x: DelqFeatures(x,3,'M0 times'),axis=1)
testData['M1FreqL6M'] = testData.apply(lambda x: DelqFeatures(x, 6, 'M1 times'), axis=1)
testData['M2FreqL3M'] = testData.apply(lambda x: DelqFeatures(x, 3, 'M2 times'), axis=1)
testData['avgUrateL1M'] = testData.apply(lambda x: UrateFeatures(x,1, 'mean utilization rate'),axis=1)
testData['avgUrateL3M'] = testData.apply(lambda x: UrateFeatures(x,3, 'mean utilization rate'),axis=1)
testData['increaseUrateL6M'] = testData.apply(lambda x: UrateFeatures(x, 6, 'increase utilization rate'),axis=1)
print("测试集特征生成完成")
print("\n对测试集进行分箱和WOE编码...")
testData['M2FreqL3M_Bin'] = testData['M2FreqL3M'].apply(lambda x: int(x>=1))
testData['maxDelqL1M_Bin'] = testData['maxDelqL1M'].apply(lambda x: scorecard_function.MergeByCondition(x, ['==0', '==1', '>=2']))
testData['maxDelqL3M_Bin'] = testData['maxDelqL3M'].apply(lambda x: scorecard_function.MergeByCondition(x, ['==0', '==1', '>=2']))

for var in numFeatures:
        newBin = var+"_Bin"
        bin = list([i.values() for i in bin_dict if var in i][0])[0]
        testData[newBin] = testData[var].apply(lambda x: scorecard_function.AssignBin(x, bin))
        print(f"  - {var} 分箱完成")

finalFeatures = [i+'_Bin' for i in numFeatures] + ['M2FreqL3M_Bin','maxDelqL1M_Bin','maxDelqL3M_Bin','M0FreqL3M']
print(f"\n需要WOE编码的特征: {finalFeatures}")

for var in finalFeatures:
    var2 = var+"_WOE"
    testData[var2] = testData[var].apply(lambda x: WOE_IV_dict[var]['WOE'][x])
    print(f"  - {var} WOE编码完成")

print("\n在测试集上应用模型...")
X_test = testData[featuresInModel].copy()
X_test['intercept'] = [1]*X_test.shape[0]
testData['pred'] = logit_result.predict(X_test)

print("\n计算测试集模型效果...")
test_ks = scorecard_function.KS(testData, 'pred', 'label')
test_auc = roc_auc_score(testData['label'],testData['pred'])
print(f"测试集 KS: {test_ks:.4f}")
print(f"测试集 AUC: {test_auc:.4f}")


##########################
#   6，在测试集上计算分数   #
##########################
print("\n" + "=" * 60)
print("第6阶段：在测试集上计算分数")
print("=" * 60)

BasePoint, PDO = 500, 50
print(f"评分基准: 基础分数 = {BasePoint}, PDO = {PDO}")

testData['score'] = testData['pred'].apply(lambda x: scorecard_function.Prob2Score(x, BasePoint, PDO))

print(f"\n测试集评分统计:")
print(f"最低分数: {testData['score'].min():.0f}")
print(f"最高分数: {testData['score'].max():.0f}")
print(f"平均分数: {testData['score'].mean():.1f}")
print(f"分数标准差: {testData['score'].std():.1f}")

# 分析不同分数段的违约率
score_ranges = [(0, 400), (400, 450), (450, 500), (500, 550), (550, 600), (600, 1000)]
score_band_rows = []
print(f"\n各分数段违约率分析:")
for low, high in score_ranges:
    mask = (testData['score'] >= low) & (testData['score'] < high)
    if mask.sum() > 0:
        bad_rate = testData[mask]['label'].mean()
        count = mask.sum()
        score_band_rows.append({'score_min': low, 'score_max': high, 'sample_count': count, 'bad_rate': bad_rate})
        print(f"  {low}-{high}: 样本数={count}, 违约率={bad_rate:.4f}")

pd.DataFrame([
    {'dataset': 'train', 'ks': train_ks, 'auc': train_auc, 'sample_count': len(trainData), 'bad_rate': trainData['label'].mean()},
    {'dataset': 'test', 'ks': test_ks, 'auc': test_auc, 'sample_count': len(testData), 'bad_rate': testData['label'].mean()},
]).to_csv(output_dir / 'metrics.csv', index=False)
pd.DataFrame({'feature': logit_result.params.index, 'coefficient': logit_result.params.values}).to_csv(
    output_dir / 'model_coefficients.csv', index=False
)
pd.DataFrame(score_band_rows).to_csv(output_dir / 'score_bands.csv', index=False)
trainData[['CUST_ID', 'label', 'pred']].to_csv(output_dir / 'train_scores.csv', index=False)
testData[['CUST_ID', 'label', 'pred', 'score']].to_csv(output_dir / 'test_scores.csv', index=False)
with open(output_dir / 'model_rules.pkl', 'wb') as f:
    pickle.dump({'features': featuresInModel, 'bin_dict': bin_dict, 'woe_iv': WOE_IV_dict,
                 'params': logit_result.params.to_dict(), 'base_point': BasePoint, 'pdo': PDO}, f)
print(f"基线结果已保存至: {output_dir}")

plt.hist(testData['score'], bins=100)
plt.title('Score Distribution')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()
print("评分分布直方图已生成")

print("\n" + "=" * 60)
print("评分卡建模完成！")
print("=" * 60)
