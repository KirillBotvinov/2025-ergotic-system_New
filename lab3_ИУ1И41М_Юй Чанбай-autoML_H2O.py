import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars
import pyarrow
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, precision_recall_curve

# ==================== 环境配置 ====================
try:
    import polars
    import pyarrow
    _multi_thread_support = True
except ImportError:
    print("警告：未安装 polars/pyarrow，将使用单线程模式")
    _multi_thread_support = False

h2o.init()


# ==================== 数据加载 ====================
def load_data():
    try:
        df = pd.read_csv(
            'модуль 3 - датасет - практика.csv',
            nrows=5000,
            usecols=['Count_subj', 'rr_interval', 'p_end', 'qrs_onset',
                     'qrs_end', 'p_axis', 'qrs_axis', 't_axis', 'Healthy_Status'],
            encoding='utf-8'
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            'модуль 3 - датасет - практика.csv',
            nrows=5000,
            usecols=['Count_subj', 'rr_interval', 'p_end', 'qrs_onset',
                     'qrs_end', 'p_axis', 'qrs_axis', 't_axis', 'Healthy_Status'],
            encoding='cp1251'
        )

    # 确保目标变量是字符串类型
    df['Healthy_Status'] = df['Healthy_Status'].astype(str)
    return df


df = load_data()

# ==================== H2O数据处理 ====================
hf = h2o.H2OFrame(df)
hf['Healthy_Status'] = hf['Healthy_Status'].asfactor()
train, test = hf.split_frame(ratios=[0.8], seed=123)

# ==================== AutoML训练 ====================
aml = H2OAutoML(
    max_models=10,
    seed=123,
    exclude_algos=["XGBoost"],
    max_runtime_secs=300,
    verbosity="info"
)
aml.train(x=hf.columns[:-1], y='Healthy_Status', training_frame=train)


# ==================== 数据转换 ====================
def convert_h2o_to_pdf(frame):
    try:
        return frame.as_data_frame(use_multi_thread=_multi_thread_support)
    except Exception as e:
        print(f"多线程转换失败: {str(e)}, 回退到单线程")
        return frame.as_data_frame()


test_df = convert_h2o_to_pdf(test)
pred = aml.leader.predict(test)
pred_df = convert_h2o_to_pdf(pred)

# ==================== 核心指标计算 ====================
cm = confusion_matrix(test_df['Healthy_Status'], pred_df['predict'])
f1 = f1_score(test_df['Healthy_Status'], pred_df['predict'], average='binary')

# ==================== 执行混淆矩阵可视化 ====================
plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})


def plot_confusion_matrix(cm):
    classes = sorted(test_df['Healthy_Status'].unique())
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title(f'Confusion Matrix\nF1 Score: {f1:.4f}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()


plot_confusion_matrix(cm)

# ==== 新增输出部分 ====
# 1. 输出最佳模型信息
best_model = aml.leader
print("\n" + "="*50)
print(f"最佳模型：{best_model.model_id}")
print("模型参数摘要：")
print(best_model.params.keys())  # 显示可用的参数配置
print("="*50 + "\n")

# 2. 显示训练模型列表
leaderboard = aml.leaderboard
print("模型排行榜：")
print(leaderboard.head())
print("\n训练的模型类型分布：")
model_types = [m.split("_")[0] for m in leaderboard.as_data_frame()['model_id']]
print(pd.Series(model_types).value_counts())

# ==================== 资源清理 ====================
h2o.cluster().shutdown(prompt=False)
