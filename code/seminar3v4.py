import pandas as pd
import matplotlib.pyplot as plt
from h2o.exceptions import H2ODependencyWarning, H2ODeprecationWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import (confusion_matrix, f1_score, roc_curve, auc,
                             precision_recall_curve, PrecisionRecallDisplay)
import h2o
from h2o.automl import H2OAutoML
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # 过滤字体类警告
warnings.filterwarnings("ignore", category=H2ODependencyWarning)  # H2O转换警告
warnings.filterwarnings("ignore", category=H2ODeprecationWarning)  # H2O弃用警告
warnings.filterwarnings("ignore", category=FutureWarning)  # 其他未来警告

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用系统支持的英文字体
plt.rcParams['axes.unicode_minus'] = False

# 数据加载与清洗
def load_clean_data():
    raw_table_data = pd.read_csv('ecg_data.csv', nrows=5000)
    numeric_cols = [
        'Count_subj', 'rr_interval', 'p_onset', 'p_end',
        'qrs_onset', 'qrs_end', 'p_axis', 'qrs_axis', 't_axis'
    ]
    # 转换目标列为数值类型
    if 'Healthy_Status' in raw_table_data:
        raw_table_data['Healthy_Status'] = LabelEncoder().fit_transform(
            raw_table_data['Healthy_Status'].astype(str)
        )
    else:
        raise KeyError("数据集中缺少目标列'Healthy_Status'")
    # 逐步过滤并打印样本量
    def apply_filter(df, condition, step_name):
        filtered = df[condition]
        # print(f"{step_name}后样本数: {len(filtered)}")
        print(f"{step_name}: {len(filtered)} samples remaining")
        return filtered
    # 第一次过滤：数值范围
    condition_1 = (raw_table_data[numeric_cols] < 2000).all(axis=1)
    full_df_filtered = apply_filter(raw_table_data, condition_1, "数值范围过滤")
    # 第二次过滤：波形时序逻辑
    condition_2 = (
            (full_df_filtered['p_onset'] < full_df_filtered['p_end']) &
            (full_df_filtered['qrs_onset'] < full_df_filtered['qrs_end'])
    )
    full_df_filtered = apply_filter(full_df_filtered, condition_2, "时序逻辑过滤")
    # 第三次过滤：列级最大值
    for col in numeric_cols:
        condition = (full_df_filtered[col] <= 10000)
        full_df_filtered = apply_filter(full_df_filtered, condition, f"{col}列过滤")
        if len(full_df_filtered) == 0:
            raise ValueError(f"过滤条件过严，{col}列导致数据清空")
    # 最终特征列选择
    selected_cols = numeric_cols + ['Healthy_Status']
    return full_df_filtered[selected_cols]

# H2O AutoML
def h2o_automl(train_df, test_df):
    h2o.init()
    train_df['Healthy_Status'] = train_df['Healthy_Status'].astype('category')
    test_df['Healthy_Status'] = test_df['Healthy_Status'].astype('category')
    # 数据转换
    hf_train = h2o.H2OFrame(train_df)
    hf_test = h2o.H2OFrame(test_df)
    # 显式设置目标列为因子类型
    hf_train['Healthy_Status'] = hf_train['Healthy_Status'].asfactor()
    hf_test['Healthy_Status'] = hf_test['Healthy_Status'].asfactor()
    # 模型训练
    automl = H2OAutoML(max_runtime_secs=60, seed=17)
    automl.train(y='Healthy_Status', training_frame=hf_train)
    # 预测与评估
    preds = automl.leader.predict(hf_test)
    y_true = hf_test['Healthy_Status'].as_data_frame().values.astype(int).flatten()
    y_pred = preds['predict'].as_data_frame().values.astype(int).flatten()
    h2o_probs = preds['p1'].as_data_frame().values.flatten()  # 获取正类概率
    # 性能指标
    evaluate_model(y_true, y_pred, h2o_probs, 'H2O')
    h2o.shutdown()


# LightAutoML
def lightautoml_pipeline(full_df):
    # 数据分割
    train, test = train_test_split(full_df, test_size=0.2,
                                   stratify=full_df['Healthy_Status'],
                                   random_state=42)
    # 模型配置
    task = Task('binary')
    automl = TabularAutoML(task=task, timeout=100,
                           cpu_limit=4,
                           reader_params={'n_jobs': 4, 'cv': 3})
    # 训练与预测
    oof_pred = automl.fit_predict(train, roles={'target': 'Healthy_Status'})
    test_pred = automl.predict(test)
    test_probs = test_pred.data[:, 0]  # 正类概率
    y_pred = (test_probs >= 0.5).astype(int)
    y_true = test['Healthy_Status'].values.astype(int)
    # 结果可视化
    evaluate_model(y_true, y_pred, test_probs, 'LightAutoML')

import seaborn as sns
def plot_confusion_matrix(y_true, y_pred, framework_name):
    """绘制混淆矩阵热力图"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Unhealthy', 'Healthy'],
                yticklabels=['Unhealthy', 'Healthy'])
    plt.title(f'{framework_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def evaluate_model(y_true, y_pred, y_probs, framework_name):
    """生成评估指标并绘制图表"""
    # 打印分类报告
    print(f"\n{framework_name} Classification Report:") # 分类报告
    print(classification_report(y_true, y_pred, target_names=['Unhealthy', 'Healthy']))
    # 计算宏观F1和加权F1
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    if y_probs is not None:
        precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)

        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, f1_scores[:-1], color='blue', lw=2)
        plt.axvline(x=0.5, color='red', linestyle='--',
                    label='Default Threshold (0.5)')
        plt.xlabel('Classification Threshold') # 分类阈值
        plt.ylabel('F1 Score')
        plt.title(f'{framework_name} F1 Score vs Threshold') # F1分数随阈值变化
        plt.legend()
        plt.grid(True)
        plt.show()
    # 计算ROC AUC
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.4f}")
    # 绘制混淆矩阵
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Unhealthy', 'Healthy'],
                yticklabels=['Unhealthy', 'Healthy'])
    plt.title(f'{framework_name} Confusion Matrix') # 混淆矩阵
    plt.xlabel('Predicted') # 预测值
    plt.ylabel('Actual') # 真实值
    plt.show()

    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC Curve (AUC = {roc_auc:.2f})') # ROC曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate') # 假阳率
    plt.ylabel('True Positive Rate') # 真阳率
    plt.title(f'{framework_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

# %% 主执行流程
if __name__ == "__main__":
    # 数据准备
    raw_df = load_clean_data()
    processed_df = raw_df
    # 数据集分割
    train_df, test_df = train_test_split(processed_df, test_size=0.3,
                                         stratify=processed_df['Healthy_Status'],
                                         random_state=17)
    # 执行AutoML
    h2o_automl(train_df, test_df)
    lightautoml_pipeline(processed_df)