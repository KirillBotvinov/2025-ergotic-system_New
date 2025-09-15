import os

os.environ["MNE_CONFIG"] = os.path.join(os.getcwd(), ".mne")  # 设置环境变量
os.makedirs(".mne", exist_ok=True)  # 创建目录（如果不存在）

import pandas as pd
import numpy as np
import mne
# print(f"配置文件路径: {mne.get_config_path()}")

import matplotlib.pyplot as plt  # 必须添加的导入
import matplotlib
matplotlib.use('Qt5Agg')

from scipy import signal
import pywt

from mpl_toolkits.mplot3d import Axes3D  # 添加3D绘图支持
from matplotlib import ticker
from matplotlib.ticker import FuncFormatter
import mpl_toolkits.axes_grid1 as axes_grid1
from matplotlib.figure import Figure
from matplotlib.colorbar import Colorbar
def plot_3d_wavelet(coefficients, frequencies, times, start_time, file_name):
    """绘制三维小波变换结果"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 下采样策略（平衡性能与细节）
    time_step = 5  # 时间方向下采样
    freq_step = 1  # 频率方向下采样
    X, Y = np.meshgrid(times[::time_step], frequencies[::freq_step])
    Z = np.abs(coefficients[::freq_step, ::time_step])
    # 创建频率颜色映射
    norm = plt.Normalize(Y.min(), Y.max())
    cmap = plt.get_cmap('Spectral')  # 光谱色图增强频率区分
    # 绘制散点图
    # 点云绘制参数
    sc = ax.scatter3D(
        X, Y, Z,
        c=Y,  # 幅度值着色
        cmap=cmap,  # 高对比度色图
        s=5,  # 增大点尺寸
        alpha=0.5,  # 提高不透明度
        edgecolor='w',  # 白色边框增强对比
        linewidth=0.2,
        depthshade=False  # 禁用深度阴影
    )
    # 频率轴优化
    ax.set_ylim(0.5, 80)  # 聚焦癫痫相关频段
    # 频率轴标注
    ax.set_ylabel('Frequency (Hz)', fontsize=12, labelpad=18)
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.tick_params(axis='y', which='major', labelsize=10, pad=8)
    # 时间轴优化
    ax.set_xlabel('Time (s)', fontsize=10, labelpad=12)
    ax.tick_params(axis='x', labelsize=10, pad=8)
    ax.set_zlabel('Amplitude', fontsize=10, labelpad=12)
    ax.tick_params(axis='z', labelsize=10, pad=8)
    # 调整平面透明度
    ax.xaxis.pane.set_alpha(0.4)
    ax.yaxis.pane.set_alpha(0.4)
    ax.zaxis.pane.set_alpha(0.4)
    # 设置平面颜色
    ax.xaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor('lightgray')
    ax.yaxis.pane.set_facecolor('whitesmoke')
    # 添加主要频带标注
    for freq in [0.5, 4, 8, 12, 30, 60]:
        ax.plot(times, [freq] * len(times), zs=0, zdir='z',
                color='dimgrey', linestyle=':', alpha=0.5, linewidth=0.8)
        ax.text(times[-1] + 0.5, freq, 0, f'{freq}Hz',
                fontsize=8, color='maroon', va='center')
    # 添加时间刻度线
    ax.xaxis._axinfo['grid'].update({'linewidth': 0.5, 'alpha': 0.3})
    # === 图例优化 ===
    cbar = fig.colorbar(sc, pad=0.1, aspect=30)
    cbar.set_label('Frequency (Hz)', rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=9)
    # === 视角优化 ===
    ax.view_init(elev=28, azim=-45)  # 最佳观察角度
    plt.savefig(file_name, dpi=200, bbox_inches='tight')
    plt.close()

def plot_dual_axis_scalogram(coef, scales, times, wavelet_name, sfreq, file_name):
    fig = plt.figure(figsize=(12, 6))
    ax1 = axes_grid1.host_axes([0.1, 0.1, 0.8, 0.8], figure=fig)
    axc = fig.add_axes([0.1, 0.05, 0.8, 0.03])
    im1 = ax1.imshow(
        np.abs(coef),
        cmap='jet',
        aspect='auto',
        interpolation='bilinear',
        extent=[times[0], times[-1], max(scales), min(scales)],
        vmax=np.abs(coef).max(),
        vmin=-np.abs(coef).max())
    Colorbar(axc, im1, orientation='horizontal')
    ax1.set_yticks(np.linspace(min(scales), max(scales), num=12))
    ax2 = ax1.twinx()
    ax2.axis["top"].toggle(ticklabels=False)
    # 频率计算器
    formatter = FuncFormatter(
        lambda x, pos: '{:0.2f}'.format(pywt.scale2frequency(wavelet_name, x) * sfreq)
    )
    ax2.yaxis.set_major_formatter(formatter)
    ax2.set_ylim(ax1.get_ylim())
    ax2.yaxis.set_major_locator(ticker.LinearLocator(numticks=15))
    ax2.set_ylabel('частота')
    ax1.set_ylabel('масштаб')
    ax2.set_xlabel('время')
    ax1.set_title("сигнал в области масштаба и частоты")
    fig.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close(fig)
def process_segment(data, sfreq, start_time, duration):
    """处理单个数据段并生成三联图并保存"""
    import matplotlib.gridspec as gridspec
    # 通道平均
    avg_signal = np.mean(data, axis=0)
    # 应用60Hz低通滤波
    sos = signal.butter(4, 60, 'lowpass', fs=sfreq, output='sos')
    filtered = signal.sosfilt(sos, avg_signal)
    # 小波变换
    scales = np.arange(1, 128)
    wavelet_name = 'morl'
    coefficients, _ = pywt.cwt(filtered, scales, wavelet_name,
                               sampling_period=1.0 / sfreq)
    center_freq = pywt.central_frequency(wavelet_name)
    frequencies = center_freq / (scales * (1.0 / sfreq))
    times = np.linspace(0, duration, coefficients.shape[1])
    # === 创建三联图排版 ===
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 1)
    # 子图1：频谱图
    ax1 = fig.add_subplot(gs[0])
    ax1.specgram(filtered, Fs=sfreq, NFFT=256, noverlap=128, cmap='viridis')
    ax1.set_title(f'Spectrogram @ {start_time}s')
    ax1.set_ylabel('Frequency (Hz)')
    # === 子图3：Dual-axis scale-frequency map ===
    from mpl_toolkits.axes_grid1 import host_subplot
    import mpl_toolkits.axisartist as AA
    from matplotlib.ticker import FuncFormatter
    host = host_subplot(gs[1], axes_class=AA.Axes)
    par = host.twinx()
    im3 = host.imshow(
        np.abs(coefficients),
        cmap='jet',
        aspect='auto',
        interpolation='bilinear',
        extent=[times[0], times[-1], max(scales), min(scales)],
        vmax=np.abs(coefficients).max(),
        vmin=-np.abs(coefficients).max()
    )
    # Axis label: left = scale, right = frequency
    host.set_ylabel('Scale')
    par.set_ylabel('Frequency (Hz)')
    host.set_xlabel('Time (s)')
    # Map scale -> frequency
    formatter = FuncFormatter(
        lambda x, pos: '{:0.2f}'.format(pywt.scale2frequency(wavelet_name, x) * sfreq)
    )
    par.yaxis.set_major_formatter(formatter)
    par.set_ylim(host.get_ylim())
    par.yaxis.set_major_locator(plt.MaxNLocator(10))
    fig.colorbar(im3, ax=host, orientation='vertical', label='Amplitude')
    # 标题
    plt.suptitle(f"EEG Signal Analysis @ {start_time}s", fontsize=14)
    # 保存图像
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"combined_analysis_{start_time}.png", dpi=300)
    plt.close(fig)
    # 仍然保留单独的 3D 图
    plot_3d_wavelet(coefficients, frequencies, times, start_time,
                    f"3d_scatter_{start_time}.png")

def find_seizure_periods(annotations):
    """将离散标注点转换为连续时间段"""
    periods = []
    start = None
    for i, val in enumerate(annotations):
        if val == 1 and start is None:
            start = i + 1  # 时间从1秒开始
        elif val == 0 and start is not None:
            periods.append((start, i))
            start = None
    if start is not None:
        periods.append((start, len(annotations)))
    return periods


def main():
    # 读取临床信息文件
    clinical = pd.read_csv("clinical_information.csv")
    eeg17_info = clinical[clinical["EEG file"] == "eeg17"].iloc[0]

    # 检查是否有发作标注
    if eeg17_info["Number of Reviewers Annotating Seizure"] == 0:
        print("该文件无癫痫发作标注")
        return

    # 读取EDF文件元数据
    raw = mne.io.read_raw_edf("eeg17.edf", preload=False, verbose="ERROR")
    total_seconds = int(raw.n_times / raw.info["sfreq"])
    print(f"EEG总时长: {total_seconds}秒")
    sfreq = raw.info['sfreq']
    # 读取三个标注文件
    annotations = []
    for expert in ["A", "B", "C"]:
        df = pd.read_csv(f"annotations_2017_{expert}.csv")
        eeg17_col = df.columns.get_loc(str(eeg17_info["ID"]))  # 根据ID定位列
        annotations.append(df.iloc[:, eeg17_col].values[:total_seconds])
    # 生成共识标注（至少两个专家同意）
    consensus = (np.array(annotations).sum(axis=0) >= 2).astype(int)
    # 转换时间段并验证
    periods = find_seizure_periods(consensus)
    valid_periods = [(s, e) for s, e in periods if e <= total_seconds]

    # 输出结果
    for idx, (start, end) in enumerate(valid_periods):
        print(f"{start} - {end}秒")
        # 计算显示范围（发作前后各延伸5秒）
        display_start = max(0, start - 5)  # 起始时间不小于0
        display_end = min(end + 5, total_seconds)  # 结束时间不超过总时长
        duration = display_end - display_start
        # 创建图形对象
        fig = raw.plot(
            start=display_start,
            duration=duration,
            title=f"癫痫发作时段 {start}-{end}秒\n（显示范围: {display_start}-{display_end}秒）",
            scalings=dict(eeg=100e-6),  # 100μV/div的缩放比例
            show=False,  # 禁止弹出窗口
            block=False,  # 非阻塞模式
            verbose="ERROR"  # 隐藏控制台输出
        )
        # 保存为高分辨率图片
        fig.savefig(
            f"seizure_{start}-{end}.png",
            dpi=300,  # 提高分辨率
            bbox_inches="tight",  # 紧凑布局
            facecolor="white"  # 白色背景
        )
        plt.close(fig)

        # 提取原始数据段
        data, _ = raw[:, int(start * sfreq):int(end * sfreq)]

        # 处理并生成分析结果
        process_segment(data, sfreq, start, duration)

if __name__ == "__main__":
    matplotlib.use('Agg')
    main()