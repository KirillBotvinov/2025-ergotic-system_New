import mne
import pywt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mne.preprocessing import ICA
from mne.datasets import eegbci

raw = mne.io.read_raw_edf('eeg56.edf', preload=True)
raw.plot(scalings = 'auto', show = False)
plt.show()
annotations = raw.annotations
print(raw.ch_names)

#从数据库注释中了解到达时间
#время приступ узнаем из аннотации из базы данных
eeg_data, eeg_times = raw.get_data(return_times=True)
print('число отчетов во временном ряду:',len(eeg_times))
print(annotations)

# 设置数据处理范围
# устанавливаем диапазон для обработки данных
start_time = 1000  # 根据实际需求调整起始时间（单位：秒）
end_time = 3000
t_index_begin = np.where(eeg_times > start_time)[0][0]
t_index_end = np.where(eeg_times > end_time)[0][0]

t = eeg_times[t_index_begin:t_index_end]
# 时间值 = 信号结束
# значение времени = конец сигнала
T = t[-1] - t[0]
# 时间序列中的元素数量
# число элементов во временном ряду
N = len(t)

# 从 EEG 数据数组中移除不必要的信号
# 在 EEG_21 信号通道 8 'Value MKR+-MKR-' 的情况下
# удалаем ненужные сигналы из массива данных ЭЭГ
# в случае сигнала EEG_21 8 канал 'Value MKR+-MKR-'
eeg_data = np.delete(eeg_data, [len(eeg_data)-1], axis = 0)

for i in range(len(raw.ch_names)-2):
    plt.plot(t, eeg_data[i, t_index_begin:t_index_end],linewidth = 0.1)
plt.show()

# 制作一个简单的平均信号 - 将所有通道放入一个数组中
# сделаем простой устредненный сигнал - все каналы в один массив
y=[]
for i in range(len(eeg_data)-2):
    y =+ eeg_data[i, t_index_begin:t_index_end]
y =y/(len(eeg_data)-1)
plt.plot(t,y)
plt.show()

# 巴特福德滤波器削减频率带宽
# фильтр Баттерфорда для вырезания полосы частот
from scipy import signal
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.lfilter(b, a, data)
    return filtered_data

# 预处理示例：信号过滤
# Пример предобработки: фильтрация сигнала
fs = len(t)/(T)  # 采样频率 частота дискретизации
lowcut = 1 # 低通滤波器截止频率 нижняя частота среза фильтра
highcut = 60 # 滤波器的上截止频率 верхняя частота среза фильтра
y_filt = np.apply_along_axis(butter_bandpass_filter, axis=0, arr=y, lowcut=lowcut, highcut=highcut, fs=fs)
plt.plot(t,y, t, y_filt)
plt.show()
y = y_filt

from scipy.fft import fft, fftfreq

window = np.hanning(N)  # 应用汉宁窗
y_windowed = y * window
yfft = fft(y_windowed)

#yfft = fft(y)
xf = fftfreq(N, T/N)[:N//2]
#yf = np.abs(yfft[0:N//2])
yf = np.abs(yfft[0:N//2]) / N  # 归一化幅值

# 设置傅立叶变换图上要显示的最大频率
# устанавливаем максимальную частоту для отображения на графике преобразования Фурье
f_viewmax = 60
if len(np.where(xf > f_viewmax)[0]) == 0:
  index_f_viewmax = len(xf)
else:
  index_f_viewmax = np.where(xf > f_viewmax)[0][0]
print('число точек в диапазоне преобразования Фурье:',len(xf))
print('максимальная частота', xf[-1])
# 傅立叶频谱关于零点是对称的，因此我们只取其右边部分
# спект Фурье симметричен относительно нуля, поэтому берем только правую его часть
plt.figure(figsize=(10, 5))
#plt.plot(xf[0:index_f_viewmax], yf[0:index_f_viewmax], label='FFT преобразование')
plt.stem(xf[0:index_f_viewmax], yf[0:index_f_viewmax],
         linefmt='b-', markerfmt=' ', basefmt=' ', label='FFT преобразование')
plt.xlabel('Частота[Герц]', fontsize=12)
plt.ylabel('спектр Фурье', fontsize=12)
#plt.xticks(np.arange(1, np.max(xf[0:100]), 2))
plt.grid()
plt.legend()
plt.show()

# 绘制功率密度频谱图
# Построение спектра плотности мощности
raw.plot_psd(fmin=1, fmax=f_viewmax, tmax=np.inf, show=False)
plt.title('Power Spectral Density (PSD)')
plt.show()

bands = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 12),
         'Beta': (12, 30), 'Gamma': (30, 60)}
for band, (f_low, f_high) in bands.items():
    mask = (xf >= f_low) & (xf <= f_high)
    power = np.trapz(yf[mask], xf[mask])
    print(f"{band} Band Power: {power:.2f} µV²/Hz")
    plt.bar(band, power)

plt.title('Energy in Different Frequency Bands')
plt.xlabel('Frequency Band')
plt.xticks(np.arange(5), ('Delta, 0.5..4', 'Theta, 4..8', 'Alpha, 8..13', 'Beta, 13..30', 'Gamma, 30..'))
plt.ylabel('Energy')
plt.show()

# 小波列表
wavlist_continuous = pywt.wavelist(kind='continuous')
wavlist_discrete = pywt.wavelist(kind='discrete')

scale_max = 300
scale_min = 3
# 带刻度的点数组（线性和对数刻度）
scales = np.linspace(scale_min, scale_max, num=25, endpoint=True)

wavelet_core = 'morl'
fs = len(t) / (T)  # 采样率
dt = 1 / fs
coef, freqs = pywt.cwt(y, scales, wavelet_core, sampling_period=dt)  #尺度，频率

f = pywt.scale2frequency(wavelet_core, scales)/dt

# 绘制频率和尺度依赖关系图
plt.figure(figsize=(7, 7))
plt.grid()
plt.yticks(np.arange(min(freqs), max(freqs), (max(freqs) - min(freqs))/10))
plt.xticks(np.arange(min(scales), max(scales), (max(scales) - min(scales))/10))
plt.ylabel('Frequency[Hertz]', fontsize=12)
plt.xlabel('scale in wavelet transform', fontsize=12)
plt.plot(scales, freqs,'.-')
plt.show()

plt.figure(figsize=(10, 7))

ax1 = plt.subplot(211)
plt.imshow(abs(coef), cmap='jet', aspect='auto', extent=[t[0], t[-1], max(scales), min(scales)], vmax=abs(coef).max(), vmin=abs(coef).min())
plt.gca().invert_yaxis ()
plt.ylabel('масштаб ', fontsize=12)

ax2 = plt.subplot(212, sharex=ax1)
plt.plot(t, y)
ax2.set_title("сигнал в области времени")

#Plotting scalogram
# 绘制示意图
plt.figure(figsize=(10, 5))
plt.imshow(abs(coef), extent=[t[0], t[-1], max(scales), min(scales)], interpolation='bilinear', cmap='plasma', aspect='auto')
plt.gca().invert_yaxis ()
plt.yticks(np.arange(min(scales), max(scales), (max(scales) - min(scales))/10))
plt.ylabel('масштаб ', fontsize=12)
plt.xlabel('время ', fontsize=12)
plt.show()

from matplotlib import colorbar
from matplotlib.ticker import FuncFormatter
import mpl_toolkits.axes_grid1 as axes_grid1
import matplotlib.ticker as ticker

#Plotting dual axis scalogram
#绘制双轴模拟图
f1 = plt.figure()
f1.set_size_inches(12, 6)


ax1 = axes_grid1.host_axes([0.1, 0.1, 0.8, 0.80])
axc = f1.add_axes([0.1, 0, 0.8, 0.05])
im1 = ax1.imshow(abs(coef), cmap='jet', aspect='auto', interpolation='bilinear',
                 extent=[t[0], t[-1], max(scales), min(scales)],
                 vmax=abs(coef).max(), vmin=-abs(coef).min())

colorbar.Colorbar(axc, im1, orientation='horizontal')

ax1.set_yticks(np.arange(min(scales), max(scales), (max(scales) - min(scales))/12))


ax2 = ax1.twinx()
# make ticklabels on the top invisible
# 隐藏顶部的标签
ax2.axis["top"].toggle(ticklabels=False)

formatter = FuncFormatter(lambda x, pos: '{:0.2f}'.format(pywt.scale2frequency(wavelet_core, x)/dt))
ax2.yaxis.set_major_formatter(formatter)
ax2.set_ylim(ax1.get_ylim())

# make number ticks what we want
# 按我们的要求进行数字勾选
ax2.yaxis.set_major_locator(ticker.LinearLocator(numticks = 15))


ax2.set_ylabel('частота')
ax1.set_ylabel('масштаб')
ax2.set_xlabel('время')

ax1.set_title("сигнал в области масштаба и частоты")

plt.show()

# Вычисляем дисперсию коэффициентов по каждому масштабу
# 计算每个量表的系数方差
dispersion = np.var(np.abs(coef), axis=1)

# Визуализируем
# 视觉化
plt.plot(scales, dispersion)
plt.xlabel('Масштаб')
plt.ylabel('Вейвлет-дисперсия')
plt.title('Дисперсия коэффициентов по масштабам')
plt.grid()
plt.show()
