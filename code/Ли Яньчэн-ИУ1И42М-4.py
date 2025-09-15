import matplotlib
matplotlib.use('TkAgg') # 设置matplotlib后端为TkAgg，确保GUI兼容性/Установите для бэкенда matplotlib значение TkAgg, чтобы обеспечить совместимость с графическим интерфейсом.
import mne # EEG数据处理库/Библиотека обработки данных ЭЭГ
import numpy as np # 数值计算/численный расчёт
import matplotlib.pyplot as plt # 绘图/чертежи
from scipy.signal import butter, filtfilt # 滤波器设计/Дизайн фильтров
import pywt # 小波变换/вейвлет-преобразование (математика)
import pandas as pd # 数据读取/操作/Считывание данных/операции
import os # 文件路径管理/Управление путями файлов


def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    # 根据信号长度动态调整滤波器阶数/Динамически настраивает порядок фильтров в зависимости от длины сигнала.
    if len(data) < 2 * order: # 动态调整滤波器阶数以适配短信号/Динамически настраивает порядок фильтрации в соответствии с номером SMS
        order = len(data) // 2
    b, a = butter_lowpass(cutoff, fs, order=order)
    # 手动设置 padlen 为信号长度的一半，但不超过 512/Вручную установите padlen равным половине длины сигнала, но не более 512
    padlen = min(len(data) // 2, 512) # 限制padlen防止短信号溢出/Ограничение padlen для предотвращения переполнения номера SMS
    y = filtfilt(b, a, data, padlen=padlen) # 零相移滤波/Фильтрация с нулевым фазовым сдвигом
    return y


# 从桌面加载 EDF 格式的脑电图记录. Загрузка записей ЭЭГ в формате EDF с рабочего стола
file_path = 'C:/Users/LYC15/Desktop/32/eeg1.edf' # 请替换为实际的文件路径. Пожалуйста, замените фактический путь к файлу
raw = mne.io.read_raw_edf(file_path, preload=True) # 读取EDF原始数据并预加载至内存. Считывание необработанных данных EDF и предварительная загрузка их в память

# 确定癫痫发作的位置/Определите местоположение приступа
# 假设注释文件是 CSV 格式，每列代表一个病人 ID，0 表示无癫痫发作，1 表示有癫痫发作
# каждый столбец представляет собой идентификатор пациента, где 0 означает отсутствие припадков, а 1 - их наличие.
annotations_file_path = 'C:/Users/LYC15/Desktop/32/annotations_2017_A.csv'
annotations_df = pd.read_csv(annotations_file_path)

patient_id = annotations_df.columns[0]
patient_annotations = annotations_df[patient_id].values

# 假设采样频率相同，根据采样频率确定每个样本对应的时间点/Определите момент времени, соответствующий каждому образцу, на основе частоты отбора образцов
fs = raw.info['sfreq'] # 获取采样频率 (Hz)/Частота выборки (Гц)
time_points = np.arange(len(patient_annotations)) / fs # 生成时间轴 (秒)/Создайте временную шкалу (секунды)

# 找出癫痫发作的索引/Выясните индекс приступов
seizure_indices = np.where(patient_annotations == 1)[0]

if len(seizure_indices) > 0:
    # 找到连续的癫痫发作段/Нахождение последовательных сегментов приступов
    seizure_segments = [] # 存储连续发作段的容器/Контейнеры для хранения последовательных сегментов
    current_segment = [seizure_indices[0]]
    for i in range(1, len(seizure_indices)):
        if seizure_indices[i] == seizure_indices[i - 1] + 1: # 连续性检测/Проверка на непрерывность
            current_segment.append(seizure_indices[i])
        else:
            seizure_segments.append(current_segment)
            current_segment = [seizure_indices[i]]
    seizure_segments.append(current_segment)

    # 处理每个癫痫发作段/Управление каждым сегментом приступа
    for idx, segment in enumerate(seizure_segments):
        tmin = time_points[segment[0]] # 段起始时间 (秒)/Время начала сегмента (секунды)
        tmax = time_points[segment[-1]]  # 段结束时间 (秒)/Время окончания сегмента (секунды)

        # 截取癫痫发作期EEG数据/Перехват данных ЭЭГ во время припадков
        raw_seizure = raw.copy().crop(tmin=tmin, tmax=tmax)

        # 通道平均降维/Уменьшение среднего значения по каналу
        data = raw_seizure.get_data() # 获取多通道数据/Получение многоканальных данных
        average_signal = np.mean(data, axis=0) # 通道平均 → (n_samples,)/Среднее значение по каналу → (n_образцов,)

        # 从信号中去除所有高于 60 Hz 的频率
        cutoff = 60
        filtered_data = butter_lowpass_filter(average_signal, cutoff, fs)

        # 创建一个包含 3 个子图的画布/Создает холст с 3 подграфами.
        fig, axes = plt.subplots(3, 1, figsize=(10, 18))

        # 绘制癫痫发作时脑电图时间依赖性图表/Построение графика временной зависимости электроэнцефалограммы во время припадков
        # 时域信号图/Карта сигналов во временной области.
        axes[0].plot(raw_seizure.times, average_signal)
        axes[0].set_xlabel('Время (секунды)')
        axes[0].set_ylabel('Амплитуда')
        axes[0].set_title(f' {patient_id} {idx + 1}  Средний сигнал ЭЭГ пациента {patient_id} во время {idx + 1}-го приступа эпилепсии')
        axes[0].grid(True)

        # 1）构建信号的频谱图/Постройте спектрограмму сигнала
        axes[1].specgram(filtered_data, Fs=fs)
        axes[1].set_xlabel('Время (секунды)')
        axes[1].set_ylabel('Частота (Герц)')
        axes[1].set_title(f' {patient_id} {idx + 1} Спектрограмма отфильтрованного сигнала ЭЭГ пациента {patient_id} во время {idx + 1}-го приступа эпилепсии')

        # 2）构建小波变换（scalegram）/Построение вейвлет-преобразования
        scales = np.arange(1, 128)
        coeffs, freqs = pywt.cwt(filtered_data, scales, 'morl', sampling_period=1 / fs)
        im = axes[2].imshow(np.abs(coeffs), extent=[0, len(filtered_data) / fs, np.min(freqs), np.max(freqs)],
                            cmap='viridis', aspect='auto', origin='lower')
        axes[2].set_xlabel('Время (секунды)')
        axes[2].set_ylabel('Частота (Герц)')
        axes[2].set_title(f'{patient_id} {idx + 1} Вейвлет - грамма отфильтрованного сигнала ЭЭГ пациента {patient_id} во время {idx + 1}-го приступа эпилепсии')
        fig.colorbar(im, ax=axes[2])

        plt.tight_layout()
        save_path = os.path.join(os.getcwd(), f'combined_seizure_{idx + 1}_{patient_id}.png')
        plt.savefig(save_path)
        plt.close()

else:
    print(f'患者 {patient_id} 没有记录到癫痫发作。 / У пациента {patient_id} не зарегистрировано приступов эпилепсии.')
