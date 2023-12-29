import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft, ifft, fftshift

# Открываем файл в бинарном режиме
with open('var08_z1.bin', 'rb') as f:
    # Читаем данные из файла
    y = np.fromfile(f, dtype=np.int16)

# Теперь 'y' содержит данные из файла в виде массива numpy


# Применяем окно Блэкмана-Ханна
window = signal.windows.blackmanharris(len(y))
y_windowed = y * window

# Вычисляем взвешенное текущее усреднение
window_size = 10000  # 10 секунд при частоте дискретизации 1 кГц
weights = np.repeat(1.0, window_size)/window_size
y_avg = np.convolve(y_windowed, weights, 'valid')

# Детрендирование (доп задание)

detr_values = [(y_windowed[s] - y_windowed[s-1]) for s in range(1, len(y_windowed))]


# Построение графика детрендирования

plt.plot(detr_values)
plt.xlabel('Отсчёты')
plt.ylabel('Значения')
plt.title('Детрендирование данных')
plt.grid()


# Строим графики
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(y_windowed)
plt.title('Исходный сигнал')

plt.subplot(2, 1, 2)
plt.plot(y_avg)
plt.title('Усредненный сигнал')

plt.tight_layout()
plt.show()