import numpy as np
import matplotlib.pyplot as plt

# Параметры
N = 100  # Количество слоев
H = 1000  # Высота столба (м)
dh = H / N  # Толщина слоя
M = 0.029  # Молярная масса воздуха (кг/моль)
g = 9.81  # Ускорение свободного падения (м/с²)
R = 8.314  # Универсальная газовая постоянная (Дж/(моль·К))
Cp = 29.1  # Удельная теплоемкость при постоянном давлении (Дж/(моль·К))
Cv = Cp - R  # Удельная теплоемкость при постоянном объеме (Дж/(моль·К))
T0 = 300  # Начальная температура (К)
P0 = 101325  # Начальное давление (Па)
A = 1  # Площадь поперечного сечения столба (м²)

# Инициализация
h = np.linspace(0, H, N)
P = np.full(N, P0)
T = np.full(N, T0)
rho = P / (R * T)
m = rho * A * dh  # Масса газа в каждом слое

# Функция для расчета полной энергии
def calculate_total_energy(h, m, T):
    U = m * Cv * T  # Внутренняя энергия
    E_pot = m * g * h  # Потенциальная энергия
    E_total = np.sum(U) + np.sum(E_pot)  # Полная энергия
    return E_total

# Итерационный процесс
E_total_initial = calculate_total_energy(h, m, T)  # Начальная полная энергия
E_total_history = [E_total_initial]  # История изменения полной энергии

for _ in range(1000):
    for i in range(N - 1):
        P[i + 1] = P[i] + rho[i] * g * dh
        T[i + 1] = T[i] - (M * g / Cp) * dh
    rho = P / (R * T)
    m = rho * A * dh  # Обновляем массу газа
    E_total = calculate_total_energy(h, m, T)
    E_total_history.append(E_total)

# Визуализация
plt.figure(figsize=(10, 10))

plt.subplot(4, 1, 1)
plt.plot(h, P, label='Давление')
plt.xlabel('Высота (м)')
plt.ylabel('Давление (Па)')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(h, T, label='Температура', color='orange')
plt.xlabel('Высота (м)')
plt.ylabel('Температура (К)')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(h, m, label='Масса газа', color='green')
plt.xlabel('Высота (м)')
plt.ylabel('Масса газа (кг)')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(E_total_history, label='Полная энергия', color='red')
plt.xlabel('Итерация')
plt.ylabel('Полная энергия (Дж)')
plt.legend()

plt.tight_layout()
plt.show()