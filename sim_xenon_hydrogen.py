import numpy as np
import matplotlib.pyplot as plt

# Параметры
N = 100  # Количество слоев
H = 1000  # Высота столба (м)
dh = H / N  # Толщина слоя
g = 9.81  # Ускорение свободного падения (м/с²)
R = 8.314  # Универсальная газовая постоянная (Дж/(моль·К))
T0 = 300  # Начальная температура (К)
P0 = 101325  # Начальное давление (Па)
A = 1  # Площадь поперечного сечения столба (м²)

# Параметры газов
M_Xe = 0.13129  # Молярная масса ксенона (кг/моль)
Cp_Xe = 20.79  # Удельная теплоемкость ксенона (Дж/(моль·К))

M_H2 = 0.002016  # Молярная масса водорода (кг/моль)
Cp_H2 = 28.82  # Удельная теплоемкость водорода (Дж/(моль·К))

# Инициализация
h = np.linspace(0, H, N)

# Ксенон
P_Xe = np.full(N, P0)
T_Xe = np.full(N, T0)
rho_Xe = P_Xe / (R * T_Xe)
m_Xe = rho_Xe * A * dh

# Водород
P_H2 = np.full(N, P0)
T_H2 = np.full(N, T0)
rho_H2 = P_H2 / (R * T_H2)
m_H2 = rho_H2 * A * dh

# Итерационный процесс
for _ in range(1000):
    for i in range(N - 1):
        # Ксенон
        P_Xe[i + 1] = P_Xe[i] + rho_Xe[i] * g * dh
        T_Xe[i + 1] = T_Xe[i] - (M_Xe * g / Cp_Xe) * dh

        # Водород
        P_H2[i + 1] = P_H2[i] + rho_H2[i] * g * dh
        T_H2[i + 1] = T_H2[i] - (M_H2 * g / Cp_H2) * dh

    # Обновляем плотность и массу
    rho_Xe = P_Xe / (R * T_Xe)
    m_Xe = rho_Xe * A * dh

    rho_H2 = P_H2 / (R * T_H2)
    m_H2 = rho_H2 * A * dh

# Визуализация
plt.figure(figsize=(12, 8))

# Давление
plt.subplot(2, 2, 1)
plt.plot(h, P_Xe, label='Ксенон', color='blue')
plt.plot(h, P_H2, label='Водород', color='red')
plt.xlabel('Высота (м)')
plt.ylabel('Давление (Па)')
plt.legend()

# Температура
plt.subplot(2, 2, 2)
plt.plot(h, T_Xe, label='Ксенон', color='blue')
plt.plot(h, T_H2, label='Водород', color='red')
plt.xlabel('Высота (м)')
plt.ylabel('Температура (К)')
plt.legend()

# Плотность
plt.subplot(2, 2, 3)
plt.plot(h, rho_Xe, label='Ксенон', color='blue')
plt.plot(h, rho_H2, label='Водород', color='red')
plt.xlabel('Высота (м)')
plt.ylabel('Плотность (кг/м³)')
plt.legend()

# Масса
plt.subplot(2, 2, 4)
plt.plot(h, m_Xe, label='Ксенон', color='blue')
plt.plot(h, m_H2, label='Водород', color='red')
plt.xlabel('Высота (м)')
plt.ylabel('Масса газа (кг)')
plt.legend()

plt.tight_layout()
plt.show()