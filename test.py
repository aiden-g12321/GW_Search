import numpy as np

time = np.linspace(0, 10, 100)
time_center = 5
indxt_around = np.where((time >= time_center - 2) & (
            time < time_center + 2))
print(indxt_around)