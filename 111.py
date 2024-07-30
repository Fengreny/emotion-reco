import matplotlib.pyplot as plt
import numpy as np

# 数据
x = np.arange(1, 6)
y = [2, 4, 6, 8, 10]
z = [1, 3, 5, 7, 9]

# 绘制折线图和条形图
fig, ax = plt.subplots()
ax.plot(x, y, color='blue', linewidth=2, linestyle='--', marker='o', markersize=8, label='折线图')
ax.bar(x, z, color='orange', alpha=0.7, label='条形图')

# 添加标题和标签
ax.set_title("折线图和条形图示例", fontsize=16, fontweight='bold')
ax.set_xlabel("X轴", fontsize=14)
ax.set_ylabel("Y轴", fontsize=14)
ax.tick_params(axis='both', labelsize=12)
ax.legend(fontsize=12)

# 显示图形
plt.show()







