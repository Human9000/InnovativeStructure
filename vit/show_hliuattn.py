import matplotlib.pyplot as plt

import numpy as np
 

plt.figure(figsize=(10, 6))

for i, d in enumerate( [64, 128, 256, 512, 768,  1024, ]): 
    n = np.linspace(0, 2*d, 1000)
    y1 = (n**2) * d
    y2 = n*(d**2)
    # y2 = np.min([y1, y2], axis=0)
    # y2 = np.min([y1, y2], axis=0)
    plt.subplot(2, 3, i+1)
    plt.plot(n, y1, 'b-', label=f'SA(d={d})')
    plt.plot(n, y2, 'b--', label=f'LA-PNF(d={d})')
    plt.scatter(d, d**3, c='r', marker='o', )
     # 设置 x 轴标签
    plt.xlabel("N", fontsize=10,  labelpad=0)
    # 设置 y 轴标签
    plt.ylabel("Flops", fontsize=10,   labelpad=0)
    plt.text(d*1.1, d**3 * 0.6, f'({d}, {d**3:.2e})', fontsize=9, color='black') # 调整坐标显示位置
    plt.legend()
    # 显示网格线
    # plt.grid(True)
plt.tight_layout()
plt.show()