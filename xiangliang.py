import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义三维点和向量的数据
points = [
    [-20.614698835973602, 19.373652302425526, -10.487693089421272],  # position1
    [-19.333608669180833, 21.027267311534096, 1.2994046686864642],  # position2
    [-14.975583272428599, 22.9876203197415, 12.705691899378316],  # position3
    [7.881531296712342, 22.549836324361276, 16.729263060478686],  # position4
    [16.119200343099227, 20.735475065959996, 7.970446349130512],  # position5
    [22.15070101208714, 18.33787377771725, -3.7083484228852397]  # position6
]
vectors = [
    [0.06931579820342743, 0.9675905512552262, -0.24282472123123894],  # vector1
    [0.09386657861388521, 0.9624498809797253, -0.25471413785150765],  # vector2
    [0.05767652440371166, 0.9719556255907315, -0.2279817545665387],  # vector3
    [0.06639506434288829, 0.9611507957765063, -0.26791947150047385],  # vector4
    [0.027319041072632055, 0.9741422124058094, -0.22427799714636779],  # vector5
    [0.019092224774177265, 0.9772534065235468, -0.21121379308959784]  # vector6
]

scale_factor = 3  # 调整缩放比例
vectors = [[scale_factor * x for x in vector] for vector in vectors]

# 分离坐标
x = [point[0] for point in points]
y = [point[1] for point in points]
z = [point[2] for point in points]
u = [vector[0] for vector in vectors]
v = [vector[1] for vector in vectors]
w = [vector[2] for vector in vectors]

# 创建三维图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制点
ax.scatter(x, y, z, color='blue', label='点',s=150)

# 绘制向量
for i in range(len(points)):
    ax.arrow(x[i], y[i], z[i], u[i], v[i], w[i], color='green', linewidth=2, length_includes_head=True, mutation_scale=20)

# 隐藏坐标轴刻度线
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# 隐藏坐标轴标签
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_zlabel("")

# 隐藏图例
# plt.legend()  # 如果不需要图例，可以注释掉这一行

# 关闭网格线显示
ax.grid(False)

# 隐藏坐标轴平面
# 注意：这里使用了正确的属性名 xaxis, yaxis, zaxis
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

ax.xaxis.line.set_visible(False)
ax.yaxis.line.set_visible(False)
ax.zaxis.line.set_visible(False)

# 显示图形
plt.show()