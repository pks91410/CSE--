# HW5 - 시간에 따른 2D φ의 변화 애니메이션 (열확산) - mp4 저장
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter  # 🔄 mp4 저장용 추가

def S(x, y):
    return 2 * (2 - x**2 - y**2)

# 파라미터 설정
alpha = 1
N = M = 20  # 격자 수
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, M)
X, Y = np.meshgrid(x, y)

h = x[1] - x[0]
dt = 0.001
beta = alpha * dt / (2 * h**2)

# 초기 조건
phi = np.zeros((N, N))
I = np.eye(N - 2)
L = np.zeros((N - 2, N - 2))

for i in range(N - 2):
    L[i, i] = -2
    if i > 0:
        L[i, i - 1] = 1
    if i < N - 3:
        L[i, i + 1] = 1

A = I - beta * L

# 시간 루프
phi_list = [phi.copy()]
n_steps = 1500

for _ in range(n_steps):
    phi_old = phi_list[-1]
    R = S(X[1:-1, 1:-1], Y[1:-1, 1:-1]) * dt + \
        (I + beta * L) @ ((I + beta * L) @ phi_old[1:-1, 1:-1].T).T
    psi = np.linalg.solve(A, R)

    phi_new = np.zeros((N, N))
    phi_new[1:-1, 1:-1] = np.dot(psi, np.linalg.inv(A.T))
    phi_list.append(phi_new)

# 색상 범위 고정
phi_min = np.min([np.min(p) for p in phi_list])
phi_max = np.max([np.max(p) for p in phi_list])

# 애니메이션 생성
fig, ax = plt.subplots(figsize=(8, 8))
contour = ax.contourf(X, Y, phi_list[0], levels=20, cmap='hot_r', vmin=phi_min, vmax=phi_max)
title = ax.set_title(f"Time step 0, Time = 0.00")

# 업데이트 함수
def update(frame):
    ax.clear()
    contour = ax.contourf(X, Y, phi_list[frame], levels=20, cmap='hot_r', vmin=phi_min, vmax=phi_max)
    ax.set_title(f"Time step {frame}, Time = {frame * dt:.2f}")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    
# 애니메이션 생성
ani = animation.FuncAnimation(fig, update, frames=len(phi_list), interval=100, blit=False)

# 🔽 mp4 저장
writer = FFMpegWriter(fps=10, metadata=dict(artist='YourName'), bitrate=1800)
ani.save("heat_diffusion_dt=0.001.mp4", writer=writer)

plt.show()

# %%
