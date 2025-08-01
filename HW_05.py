#HW5
# %% (2)
import numpy as np
import matplotlib.pyplot as plt

def S(x,y):
    s = 2*(2-x**2-y**2)
    return s 

#parameters
alpha = 1 #열확산 계수
N = M = 20 # 격자 수

# 격자 생성
x = np.linspace(-1,1,N)
y = np.linspace(-1,1,M)

# 격자 좌표 생성
X,Y = np.meshgrid(x,y) 

h = x[1] - x[0] # 격자 간격

t = 0 #initial time
dt = 0.01 #time step

beta = alpha * dt / (2*h**2)

# 초기 phiϕ 설정
phi = np.zeros((N, N)) # 초기값 설정
I = np.eye(N-2) # 내부 격자에 대한 단위 행렬
# 외부 격자는 phi=0이라는 조건이 주어졌으므로, 해를 구해야 할 내부 격자점만 (n-2)개 (경계 제외)

# 내부 격자에 대한 행렬 (N-2)x(N-2)
L = np.zeros((N-2, N-2)) 
for i in range(N-2):
    L[i,i] = -2
    if i-1 >= 0:
        L[i,i-1] = 1
    if i+1 < N-2:
        L[i,i+1] = 1    

# 행렬 A 정의
A = I - beta * L 

# 시간에 따른 phiϕ값을 저장할 리스트
phi_list = [phi,]

# 시각화할 인덱스 선택 (1~149 중 10개)
j_list = np.linspace(1,149,10, dtype=int)  

for j in range(150):
    t = t + dt

    R = S(X[1:-1,1:-1], Y[1:-1,1:-1])*dt + (I + beta * L) @ ((I + beta * L) @ phi_list[j][1:-1, 1:-1].T).T
    psi = np.linalg.solve(A, R)  # solve the linear system
    phi_new = np.dot(psi, np.linalg.inv(A.T))

    # 전체 phi 행렬에 내부 영역만 채워 넣음
    phi_full = np.zeros((N, N))  # initialize full pi
    phi_full[1:-1, 1:-1] = phi_new
    phi_list.append(phi_full)

# phi 전체 범위 계산 (루프 이후)
phi_min = np.min([np.min(p) for p in phi_list])
phi_max = np.max([np.max(p) for p in phi_list])

# 시각화
for j in j_list:
    plt.figure(figsize=(8,6))
    plt.contourf(X, Y, phi_list[j], levels=20, cmap='hot_r', vmin=phi_min, vmax=phi_max)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Time = {j*dt:.2f}')
    plt.show()

def phi_exact(x,y):
    return (1-x**2)*(1-y**2)

# exact solution 시각화
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, phi_exact(X,Y), levels=20, cmap='hot_r', vmin=0, vmax=1)
plt.colorbar(label='ϕ exact')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Exact Solution')
plt.show()

# %% (3)_space
# Space
import numpy as np
import matplotlib.pyplot as plt

def S(x,y):
    s = 2*(2-x**2-y**2)
    return s 

def phi_exact(x,y):
    return (1-x**2)*(1-y**2)

h_list = []
error_list = []

for N in range(5,20,1):

    alpha = 1 
    x_list = np.linspace(-1, 1, N) # x grid points 
    y_list = np.linspace(-1, 1, N) # y grid points 
    X, Y = np.meshgrid(x_list, y_list) # create a meshgrid
    h = x_list[1] - x_list[0] # grid spacing
    dt = 0.005 # time step
    t = 0 # initial time
    beta = alpha * dt / (h**2) / 2 

    phi = np.zeros((N, N)) # initialize pi       
    I = np.eye(N-2) # identity matrix
    L = np.zeros((N-2, N-2)) # initialize pi

    phi_list = [phi,]

    for i in range(N-2):
        L[i, i] = -2  # 주대각선

        if i - 1 >= 0:
            L[i, i-1] = 1  # 왼쪽 이웃

        if i + 1 < (N-2):
            L[i, i+1] = 1  # 오른쪽 이웃

    A = (I - beta * L)

    for j in range(1000):
        t = t + dt
        
        R = S(X[1:-1,1:-1], Y[1:-1,1:-1])*dt \
         + (I + beta * L) @ ((I + beta * L) @ phi_list[j][1:-1, 1:-1].T).T
        psi = np.linalg.solve(A, R)  # solve the linear system
        phi_new = np.dot(psi, np.linalg.inv(A.T))
        phi_full = np.zeros((N, N))  # initialize full pi
        phi_full[1:-1, 1:-1] = phi_new
        phi_list.append(phi_full)
        
    error_list.append(np.linalg.norm(phi_list[-1] - phi_exact(X, Y), 2)*h)
    h_list.append(h)

plt.figure(figsize=(8,6))
plt.plot(np.log10(h_list), np.log10(error_list), marker='o', label='Error')
plt.xlabel('Grid Spacing (h)')
plt.ylabel('Error (L2 norm)') 
plt.plot(np.log10(h_list), 2*(np.log10(h_list)-np.log10(h_list[0]))+np.log10(error_list[0]), linestyle='--', color='r', label='Slope 2')
plt.title('Error Analysis (Space)')
plt.grid()


p = (np.log(error_list[0]) - np.log(error_list[1])) / \
    (np.log(h_list[0]) - np.log(h_list[1]))

print("공간 수렴 차수 p =", p)
# %% (3)_time
import numpy as np
import matplotlib.pyplot as plt

def S(x,y):
    s = 2*(2-x**2-y**2)
    return s 

def phi_exact(x,y):
    return (1-x**2)*(1-y**2)

#time
t_list = []
error_list = []
N = M = 20  # Fixed grid size for time analysis

for p in range(5, 10, 1):
    alpha = 1
    x_list = np.linspace(-1, 1, N)  # x grid points
    y_list = np.linspace(-1, 1, M)  # y grid points
    X, Y = np.meshgrid(x_list, y_list)  # create a meshgrid
    h = x_list[1] - x_list[0]  # grid spacing
    dt = 0.004 * (2 ** p)  # time step, increasing with p
    t = 0  # initial time
    beta = alpha * dt / (h ** 2) / 2

    phi = np.zeros((N, M))  # initialize phi
    I = np.eye(N - 2)  # identity matrix
    L = np.zeros((N - 2, N - 2))  # initialize L

    phi_list = [phi, ]

    for i in range(N - 2):
        L[i, i] = -2  # main diagonal

        if i - 1 >= 0:
            L[i, i - 1] = 1  # left neighbor

        if i + 1 < (N - 2):
            L[i, i + 1] = 1  # right neighbor

    A = (I - beta * L)

    for j in range(1000):
        t = t + dt
        
        R = S(X[1:-1,1:-1], Y[1:-1,1:-1])*dt + (I + beta * L) @ ((I + beta * L) @ phi_list[j][1:-1, 1:-1].T).T
        psi = np.linalg.solve(A, R)  # solve the linear system
        phi_new = np.dot(psi, np.linalg.inv(A.T))
        phi_full = np.zeros((N, M))  # initialize full pi
        phi_full[1:-1, 1:-1] = phi_new
        phi_list.append(phi_full)
    
    error_list.append(np.linalg.norm(phi_list[-1] - phi_exact(X, Y), 2) * h)
    t_list.append(dt)

# analysis order of accuracy
plt.figure(figsize=(8, 6))
plt.plot(np.log10(t_list), np.log10(error_list), marker='o', label='Error')
plt.xlabel('Grid Spacing (t)')
plt.ylabel('Error (L2 norm)')
plt.plot(np.log10(t_list), 2 * (np.log10(t_list) - np.log10(t_list[0])) + np.log10(error_list[0]), linestyle='--', color='r', label='Slope 2')
plt.title('Error Analysis (Time)')
plt.grid()
plt.legend()
plt.show()