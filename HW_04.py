# %%

#HW4 Stokes second problem

#(1)
from numpy import *
import matplotlib.pyplot as plt

# Parameters
vorticity = 1
n = 2
U0 = 1
L = 10
T = 10*pi
dt = 0.001

# nt list
nt = [0, pi/2, pi, 3*pi/2, 2*pi]
nt_labels = ['0', 'π/2', 'π', '3π/2', '2π']

# 인덱스 번호 계산
nt_idx = [int(round((nt_val/n)/dt)) for nt_val in nt] 
nt_quasi_idx = [int(round(((nt_val + T) / n ) / dt)) for nt_val in nt]
#u_FTCS[nt[i]/n] -> 에러
#인덱스는 정수이여야 하므로 실수 시간값을 그대로 사용할 수 없음.->int()
# ex) 인덱스 = 0.784 / 0.001 = 784 이므로 /dt한 것.
#round() -> 0.785/0.001 = 784.99999이므로 가장 가까운 인덱스를 찾기 위해서 round()를 먼저씀
#round(x): 숫자 x를 가장 가까운 정수로 반올림하는 함수임.
#int(round())-> 반올림해서 정확한 인덱스 정수값을 받기 위함.

# t grid 
t = arange(0,2*pi/n+dt,dt) 
t_quasi = arange(0, (T+2*pi)/n+dt, dt) #t나 t_quasi는 0.001 간격의 float 배열임.

dt_quasi = t_quasi[1] - t_quasi[0]
#위에서 dt를 0.001로 정의를 하였는데 t와 t_quasi 정의시 dt로 간격을 똑같이 맞춰줌.

# y grid(공간격자)
y = linspace(0,10,200) 
dy = y[1] - y[0]


#stability (안정성 기준 alpha <= 0.5)
alpha = vorticity * dt / dy**2  
print('Stability alpha =', alpha)
if alpha > 0.5:
    print("Unstable scheme")


def n_s(y):
    return sqrt(n/(2*vorticity))*y

def u_exact(nt_val,n_s):
    return U0*exp(-n_s)*cos(nt_val - n_s) #nt_val은 각profile당 해당하는 nt값을 넣겠다는 것.

def u_numeric(t, y):
    #초기화
    u = zeros((len(t),len(y)))
    u[0,:] = 0 # t = 0일때 속도 0 조건

    for p in range(len(t)-1):
        u[p+1,0]= cos(n*t[p]) #아래벽 y=0(y[0]) 일때 진동조건
        u[p+1,-1] = 0       #위쪽 벽 y=:(y[-1]) 고정(u=0) 
        for i in range(1,len(y)-1):
            u[p+1,i] = u[p,i] + alpha * (u[p,i+1] - 2*u[p,i] + u[p,i-1])

    return u

u_FTCS = u_numeric(t,y)
u_quasi = u_numeric(t_quasi,y)

for i in range(len(nt)): #nt = [0, pi/2, pi, 3*pi/2, 2*pi] 리스트임. 총5개 -> i = 0,1,2,3,4
    plt.plot(u_FTCS[nt_idx[i]],y,'-', label=f'FTCS t={nt_labels[i]}')
    plt.plot(u_quasi[nt_quasi_idx[i]],y,'--',label=f'Quasi-steady t={nt_labels[i]}')

plt.xlabel("u")
plt.ylabel('y')
plt.legend(loc='upper left')
plt.title('Velocity Profiles using FTCS and Quasi-steady State')
plt.grid(True)
plt.show()
            

# %%
#(2) Crank-Nicolson scheme (무조건 안정)

from numpy import *
import matplotlib.pyplot as plt

# Parameters
vorticity = 1
n = 2
U0 = 1
L = 10
T = 10*pi
dt = 0.01

# nt list
nt = [0, pi/2, pi, 3*pi/2, 2*pi]
nt_labels = ['0', 'π/2', 'π', '3π/2', '2π']

# t grid
t = arange(0, 2*pi/n+dt,dt)  #transient 과도상태
t_quasi = arange(0,(T + 2*pi)/n + dt, dt)
dt_quasi = t_quasi[1]-t_quasi[0]

# y grid(공간격자)
y = linspace(0,10,200) 
dy = y[1] - y[0]
ny = len(y)

# r 정의
r = vorticity * dt / dy**2 #t와 dt_quasi의 간격은 똑같이 dt로 정의로 해서 둘다 만족함.

#u배열 초기화(quasi만 저장해도 transient도 포함됨.)
u = zeros((len(t_quasi),ny))

# Matrix A,B
A = diag((1+r)*ones(ny)) + diag(-r/2 * ones(ny-1),1) + diag(-r/2*ones(ny-1),-1)
B = diag((1-r)*ones(ny)) + diag(r/2 *ones(ny-1),1) + diag(r/2 * ones(ny-1),-1)          
#만약 nyXny 의 정사각행렬일때 , 주 대각선은 ny개라면 위/아래 대각선은 ny-1개 임.
#diag()함수는 1차원 배열을 넘기면 2차우너 정방 행렬로 자동으로 확장을 해주기 떄문에,
#ones(ny=5)이면 자동으로 5X5 정방 행렬을 생성해줌.


#경계조건
A[0,:] = 0 ; A[0,0] = 1  #A행렬 1행 [1, 0, 0, 0]   
A[-1,:] = 0;  A[-1,-1] = 1 #A행렬 -1행 [ 0, 0, 0, 1]

B[0,:] = 0; B[0,0] = 1
B[-1,:] = 0;  B[-1,-1] = 1 
#쉽게 말하면 B의 행렬에서 경계조건이 들어가 위치에 1, 나머지는 0으로 만들어주는거다.

# time marching
for p in range(len(t_quasi)-1):
    u[p,0] = U0 * cos(n * t_quasi[p]) #경계조건 부여
    u[p+1,0] = U0 * cos(n* t_quasi[p+1]) 
#Crank-Nicolson법 식을 보면 t = p, t= p+1일떄 중앙차분을 이용함. 경계 내부 값들을 구하기 위해선, t= p, p+1 일때 양 끝 경계에서의 값을 알고 있어야 함. 

    u[p, -1] = 0 #경계조건 부여
    u[p+1,-1] = 0

    b = dot(B, u[p, :]) #Bu^n 계산
    b[0] = u[p+1,0] #B행렬에서 경계조건이 들어갈 위치에 1 넣어뒀으니까,
    b[-1] = 0 #그 위치에 해당하는 정확한 경계조건 값을 넣어주는 것.

    u[p+1,:] = linalg.solve(A,b) # Au^{n+1} = b  선형시스템 풀어서 그 해인 u^{p+1}값을 좌항 변수에 저장.

#index 계산
#nt의 값이 해당되는 부분의 인덱스 번호 계산
nt_idx = [int(round((nt_val/n)/dt)) for nt_val in nt] 
nt_quasi_idx = [int(round(((nt_val + T) / n ) / dt)) for nt_val in nt]

#Transient 상태
for i, idx in enumerate(nt_idx):
    plt.plot(u[idx, :], y, label=f't = {nt_labels[i]} (transient)', linestyle='-')
#enumerate()는 리스트나 배열을 동시에 반복하면서 동시에 인덱스와 값을 함께 꺼낼 때 사용.
# nt_idx의 경우 i =1 , idx = 0 / i = 1, idx = 79 을 뽑아서 쓴다~

# Quasi-Steady 상태 프로파일
for i, idx in enumerate(nt_quasi_idx):
    plt.plot(u[idx, :], y, label=f't = {nt_labels[i]} (quasi)', linestyle='--')

plt.title('Velocity Profiles use Crank-Nicolson (Transient vs Quasi-Steady)')
plt.xlabel('u(y,t)')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# %%
# (3) 각각 다른 두 scheme에 대하여 시간의 변화에 따른 해의 수렴율 구하시오.
#log(del t) vs log(l2 norm). FTCS는 시간에 대한 1차, Crank-Nicolson은 2차의 수렴율을 보여야 함.

from numpy import *
import matplotlib.pyplot as plt

# Parameters
vorticity = 1
n = 2
U0 = 1
L = 10
T = 10*pi


# nt list
nt = [0, pi/2, pi, 3*pi/2, 2*pi]
nt_labels = ['0', 'π/2', 'π', '3π/2', '2π']

# t grid 
dt_list = [0.001, 0.0005, 0.0001]
for dt in dt_list:
    t = arange(0,2*pi/n+dt,dt) 
    t_quasi = arange(0, (T+2*pi)/n+dt, dt) #t나 t_quasi는 0.001 간격의 float 배열임.
    nt_idx = [int(round((nt_val/n)/dt)) for nt_val in nt] #인덱스 번호 계산

# y grid(공간격자)
y = linspace(0,10,100) 
dy = y[1] - y[0]

alpha = vorticity * dt / dy**2 

def n_s(y):
    return sqrt(n/(2*vorticity))*y

def u_exact(nt_val,n_s):
    return U0*exp(-n_s)*cos(nt_val - n_s) #nt_val은 각profile당 해당하는 nt값을 넣겠다는 것.

def u_numeric(t, y,dt):
    #초기화
    u = zeros((len(t),len(y)))
    u[0,:] = 0 # t = 0일때 속도 0 조건
    alpha = vorticity * dt / dy**2  

    for p in range(len(t)-1):
        u[p+1,0]= cos(n*t[p]) #아래벽 y=0(y[0]) 일때 진동조건
        u[p+1,-1] = 0       #위쪽 벽 y=:(y[-1]) 고정(u=0) 
        for i in range(1,len(y)-1):
            u[p+1,i] = u[p,i] + alpha * (u[p,i+1] - 2*u[p,i] + u[p,i-1])

    return u

def crank_nicolson(t_grid, y_grid, n, U0,dt):
    # dy = y_grid[1] - y_grid[0]
    # dt = t_grid[1] - t_grid[0]
    ny = len(y_grid)
    u = zeros((len(t_grid), ny))
    r = vorticity * dt / dy**2  # stability condition

    # Matrix A, B
    A = diag((1 + r) * ones(ny)) + diag(-r / 2 * ones(ny - 1), 1) + diag(-r / 2 * ones(ny - 1), -1)
    B = diag((1 - r) * ones(ny)) + diag(r / 2 * ones(ny - 1), 1) + diag(r / 2 * ones(ny - 1), -1)

    A[0,:] = 0; A[0,0] = 1
    A[-1,:] = 0; A[-1,-1] = 1
    B[0,:] = 0; B[0,0] = 1
    B[-1,:] = 0; B[-1,-1] = 1

    for p in range(len(t_grid)-1):
        u[p,0] = U0 * cos(n * t_grid[p])
        u[p+1,0] = U0 * cos(n * t_grid[p+1])
        u[p,-1] = u[p+1,-1] = 0

        b = dot(B, u[p,:])
        b[0] = u[p+1,0]
        b[-1] = 0

        u[p+1,:] = linalg.solve(A, b)

    return u

####
def l2_norm_error(u_numeric,u_exact, dy):
    return sqrt(sum((u_numeric - u_exact)**2) * dy)

errors_FTCS = []
errors_CN = []

t_final = 2*pi/n
# y = linspace(0,10,200) 위에서 정의함.
#dy = y[1]- dy[0]
eta = n_s(y)

for dt in dt_list:
    t = arange(0,t_final + dt, dt)
    t_idx = int(round(t_final/dt))

    #수치해 계산
    u_FTCS = u_numeric(t, y,dt)
    u_CN = crank_nicolson(t, y, n, U0,dt)

    #정확해 계산
    t_exact = t[t_idx]  # 수치해와 정확히 같은 시간에서 정확해 평가
    eta = n_s(y)
    u_ex = u_exact(t_exact, eta)

    #L2 error 계산
    err_FTCS = l2_norm_error(u_FTCS[t_idx,:],u_ex,dy)
    err_CN = l2_norm_error(u_CN[t_idx,:],u_ex,dy)

    errors_FTCS.append(err_FTCS)
    errors_CN.append(err_CN)

log_dt = log10(dt_list)
log_err_ftcs = log10(errors_FTCS)
log_err_cn = log10(errors_CN)

p_ftcs = polyfit(log_dt, log_err_ftcs, 1)[0]
p_cn = polyfit(log_dt, log_err_cn, 1)[0]

# 수렴차수 출력
print(f"FTCS 수렴차수 p ≈ {p_ftcs:f}")
print(f"CN   수렴차수 p ≈ {p_cn:f}")

# FTCS 그래프
plt.figure()
plt.plot(log_dt, log_err_ftcs, 'o-', color='blue', label=f'FTCS (p ≈ {p_ftcs:.2f})')
plt.xlabel('log(Δt)')
plt.ylabel('log(L2 Error)')
plt.title('FTCS Convergence Rate')
plt.legend()
plt.grid(True)
plt.show()

# Crank-Nicolson 그래프
plt.figure()
plt.plot(log_dt, log_err_cn, 's--', color='orange', label=f'Crank-Nicolson (p ≈ {p_cn:.2f})')
plt.xlabel('log(Δt)')
plt.ylabel('log(L2 Error)')
plt.title('Crank-Nicolson Convergence Rate')
plt.legend()
plt.grid(True)
plt.show()

for dt, err in zip(dt_list, errors_CN):
    print(f"dt = {dt:.5f}, CN L2 Error = {err:.12e}")


# %%
#(4) L = 2 일때, 두 평판의 거리가 속도 profile에 미치는 영향]

from numpy import *
import matplotlib.pyplot as plt

# Parameters
vorticity = 1
n = 2
U0 = 1
L = 2
T = 10*pi
dt = 0.001

# nt list
nt = [0, pi/2, pi, 3*pi/2, 2*pi]
nt_labels = ['0', 'π/2', 'π', '3π/2', '2π']

# t grid
t = arange(0, 2*pi/n+dt,dt)  #transient 과도상태
t_quasi = arange(0,(T + 2*pi)/n + dt, dt)
dt_quasi = t_quasi[1]-t_quasi[0]

# y grid(공간격자)
y = linspace(0,L,200) 
dy = y[1] - y[0]
ny = len(y)

# r 정의
r = vorticity * dt / dy**2 #t와 dt_quasi의 간격은 똑같이 dt로 정의로 해서 둘다 만족함.

#u배열 초기화(quasi만 저장해도 transient도 포함됨.)
u = zeros((len(t_quasi),ny))

# Matrix A,B
A = diag((1+r)*ones(ny)) + diag(-r/2 * ones(ny-1),1) + diag(-r/2*ones(ny-1),-1)
B = diag((1-r)*ones(ny)) + diag(r/2 *ones(ny-1),1) + diag(r/2 * ones(ny-1),-1)          
#만약 nyXny 의 정사각행렬일때 , 주 대각선은 ny개라면 위/아래 대각선은 ny-1개 임.
#diag()함수는 1차원 배열을 넘기면 2차우너 정방 행렬로 자동으로 확장을 해주기 떄문에,
#ones(ny=5)이면 자동으로 5X5 정방 행렬을 생성해줌.


#경계조건
A[0,:] = 0 ; A[0,0] = 1  #A행렬 1행 [1, 0, 0, 0]   
A[-1,:] = 0;  A[-1,-1] = 1 #A행렬 -1행 [ 0, 0, 0, 1]

B[0,:] = 0; B[0,0] = 1
B[-1,:] = 0;  B[-1,-1] = 1 
#쉽게 말하면 B의 행렬에서 경계조건이 들어가 위치에 1, 나머지는 0으로 만들어주는거다.

# time marching
for p in range(len(t_quasi)-1):
    u[p,0] = U0 * cos(n * t_quasi[p]) #경계조건 부여
    u[p+1,0] = U0 * cos(n* t_quasi[p+1]) 
#Crank-Nicolson법 식을 보면 t = p, t= p+1일떄 중앙차분을 이용함. 경계 내부 값들을 구하기 위해선, t= p, p+1 일때 양 끝 경계에서의 값을 알고 있어야 함. 

    u[p, -1] = 0 #경계조건 부여
    u[p+1,-1] = 0

    b = dot(B, u[p, :]) #Bu^n 계산
    b[0] = u[p+1,0] #B행렬에서 경계조건이 들어갈 위치에 1 넣어뒀으니까,
    b[-1] = 0 #그 위치에 해당하는 정확한 경계조건 값을 넣어주는 것.

    u[p+1,:] = linalg.solve(A,b) # Au^{n+1} = b  선형시스템 풀어서 그 해인 u^{p+1}값을 좌항 변수에 저장.

#index 계산
#nt의 값이 해당되는 부분의 인덱스 번호 계산
nt_idx = [int(round((nt_val/n)/dt)) for nt_val in nt] 
nt_quasi_idx = [int(round(((nt_val + T) / n ) / dt)) for nt_val in nt]

#Transient 상태
for i, idx in enumerate(nt_idx):
    plt.plot(u[idx, :], y, label=f't = {nt_labels[i]} (transient)', linestyle='-')
#enumerate()는 리스트나 배열을 동시에 반복하면서 동시에 인덱스와 값을 함께 꺼낼 때 사용.
# nt_idx의 경우 i =1 , idx = 0 / i = 1, idx = 79 을 뽑아서 쓴다~

# Quasi-Steady 상태 프로파일
for i, idx in enumerate(nt_quasi_idx):
    plt.plot(u[idx, :], y, label=f't = {nt_labels[i]} (quasi)', linestyle='--')


plt.title('Velocity Profiles use Crank-Nicolson (Transient vs Quasi-Steady)')
plt.xlabel('u(y,t)')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

#그래프를 보면 t=0일떄 탁 튀는 현상을 확인할 수 있음.
#이것은t=0까지 속도가0이다가, t가 0이후부터 U0cos(n*t)로 정의되기 때문에 발생하는 현상이다.



