#%%
from numpy import *
import matplotlib.pyplot as plt

def f(x):
    return sin((4-x)*(4+x))
def fp_2(x): #f"(x)
    return -2*cos(16-x**2) -4*x**2*sin(16-x**2)

#[0,8] uniform nodes를 사용하여 33개의 격자점 사용
N_value = [33,65,129]
dx_list = []
error_norm_list = []

for N in N_value :
    x = linspace(0,8,N)
    h = x[1] - x[0]

#1 second-order one-sides difference scheme 사용하여 경계에서 f''(x) 유도
# x=0에서 전진차분 / x=8 에서 후진차분 사용
    fx= f(x)
    fpp_0 = (2*fx[0] - 5*fx[1] + 4*fx[2] - fx[3])/ h**2
    fpp_8 = (2*fx[-1] -5*fx[-2] +4*fx[-3] - fx[-4])/ h**2

    print('N =%d 에서 f"(x)는' %N)
    print('x=0에서 f"(x)=',fpp_0)
    print('x=8에서 f"(x)=',fpp_8)
    print()

#2 second - order central difference scheme 이용해 exact solution과 함께 f"(x) 그리기
    
    #x와 동일한 크기를 가지는 배열을 만들되 ,그 안의 값을 전부 0으로 채운다.
    fpp_central = zeros(len(x))
    
    for i in range(1,len(x)-1):
        fpp_central[i] = (fx[i+1] - 2*fx[i] + fx[i-1]) / h**2
    # fpp_central[0] = fpp_0
    # fpp_central[-1] = fpp_8

    x_plot = linspace(0, 8, 1000)
    fpp_exact_plot = fp_2(x_plot)

    plt.plot(x_plot,fpp_exact_plot, 'r', label = 'Exact Solution')
    plt.plot(x,fpp_central, 'b', label = 'Central difference')
    plt.plot(x,fpp_central,'go')
    plt.xlabel('x')
    plt.ylabel('f"(x)')
    plt.title('N = %d Numerical differentiation' %N )
    plt.grid(True)
    plt.legend()
    plt.show()

# ==============================================================

#L2 norm 오차 계산
    fpp_exact_at_nodes = fp_2(x)
    #경계를 제외하고 오차 계산 ->>>> backup 발표 시 질문하기!!!!!
    error = fpp_central[1:-1] - fpp_exact_at_nodes[1:-1]
    error_norm = sqrt(sum(error**2)*h)

# 리스트에 저장
    dx_list.append(h)
    error_norm_list.append(error_norm)

# 로그값으로 변환
    log_dx = log10(dx_list)
    log_error = log10(error_norm_list)

#기울기(=p) 2인 reference line
ref_log_dx = log_dx[-1]   #기준점을 가장 작은 dx에서 시작하기 위해
ref_log_err = log_error[-1]
ref_line_y = [ref_log_err + 2 * (dx - ref_log_dx) for dx in log_dx]

plt.plot(log_dx, ref_line_y, '--', label = 'Reference solution p=2')
plt.plot(log_dx, log_error,label = 'L2 norm error') 
plt.plot(log_dx, log_error, 'ro')
plt.title('log-log L2 norm error')
plt.xlabel('log(Δx)')
plt.ylabel('log||e||')
plt.grid(True)
plt.legend()
plt.show()

print(list(map(float, error_norm_list)))

# %%
