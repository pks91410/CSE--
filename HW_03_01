 # 1_(1) second-order Runge-Kutta method with h = 0.01
# Heun법 사용.
# %%
import numpy as np
import matplotlib.pyplot as plt

def f(t,x):
    return t + 2*x*t

def exact(t):
    return 0.5*(np.exp(t**2)-1)

def Heun(t,x0):
    n = len(t)  #h=0.1이라면  n = 21개 생성 됨
    xp = np.zeros(n)  #예측점 배열 생성
    x = np.zeros(n)   #실제x값 배열 생성
    x[0] = x0
    for i in range(n-1):  # i=20일때  t[21]에 접근-> t[20]이 마지막 인덱스 이므로 오류발생.
        xp[i+1] = x[i] + f(t[i],x[i])*h
        x[i+1] = x[i] + (f(t[i],x[i]) + f(t[i+1],xp[i+1]))*0.5*h
       #K1 = f(t[i], x[i])
       #K2 = f(t[i] + h, x[i] + h * K1)
       #x[i+1] = x[i] + 0.5 * h * (K1 + K2)
    return x

 # 조건
h = 0.01
t = np.arange(0,2+h,h) # arange(시작값, 끝값(포함X), 간격)
x0 = 0
x = Heun(t,x0)

t_plot = np.linspace(0,2,300)
x_plot = exact(t_plot)

# print('t=' ,t,'x=', x)
for i in range(len(t)):
    print("t = %.2f, x = %.6f" % (t[i], x[i]))

plt.plot(t_plot,x_plot,'r',label= 'Exact Solution')
plt.plot(t,x,'b--',label = '2nd order Runge-Kutta Method')
plt.title("Comparison of Exact and Numerical Solutions (Heun's Method)")
plt.legend()
plt.show()

#%%
# =======================================================================
# 1_(2) fourth - order  Runge - Kutta method h= 0.01

import numpy as np
import matplotlib.pyplot as plt

def f(t,x):
    return t+2*x*t

def exact(t):
    return 0.5*(np.exp(t**2)-1)

def RK4(t,x0,h):
    n = len(t)
    x = np.zeros(n)
    x[0] = x0
    for i in range(n-1):
        K1 = f( t[i], x[i] )
        K2 = f( t[i]+0.5*h, x[i]+0.5*K1*h )
        K3 = f( t[i]+0.5*h, x[i] + 0.5*K2*h )
        K4 = f( t[i+1], x[i] + K3*h )
        x[i+1] = x[i] + ( K1 + 2*K2 + 2*K3 + K4 ) * h / 6.0
    return x

h = 0.01
t = np.arange(0,2+h,h)
x0= 0
x = RK4(t,x0,h)

t_plot = np.linspace(0,2,300)
x_plot = exact(t_plot)

# print('t=',t, 'x=', x)
for i in range(len(t)):
    print("t = %.2f, x = %.6f" % (t[i], x[i]))

plt.plot(t_plot,x_plot,'r',label = 'Exact Solution')
plt.plot(t,x,'b--',label = '4th order Runge Kutta Method')
plt.title("Comparison of Exact and Numerical Solutions (4th Order Runge-Kutta Method)")
plt.legend()
plt.show()


# ==================================================================
# %%
# 1_(3)
# 각 Runge Kutta법의 Order of accuracy 분석

import numpy as np
import matplotlib.pyplot as plt


def f(t,x):
    return t + 2*x*t

def exact(t):
    return 0.5*(np.exp(t**2)-1)

def Heun(t,x0,h):
    n = len(t)  # n = 21개 생성 됨
    xp = np.zeros(n)  #예측점 배열 생성
    x = np.zeros(n)   #실제x값 배열 생성
    x[0] = x0
    for i in range(n-1):  # i=20일때  t[21]에 접근-> t[20]이 마지막 인덱스 이므로 오류발생.
        K1 = f(t[i], x[i])
        K2 = f(t[i] + h, x[i] + h * K1)
        x[i+1] = x[i] + 0.5 * h * (K1 + K2)
    return x

def RK4(t,x0,h):
    n = len(t)
    x = np.zeros(n)
    x[0] = x0
    for i in range(n-1):
        K1 = f( t[i], x[i] )
        K2 = f( t[i]+0.5*h, x[i]+0.5*K1*h )
        K3 = f( t[i]+0.5*h, x[i] + 0.5*K2*h )
        K4 = f( t[i+1], x[i] + K3*h )
        x[i+1] = x[i] + ( K1 + 2*K2 + 2*K3 + K4 ) * h / 6.0
    return x

# 조건
h_list = [0.1, 0.05, 0.01,0.0125]
x0 = 0
x_exact_2 = exact(2)

heun_errors = []
rk4_errors = []

for h in h_list:
    t = np.arange(0,2+h,h) # arange(시작값, 끝값(포함X), 간격)

    x_heun = Heun(t,x0,h)
    err_heun = abs(x_heun[-1]-x_exact_2)
    heun_errors.append(err_heun)

    x_rk4 = RK4(t,x0,h)
    err_rk4 = abs(x_rk4[-1]-x_exact_2)
    rk4_errors.append(err_rk4)

# 로그 변환
log_h = np.log10(h_list)
log_heun = np.log10(heun_errors)
log_rk4 = np.log10(rk4_errors)

#정확도 계산(log-log 기울기)
p_heun = np.polyfit(log_h,log_heun,1)[0]
p_rk4 = np.polyfit(log_h, log_rk4,1)[0]
#[0]은 polyfit()의 출력에서 기울기만 뽑아내기 위해서 사용

print("Heun Method order of accuracy p = {:.4f}".format(p_heun))
print("RK4 Method order of accuracy p= {:.4f}".format(p_rk4))

plt.figure()
plt.plot(log_h, log_heun, label = "Heun method's p= {:.4f}".format(p_heun))
plt.plot(log_h, log_rk4,label = "RK4 method's p= {:.4f}".format(p_rk4))
plt.xlabel('Step size h')
plt.ylabel('Error at t = 2')
plt.title('Order of Accuracy: Heun VS RK4')
plt.legend()
plt.grid(True)
plt.legend()
plt.show()

# ==================================================================================================

# %%
#1_4
# 1_(2) fourth - order  Runge - Kutta method

import numpy as np
import matplotlib.pyplot as plt

def f(t,x):
    return t+2*x*t

def exact(t):
    return 0.5*(np.exp(t**2)-1)

def RK4(t,x0,h):
    n = len(t)
    x = np.zeros(n)
    x[0] = x0
    for i in range(n-1):
        K1 = f( t[i], x[i] )
        K2 = f( t[i]+0.5*h, x[i]+0.5*K1*h )
        K3 = f( t[i]+0.5*h, x[i] + 0.5*K2*h )
        K4 = f( t[i+1], x[i] + K3*h )
        x[i+1] = x[i] + ( K1 + 2*K2 + 2*K3 + K4 ) * h / 6.0
    return x

h_list = [0.01,0.05,0.1]
x0= 0
x_exact_2 = exact(2)
rk4_errors = []

for h in h_list:
    t = np.arange(0,2+h,h) # arange(시작값, 끝값(포함X), 간격)
    x_rk4 = RK4(t,x0,h)
    err_rk4 = abs(x_rk4[-1]-x_exact_2)
    rk4_errors.append(err_rk4)
    print(f'h = {h:.4f}, error = {err_rk4:.6e}')

log_h = np.log10(h_list)
log_rk4 = np.log10(rk4_errors)

# 정확도 계산(log-log 기울기)
p_rk4 = np.polyfit(log_h, log_rk4,1)[0]
print("RK4 Method order of accuracy p= {:.4f}".format(p_rk4))


# plt.plot(log_h,log_rk4,'ro')
# plt.plot(log_h, log_rk4,label = "RK4 method's p= {:.4f}".format(p_rk4))

plt.plot(h_list, rk4_errors,'b',label = 'Analysis of error at t=2')
plt.plot(h_list, rk4_errors,'ro')

#x축 눈금 표시
plt.xticks(h_list)

#각 점에 좌표 텍스트 표시
for h, err in zip(h_list, rk4_errors):
    plt.text(h, err + 0.0002, f"{err:.2e}", ha='center', fontsize=9)

plt.xlabel('Step size h')
plt.ylabel('Error at t = 2')
plt.title('Analysis of Step Size Effect on RK4 Accuracy at t = 2')
plt.legend()
plt.grid(True)
plt.show()


# %%
