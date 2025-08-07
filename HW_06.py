#HW06_01
# %%
#Jacobi Method
import numpy as np
import matplotlib.pyplot as plt

#parameter
N = 21
x_list = np.linspace(0,1,N)
y_list = np.linspace(0,1,N)
X,Y = np.meshgrid(x_list,y_list)
h = x_list[1]-x_list[0]
u_0 = np.zeros((N,N))

def u_exact(x, y):
    return -np.sin(np.pi * x) * np.sin(np.pi * y)/(2*np.pi**2)

def f1(x,y):
    f1 = np.sin(np.pi* x) * np.sin(np.pi*y)
    return f1

#Jacobi Method
def jacobi(u,h):
    u_new=np.copy(u)
    for i in range(1,N-1):
        for j in range(1,N-1):
            u_new[i,j] = (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1]\
                           - h**2 * f1(X[i,j],Y[i,j])) / 4
    return u_new

u_list_jacobi = []
u_list_jacobi.append(u_0)
u_list_jacobi.append(jacobi(u_0,h))

# count iteration i
i = 0
tol = 1e-5
i_list=[i,]

while np.linalg.norm(u_list_jacobi[-1]-u_list_jacobi[-2]) > tol:
    i += 1
    u_jacobi = jacobi(u_list_jacobi[i],h)
    u_list_jacobi.append(u_jacobi)
    i_list.append(i)
    print(i)

#Residual
residual = np.zeros((N, N))
for i in range(1, N-1):
    for j in range(1, N-1):
        laplacian = (u_list_jacobi[-1][i+1,j] + u_list_jacobi[-1][i-1,j] + u_list_jacobi[-1][i,j+1] + u_list_jacobi[-1][i,j-1] \
            - 4*u_list_jacobi[-1][i,j]) / (h*h)
        residual[i,j] = laplacian - f1(X[i,j], Y[i,j])

#시각화
plt.figure(figsize=(8,6))
plt.contourf(X, Y, u_list_jacobi[-1], levels=20, cmap='hot_r')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Jacobi Method Solution (Iterations: {i_list[-1]})')
plt.show()

#performance
print('Jacobi Method')
print('iteration:', i_list[-1] )
print('norm of residual:', np.linalg.norm(residual))
print("error", np.linalg.norm(u_list_jacobi[-1] - u_exact(X, Y),2))

# %%
#Gauss-Seidel
import numpy as np
import matplotlib.pyplot as plt

#parameter
N = 21
x_list = np.linspace(0,1,N)
y_list = np.linspace(0,1,N)
X,Y = np.meshgrid(x_list,y_list)
h = x_list[1]-x_list[0]
u_0 = np.zeros((N,N))

def u_exact(x, y):
    return -np.sin(np.pi * x) * np.sin(np.pi * y)/(2*np.pi**2)

def f1(x,y):
    f1 = np.sin(np.pi* x) * np.sin(np.pi*y)
    return f1

# Gauss-Seidel Method
def gauss_seidel(u,h):
    u_new = np.copy(u)
    for i in range(1,N-1):
        for j in range(1,N-1):
            u_new[i,j] = (u[i+1,j] + u_new[i-1,j] + u[i,j+1] + u_new[i,j-1] \
                - h**2 * f1(X[i,j],Y[i,j]) ) / 4
    return u_new

u_list_gauss = []
u_list_gauss.append(u_0)
u_list_gauss.append(gauss_seidel(u_0, h))

i = 0
tol = 1e-5
i_list = [i,]

while np.linalg.norm(u_list_gauss[-1] - u_list_gauss[-2]) > 1e-5: 
    i += 1
    u_gauss = gauss_seidel(u_list_gauss[i], h)
    u_list_gauss.append(u_gauss)
    i_list.append(i)

residual = np.zeros((N, N))
for i in range(1, N-1):
    for j in range(1, N-1):
        laplacian = (u_list_gauss[-1][i+1,j] + u_list_gauss[-1][i-1,j] + u_list_gauss[-1][i,j+1] + u_list_gauss[-1][i,j-1] \
                     - 4*u_list_gauss[-1][i,j]) / (h*h)
        residual[i,j] = laplacian - f1(X[i,j], Y[i,j])

plt.figure(figsize=(8,6))
plt.contourf(X, Y, u_list_gauss[-1], levels=20, cmap='hot_r')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Gauss-Seidel Method Solution (Iterations: {i_list[-1]})')
plt.show()

print('Gauss-Seidel Method')
print('iteration:',i_list[-1])
print('Residual norm:', np.linalg.norm(residual))
print("error", np.linalg.norm(u_list_gauss[-1] - u_exact(X, Y),2))


# %%
#SOR Method

import numpy as np
import matplotlib.pyplot as plt

#parameter
N = 21
x_list = np.linspace(0,1,N)
y_list = np.linspace(0,1,N)
X,Y = np.meshgrid(x_list,y_list)
h = x_list[1]-x_list[0]
u_0 = np.zeros((N,N))

def u_exact(x, y):
    return -np.sin(np.pi * x) * np.sin(np.pi * y)/(2*np.pi**2)

def f1(x,y):
    f1 = np.sin(np.pi* x) * np.sin(np.pi*y)
    return f1

# SOR Method
def SOR_method(u,h,omega = 1.5):  #omega 값이 2에 가까워질 수록 수렴속도 빨라지는 것을 볼 수 있음.
    u_new = np.copy(u)
    for i in range(1,N-1):
        for j in range(1,N-1):
            u_new[i,j] = (1-omega) * u[i,j] + omega * (u[i+1,j] + u_new[i-1,j] + u[i,j+1] + u_new[i,j-1] - h**2 * f1(X[i,j],Y[i,j]) ) / 4
    return u_new

u_list_SOR = []
u_list_SOR.append(u_0)
u_list_SOR.append(SOR_method(u_0, h))

i = 0
i_list = [i,]

while np.linalg.norm(u_list_SOR[-1] - u_list_SOR[-2]) > 1e-5: 
    i += 1
    u_SOR = SOR_method(u_list_SOR[i], h)
    u_list_SOR.append(u_SOR)
    i_list.append(i)

residual = np.zeros((N, N))
for i in range(1, N-1):
    for j in range(1, N-1):
        laplacian = (u_list_SOR[-1][i+1,j] + u_list_SOR[-1][i-1,j] + u_list_SOR[-1][i,j+1] + u_list_SOR[-1][i,j-1] - 4*u_list_SOR[-1][i,j]) / (h*h)
        residual[i,j] = laplacian - f1(X[i,j], Y[i,j])

plt.figure(figsize=(8,6))
plt.contourf(X, Y, u_list_SOR[-1], levels=20, cmap='hot_r')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'SOR Method Solution (Iterations: {i_list[-1]})')
plt.show()

print('SOR Method')
print('iteration:',i_list[-1])
print('Residual norm:', np.linalg.norm(residual))
print("error", np.linalg.norm(u_list_SOR[-1] - u_exact(X, Y),2))

# %%
#HW_06_02_(1)

import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()
#parameter
N = 21
x_list = np.linspace(0,1,N)
y_list = np.linspace(0,1,N)
X,Y = np.meshgrid(x_list,y_list)
h = x_list[1]-x_list[0]
u_0 = np.zeros((N,N))


def f1(x,y):
    f1 = np.sin(np.pi* x) * np.sin(np.pi*y)
    return f1

def f2(x,y):
    return np.exp(-100*((x-0.5)**2 + (y-0.5)**2))

# f1 + f2
def SOR_method2(u,h,omega = 1.5):  #omega 값이 2에 가까워질 수록 수렴속도 빨라지는 것을 볼 수 있음.
    u_new = np.copy(u)
    for i in range(1,N-1):
        for j in range(1,N-1):
            u_new[i,j] = (1-omega) * u[i,j] \
                + omega * (u[i+1,j] + u_new[i-1,j] \
                           + u[i,j+1] + u_new[i,j-1] \
                            - h**2 * (f1(X[i,j],Y[i,j]) + f2(X[i,j],Y[i,j]))) / 4
    return u_new

u_list_SOR2 = []
u_list_SOR2.append(u_0)
u_list_SOR2.append(SOR_method2(u_0, h))

i = 0
i_list = [i,]
tol = 1e-5

while np.linalg.norm(u_list_SOR2[-1] - u_list_SOR2[-2]) > tol: 
    i += 1
    u_SOR2 = SOR_method2(u_list_SOR2[i], h)
    u_list_SOR2.append(u_SOR2)
    i_list.append(i)


print('SOR Method(f1+f2)')
print('iteration:',i_list[-1])

plt.figure(figsize=(8,6))
plt.contourf(X, Y, u_list_SOR2[-1], levels=20, cmap='hot_r')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'SOR Method Solution(f1+f2) (Iterations: {i})')
plt.show()
end=time.time()

print(f"계산 시간: {end - start:.8f}초")
# %%
#HW06_02_(2)
import numpy as np
import matplotlib.pyplot as plt

#parameter
N = 21
x_list = np.linspace(0,1,N)
y_list = np.linspace(0,1,N)
X,Y = np.meshgrid(x_list,y_list)
h = x_list[1]-x_list[0]
u_0 = np.zeros((N,N))


def f1(x,y):
    f1 = np.sin(np.pi* x) * np.sin(np.pi*y)
    return f1

def f2(x,y):
    return np.exp(-100*((x-0.5)**2 + (y-0.5)**2))

# f2
def SOR_method3(u,h,omega = 1.5):  #omega 값이 2에 가까워질 수록 수렴속도 빨라지는 것을 볼 수 있음.
    u_new = np.copy(u)
    for i in range(1,N-1):
        for j in range(1,N-1):
            u_new[i,j] = (1-omega) * u[i,j] \
                + omega * (u[i+1,j] + u_new[i-1,j] \
                           + u[i,j+1] + u_new[i,j-1] \
                            - h**2 *  (f2(X[i,j],Y[i,j]))) / 4
    return u_new

u_list_SOR3 = []
u_list_SOR3.append(u_0)
u_list_SOR3.append(SOR_method3(u_0, h))

i = 0
i_list = [i,]
tol = 1e-5

while np.linalg.norm(u_list_SOR3[-1] - u_list_SOR3[-2]) > tol: 
    i += 1
    u_SOR3 = SOR_method3(u_list_SOR3[i], h)
    u_list_SOR3.append(u_SOR3)
    i_list.append(i)


print('SOR Method(f2)')
print('iteration:',i_list[-1])

plt.figure(figsize=(8,6))
plt.contourf(X, Y, u_list_SOR3[-1], levels=20, cmap='hot_r')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'SOR Method Solution(f2) (Iterations: {i_list[-1]})')
plt.show()

# %%
# HW06_02_(3)
#SOR_method2(f1+f2)와 비교해보자.

import numpy as np
import matplotlib.pyplot as plt
import time

start=time.time()
#parameter
N = 21
x_list = np.linspace(0,1,N)
y_list = np.linspace(0,1,N)
X,Y = np.meshgrid(x_list,y_list)
h = x_list[1]-x_list[0]
u_0 = np.zeros((N,N))
tol = 1e-5

def f1(x,y):
    f1 = np.sin(np.pi* x) * np.sin(np.pi*y)
    return f1

def f2(x,y):
    return np.exp(-100*((x-0.5)**2 + (y-0.5)**2))

# SOR Method f1
def SOR_method(u,h,omega = 1.5):  #omega 값이 2에 가까워질 수록 수렴속도 빨라지는 것을 볼 수 있음.
    u_new = np.copy(u)
    for i in range(1,N-1):
        for j in range(1,N-1):
            u_new[i,j] = (1-omega) * u[i,j] \
                + omega * (u[i+1,j] + u_new[i-1,j] \
                           + u[i,j+1] + u_new[i,j-1] \
                            - h**2 * f1(X[i,j],Y[i,j]) ) / 4
    return u_new

# f2
def SOR_method3(u,h,omega = 1.5):  #omega 값이 2에 가까워질 수록 수렴속도 빨라지는 것을 볼 수 있음.
    u_new = np.copy(u)
    for i in range(1,N-1):
        for j in range(1,N-1):
            u_new[i,j] = (1-omega) * u[i,j] \
                + omega * (u[i+1,j] + u_new[i-1,j] \
                           + u[i,j+1] + u_new[i,j-1] \
                            - h**2 *  (f2(X[i,j],Y[i,j]))) / 4
    return u_new

u_list_SOR = []
u_list_SOR.append(u_0)
u_list_SOR.append(SOR_method(u_0, h))

u_list_SOR3 = []
u_list_SOR3.append(u_0)
u_list_SOR3.append(SOR_method3(u_0, h))

i1 = 0
i2 = 0
i_list_SOR = [i1,]
i_list_SOR3=[i2,]

while np.linalg.norm(u_list_SOR[-1] - u_list_SOR[-2]) > 1e-5: 
    i1 += 1
    u_SOR = SOR_method(u_list_SOR[i1], h)
    u_list_SOR.append(u_SOR)
    i_list_SOR.append(i1)

while np.linalg.norm(u_list_SOR3[-1] - u_list_SOR3[-2]) > tol: 
    i2 += 1
    u_SOR3 = SOR_method3(u_list_SOR3[i2], h)
    u_list_SOR3.append(u_SOR3)
    i_list_SOR3.append(i2)


u_list_sum = u_list_SOR3[-1]+u_list_SOR[-1]

plt.figure(figsize=(8,6))
plt.contourf(X, Y, u_list_sum, levels=20, cmap='hot_r')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'SOR Method Solution u1 + u2')
plt.show()

print('SOR Method(f1)')
print('iteration:',i_list_SOR[-1])
print('SOR Method(f2)')
print('iteration:',i_list_SOR3[-1])
end=time.time()

print(f"계산 시간: {end - start:.8f}초")
