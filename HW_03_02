# (1)
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

colors = {
    "fastball": "blue",
    "curveball": "orange",
    "slider": "green",
    "screwball": "red"
}

pitches = {
    "fastball": {"v0": 40 ,"phi": radians(225)},
    "curveball": {"v0": 30 ,"phi": radians(45)},
    "slider": {"v0": 30 ,"phi": radians(0)},
    "screwball": {"v0": 30 ,"phi": radians(135)},
                 }

#RK4법
def RK4(t, X, h, B, omega, phi):
    K1 = dXdt(t, X, B, omega, phi)
    K2 = dXdt(t + 0.5 * h, X + 0.5 * K1 * h, B, omega, phi)
    K3 = dXdt(t + 0.5 * h, X + 0.5 * K2 * h, B, omega, phi)
    K4 = dXdt(t + h, X + h * K3, B, omega, phi)
    return X + (K1 + 2*K2 + 2*K3 + K4) * h / 6

def F(V):
    return 0.0039 + 0.0058 / (1 + exp((V-35)/5))

def dXdt(t, X, B, omega, phi):
    x,y,z,vx,vy,vz = X  # X 안에 각 원소를 변수6개에 한번에 할당 한다는 뜻.
    V = sqrt(vx**2 + vy**2 + vz**2)
    Fv = F(V)

    dxdt = vx
    dydt = vy
    dzdt = vz

    dvxdt = -Fv*V*vx + B*omega*(vz*sin(phi)-vy*cos(phi))
    dvydt = -Fv*V*vy + B*omega*vx*cos(phi)
    dvzdt = -9.81-Fv*V*vz - B*omega*vx*sin(phi)

    return array([dxdt,dydt,dzdt,dvxdt,dvydt,dvzdt])

#초기조건 t=0일때
trajectories = {}
#.items()는 딕셔너리(key,value)쌍을 하나씩 꺼냄. pitch_name = "fastball"/ params = {"v0": 40 ,"phi": radians(225)}
for pitch_name, params in pitches.items():
    v0 = params['v0']
    phi = params['phi']

    #공통값
    theta = radians(1)
    h = 1.7
    omega = 1600 * 2*pi / 60
    B = 4.1e-4

    #초기조건 계산
    vx0 = v0 * cos(theta)
    vy0 = 0
    vz0 = v0 * sin(theta)
    x0 = array([0,0,h,vx0,vy0,vz0])

    # 시간 설정
    t = 0
    dt = 0.001
    X = x0.copy() #?????????????????????????????????/

    #trajectory 저장용 리스트 초기화
    x_list = []
    y_list = []
    z_list = []

    if X[0] > 18.39:
        print(f"{pitch_name} 시작 위치가 잘못됨: x0 = {X[0]}")
        continue

    while X[0] <= 18.39:
        x_list.append(X[0])
        y_list.append(X[1])
        z_list.append(X[2])

        X = RK4(t, X, dt, B, omega, phi)
        t += dt

    trajectories[pitch_name] = (x_list, y_list, z_list)

for pitch, (x, y, z) in trajectories.items():
    fig = plt.figure()  # 3차원 공간으로 표현
    ax = fig.add_subplot(111, projection='3d')  #새로운 3d subplot 생성

    ax.plot(x, y, z, label=pitch, color=colors[pitch], linewidth=2)

    #출발 좌표
    ax.scatter(x[0], y[0], z[0], color='black', marker='o', s=50)
    ax.text(x[0] + 0.3, y[0] + 0.05, z[0] + 0.05,
            f"Start\n({x[0]:.2f}, {y[0]:.2f}, {z[0]:.2f})",
            color='black', fontsize=9,ha='left')


    #도착 좌표
    ax.scatter(x[-1], y[-1], z[-1], color='purple', marker='^', s=50)
    ax.text(x[-1] + 0.3, y[-1] + 0.05, z[-1] + 0.05,
            f"End\n({x[-1]:.2f}, {y[-1]:.2f}, {z[-1]:.2f})",
            color='black', fontsize=9,ha='right')

    print("도착지점 ({}):".format(pitch))
    print(f"x좌표: {x[-1]:.4f}, y좌표: {y[-1]:.4f}, z좌표: {z[-1]:.4f}\n")

    ax.set_xlabel("x (Distance to catcher)")
    ax.set_ylabel("y (Side deviation)")
    ax.set_zlabel("z (Height)")
    ax.set_title("3D Trajectory of Baseball Pitches")
    ax.legend(loc='upper left')
    plt.show()
    

