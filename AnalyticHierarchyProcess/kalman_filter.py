import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter:
    def __init__(self, dt, A, H, Q, R, x0, P0):
        """
        初始化卡尔曼滤波器。

        参数：
        dt: 时间步长
        A: 状态转移矩阵
        H: 观测矩阵
        Q: 系统噪声协方差矩阵
        R: 测量噪声协方差矩阵
        x0: 初始状态估计值
        P0: 初始状态协方差矩阵
        """
        self.dt = dt
        self.A = A
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

    def predict(self):
        """
        预测下一个状态估计值和协方差矩阵。
        """
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        """
        使用测量值更新状态估计值和协方差矩阵。

        参数：
        z: 测量值
        """
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(np.dot(np.dot(self.H, self.P), self.H.T) + self.R))
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))
        self.P = np.dot((np.eye(self.A.shape[0]) - np.dot(K, self.H)), self.P)

    def filter(self, measurements):
        """
        使用卡尔曼滤波器对测量值序列进行滤波，并返回状态估计值序列。

        参数：
        measurements: 测量值序列

        返回：
        一个二维数组，包含状态估计值序列。
        """
        estimates = []
        for z in measurements:
            self.predict()
            self.update(z)
            estimates.append(self.x)
        return np.array(estimates)


# define the Kalman filter parameters
dt = 0.1  # time step
A = np.array([[1, dt], [0, 1]])  # state transition matrix
H = np.array([[1, 0]])  # observation matrix
Q = np.array([[1e-4, 0], [0, 1e-2]])  # process noise covariance matrix
R = np.array([[0.01]])  # measurement noise covariance matrix
x0 = np.array([0, 0])  # initial state estimate
P0 = np.array([[1, 0], [0, 1]])  # initial state covariance matrix

# generate simulated data
t = np.arange(0, 10, dt)  # time vector
x_true = np.sin(2 * np.pi * 0.2 * t)  # true state
x_meas = x_true + np.random.normal(0, np.sqrt(R[0][0]), len(t))  # noisy measurements

# run the Kalman filter on the data
kf = KalmanFilter(dt, A, H, Q, R, x0, P0)
x_est = kf.filter(x_meas)



# plot the results
plt.plot(t, x_true, 'b-', label='真实状态')
plt.plot(t, x_meas, 'r.', label='测量状态')
plt.plot(t, x_est[:, 0], 'g-', label='估计状态')
plt.legend(loc='best')
plt.xlabel('Time (s)')
plt.ylabel('State')
plt.show()