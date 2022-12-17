% 2维线性卡尔曼滤波(Dr.CAN示例)
clc
clear

% 采样点数
N = 30;

% 状态转移矩阵,取dt为1
A = [1 1; 0 1]; % x_{k} = x_{k-1} + v_{k-1} * dt; v_{k} = v_{k-1}

% 状态观测矩阵
H = [1 0; 0 1]; 

% 过程噪声协方差矩阵
q = 1;
Q = [q 0; 0 q];

% 观测噪声协方差矩阵
r = 0.1;
R = [r 0; 0 r];

% 单位矩阵
I = eye(2);

% 实际状态矩阵
X_true = zeros(2, N+1);
X_true(:, 1) = [0; 1]; % 初始位置和速度

% 初始化状态估计协方差矩阵
P = [1 0; 0 1];

X_posterior = [0; 1];
P_posterior = P;

speed_true = zeros(1, N+1);
position_true = zeros(1, N+1);

speed_measure = zeros(1, N+1);
position_measure = zeros(1, N+1);

speed_prior_est = zeros(1, N+1);
position_prior_est = zeros(1, N+1);

speed_posterior_est = zeros(1, N+1);
position_posterior_est = zeros(1, N+1);

% 卡尔曼滤波算法
for k = 1:N+1

    % 生成真实值
    w = [normrnd(0,sqrt(q)); normrnd(0,sqrt(q))]; % 生成过程噪声
    X_true(:,k+1) = A * X_true(:,k) + w; % 当前时刻状态
    position_true(k) = X_true(1,k);
    speed_true(k) = X_true(2,k);

    % 生成观测值
    v = [normrnd(0,sqrt(r)); normrnd(0,sqrt(r))]; % 生成观测噪声
    Z_measure = H * X_true(:,k) + v; % 观测值
    position_measure(k+1) = Z_measure(1,1);
    speed_measure(k+1) = Z_measure(2,1);

    % 先验估计
    X_prior = A * X_posterior;
    position_prior_est(k+1) = X_prior(1,1);
    speed_prior_est(k+1) = X_prior(2,1);

    % 计算状态估计协方差矩阵P
    P_prior = A * P_posterior * A' + Q;

    % 计算卡尔曼增益
    K = P_prior * H' / (H * P_prior * H' + R);

    % 后验估计
    X_posterior = X_prior + K * (Z_measure - H * X_prior);
    position_posterior_est(k+1) = X_posterior(1,1);
    speed_posterior_est(k+1) = X_posterior(2,1);

    % 更新状态估计协方差矩阵P
    P_posterior = (I - K * H) * P_prior;

end

% 画图
figure(1);
plot(speed_true, 'r', 'DisplayName', '实际速度', linewidth=1);
hold on
plot(speed_measure(2:32), 'g', 'DisplayName', '测量速度', linewidth=1);
hold on
plot(speed_prior_est(2:32), 'b', 'DisplayName', '先验估计速度', linewidth=1);
hold on
plot(speed_posterior_est(2:32), 'c', 'DisplayName', '后验估计速度', linewidth=1);
hold on
title("速度比较")
xlabel('k');
legend;

figure(2);
plot(position_true, 'r', 'DisplayName', '实际位置', linewidth=1);
hold on
plot(position_measure(2:32), 'g', 'DisplayName', '测量位置', linewidth=1);
hold on
plot(position_prior_est(2:32), 'b', 'DisplayName', '先验估计位置', linewidth=1);
hold on
plot(position_posterior_est(2:32), 'c', 'DisplayName', '后验估计位置', linewidth=1);
hold on
title("位置比较")
xlabel('k')
legend;

