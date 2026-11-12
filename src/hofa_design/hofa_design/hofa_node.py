#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import VehicleOdometry, OffboardControlMode, VehicleRatesSetpoint, VehicleCommand
import numpy as np
import scipy.linalg as la
import cvxpy as cp

# ==============================================================================
# 模块 1: 普通中间估计器 (Standard IE) 
# 论文出处: 第三章 3.1 节 (用于在线观测系统未建模动态和外部风场等低频复合扰动)
# ==============================================================================
class StandardIE:
    def __init__(self, Ts, K0, K1, A0, A1):
        # Ts: 采样时间; dim: 广义状态维度(9维); x_hat: 状态估计向量
        self.Ts = Ts; self.dim = 9; self.x_hat = np.zeros(self.dim)
        
        # 依据论文构建广义状态空间系统矩阵 A_z, B_z, C_z (对应论文 Eq. 3-3, 3-4, 3-5)
        self.A_z = np.block([[np.zeros((3,3)), np.eye(3), np.zeros((3,3))], [K0 - A0, K1 - A1, np.eye(3)], [np.zeros((3,3)), np.zeros((3,3)), np.eye(3)]])
        self.B_z = np.block([[np.zeros((3,3))], [np.eye(3)], [np.zeros((3,3))]])
        self.C_z = np.block([[np.eye(3), np.zeros((3,3)), np.zeros((3,3))], [np.zeros((3,3)), np.eye(3), np.zeros((3,3))]])
        
        # Q_ie, R_ie: 观测器过程噪声与测量噪声协方差矩阵参数
        Q_ie = np.eye(self.dim) * 10.0; R_ie = np.eye(6) * 0.01         
        # 求解离散代数黎卡提方程 (DARE) 获得稳态 Pie (对应论文 Eq. 3-6)
        P_ie = la.solve_discrete_are(self.A_z.T, self.C_z.T, Q_ie, R_ie)
        # 计算最优稳态卡尔曼增益矩阵 L_g (对应论文 Eq. 3-7)
        self.L_g = P_ie @ self.C_z.T @ np.linalg.inv(self.C_z @ P_ie @ self.C_z.T + R_ie)
        self.d_filtered = np.zeros(3)

    def update(self, e_k, e_k1, v_last):
        # innovation: 计算新息 (真实测量残差 - 模型先验预测残差) (对应论文 Eq. 3-10)
        innovation = np.concatenate((e_k, e_k1)) - self.C_z @ (self.A_z @ self.x_hat + self.B_z @ v_last)
        # 状态后验更新方程 (对应论文 Eq. 3-11)
        self.x_hat = self.A_z @ self.x_hat + self.B_z @ v_last + self.L_g @ innovation
        # 提取最后三维作为扰动估计值，并进行限幅防爆震
        d_est_clipped = np.clip(self.x_hat[6:9], -5.0, 5.0)
        # 一阶低通滤波平滑扰动输出
        self.d_filtered = (1.0 - 0.4) * self.d_filtered + 0.2 * d_est_clipped
        return self.d_filtered

# ==============================================================================
# 模块 2: 模型预测控制器 (Nominal MPC) 
# 论文出处: 第三章 3.2 节 (基于QP求解器，结合IE前馈补偿计算最优控制增量)
# ==============================================================================
class NominalMPC:
    def __init__(self, Ts, K0, K1, A0, A1, N=5):
        # N: 预测时域; Ts: 采样时间
        self.N = N; self.Ts = Ts
        # 构建 MPC 增广状态空间矩阵 A_bar, B_bar (对应论文 Eq. 3-16)
        self.A_bar = np.block([[np.zeros((3, 3)), np.eye(3)], [K0 - A0, K1 - A1]])
        self.B_bar = np.vstack([np.zeros((3, 3)), np.eye(3)])
        
        # v: 预测控制输入序列; s: 预测状态序列 (CVXPY优化变量)
        self.v = cp.Variable((3, N)); self.s = cp.Variable((6, N+1)) 
        # s_init: 初始状态参数; d_hat: IE前馈的常量扰动参数
        self.s_init = cp.Parameter(6); self.d_hat = cp.Parameter(3)
        
        # Q_mpc, R_mpc: 非对称状态惩罚与控制增量惩罚矩阵
        self.Q_mpc = np.diag([2.0, 1.0, 10.0, 0.1, 0.1, 0.1]) 
        self.R_mpc = np.diag([5000.0, 7000.0, 2000.0])    
        v_bound = 0.0015  # 虚拟控制量 v 的安全边界约束
        
        # 构建二次规划 (QP) 问题的代价函数与约束条件 (对应论文 Eq. 3-18 ~ 3-21)
        cost = 0; constraints = [self.s[:, 0] == self.s_init]
        for k in range(N):
            # 累加预测视界内的状态代价与输入代价
            cost += cp.quad_form(self.s[:, k], self.Q_mpc) + cp.quad_form(self.v[:, k], self.R_mpc)
            # 状态转移约束，无缝嵌入 IE 观测的外部扰动 d_hat (对应 Eq. 3-20)
            constraints += [self.s[:, k+1] == self.A_bar @ self.s[:, k] + self.B_bar @ (self.v[:, k] + self.d_hat)]
            # 控制动作上下限幅度约束
            constraints += [self.v[:, k] <= v_bound, self.v[:, k] >= -v_bound]
            
        self.prob = cp.Problem(cp.Minimize(cost), constraints)

    def solve(self, s_current, d_current):
        # 更新当前时刻的系统状态与扰动观测值
        self.s_init.value = s_current; self.d_hat.value = d_current
        try:
            # 调用 OSQP 求解器实时寻优，开启热启动 (warm_start) 保障 50Hz 实时性
            self.prob.solve(solver=cp.OSQP, warm_start=True)
            # 滚动优化机理：仅提取最优预测序列的第一步作为实际输出 (对应论文 Eq. 3-23)
            if self.prob.status in ["optimal", "optimal_inaccurate"]: return self.v.value[:, 0]
        except Exception: pass
        return np.zeros(3)

# ==============================================================================
# ROS 2 主节点 - HOFA Square
# 论文出处: 第一章 系统总体架构 & 算法映射层
# ==============================================================================
class HOFASquareNode(Node):
    def __init__(self):
        super().__init__('hofa_square_node')
        # 配置底层通信 QoS
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)

        # 订阅位姿话题与发布控制指令话题
        self.odom_sub = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_callback, qos)
        self.ctrl_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.rates_pub = self.create_publisher(VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', qos)
        self.command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)

        # 初始化 CSV 数据记录日志
        self.log_file = open("hofa_square.csv", "w")
        self.log_file.write("timestamp,ref_x,act_x,ref_y,act_y,ref_z,act_z\n")

        # 物理环境与无人机基础参数
        self.Ts = 0.02; self.m = 2.0; self.g = 9.81; self.hover_thrust_base = 0.57; self.K_att = 4.0  
        p = 0.95  # 极点配置参数
        
        # 高阶全驱系统特征参数矩阵设计 (对应论文 Eq. 2-6)
        self.A1 = np.diag([-2.0, -2.0, -2.0]); self.A0 = np.diag([1.0, 1.0, 1.0])
        # 闭环镇定反馈矩阵解析计算 (对应论文 Eq. 2-10, 2-11)
        self.K1 = np.diag([2.0 * p - 2.0, 2.0 * p - 2.0, 2.0 * p - 2.0]); self.K0 = np.diag([1.0 - p**2, 1.0 - p**2, 1.0 - p**2])
        # Bp 的逆矩阵，用于计算虚拟控制量
        self.Bp_inv = np.eye(3) * (self.m / (self.Ts**2))

        # 实例化级联架构核心模块
        self.ie = StandardIE(self.Ts, self.K0, self.K1, self.A0, self.A1)
        self.mpc = NominalMPC(self.Ts, self.K0, self.K1, self.A0, self.A1, N=5)
        
        # 历史控制指令缓存
        self.v_p_last = np.zeros(3)
        self.v_p_raw_last = np.zeros(3)

        # 定时器触发 50Hz 的控制循环
        self.step_cnt = 0; self.timer = self.create_timer(self.Ts, self.cmdloop_callback)
        self.current_pos = np.zeros(3); self.current_vel = np.zeros(3)
        self.current_yaw = 0.0; self.current_roll = 0.0; self.current_pitch = 0.0
        
        # ★ 降落标志位
        self.is_landing_triggered = False

    def odom_callback(self, msg):
        # 实时获取当前位姿及速度
        self.current_pos = np.array(msg.position); self.current_vel = np.array(msg.velocity)
        # 四元数解析为欧拉角
        q = msg.q 
        self.current_roll = np.arctan2(2*(q[0]*q[1]+q[2]*q[3]), 1-2*(q[1]*q[1]+q[2]*q[2]))
        self.current_pitch = np.arcsin(np.clip(2*(q[0]*q[2]-q[3]*q[1]), -1.0, 1.0))
        self.current_yaw = np.arctan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]*q[2]+q[3]*q[3]))

    def cmdloop_callback(self):
        # 维持 Offboard 模式的心跳
        self.publish_offboard_mode()
        if self.step_cnt == 50: self.arm()
        if self.step_cnt == 60: self.set_offboard()

        msg = VehicleRatesSetpoint(); msg.roll = 0.0; msg.pitch = 0.0; msg.yaw = 0.0; msg.thrust_body = [0.0, 0.0, -0.15]
        # 缓冲阶段：待系统稳定后再启动轨迹跟踪
        if self.step_cnt < 100: 
            self.rates_pub.publish(msg); self.step_cnt += 1; return

        t_flight = (self.step_cnt - 100) * self.Ts

        # ======================================================================
        # 参考轨迹生成生成器 (方形轨迹)
        # ======================================================================
        def get_ref(time):
            if time <= 4.0:
                tau = time / 4.0
                z = -5.0 * 0.5 * (1.0 - np.cos(np.pi * tau))
                return np.array([0.0, 0.0, z])
            elif time <= 6.0: 
                return np.array([0.0, 0.0, -5.0])
            elif time <= 10.0: 
                tau = (time - 6.0) / 4.0
                x = 5.0 * 0.5 * (1.0 - np.cos(np.pi * tau))
                return np.array([x, 0.0, -5.0])
            elif time <= 14.0: 
                tau = (time - 10.0) / 4.0
                y = 5.0 * 0.5 * (1.0 - np.cos(np.pi * tau))
                return np.array([5.0, y, -5.0])
            elif time <= 18.0: 
                tau = (time - 14.0) / 4.0
                x = 5.0 - 5.0 * 0.5 * (1.0 - np.cos(np.pi * tau))
                return np.array([x, 5.0, -5.0])
            elif time <= 22.0: 
                tau = (time - 18.0) / 4.0
                y = 5.0 - 5.0 * 0.5 * (1.0 - np.cos(np.pi * tau))
                return np.array([0.0, y, -5.0])
            else: 
                return np.array([0.0, 0.0, -5.0])
                
        def get_ref_vel(time):
            if time <= 4.0:
                tau = time / 4.0
                vz = -5.0 * (np.pi / 8.0) * np.sin(np.pi * tau)
                return np.array([0.0, 0.0, vz])
            elif time <= 6.0:
                return np.zeros(3)
            elif time <= 10.0:
                tau = (time - 6.0) / 4.0
                vx = 5.0 * (np.pi / 8.0) * np.sin(np.pi * tau)
                return np.array([vx, 0.0, 0.0])
            elif time <= 14.0:
                tau = (time - 10.0) / 4.0
                vy = 5.0 * (np.pi / 8.0) * np.sin(np.pi * tau)
                return np.array([0.0, vy, 0.0])
            elif time <= 18.0:
                tau = (time - 14.0) / 4.0
                vx = -5.0 * (np.pi / 8.0) * np.sin(np.pi * tau)
                return np.array([vx, 0.0, 0.0])
            elif time <= 22.0:
                tau = (time - 18.0) / 4.0
                vy = -5.0 * (np.pi / 8.0) * np.sin(np.pi * tau)
                return np.array([0.0, vy, 0.0])
            else:
                return np.zeros(3)

        # ======================================================================

        # 提取当前及未来两拍的期望位置
        sigma_rk = get_ref(t_flight); sigma_rk1 = get_ref(t_flight + self.Ts); sigma_rk2 = get_ref(t_flight + 2 * self.Ts)

        # 解算位置跟踪误差与外推预测误差
        err_p = self.current_pos - sigma_rk  
        err_v = self.current_vel - get_ref_vel(t_flight)
        ref_acc = (get_ref_vel(t_flight + self.Ts) - get_ref_vel(t_flight)) / self.Ts
        err_p_next = err_p + err_v * self.Ts + 0.5 * ref_acc * (self.Ts ** 2)
        
        # 步骤二：IE 扰动观测模块
        d_est = self.ie.update(err_p, err_p_next, self.v_p_last)

        # 步骤三：集成 MPC 最优增量解算
        s_current = np.concatenate((err_p, err_p_next))
        v_p_raw = self.mpc.solve(s_current, d_est)
        
        # 信号后处理：通过超前补偿抑制指令高频抖动
        v_p_diff = (v_p_raw - self.v_p_raw_last) / self.Ts
        self.v_p_raw_last = v_p_raw
        T_lead = 0.03
        v_p_lead = v_p_raw + T_lead * v_p_diff
        
        # 深度低通滤波平滑最终虚拟增量
        alpha = 0.05
        self.v_p_last = (1.0 - alpha) * self.v_p_last + alpha * v_p_lead
        
        # 步骤四：HOFA 代数解耦逆向映射 (对应论文 Eq. 2-7, 3-24)
        # 计算前馈加速度项
        hofa_ref_term = sigma_rk2 + self.A1 @ sigma_rk1 + self.A0 @ sigma_rk
        ff_acc = self.Bp_inv @ hofa_ref_term / self.m

        # 计算状态反馈项并融入 MPC 抗扰增量
        hofa_error_term = self.K1 @ err_p_next + self.K0 @ err_p
        fb_acc_raw = self.Bp_inv @ (hofa_error_term + self.v_p_last) / self.m
        
        # 合成总期望加速度并执行幅度限制
        fb_acc = np.array([np.clip(fb_acc_raw[0], -2.0, 2.0), np.clip(fb_acc_raw[1], -2.0, 2.0), np.clip(fb_acc_raw[2], -10.0, 10.0)])
        des_acc_raw = ff_acc + fb_acc

        # 步骤五：非线性代数映射与下发 (对应论文 Eq. 2-12 ~ 2-15)
        # 投影至机体坐标系，解耦偏航旋转引起的推力交叉
        c_yaw = np.cos(self.current_yaw); s_yaw = np.sin(self.current_yaw)
        acc_body_x = des_acc_raw[0] * c_yaw + des_acc_raw[1] * s_yaw
        acc_body_y = -des_acc_raw[0] * s_yaw + des_acc_raw[1] * c_yaw
        acc_body_z = des_acc_raw[2]
        
        # 线性化映射目标姿态角 (Roll, Pitch) 及推力
        des_pitch = np.clip(-acc_body_x / self.g, -0.35, 0.35); des_roll  = np.clip(acc_body_y / self.g, -0.35, 0.35)
        thrust = np.clip(self.hover_thrust_base - (acc_body_z * 0.058), 0.2, 0.9)

        # 偏航角最短环绕限制逻辑
        dyaw_ctrl = 0.0 - self.current_yaw
        if dyaw_ctrl > np.pi: dyaw_ctrl -= 2*np.pi
        if dyaw_ctrl < -np.pi: dyaw_ctrl += 2*np.pi

        # 外环姿态 P 控制器转化为角速度设定点 (对应论文 Eq. 2-16)
        msg.roll = float((des_roll - self.current_roll) * self.K_att); msg.pitch = float((des_pitch - self.current_pitch) * self.K_att)
        msg.yaw = float(dyaw_ctrl * 1.5); msg.thrust_body = [0.0, 0.0, -float(thrust)]

        # 步骤六：封装协议下发控制指令与数据记录
        if self.step_cnt > 100: 
            self.rates_pub.publish(msg)
            self.log_file.write(f"{t_flight},{sigma_rk[0]},{self.current_pos[0]},{sigma_rk[1]},{self.current_pos[1]},{sigma_rk[2]},{self.current_pos[2]}\n")

       

        self.step_cnt += 1

    def publish_offboard_mode(self): m = OffboardControlMode(); m.position = False; m.velocity = False; m.acceleration = False; m.attitude = False; m.body_rate = True; self.ctrl_mode_pub.publish(m)
    def arm(self): self.send_cmd(400, 1.0)
    def set_offboard(self): self.send_cmd(176, 1.0, 6.0)
    def send_cmd(self, c, p1, p2=0.0): m = VehicleCommand(); m.command=c; m.param1=p1; m.param2=p2; m.target_system=1; m.target_component=1; m.source_system=1; m.from_external=True; self.command_pub.publish(m)

def main(): rclpy.init(); node = HOFASquareNode(); rclpy.spin(node); node.destroy_node(); rclpy.shutdown()
if __name__ == '__main__': main()