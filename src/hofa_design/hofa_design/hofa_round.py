#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import VehicleOdometry, OffboardControlMode, VehicleRatesSetpoint, VehicleCommand
import numpy as np
import scipy.linalg as la
import cvxpy as cp

# ==============================================================================
# 模块 1: 标准中间估计器 (Standard IE) 
# 论文对应：第3章 3.1 节，用于在线观测系统未建模动态和低频复合扰动
# ==============================================================================
class StandardIE:
    def __init__(self, Ts, K0, K1, A0, A1):
        # Ts: 采样时间; dim: 广义状态维度; x_hat: 状态估计向量
        self.Ts = Ts; self.dim = 9; self.x_hat = np.zeros(self.dim)
        
        # 依据论文 Eq. 3-3, 3-4, 3-5 构建广义状态空间系统矩阵 A_z, B_z, C_z
        self.A_z = np.block([[np.zeros((3,3)), np.eye(3), np.zeros((3,3))], [K0 - A0, K1 - A1, np.eye(3)], [np.zeros((3,3)), np.zeros((3,3)), np.eye(3)]])
        self.B_z = np.block([[np.zeros((3,3))], [np.eye(3)], [np.zeros((3,3))]])
        self.C_z = np.block([[np.eye(3), np.zeros((3,3)), np.zeros((3,3))], [np.zeros((3,3)), np.eye(3), np.zeros((3,3))]])
        
        # 观测器过程噪声与测量噪声协方差矩阵参数
        Q_ie = np.eye(self.dim) * 10.0; R_ie = np.eye(6) * 0.01         
        # 求解离散代数黎卡提方程 (DARE) 获得稳态误差协方差矩阵 P_ie
        P_ie = la.solve_discrete_are(self.A_z.T, self.C_z.T, Q_ie, R_ie)
        # 计算最优稳态卡尔曼增益矩阵 L_g
        self.L_g = P_ie @ self.C_z.T @ np.linalg.inv(self.C_z @ P_ie @ self.C_z.T + R_ie)
        self.d_filtered = np.zeros(3)

    def update(self, e_k, e_k1, v_last):
        # 计算新息 (真实测量残差 - 模型先验预测残差)
        innovation = np.concatenate((e_k, e_k1)) - self.C_z @ (self.A_z @ self.x_hat + self.B_z @ v_last)
        # 状态后验更新方程
        self.x_hat = self.A_z @ self.x_hat + self.B_z @ v_last + self.L_g @ innovation
        # 提取最后三维作为扰动估计值，并进行限幅防爆震
        d_est_clipped = np.clip(self.x_hat[6:9], -5.0, 5.0)
        # 一阶低通滤波平滑扰动输出
        self.d_filtered = (1.0 - 0.6) * self.d_filtered + 0.2 * d_est_clipped
        return self.d_filtered

# ==============================================================================
# 模块 2: 模型预测控制器 (Nominal MPC) 
# 论文对应：第3章 3.2 节，基于二次规划(QP)求解器，结合IE前馈补偿计算最优控制增量
# ==============================================================================
class NominalMPC:
    def __init__(self, Ts, K0, K1, A0, A1, N=5):
        # N: 预测时域步长; Ts: 采样时间
        self.N = N; self.Ts = Ts
        # 构建 MPC 增广离散状态空间预测模型矩阵 A_bar, B_bar
        self.A_bar = np.block([[np.zeros((3, 3)), np.eye(3)], [K0 - A0, K1 - A1]])
        self.B_bar = np.vstack([np.zeros((3, 3)), np.eye(3)])
        
        # v: 预测控制输入序列; s: 预测状态序列 (CVXPY优化变量)
        self.v = cp.Variable((3, N)); self.s = cp.Variable((6, N+1)) 
        # s_init: 初始状态参数; d_hat: IE前馈的常量扰动参数
        self.s_init = cp.Parameter(6); self.d_hat = cp.Parameter(3)
        
        # Q_mpc, R_mpc: 状态误差惩罚矩阵与控制增量惩罚矩阵
        self.Q_mpc = np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1]) 
        self.R_mpc = np.eye(3) * 7000.0   
        
        # 虚拟控制量 v 的安全边界约束
        v_bound = 0.0015  
        
        # 构建二次规划 (QP) 问题的代价函数与约束条件
        cost = 0; constraints = [self.s[:, 0] == self.s_init]
        for k in range(N):
            # 累加预测视界内的状态代价与输入代价
            cost += cp.quad_form(self.s[:, k], self.Q_mpc) + cp.quad_form(self.v[:, k], self.R_mpc)
            # 状态转移约束，无缝嵌入 IE 观测的外部扰动 d_hat
            constraints += [self.s[:, k+1] == self.A_bar @ self.s[:, k] + self.B_bar @ (self.v[:, k] + self.d_hat)]
            # 控制动作上下限幅度约束
            constraints += [self.v[:, k] <= v_bound, self.v[:, k] >= -v_bound]
            
        self.prob = cp.Problem(cp.Minimize(cost), constraints)

    def solve(self, s_current, d_current):
        # 更新当前时刻的系统状态与扰动观测值
        self.s_init.value = s_current; self.d_hat.value = d_current
        try:
            # 调用 OSQP 求解器实时寻优，开启热启动 (warm_start) 保障高频实时性
            self.prob.solve(solver=cp.OSQP, warm_start=True)
            # 滚动优化机理：仅提取最优预测序列的第一步作为实际输出
            if self.prob.status in ["optimal", "optimal_inaccurate"]: return self.v.value[:, 0]
        except Exception: pass
        return np.zeros(3)

# ==============================================================================
# ROS 2 主节点 - 螺旋上升轨迹基础跟踪
# ==============================================================================
class HOFABaselineNode(Node):
    def __init__(self):
        super().__init__('hofa_baseline_node')
        # 配置底层通信 QoS 策略
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)

        # 订阅位姿反馈与发布控制指令
        self.odom_sub = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_callback, qos)
        self.ctrl_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.rates_pub = self.create_publisher(VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', qos)
        self.command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)

        # 初始化 CSV 数据记录文件
        self.log_file = open("hofa_round.csv", "w")
        self.log_file.write("timestamp,ref_x,act_x,ref_y,act_y,ref_z,act_z\n")

        # 物理系统参数与无人机基础控制参数
        self.Ts = 0.02; self.m = 2.0; self.g = 9.81; self.hover_thrust_base = 0.57; self.K_att = 4.0  

        # 依据高阶全驱系统(HOFA)理论设定特征参数与极点配置
        p = 0.95  
        self.A1 = np.diag([-2.0, -2.0, -2.0]); self.A0 = np.diag([1.0, 1.0, 1.0])
        self.K1 = np.diag([2.0 * p - 2.0, 2.0 * p - 2.0, 2.0 * p - 2.0]); self.K0 = np.diag([1.0 - p**2, 1.0 - p**2, 1.0 - p**2])
        self.Bp_inv = np.eye(3) * (self.m / (self.Ts**2))

        # 实例化级联架构核心算法模块
        self.ie = StandardIE(self.Ts, self.K0, self.K1, self.A0, self.A1)
        self.mpc = NominalMPC(self.Ts, self.K0, self.K1, self.A0, self.A1, N=5)
        
        # 历史控制指令缓存
        self.v_p_last = np.zeros(3)
        self.v_p_raw_last = np.zeros(3)
        self.v_p_diff_filtered = np.zeros(3)

        # 定时器触发 50Hz (Ts=0.02) 的主控制循环
        self.step_cnt = 0; self.timer = self.create_timer(self.Ts, self.cmdloop_callback)
        self.state = "TAKEOFF"; self.start_time = 0.0
        self.current_pos = np.zeros(3); self.current_vel = np.zeros(3)
        self.current_yaw = 0.0; self.current_roll = 0.0; self.current_pitch = 0.0

    def odom_callback(self, msg):
        # 实时获取 NED 坐标系下位姿及速度
        self.current_pos = np.array(msg.position); self.current_vel = np.array(msg.velocity)
        # 四元数转欧拉角解算 (Roll, Pitch, Yaw)
        q = msg.q 
        self.current_roll = np.arctan2(2*(q[0]*q[1]+q[2]*q[3]), 1-2*(q[1]*q[1]+q[2]*q[2]))
        self.current_pitch = np.arcsin(np.clip(2*(q[0]*q[2]-q[3]*q[1]), -1.0, 1.0))
        self.current_yaw = np.arctan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]*q[2]+q[3]*q[3]))

    def cmdloop_callback(self):
        # 维持 Offboard 模式的心跳连接与阶段起飞判定
        self.publish_offboard_mode()
        if self.step_cnt == 50: self.arm()
        if self.step_cnt == 60: self.set_offboard()

        msg = VehicleRatesSetpoint(); msg.roll = 0.0; msg.pitch = 0.0; msg.yaw = 0.0; msg.thrust_body = [0.0, 0.0, -0.15]
        # 前100帧用于系统姿态稳定
        if self.step_cnt < 100: 
            self.rates_pub.publish(msg); self.step_cnt += 1; return

        t_flight = (self.step_cnt - 100) * self.Ts

        # ======================================================================
        # 螺旋上升轨迹规划 (圆周运动 + 匀速爬升)
        # ======================================================================
        def get_ref(time):
            if time <= 4.0: return np.array([0.0, 0.0, np.clip(-5.0 * (time / 4.0), -5.0, 0.0)])
            elif time <= 6.0: return np.array([0.0, 0.0, -5.0])
            else:
                t_traj = time - 6.0; R = 3.0; omega = 0.3; vz = 0.15; fade = np.clip(t_traj / 5.0, 0.0, 1.0)
                return np.array([R * fade * np.sin(omega * t_traj), R * fade * (1.0 - np.cos(omega * t_traj)), np.clip(-5.0 - vz * t_traj, -15.0, -5.0)])
                
        def get_ref_vel(time):
            if time <= 4.0: return np.array([0.0, 0.0, -1.25])
            elif time <= 6.0: return np.zeros(3)
            else:
                t_traj = time - 6.0; R = 3.0; omega = 0.3; vz = 0.15; fade = np.clip(t_traj / 5.0, 0.0, 1.0)
                return np.array([R * fade * omega * np.cos(omega * t_traj), R * fade * omega * np.sin(omega * t_traj), -vz if (-5.0 - vz * t_traj) > -15.0 else 0.0])

        # 提取当前及未来两拍的期望参考状态
        sigma_rk = get_ref(t_flight); sigma_rk1 = get_ref(t_flight + self.Ts); sigma_rk2 = get_ref(t_flight + 2 * self.Ts)

        # 误差计算与下一步位置外推
        err_p = self.current_pos - sigma_rk  
        err_v = self.current_vel - get_ref_vel(t_flight)
        ref_acc = (get_ref_vel(t_flight + self.Ts) - get_ref_vel(t_flight)) / self.Ts
        err_p_next = err_p + err_v * self.Ts + 0.5 * ref_acc * (self.Ts ** 2)
        
        # 步骤 1: 中间估计器 (IE) 状态更新与扰动观测
        d_est = self.ie.update(err_p, err_p_next, self.v_p_last)

        # 步骤 2: MPC 求解器滚动优化，计算最优控制增量
        s_current = np.concatenate((err_p, err_p_next))
        v_p_raw = self.mpc.solve(s_current, d_est)
        
        # 信号后处理：超前预判补偿 (减小 T_lead 以抑制轨迹内切超调)
        v_p_diff = (v_p_raw - self.v_p_raw_last) / self.Ts
        self.v_p_raw_last = v_p_raw
        T_lead = 0.03
        v_p_lead = v_p_raw + T_lead * v_p_diff
        
        # 低通滤波，平滑虚拟控制增量
        alpha = 0.05
        self.v_p_last = (1.0 - alpha) * self.v_p_last + alpha * v_p_lead

        # =========================================================
        
        # 步骤 3: HOFA 代数解耦逆向映射
        # 计算消除延时的轨迹前馈加速度
        hofa_ref_term = sigma_rk2 + self.A1 @ sigma_rk1 + self.A0 @ sigma_rk
        ff_acc = self.Bp_inv @ hofa_ref_term / self.m

        # 计算状态反馈加速度，并代数合成 MPC 虚拟抗扰增量
        hofa_error_term = self.K1 @ err_p_next + self.K0 @ err_p
        fb_acc_raw = self.Bp_inv @ (hofa_error_term + self.v_p_last) / self.m
        
        # 幅度限幅与总体虚拟期望加速度合成
        fb_acc = np.array([np.clip(fb_acc_raw[0], -2.0, 2.0), np.clip(fb_acc_raw[1], -2.0, 2.0), np.clip(fb_acc_raw[2], -10.0, 10.0)])
        des_acc_raw = ff_acc + fb_acc

        # 步骤 4: 非线性代数映射与指令下发
        # 投影至偏航坐标系消除三维耦合
        c_yaw = np.cos(self.current_yaw); s_yaw = np.sin(self.current_yaw)
        acc_body_x = des_acc_raw[0] * c_yaw + des_acc_raw[1] * s_yaw
        acc_body_y = -des_acc_raw[0] * s_yaw + des_acc_raw[1] * c_yaw
        acc_body_z = des_acc_raw[2]
        
        # 加速度向几何姿态角与推力映射
        des_pitch = np.clip(-acc_body_x / self.g, -0.35, 0.35); des_roll  = np.clip(acc_body_y / self.g, -0.35, 0.35)
        thrust = np.clip(self.hover_thrust_base - (acc_body_z * 0.058), 0.2, 0.9)

        # 偏航角最短环绕修正逻辑
        dyaw_ctrl = 0.0 - self.current_yaw
        if dyaw_ctrl > np.pi: dyaw_ctrl -= 2*np.pi
        if dyaw_ctrl < -np.pi: dyaw_ctrl += 2*np.pi

        # 比例（P）外环控制：解算底层所需的目标角速度指令
        msg.roll = float((des_roll - self.current_roll) * self.K_att); msg.pitch = float((des_pitch - self.current_pitch) * self.K_att)
        msg.yaw = float(dyaw_ctrl * 1.5); msg.thrust_body = [0.0, 0.0, -float(thrust)]

        # Offboard 协议发送与数据日志记录
        if self.step_cnt > 100: 
            self.rates_pub.publish(msg)
            self.log_file.write(f"{t_flight},{sigma_rk[0]},{self.current_pos[0]},{sigma_rk[1]},{self.current_pos[1]},{sigma_rk[2]},{self.current_pos[2]}\n")

            if self.step_cnt % 20 == 0:
                self.get_logger().info(
                    f"Wind EstX: {d_est[0]:.2f} | MPC OutX: {self.v_p_last[0]:.4f}"
                )

        self.step_cnt += 1

    def publish_offboard_mode(self): m = OffboardControlMode(); m.position = False; m.velocity = False; m.acceleration = False; m.attitude = False; m.body_rate = True; self.ctrl_mode_pub.publish(m)
    def arm(self): self.send_cmd(400, 1.0)
    def set_offboard(self): self.send_cmd(176, 1.0, 6.0)
    def send_cmd(self, c, p1, p2=0.0): m = VehicleCommand(); m.command=c; m.param1=p1; m.param2=p2; m.target_system=1; m.target_component=1; m.source_system=1; m.from_external=True; self.command_pub.publish(m)

def main(): rclpy.init(); node = HOFABaselineNode(); rclpy.spin(node); node.destroy_node(); rclpy.shutdown()
if __name__ == '__main__': main()