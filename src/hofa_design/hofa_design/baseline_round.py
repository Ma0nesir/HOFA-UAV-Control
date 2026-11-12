#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import VehicleOdometry, OffboardControlMode, TrajectorySetpoint, VehicleCommand
import numpy as np

# ==============================================================================
# ROS 2 主节点 - PX4 原生固件对照组 (纯 PID 位置控制 Baseline)
# ==============================================================================
class PX4BaselineNode(Node):
    def __init__(self):
        super().__init__('baseline_round_node')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)

        self.odom_sub = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_callback, qos)
        self.ctrl_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        
        # ★ 注意：这里不再是 VehicleRatesSetpoint，而是原生的 TrajectorySetpoint
        self.traj_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)

        # 记录相同的日志格式，命名为 baseline_round.csv
        self.log_file = open("baseline_round.csv", "w")
        self.log_file.write("timestamp,ref_x,act_x,ref_y,act_y,ref_z,act_z\n")

        self.Ts = 0.02
        self.step_cnt = 0
        self.timer = self.create_timer(self.Ts, self.cmdloop_callback)
        
        self.current_pos = np.zeros(3)

    def odom_callback(self, msg):
        self.current_pos = np.array(msg.position)

    def cmdloop_callback(self):
        self.publish_offboard_mode()
        
        if self.step_cnt == 50: self.arm()
        if self.step_cnt == 60: self.set_offboard()

        msg = TrajectorySetpoint()
        # PX4 的 TrajectorySetpoint 必须带上时间戳
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        
        # 起飞前保持原地
        if self.step_cnt < 100: 
            msg.position = [0.0, 0.0, -0.15]
            msg.yaw = 0.0
            self.traj_pub.publish(msg)
            self.step_cnt += 1
            return

        t_flight = (self.step_cnt - 100) * self.Ts

        # =========================================================
        # ★ 绝对公平：使用完全相同的解析几何轨迹
        # =========================================================
        def get_ref(time):
            if time <= 4.0: 
                return np.array([0.0, 0.0, np.clip(-5.0 * (time / 4.0), -5.0, 0.0)])
            elif time <= 6.0: 
                return np.array([0.0, 0.0, -5.0])
            else:
                t_traj = time - 6.0
                R = 3.0; omega = 0.3; vz = 0.15
                fade = np.clip(t_traj / 5.0, 0.0, 1.0) 
                x = R * fade * np.sin(omega * t_traj)
                y = R * fade * (1.0 - np.cos(omega * t_traj))
                z = np.clip(-5.0 - vz * t_traj, -15.0, -5.0) 
                return np.array([x, y, z])

        def get_ref_vel(time):
            if time <= 4.0: return np.array([0.0, 0.0, -1.25])
            elif time <= 6.0: return np.zeros(3)
            else:
                t_traj = time - 6.0; R = 3.0; omega = 0.3; vz = 0.15; fade = np.clip(t_traj / 5.0, 0.0, 1.0)
                vx = R * fade * omega * np.cos(omega * t_traj)
                vy = R * fade * omega * np.sin(omega * t_traj)
                vz_vel = -vz if (-5.0 - vz * t_traj) > -15.0 else 0.0
                return np.array([vx, vy, vz_vel])

        sigma_rk = get_ref(t_flight)
        ref_vel = get_ref_vel(t_flight)

        # =========================================================
        # ★ 喂给 PX4 原生 PID 控制器
        # =========================================================
        msg.position = [float(sigma_rk[0]), float(sigma_rk[1]), float(sigma_rk[2])]
        msg.velocity = [float(ref_vel[0]), float(ref_vel[1]), float(ref_vel[2])]
        msg.yaw = 0.0
        
        self.traj_pub.publish(msg)

        # 写入相同的日志文件格式
        self.log_file.write(f"{t_flight},{sigma_rk[0]},{self.current_pos[0]},"
                            f"{sigma_rk[1]},{self.current_pos[1]},"
                            f"{sigma_rk[2]},{self.current_pos[2]}\n")

        if self.step_cnt % 50 == 0:
            self.get_logger().info(f"PX4 Native Mode -> Z: {self.current_pos[2]:.2f}m")

        self.step_cnt += 1

    def publish_offboard_mode(self):
        m = OffboardControlMode()
        m.position = True       # ★ 开启原生位置控制
        m.velocity = True       # ★ 开启原生速度控制
        m.acceleration = False
        m.attitude = False
        m.body_rate = False
        m.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.ctrl_mode_pub.publish(m)

    def arm(self): self.send_cmd(400, 1.0)
    def set_offboard(self): self.send_cmd(176, 1.0, 6.0)
    def send_cmd(self, c, p1, p2=0.0): 
        m = VehicleCommand(); m.command=c; m.param1=p1; m.param2=p2
        m.target_system=1; m.target_component=1; m.source_system=1; m.from_external=True
        m.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.command_pub.publish(m)

def main(): 
    rclpy.init()
    node = PX4BaselineNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: 
        node.log_file.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': main()