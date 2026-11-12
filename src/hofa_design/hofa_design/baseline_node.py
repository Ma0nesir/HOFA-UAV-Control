#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import VehicleOdometry, OffboardControlMode, TrajectorySetpoint, VehicleCommand
import numpy as np

# ==============================================================================
# ROS 2 主节点 - PX4 原生固件对照组 (Square Baseline)
# ==============================================================================
class PX4BaselineSquareNode(Node):
    def __init__(self):
        super().__init__('baseline_square_node')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)

        self.odom_sub = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_callback, qos)
        self.ctrl_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        
        self.traj_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)

        # ★ 日志名
        self.log_file = open("baseline_square.csv", "w")
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
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        
        if self.step_cnt < 100: 
            msg.position = [0.0, 0.0, -0.15]
            msg.yaw = 0.0
            self.traj_pub.publish(msg)
            self.step_cnt += 1
            return

        t_flight = (self.step_cnt - 100) * self.Ts

        # ======================================================================
        # ★ 轨迹生成：与 hofa_square.py 保持绝对一致的 1x1 米平滑正方形
        # ======================================================================
        def get_ref(time):
            if time <= 4.0: # 0-4秒：平滑起飞到 5m (Z=-5.0)
                tau = time / 4.0
                z = -5.0 * 0.5 * (1.0 - np.cos(np.pi * tau))
                return np.array([0.0, 0.0, z])
            elif time <= 6.0: # 4-6秒：悬停稳定
                return np.array([0.0, 0.0, -5.0])
            elif time <= 10.0: # 6-10秒：向前飞 5m
                tau = (time - 6.0) / 4.0
                x = 5.0 * 0.5 * (1.0 - np.cos(np.pi * tau))
                return np.array([x, 0.0, -5.0])
            elif time <= 14.0: # 10-14秒：向右飞 5m
                tau = (time - 10.0) / 4.0
                y = 5.0 * 0.5 * (1.0 - np.cos(np.pi * tau))
                return np.array([5.0, y, -5.0])
            elif time <= 18.0: # 14-18秒：向后飞 5m
                tau = (time - 14.0) / 4.0
                x = 5.0 - 5.0 * 0.5 * (1.0 - np.cos(np.pi * tau))
                return np.array([x, 5.0, -5.0])
            elif time <= 22.0: # 18-22秒：向左飞 5m 回到起点
                tau = (time - 18.0) / 4.0
                y = 5.0 - 5.0 * 0.5 * (1.0 - np.cos(np.pi * tau))
                return np.array([0.0, y, -5.0])
            else: # 飞完悬停
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

        sigma_rk = get_ref(t_flight)
        ref_vel = get_ref_vel(t_flight)

        msg.position = [float(sigma_rk[0]), float(sigma_rk[1]), float(sigma_rk[2])]
        msg.velocity = [float(ref_vel[0]), float(ref_vel[1]), float(ref_vel[2])]
        msg.yaw = 0.0
        
        self.traj_pub.publish(msg)

        self.log_file.write(f"{t_flight},{sigma_rk[0]},{self.current_pos[0]},"
                            f"{sigma_rk[1]},{self.current_pos[1]},"
                            f"{sigma_rk[2]},{self.current_pos[2]}\n")

        self.step_cnt += 1

    def publish_offboard_mode(self):
        m = OffboardControlMode()
        m.position = True       
        m.velocity = True       
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
    node = PX4BaselineSquareNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: 
        node.log_file.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': main()