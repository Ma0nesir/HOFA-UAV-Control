#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# 导入必要的消息类型
from px4_msgs.msg import VehicleOdometry
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path
import tf2_ros  
import numpy as np

class PX4RVizBridge(Node):
    def __init__(self):
        super().__init__('px4_rviz_bridge')

        # 1. 定义正确的 QoS 配置 (适配 PX4 的 Best Effort 模式)
        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 2. 订阅 PX4 原始里程计数据
        self.sub_odom = self.create_subscription(
            VehicleOdometry, 
            '/fmu/out/vehicle_odometry', 
            self.odom_callback, 
            qos_sub
        )

        # 3. 发布可视化数据
        self.pub_pose = self.create_publisher(PoseStamped, '/rviz/uav_pose', 10)
        self.pub_path = self.create_publisher(Path, '/rviz/uav_path', 10)
        
        # 4. 初始化 TF 广播器 (建立 map -> base_link 的连接)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # 初始化轨迹消息
        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"
        
        self.get_logger().info("🌉 坐标桥接器已就绪：NED -> ENU 转换中...")

    def odom_callback(self, msg):
        now = self.get_clock().now().to_msg()
        
        # --- 坐标转换 (PX4 NED -> ROS ENU) ---
        # NED: X-北, Y-东, Z-下
        # ENU: X-东, Y-北, Z-上
        pos_x = float(msg.position[1])  # 东 -> X
        pos_y = float(msg.position[0])  # 北 -> Y
        pos_z = float(-msg.position[2]) # -下 -> Z

        # --- 填充位姿消息 ---
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = now
        pose.pose.position.x = pos_x
        pose.pose.position.y = pos_y
        pose.pose.position.z = pos_z
        
        # 四元数转换 (注意顺序调整)
        pose.pose.orientation.w = float(msg.q[0])
        pose.pose.orientation.x = float(msg.q[2])
        pose.pose.orientation.y = float(msg.q[1])
        pose.pose.orientation.z = float(-msg.q[3])

        # 发布当前位姿
        self.pub_pose.publish(pose)

        # --- 发布 TF 变换 (让 RViz 承认 map 坐标系) ---
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = pos_x
        t.transform.translation.y = pos_y
        t.transform.translation.z = pos_z
        t.transform.rotation = pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)

        # --- 更新并发布轨迹 ---
        self.path_msg.header.stamp = now
        self.path_msg.poses.append(pose)
        
        # 限制长度防止卡顿 (保留最后 1000 个点)
        if len(self.path_msg.poses) > 1000:
            self.path_msg.poses.pop(0)
            
        self.pub_path.publish(self.path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PX4RVizBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()