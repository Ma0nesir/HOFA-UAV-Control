import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 1. 获取包路径
    package_name = 'hofa_design'
    
    # 2. 寻找 RViz 配置文件
    # 确保你的 rviz 配置文件放在 src/hofa_design/rviz/rviz.rviz
    rviz_config_path = os.path.join(
        get_package_share_directory(package_name),
        'rviz',
        'rviz.rviz'
    )

    return LaunchDescription([
        # 启动桥接节点
        Node(
            package=package_name,
            executable='px4_rviz_bridge',
            name='rviz_bridge',
            output='screen'
        ),

        # 启动 RViz2 界面
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_path],
            output='screen'
        )
    ])