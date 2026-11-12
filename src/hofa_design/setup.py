from setuptools import find_packages, setup
import os               
from glob import glob   

package_name = 'hofa_design'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xu',
    maintainer_email='xu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'hofa_node = hofa_design.hofa_node:main',
            'baseline_node = hofa_design.baseline_node:main',
            'hofa_wind_node = hofa_design.hofa_wind_node:main',
            'baseline_wind_node = hofa_design.baseline_wind_node:main',
            'px4_rviz_bridge = hofa_design.px4_rviz_bridge:main',
            'hofa_round = hofa_design.hofa_round:main',
            'baseline_round_node = hofa_design.baseline_round:main',
        ],
    },
)
