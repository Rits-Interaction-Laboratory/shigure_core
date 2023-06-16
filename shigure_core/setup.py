import os
from glob import glob

from setuptools import setup

package_name = 'shigure_core'
node_package_name = 'shigure_core.nodes'
util_package_name = 'shigure_core.util'
enum_package_name = 'shigure_core.enum'
db_package_name = 'shigure_core.db'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name, node_package_name, node_package_name + '.common_model',
              node_package_name + '.bg_subtraction', node_package_name + '.people_mask_buffer',
              node_package_name + '.subtraction_analysis', node_package_name + '.people_tracking',
              node_package_name + '.object_detection', node_package_name + '.object_tracking',
              node_package_name + '.contact_detection', node_package_name + '.record_event',
              node_package_name + '.yolox_object_detection',
              util_package_name, enum_package_name, db_package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('shigure_core/launch/*_launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yukiho-YOSHIEDA',
    maintainer_email='is0436er@ed.ritsumei.ac.jp',
    description='インタラクション研究室 室内監視システム',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bg_subtraction = shigure_core.nodes.node_bg_subtraction:main',
            'people_mask_buffer = shigure_core.nodes.node_people_mask_buffer:main',
            'subtraction_analysis = shigure_core.nodes.node_subtraction_analysis:main',
            'object_detection = shigure_core.nodes.node_object_detection:main',
            'object_tracking = shigure_core.nodes.node_object_tracking:main',
            'contact_detection = shigure_core.nodes.node_contact_detection:main',
            'people_tracking = shigure_core.nodes.node_people_tracking:main',
            'record_event = shigure_core.nodes.node_record_event:main',
            'pose_save = shigure_core.nodes.node_pose_save:main',
            'yolox_object_detection = shigure_core.nodes.node_yolox_object_detection:main',
        ],
    },
)
