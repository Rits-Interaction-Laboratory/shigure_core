from setuptools import setup

package_name = 'shigure'
node_package_name = 'shigure.nodes'
util_package_name = 'shigure.util'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name, node_package_name, node_package_name + '.bg_subtraction',
              node_package_name + '.subtraction_analysis', node_package_name + '.frame_combiner',
              node_package_name + '.people_detection', util_package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'bg_subtraction = shigure.nodes.node_bg_subtraction:main',
            'subtraction_analysis = shigure.nodes.node_subtraction_analysis:main',
        ],
    },
)
