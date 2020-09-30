from setuptools import setup

package_name = 'shigure'
node_package_name = 'shigure.nodes'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name, node_package_name, node_package_name + '.bg_subtraction'],
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
        ],
    },
)
