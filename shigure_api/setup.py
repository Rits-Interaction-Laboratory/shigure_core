from setuptools import setup

package_name = 'shigure_api'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yukiho-YOSHIEDA',
    maintainer_email='is0436er@ed.ritsumei.ac.jp',
    description='FastAPI bridge for Shigure face recognition events',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'shigure_api = shigure_api.main:main',
        ],
    },
)
