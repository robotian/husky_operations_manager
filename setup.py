from glob import glob

from setuptools import find_packages, setup

package_name = 'husky_operations_manager'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name + '/data', glob('data/*')),
        ('share/' + package_name + '/mesh', glob('mesh/*')),
        ('share/' + package_name + '/launch', glob('launch/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='vupadhye@mtu.edu',
    description='Clearpath Husky A300 Operations Manager. This package handles task management and system monitoring.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'husky_operations_manager = husky_operations_manager.husky_operations_manager:main',
            # Action client test nodes
            'test_navigation_client = husky_operations_manager.unit_test.test_navigation_client:main',
            'test_docking_client = husky_operations_manager.unit_test.test_docking_client:main',
            'test_undocking_client = husky_operations_manager.unit_test.test_undocking_client:main',
            'test_harvest_client = husky_operations_manager.unit_test.test_harvest_client:main',
            # Reverse Navigation Test
            'reverse_navigation_node = husky_operations_manager.reverse_navigation_node:main',
            'docking_param_fetcher = husky_operations_manager.test_docking_param_fetcher:main',
        ],
    },
)
