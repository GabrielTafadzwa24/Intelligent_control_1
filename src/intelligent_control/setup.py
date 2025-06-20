from setuptools import find_packages, setup

package_name = 'intelligent_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tafadzwa',
    maintainer_email='tafadzwa@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    # tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fuzzy_logic_controller = intelligent_control.fuzzy_logic_controller:main',
            'neuro_fuzzy_controller = intelligent_control.neuro_fuzzy_controller:main',
            'battery_monitor = intelligent_control.battery_monitor:main'
        ],
    },
)
