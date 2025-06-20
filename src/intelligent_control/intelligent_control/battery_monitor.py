#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import BatteryState
from mavros_msgs.msg import State, GlobalPositionTarget
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from geographic_msgs.msg import GeoPoseStamped
from geometry_msgs.msg import TwistStamped
import numpy as np
import time
import math

class BatteryMonitorNode(Node):
    def __init__(self):
        super().__init__('battery_monitor_mavros')

        # QoS profile for reliable communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Parameters
        self.declare_parameter('battery_critical_threshold', 20.0)  # battery percentage to trigger RTH
        self.declare_parameter('battery_warning_threshold', 30.0)   # battery percentage to warn
        self.declare_parameter('home_position_lat', 0.0)    # home latitude
        self.declare_parameter('home_position_lon', 0.0)    # home longitude
        self.declare_parameter('home_position_alt', 0.0)    # home altitude (AMSL)
        self.declare_parameter('solar_power_estimation', True)  # enable solar power estimation
        self.declare_parameter('min_solar_power', 5.0)  # minimum solar power in watts
        self.declare_parameter('max_solar_power', 100.0) # maximum solar power in watts
        self.declare_parameter('energy_per_meter', 0.05)    # energy needed per meter in Wh
        self.declare_parameter('battery_capacity_wh', 1404.0)  # battery capacity in Wh (108Ah * 13V)
        self.declare_parameter('return_speed_ms', 1.5)  # estimated return speed in m/s

        # Get parameter values
        self.battery_critical = self.get_parameter('battery_critical_threshold').value
        self.battery_warning = self.get_parameter('battery_warning_threshold').value
        self.home_lat = self.get_parameter('home_position_lat').value
        self.home_lon = self.get_parameter('home_position_lon').value
        self.home_alt = self.get_parameter('home_position_alt').value
        self.solar_power_estimation = self.get_parameter('solar_power_estimation').value
        self.min_solar_power = self.get_parameter('min_solar_power').value
        self.max_solar_power = self.get_parameter('max_solar_power').value
        self.energy_per_meter = self.get_parameter('energy_per_meter').value
        self.battery_capacity_wh = self.get_parameter('battery_capacity_wh').value
        self.return_speed_ms = self.get_parameter('return_speed_ms').value

        # Initialize internal variables
        self.battery_remaining = 100.0
        self.battery_voltage = 0.0
        self.battery_current = 0.0
        self.current_lat = 0.0
        self.current_lon = 0.0
        self.current_alt = 0.0
        self.vehicle_armed = False
        self.vehicle_connected = False
        self.vehicle_mode = ""
        self.rth_initiated = False
        self.last_battery_warn_time = 0
        self.home_position_set = False
        self.solar_power_current = 0.0

        # Subscribers
        self.battery_sub = self.create_subscription(
            BatteryState, '/mavros/battery', self.battery_callback, qos_profile)
        
        self.state_sub = self.create_subscription(
            State, '/mavros/state', self.state_callback, qos_profile)
        
        self.global_position_sub = self.create_subscription(
            GeoPoseStamped, '/mavros/global_position/global', self.global_position_callback, qos_profile)

        # Service clients
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')
        self.land_client = self.create_client(CommandTOL, '/mavros/cmd/land')

        # Publishers
        self.global_position_pub = self.create_publisher(
            GlobalPositionTarget, '/mavros/setpoint_raw/global', qos_profile)

        # Timer for regular system checks
        self.timer = self.create_timer(1.0, self.system_check)

        # Wait for MAVROS connection
        self.get_logger().info('Waiting for MAVROS connection...')
        self.connection_timer = self.create_timer(0.1, self.check_connection)

    def check_connection(self):
        """Check if MAVROS is connected and services are available"""
        if (self.arming_client.service_is_ready() and 
            self.set_mode_client.service_is_ready() and
            self.vehicle_connected):
            self.connection_timer.cancel()
            self.get_logger().info('Battery Monitor Node initialized - MAVROS connected')

    def battery_callback(self, msg):
        """Process battery status from MAVROS"""
        # BatteryState percentage is from 0.0 to 1.0, convert to percentage
        if not math.isnan(msg.percentage):
            self.battery_remaining = msg.percentage * 100.0
        else:
            # Fallback: estimate from voltage if percentage not available
            self.battery_remaining = self.estimate_battery_from_voltage(msg.voltage)
        
        self.battery_voltage = msg.voltage
        self.battery_current = msg.current

        # Estimate solar power input (simplified model)
        if self.solar_power_estimation and not math.isnan(msg.current):
            if msg.current < 0:  # Negative current typically means charging
                self.solar_power_current = abs(msg.current) * msg.voltage
            else:
                self.solar_power_current = 0.0

            self.get_logger().debug(
                f'Solar power: {self.solar_power_current:.2f}W, '
                f'Battery: {self.battery_remaining:.1f}%, '
                f'Voltage: {msg.voltage:.2f}V, Current: {msg.current:.2f}A'
            )

    def estimate_battery_from_voltage(self, voltage):
        """Estimate battery percentage from voltage (LiFePO4 approximation)"""
        if voltage <= 0:
            return 0.0
        
        # LiFePO4 voltage curve approximation (per cell, assuming 4S)
        cell_voltage = voltage / 4.0
        
        if cell_voltage >= 3.4:
            return 100.0
        elif cell_voltage >= 3.3:
            return 80.0 + (cell_voltage - 3.3) * 200.0  # 80-100%
        elif cell_voltage >= 3.2:
            return 20.0 + (cell_voltage - 3.2) * 600.0  # 20-80%
        elif cell_voltage >= 2.8:
            return (cell_voltage - 2.8) * 50.0           # 0-20%
        else:
            return 0.0

    def state_callback(self, msg):
        """Process vehicle state from MAVROS"""
        self.vehicle_armed = msg.armed
        self.vehicle_connected = msg.connected
        self.vehicle_mode = msg.mode

        # Set home position on first arm if not manually set
        if (self.vehicle_armed and not self.home_position_set and 
            self.home_lat == 0.0 and self.home_lon == 0.0 and
            self.current_lat != 0.0 and self.current_lon != 0.0):
            
            self.home_lat = self.current_lat
            self.home_lon = self.current_lon
            self.home_alt = self.current_alt
            self.home_position_set = True

            self.get_logger().info(
                f'Home position set automatically: '
                f'Lat: {self.home_lat:.7f}, Lon: {self.home_lon:.7f}, Alt: {self.home_alt:.2f}m'
            )

    def global_position_callback(self, msg):
        """Process global position from MAVROS"""
        self.current_lat = msg.pose.position.latitude
        self.current_lon = msg.pose.position.longitude
        self.current_alt = msg.pose.position.altitude

    def system_check(self):
        """Main system monitoring loop"""
        if not self.vehicle_armed or not self.vehicle_connected:
            return
        
        # Calculate distance to home
        distance_to_home = self.calculate_distance_to_home()

        # Calculate energy needed to return home
        energy_needed_wh = distance_to_home * self.energy_per_meter

        # Calculate current energy available in battery
        available_energy_wh = self.battery_capacity_wh * (self.battery_remaining / 100.0)

        # Adjust for solar power if available
        solar_contribution = 0.0
        if self.solar_power_current > 0:
            # Estimate time to home in hours
            time_to_home_hrs = distance_to_home / (self.return_speed_ms * 3600)
            solar_contribution = self.solar_power_current * time_to_home_hrs

        total_available_energy = available_energy_wh + solar_contribution

        # Safety margin (20%)
        energy_with_margin = energy_needed_wh * 1.2

        self.get_logger().debug(
            f'Battery: {self.battery_remaining:.1f}%, '
            f'Distance home: {distance_to_home:.1f}m, '
            f'Energy needed: {energy_with_margin:.1f}Wh, '
            f'Available: {total_available_energy:.1f}Wh, '
            f'Solar: {self.solar_power_current:.1f}W'
        )

        # Check for critical conditions
        should_return = False
        reason = ""

        if self.battery_remaining <= self.battery_critical:
            should_return = True
            reason = f"Battery critical ({self.battery_remaining:.1f}%)"
        elif total_available_energy < energy_with_margin:
            should_return = True
            reason = f"Insufficient energy (need {energy_with_margin:.1f}Wh, have {total_available_energy:.1f}Wh)"
        elif self.battery_remaining <= self.battery_warning:
            # Battery warning (not critical yet)
            current_time = time.time()
            if current_time - self.last_battery_warn_time > 30:
                self.get_logger().warn(
                    f'Battery warning ({self.battery_remaining:.1f}%)! '
                    f'Energy margin: {total_available_energy - energy_with_margin:.1f}Wh'
                )
                self.last_battery_warn_time = current_time

        if should_return and not self.rth_initiated:
            self.get_logger().warn(f'{reason}! Initiating return to home.')
            self.return_to_home()
            self.rth_initiated = True

    def return_to_home(self):
        """Command the vehicle to return to home position using MAVROS"""
        self.get_logger().info('Commanding return to home via MAVROS')

        # Method 1: Try to set RTL mode
        if self.set_mode_client.service_is_ready():
            mode_request = SetMode.Request()
            mode_request.custom_mode = "RTL"  # Return to Launch mode
            
            future = self.set_mode_client.call_async(mode_request)
            future.add_done_callback(self.mode_set_callback)
        else:
            self.get_logger().error('Set mode service not available')

        # Log the RTH initiation
        distance = self.calculate_distance_to_home()
        self.get_logger().warn(
            f'RETURNING TO HOME: Battery {self.battery_remaining:.1f}%, '
            f'Distance {distance:.1f}m, Mode: {self.vehicle_mode}'
        )

    def mode_set_callback(self, future):
        """Callback for mode setting service"""
        try:
            response = future.result()
            if response.mode_sent:
                self.get_logger().info('RTL mode command sent successfully')
            else:
                self.get_logger().error('Failed to set RTL mode')
                # Fallback: try to send global position setpoint to home
                self.send_home_setpoint()
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
            self.send_home_setpoint()

    def send_home_setpoint(self):
        """Send a global position setpoint to home as fallback"""
        if self.home_lat == 0.0 and self.home_lon == 0.0:
            self.get_logger().error('Home position not set, cannot return home')
            return

        # Create global position target
        target = GlobalPositionTarget()
        target.header.stamp = self.get_clock().now().to_msg()
        target.header.frame_id = "map"
        
        # Set coordinate frame (global frame, relative altitude)
        target.coordinate_frame = GlobalPositionTarget.FRAME_GLOBAL_REL_ALT
        
        # Set position
        target.latitude = self.home_lat
        target.longitude = self.home_lon
        target.altitude = self.home_alt
        
        # Set type mask (position only)
        target.type_mask = (
            GlobalPositionTarget.IGNORE_VX |
            GlobalPositionTarget.IGNORE_VY |
            GlobalPositionTarget.IGNORE_VZ |
            GlobalPositionTarget.IGNORE_AFX |
            GlobalPositionTarget.IGNORE_AFY |
            GlobalPositionTarget.IGNORE_AFZ |
            GlobalPositionTarget.IGNORE_YAW |
            GlobalPositionTarget.IGNORE_YAW_RATE
        )

        self.global_position_pub.publish(target)
        self.get_logger().info(f'Sent home position setpoint: {self.home_lat:.7f}, {self.home_lon:.7f}')

    def calculate_distance_to_home(self):
        """Calculate the great circle distance between current position and home"""
        if self.home_lat == 0.0 and self.home_lon == 0.0:
            return 0.0

        # Convert to radians
        lat1 = math.radians(self.current_lat)
        lon1 = math.radians(self.current_lon)
        lat2 = math.radians(self.home_lat)
        lon2 = math.radians(self.home_lon)

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000  # Radius of earth in meters

        # Calculate horizontal distance
        horizontal_distance = c * r

        # Calculate vertical distance (altitude difference)
        vertical_distance = abs(self.home_alt - self.current_alt)

        # Calculate 3D distance using Pythagorean theorem
        total_distance = math.sqrt(horizontal_distance**2 + vertical_distance**2)

        return total_distance

def main(args=None):
    rclpy.init(args=args)
    battery_monitor_node = BatteryMonitorNode()
    
    try:
        rclpy.spin(battery_monitor_node)
    except KeyboardInterrupt:
        battery_monitor_node.get_logger().info('Node stopped cleanly')
    except Exception as e:
        battery_monitor_node.get_logger().error(f'Error: {e}')
    finally:
        battery_monitor_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()