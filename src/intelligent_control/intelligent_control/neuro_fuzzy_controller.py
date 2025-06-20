#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from collections import deque
from scipy.spatial.transform import Rotation
import math
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State, WaypointList, Waypoint, ActuatorControl, Altitude
from mavros_msgs.srv import CommandBool, SetMode, WaypointPush
from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import Float64
from transforms3d.euler import quat2euler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import time
import threading
from threading import Lock

class NeuroFuzzyModel(nn.Module):
    """PyTorch Neural Network model for the neuro-fuzzy controller"""

    def __init__(self):
        super(NeuroFuzzyModel, self).__init__()

        # common layers
        self.common = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # output layers
        self.speed = nn.Linear(16, 1)
        self.yaw_rate = nn.Linear(16, 1)

    def forward(self, x):
        common_features = self.common(x)
        speed = self.speed(common_features)
        yaw_rate = self.yaw_rate(common_features)
        return speed, yaw_rate

class NeuroFuzzyNavigation(Node):
    """Node that uses a Neuro-Fuzzy Controller to optimize PX4 mission navigation"""

    def __init__(self):
        super().__init__('fuzzy_navigation')

        # setting device for PyTorch (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # Initialize communication infrastracture
        self._init_quality_of_service_profiles()
        self._init_publishers()
        self._init_subscribers()
        self._init_services()

        # Initializing navigation state variables
        self._init_vehicle_state()
        self._init_navigation_parameters()

        # thread safety mechanisms
        self.lock = threading.Lock()  # lock for shared resources
        self.stop_event = threading.Event() # flag to stop the training thread

        # initialize parameters for backpropagation
        self._init_parameters_for_backpropagation()

        # initialize fuzzy membership functions
        self._init_fuzzy_membership()

        # create neuro-fuzzy model
        self.create_neuro_fuzzy_model()

        self._init_training_paramaters()

        # timer to adjust mission parameters
        self.timer = self.create_timer(0.5, self.navigation_timer_callback)

        # timer for continuous training (every 20 seconds)
        self.training_timer = self.create_timer(20.0, self.training_timer_callback)

        self.get_logger().info("Neuro-Fuzzy Navigation node started.")

    def _init_quality_of_service_profiles(self):
        """Initializing QoS profile for reliable communication."""
        self.uxrQoS_pub = QoSProfile(
            # """QoS settings for publishers"""
            reliability = ReliabilityPolicy.BEST_EFFORT,
            durability = DurabilityPolicy.TRANSIENT_LOCAL,
            history = HistoryPolicy.KEEP_LAST,
            depth = 0
        )
        self.uxrQoS_sub = QoSProfile(
            # """QoS settings for subscribers"""
            reliability = ReliabilityPolicy.BEST_EFFORT,
            durability = DurabilityPolicy.VOLATILE,
            history = HistoryPolicy.KEEP_LAST,
            depth = 10
        )

    def _init_publishers(self):
        """Initializing all publishers."""
        self.motor_command_publisher = self.create_publisher(
            ActuatorControl, '/mavros/actuator_control', self.uxrQoS_pub)
        
    def _init_subscribers(self):
        """Initialize subscribers."""
        self.local_position_subscriber = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self.local_position_callback, self.uxrQoS_sub)


        self.global_position_subscriber = self.create_subscription(
            NavSatFix, "/mavros/global_position/global", self.global_position_callback, self.uxrQoS_sub)
        
        
        self.waypoint_subscriber = self.create_subscription(
            WaypointList, '/mavros/mission/waypoints', self.waypoint_callback, self.uxrQoS_sub)
        
        self.state_subscriber = self.create_subscription(
            State, '/mavros/state', self.state_callback, self.uxrQoS_sub)
        
        self.imu_subscriber = self.create_subscription(
            Imu, 'mavros/imu/data', self.imu_callback, self.uxrQoS_sub)

        self.altitude_subscriber = self.create_subscription(
            Altitude, '/mavros/altitude', self.altitude_callback, self.uxrQoS_sub)
        
    def _init_services(self):
        """Initialize service clients"""
        self.arm_service = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_service = self.create_client(SetMode, '/mavros/set_mode')
        self.waypoint_push_service = self.create_client(WaypointPush, '/mavros/mission/push')

    def _init_vehicle_state(self):
        """Initializing vehicle state variables"""
        self.current_position = PoseStamped()
        self.global_position = NavSatFix()
        self.vehicle_heading = 0.0     # Current heading in radians
        self.vehicle_state = State()

    def _init_navigation_parameters(self):
        """Initializing navigation parameters and declare ROS parameters"""
        # Navigation targets
        self.target_x = 0.0     # Default target waypoint X-coordinate
        self.target_y = 0.0     # Default target waypoint Y-coordinate

        # Path tracking variables        
        self.distance_to_target_wp = 0.0    # Distance to waypoint
        self.distance_from_prev_wp = 0.0    # Distance from previous waypoint
        self.distance_prev_target_wp = 0.0  # Distance between waypoints
        self.cross_track_error = 0.0        # Cross track error of the vehicle
        self.theta_actual_path = 0.0        # The angle between waypoin path and actual path of the vehicle

        # Waypoing management
        self.waypoints = []
        self.current_wp_index = 0
        self.previous_waypoint = (0.0, 0.0)    # Store previous waypoint for path calculation

        # Parameters
        self.declare_parameter('~wave_compensation_gain', 30)
        self.wave_compensation_gain = self.get_parameter('~wave_compensation_gain').value
        self.declare_parameter('~altitude_smoothing_window', 10)
        self.altitude_smoothing_window = self.get_parameter('~altitude_smoothing_window').value
        self.declare_parameter('~pitch_threshold', 0.1)
        self.pitch_threshold = self.get_parameter('~pitch_threshold').value 

        # State variables
        self.altitude_history = deque(maxlen=self.altitude_smoothing_window)
        self.altitude_rate = 0.0
        self.wave_phase = 0         # -1: going down, 0: stable, 1: going up
        self.current_altitude = 0.0
        self.current_pitch = 0.0

        # initialize data collection for training
        self.training_data = {
            'fuzzified_inputs': [],
            'speed': [],
            'yaw_rate': []
        }

        # Declare ROS parameters
        self._declare_ros_parameters()

    def _declare_ros_parameters(self):
        """Declaring all ROS parameters."""
        # Home position parameters
        self.declare_parameter('home_latitude', 0.0)
        self.declare_parameter('home_longitude', 0.0)
        self.declare_parameter('home_altitude', 0.0)

        # Debug parameters
        self.declare_parameter('debug_coordinates', False)
        self.declare_parameter('debug_navigation', False)

        # Navigation parameters
        self.declare_parameter('waypoint_reached_threshold', 5.0)
        self.waypoint_reached_threshold = self.get_parameter('waypoint_reached_threshold').value

    def _init_parameters_for_backpropagation(self):
        """Initializing parameters for navigation."""
        self.learning_rate = 0.01
        self.batch_size = 16
        self.epochs = 150
        self.validation_split = 0.2

    def _init_training_paramaters(self):
        """Initializing training parameters for neuro-fuzzy logic controller"""
        # flag to indicate if model has been trained
        self.model_trained = False

        # counter for collecting training data
        self.data_collect_counter = 0

        # custom training metrics
        self.training_losses = []
        self.validation_losses = []

        # counter for traininf iterations
        self.training_iteration = 0

        # flag to ensure continuous training
        self.continuous_training = True

        # early stopping settings
        self.patient = 5    # reduced patience for more aggressive training

        # Model confidence level (starts with low confidence)
        self.model_confidence = 0.0

    # Callback methods
    def local_position_callback(self, msg):
        """Callback to update vehicle position."""
        self.current_position = msg
        self._update_vehicle_heading_from_quaternion(msg)        
        self.check_waypoint_reached()

    def _update_vehicle_heading_from_quaternion(self, msg):
        """Extracting heading from pose message quaternion."""
        q = [
            msg.pose.orientation.w, 
            msg.pose.orientation.x, 
            msg.pose.orientation.y, 
            msg.pose.orientation.z
        ]
        
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        _, _, self.vehicle_heading = quat2euler([q[0], q[1], q[2], q[3]])

    def global_position_callback(self, msg):
        self.global_position = msg
        self.check_waypoint_reached()

    def state_callback(self, msg):
        """Callback to update vehicle state."""
        self.vehicle_state = msg

    def imu_callback(self, msg):
        """Process IMU data to extract pitch angle."""
        # Convert quatermiom to euler angles
        q = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        rotation = Rotation.from_quat(q)
        euler = rotation.as_euler('xyz', degrees=False)
        self.current_pitch = euler[1]  # Pitch angle

    def altitude_callback(self, msg):
        """Process barometric altitude data."""
        self.current_altitude = msg.amsl
        self.altitude_history.append(self.current_altitude)

        # Calculate altitude rate of change for wave detection
        if len(self.altitude_history) >= 2:
            dt = 0.1    # Assuming 10Hz update rate
            self.altitude_rate = (self.altitude_history[-1] - self.altitude_history[-2]) / dt

            # Determining wave phase based on altitude rate and pitch
            self.update_wave_phase()

    def update_wave_phase(self):
        """Determining wave phase based on pitch angle and altitude rate"""
        pitch_deg = np.degrees(abs(self.current_pitch))

        # use combination of pitch angle and altitude rate
        if abs(self.current_pitch) > self.pitch_threshold:
            if self.current_pitch > 0 and self.altitude_rate > 0.02:    # Bow up, climbing
                self.wave_phase = 1 # Going up wave
            elif self.current_pitch < 0 and self.altitude_rate < -0.02: # Bow down, descending
                self.wave_phase = -1    # Going down wave
            else:
                self.wave_phase = 0 # Transitional/stable
        else:
            self.wave_phase = 0 # Stable condition
   
    # Distance calculation methods        
    def distance_calculation(self, lat, lon):
        """alt(amsl): meters"""
        R = 6371000  # metres
        rlat1 = math.radians(lat)
        rlat2 = math.radians(self.global_position.latitude)

        rlat_d = math.radians(self.global_position.latitude- lat)
        rlon_d = math.radians(self.global_position.longitude - lon)

        a = (math.sin(rlat_d / 2) * math.sin(rlat_d / 2) + math.cos(rlat1) *
             math.cos(rlat2) * math.sin(rlon_d / 2) * math.sin(rlon_d / 2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        d = R * c
        
        return d
    
    def waypoint_path(self, prev_lat, prev_lon, target_lat, target_lon):
        """alt(amsl): meters"""
        R = 6371000  # metres
        rlat1 = math.radians(target_lat)
        rlat2 = math.radians(prev_lat)

        rlat_d = math.radians(prev_lat - target_lat)
        rlon_d = math.radians(prev_lon - target_lon)

        a = (math.sin(rlat_d / 2) * math.sin(rlat_d / 2) + math.cos(rlat1) *
             math.cos(rlat2) * math.sin(rlon_d / 2) * math.sin(rlon_d / 2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        d = R * c
        
        return d
    
    # Waypoint handling methods        
    def waypoint_callback(self, msg):
        """
        Callback for receiving mission waypoints from MAVROS.
        
        Processes waypoint list messages, converts global coordinates to local coordinates,
        and updates the navigation path parameters.
        
        Args:
            msg (WaypointList): Message containing the list of waypoints from MAVROS
        """
        # Skip processing if message is empty
        if len(msg.waypoints) == 0:
            self.get_logger().warn("Received empty waypoint list")
            return
            
        # Log receipt of waypoints
        self.get_logger().info(f"Received waypoint list with {len(msg.waypoints)} waypoints")

        # Process waypoints
        self._process_waypoint_list(msg.waypoints)
        
        # Update navigation parameters based on current waypoint
        self.update_navigation_targets()

        # log current navigation status
        self._log_navigation_status()
        
    def _process_waypoint_list(self, raw_waypoints):
        """Process and filter a list of raw waypoints.
        
        Args:
            raw_waypoints (list): List of Waypoint messages from MAVROS"""
        # Store the original waypoints for potential future reference
        self.original_waypoints = raw_waypoints.copy()
        
        # Clear existing waypoints and reset navigation state
        self.waypoints = []
        
        # Flag to track if the current waypoint was identified in the message
        current_wp_found = False
        current_wp_index = 0
        
        # Process each waypoint in the message
        for i, wp in enumerate(raw_waypoints):
            # Skip certain waypoints based on command type
            if wp.command not in [16, 22, 82, 84]:  # Navigation commands: waypoint, takeoff, spline, loiter
                self.get_logger().debug(f"Skipping non-navigation waypoint at index {i} (cmd={wp.command})")
                continue
                
            # Check if this is marked as the current waypoint
            if wp.is_current:
                current_wp_index = len(self.waypoints)
                current_wp_found = True
                self.get_logger().info(f"Current waypoint identified at mission index {i}")
            
            # Store local coordinates and metadata in a dictionary for richer waypoint representation
            waypoint_data = {
                'global_coordinates': (wp.x_lat, wp.y_long, wp.z_alt),
                'command': wp.command,
                'param1': wp.param1,  # Hold time for waypoints
                'param2': wp.param2,  # Acceptance radius
                'param3': wp.param3,  # Pass through waypoint (0 = no)
                'param4': wp.param4,  # Desired yaw angle
                'autocontinue': wp.autocontinue,
                'frame': wp.frame,
                'original_index': i
            }
            
            self.waypoints.append(waypoint_data)
            self.get_logger().debug(
                f"Added waypoint: global({wp.x_lat:.7f}, {wp.y_long:.7f}, {wp.z_alt:.2f})")
        
        # Handle empty waypoint list after filtering
        if not self.waypoints:
            self.get_logger().warn("No valid waypoints found after filtering")
            return
            
        # If no current waypoint was found, set it to the first waypoint
        if not current_wp_found:
            current_wp_index = 0
            self.get_logger().info("No current waypoint flag found, defaulting to first waypoint")
        
        # Update the current waypoint index
        self.current_wp_index = current_wp_index

    def _log_navigation_status(self):
        """Log current navigation status and targets."""
        if self.current_wp_index >= len(self.waypoints):
            return
        
        self.get_logger().debug(
            f"Navigation status: targeting waypoint {self.current_wp_index} at local position "
            f"({self.target_x:.2f}, {self.target_y:.2f}, {self.target_z:.2f})")

    def check_waypoint_reached(self):
        """Check if vehicle has reached the current waypoint and advance to the next one."""
        if self.current_wp_index >= len(self.waypoints):
            return
    
        # Get waypoint coordinates from the dictionary structure
        target_lat, target_lon = self.waypoints[self.current_wp_index]['global_coordinates'][:2]
        
        if self.distance_to_target_wp < self.waypoint_reached_threshold:
            self._advance_to_next_waypoint(target_lat, target_lon)

    def _advance_to_next_waypoint(self, prev_x, prev_y):
        """Advance to the next waypoint in the mission."""
        # Increment waypoint index
        prev_index = self.current_wp_index

        # Increment with bounds checking 
        if self.current_wp_index < len(self.waypoints) - 1:
            self.current_wp_index += 1
            self.get_logger().info(f"Advanced to waypoint {self.current_wp_index}")

            # Update navigation targets for new waypoint
            self.update_navigation_targets()
        else:
            self.get_logger().debug("Reached final waypoint.")

    def _set_previous_waypoint(self):
        """Set the previous waypoint for path calculation."""
        if self.current_wp_index > 0:
            # Use the previous waypoint in the list
            prev_wp = self.waypoints[self.current_wp_index - 1]
            self.previous_waypoint = prev_wp['global_coordinates'][:2]  # Just x and y
        else:
            # For the first waypoint, use current position as previous
            self.previous_waypoint = (
                self.global_position.latitude,
                self.global_position.longitude
            )
    
    def update_navigation_targets(self):
        """ Update navigation targets based on the current waypoint index."""
        # Validate waypoint index
        if self.current_wp_index >= len(self.waypoints):
            self.get_logger().error(f"Invalid waypoint index: {self.current_wp_index}")
            return False
            
        # Get current target waypoint coordinates
        current_wp = self.waypoints[self.current_wp_index]
        self.target_x, self.target_y, self.target_z = current_wp['global_coordinates']
        
        # Set the previous waypoint for path calculations
        self._set_previous_waypoint()

        # Update acceptance radious based on waypoing parameter
        self._update_acceptance_radius(current_wp)
        
        self.get_logger().debug(
            f"Navigation target updated: pos=({self.target_x:.2f}, {self.target_y:.2f}, {self.target_z:.2f}), "
            f"acceptance radius={self.waypoint_reached_threshold:.2f}m")
        
        return True

    def _update_acceptance_radius(self, waypoint):
        """Updating waypoint acceptance radious based on waypooint parameters."""
        self.waypoint_reached_threshold = waypoint['param2']
        if self.waypoint_reached_threshold <= 0.0:
            # If not specified, use the default parameter value
            self.waypoint_reached_threshold = self.get_parameter('waypoint_reached_threshold').value

    # Neuro-Fuzzy controller methods
    def _create_fuzzy_variables(self):
        """Creating fuzzy variables for inputs and outputs."""
        # Input variables
        distance_to_wp = ctrl.Antecedent(np.arange(0.0, 5000.0, 1.0), 'distance_to_wp')
        cross_track_error = ctrl.Antecedent(np.arange(-180.0, 180.0, 1.0), 'cross_track_error')
        wave_condition = ctrl.Antecedent(np.arange(-1.0, 1.1, 0.1), 'wave_condition')

        # Output variables - controlling velocity and yaw rate
        speed = ctrl.Consequent(np.arange(0.0, 2.5, 0.1), 'speed')
        yaw_rate = ctrl.Consequent(np.arange(-1.0, 1.0, 0.1), 'yaw_rate')

        # Store variable in a dictionary for wasy access
        fuzzy_vars = {
            'distance_to_wp': distance_to_wp,
            'cross_track_error': cross_track_error,
            'wave_condition': wave_condition,
            'speed': speed,
            'yaw_rate': yaw_rate
        }
        
        return fuzzy_vars
    
    def _init_fuzzy_membership(self):
        """Initialize fuzzy membership functions for fuzzification"""

        self.fuzzy_variables = self._create_fuzzy_variables()

        # universe of discourse
        self.distance_universe = self.fuzzy_variables['distance_to_wp'].universe
        self.cross_track_universe = self.fuzzy_variables['cross_track_error'].universe
        self.wave_condition_universe = self.fuzzy_variables['wave_condition'].universe

        # distance to waypoint membership function
        self.distance_close = fuzz.trapmf(self.distance_universe, [0.0, 0.0, 3.0, 6.0])
        self.distance_moderate = fuzz.trimf(self.distance_universe, [3.0, 6.0, 9.0])
        self.distance_far = fuzz.trimf(self.distance_universe, [6.0, 9.0, 12.0])
        self.distance_very_far = fuzz.trapmf(self.distance_universe, [9.0, 12.0, 5000.0, 5000.0])

        # cross-track error membership function
        self.ctr_large_left = fuzz.trapmf(self.cross_track_universe, [-180, -180, -1.2, -0.6])
        self.ctr_small_left = fuzz.trimf(self.cross_track_universe, [-1.2, -0.6, 0])
        self.ctr_center = fuzz.trimf(self.cross_track_universe, [-1.0, 0.0, 1.0])
        self.ctr_small_right = fuzz.trimf(self.cross_track_universe, [0.0, 0.6, 1.2])
        self.ctr_large_right = fuzz.trapmf(self.cross_track_universe, [0.6, 1.2, 180.0, 180.0])

        # Membership functions for wave conditions
        self.wave_condition_going_down = fuzz.trimf(self.wave_condition_universe, [-1, -1, -0.2])
        self.wave_condition_stable = fuzz.trimf(self.wave_condition_universe, [-0.3, 0, 0.3])
        self.wave_condition_going_up = fuzz.trimf(self.wave_condition_universe, [0.2, 1, 1])

        # Defining membership functions for fuzzy simulator
        distance_to_wp = self.fuzzy_variables['distance_to_wp']
        cross_track_error = self.fuzzy_variables['cross_track_error']
        wave_condition = self.fuzzy_variables['wave_condition']
        speed = self.fuzzy_variables['speed']
        yaw_rate = self.fuzzy_variables['yaw_rate']

        # Distance membership functions
        distance_to_wp['close'] = fuzz.trapmf(distance_to_wp.universe, [0.0, 0.0, 3.0, 6.0])
        distance_to_wp['medium'] = fuzz.trimf(distance_to_wp.universe, [3.0, 6.0, 9.0])
        distance_to_wp['far'] = fuzz.trimf(distance_to_wp.universe, [6.0, 9.0, 12.0])
        distance_to_wp['very_far'] = fuzz.trapmf(distance_to_wp.universe, [9.0, 12.0, 5000.0, 5000.0])
        
        # Cross-track error membership functions
        cross_track_error['large_left'] = fuzz.trapmf(cross_track_error.universe, [-180, -180, -1.2, -0.6])
        cross_track_error['left'] = fuzz.trimf(cross_track_error.universe, [-1.2, -0.6, 0])
        cross_track_error['center'] = fuzz.trimf(cross_track_error.universe, [-1.0, 0.0, 1.0])
        cross_track_error['right'] = fuzz.trimf(cross_track_error.universe, [0.0, 0.6, 1.2])
        cross_track_error['large_right'] = fuzz.trapmf(cross_track_error.universe, [0.6, 1.2, 180.0, 180.0])

        # Membership functions for wave conditions
        wave_condition['going_down'] = fuzz.trimf(wave_condition.universe, [-1, -1, -0.2])
        wave_condition['stable'] = fuzz.trimf(wave_condition.universe, [-0.3, 0, 0.3])
        wave_condition['going_up'] = fuzz.trimf(wave_condition.universe, [0.2, 1, 1])

        # Membership function for forward velocity
        speed['slow'] = fuzz.trapmf(speed.universe, [0.0, 0.0, 0.5, 1.0])
        speed['moderate'] = fuzz.trimf(speed.universe, [0.5, 1.0, 1.5])
        speed['fast'] = fuzz.trimf(speed.universe, [1.0, 1.5, 2.0])
        speed['very_fast'] = fuzz.trapmf(speed.universe, [1.5, 2.0, 2.5, 2.5])
        
        # Membership function for yaw rate
        yaw_rate['hard_left'] = fuzz.trapmf(yaw_rate.universe, [-1.0, -1.0, -0.5, -0.3])
        yaw_rate['left'] = fuzz.trimf(yaw_rate.universe, [-0.5, -0.3, -0.1])
        yaw_rate['straight'] = fuzz.trimf(yaw_rate.universe, [-0.2, 0.0, 0.2])
        yaw_rate['right'] = fuzz.trimf(yaw_rate.universe, [0.1, 0.3, 0.5])
        yaw_rate['hard_right'] = fuzz.trapmf(yaw_rate.universe, [0.3, 0.5, 1.0, 1.0])

    def fuzzify_inputs(self):
        """convert crisp input to fuzzy membership degrees"""

        # fuzzify distance to wayypoint
        distance_close_degree = fuzz.interp_membership(self.distance_universe, self.distance_close, self.distance_to_target_wp)
        distance_medium_degree = fuzz.interp_membership(self.distance_universe, self.distance_moderate, self.distance_to_target_wp)
        distance_far_degree = fuzz.interp_membership(self.distance_universe, self.distance_far, self.distance_to_target_wp)
        distance_very_far_degree = fuzz.interp_membership(self.distance_universe, self.distance_very_far, self.distance_to_target_wp)

        # fuzzify cross-track error
        cte_large_left_degree = fuzz.interp_membership(self.cross_track_universe, self.ctr_large_left, self.cross_track_error )
        cte_small_left_degree = fuzz.interp_membership(self.cross_track_universe, self.ctr_small_left, self.cross_track_error )
        cte_center_degree = fuzz.interp_membership(self.cross_track_universe, self.ctr_center, self.cross_track_error )
        cte_small_right_degree = fuzz.interp_membership(self.cross_track_universe, self.ctr_small_right, self.cross_track_error) 
        cte_large_right_degree = fuzz.interp_membership(self.cross_track_universe, self.ctr_large_right, self.cross_track_error )

        # fuzzify wave condition
        wv_going_up = fuzz.interp_membership(self.wave_condition_universe, self.wave_condition_going_up, self.wave_phase)
        wv_stable = fuzz.interp_membership(self.wave_condition_universe, self.wave_condition_stable, self.wave_phase)
        wv_going_down = fuzz.interp_membership(self.wave_condition_universe, self.wave_condition_going_down, self.wave_phase)

        # return all fuzzy membership degrees as features for the neural network
        return np.array([
            distance_close_degree, distance_medium_degree, distance_far_degree, distance_very_far_degree, cte_large_left_degree,
            cte_small_left_degree, cte_center_degree, cte_small_right_degree, cte_large_right_degree, wv_going_up, wv_stable,
            wv_going_down
        ]).reshape(1, -1)

    def create_neuro_fuzzy_model(self):
        """creating a neuro-fuzzy model using PyTorch"""

        # create PyTorch model
        self.model = NeuroFuzzyModel().to(self.device)

        # create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # define loss function
        self.criterion = nn.MSELoss()

        # create loss history trackers
        self.left_speed_loss_history = []
        self.right_speed_lost_history = []

        self.get_logger().info("Neuro-fuzzy model created and explicit backpropagation training.")

    def collect_training_data_from_fuzzy(self):
        """collecting training data using the original fuzzy rules"""

        # get fuzzified inputs
        fuzzified_inputs = self.fuzzify_inputs().flatten()

        distance = self.fuzzy_variables['distance_to_wp']
        cte = self.fuzzy_variables['cross_track_error']
        wave_condition = self.fuzzy_variables['wave_condition']
        speed = self.fuzzy_variables['speed']
        yaw_rate = self.fuzzy_variables['yaw_rate']

        # Creating rules
        rules = [
            # Center heading rules and straight line navigation
            ctrl.Rule(cte['center'] & distance['very_far'] & wave_condition['stable'], speed['very_fast']),
            ctrl.Rule(cte['center'] & distance['very_far'] & wave_condition['stable'], yaw_rate['straight']),

            # Distance-based speed adjustments
            ctrl.Rule(cte['center'] & distance['far'] & wave_condition['stable'], speed['fast']),
            ctrl.Rule(cte['center'] & distance['far'] & wave_condition['stable'], yaw_rate['straight']),

            ctrl.Rule(cte['center'] & distance['medium'] & wave_condition['stable'], speed['moderate']),
            ctrl.Rule(cte['center'] & distance['medium'] & wave_condition['stable'], yaw_rate['straight']),

            ctrl.Rule(cte['center'] & distance['close'] & wave_condition['stable'], speed['slow']),
            ctrl.Rule(cte['center'] & distance['close'] & wave_condition['stable'], yaw_rate['straight']),

            # Wave compensation - going up
            ctrl.Rule(cte['center'] & distance['very_far'] & wave_condition['going_up'], speed['very_fast']),
            ctrl.Rule(cte['center'] & distance['very_far'] & wave_condition['going_up'], yaw_rate['straight']),

            ctrl.Rule(cte['center'] & distance['far'] & wave_condition['going_up'], speed['very_fast']),
            ctrl.Rule(cte['center'] & distance['far'] & wave_condition['going_up'], yaw_rate['straight']),

            ctrl.Rule(cte['center'] & distance['medium'] & wave_condition['going_up'], speed['fast']),
            ctrl.Rule(cte['center'] & distance['medium'] & wave_condition['going_up'], yaw_rate['straight']),

            ctrl.Rule(cte['center'] & distance['close'] & wave_condition['going_up'], speed['moderate']),
            ctrl.Rule(cte['center'] & distance['close'] & wave_condition['going_up'], yaw_rate['straight']),

            # Wave compensation - going down
            ctrl.Rule(cte['center'] & distance['very_far'] & wave_condition['going_down'], speed['moderate']),
            ctrl.Rule(cte['center'] & distance['very_far'] & wave_condition['going_down'], yaw_rate['straight']),

            ctrl.Rule(cte['center'] & distance['far'] & wave_condition['going_down'], speed['moderate']),
            ctrl.Rule(cte['center'] & distance['far'] & wave_condition['going_down'], yaw_rate['straight']),

            ctrl.Rule(cte['center'] & distance['medium'] & wave_condition['going_down'], speed['slow']),
            ctrl.Rule(cte['center'] & distance['medium'] & wave_condition['going_down'], yaw_rate['straight']),

            ctrl.Rule(cte['center'] & distance['close'] & wave_condition['going_down'], speed['slow']),
            ctrl.Rule(cte['center'] & distance['close'] & wave_condition['going_down'], yaw_rate['straight']),

            # Heading correction rules
                # Right turn corrections (reduce right motor speed)
            ctrl.Rule(cte['large_left'], speed['slow']),
            ctrl.Rule(cte['large_left'], yaw_rate['hard_right']),

            ctrl.Rule(cte['left'], speed['moderate']),
            ctrl.Rule(cte['left'], yaw_rate['right']),

                # Right turn corrections (reduce right motor speed)
            ctrl.Rule(cte['right'], speed['moderate']),
            ctrl.Rule(cte['right'], yaw_rate['left']),

            ctrl.Rule(cte['large_right'], speed['slow']),
            ctrl.Rule(cte['large_right'], yaw_rate['hard_left'])
            ]
        
        # Create control system
        system = ctrl.ControlSystem(rules)
        self.fuzzy_simulator = ctrl.ControlSystemSimulation(system)

        # Convert speed and heading to left and right motor commands
        speed, yaw_rate = self.calculate_differential_speeds()

        # store data for training if vehicle is armed
        if self.vehicle_state.armed:
            with self.lock:     # thread-safe access to training data
                self.training_data['fuzzified_inputs'].append(fuzzified_inputs)
                self.training_data['speed'].append(speed)
                self.training_data['yaw_rate'].append(yaw_rate)

        return speed, yaw_rate
    
    def calculate_differential_speeds(self):
        """Calculating differential steering motor commands."""
        # Extract the global coordinates
        target_lat, target_lon, _ = self.waypoints[self.current_wp_index]['global_coordinates']
        # prev_lat, prev_lon, _ = self.waypoints[self.current_wp_index - 1]['global_coordinates']
        prev_lat, prev_lon = self.previous_waypoint

        self.distance_to_target_wp = self.distance_calculation(target_lat, target_lon)
        self.distance_from_prev_wp = self.distance_calculation(prev_lat, prev_lon)
        self.distance_prev_target_wp = self.waypoint_path(prev_lat, prev_lon, target_lat, target_lon)

        #Calculate heading parameters and the cross-track error
        self.compute_heading_parameters()
        self.cross_track_error = self.compute_cross_track_error(prev_lat, prev_lon, target_lat, target_lon)

        wave_condition = np.clip(self.wave_phase, -1, 1)

        # Fuzzy input
        self.fuzzy_simulator.input['cross_track_error'] = self.cross_track_error
        self.fuzzy_simulator.input['distance_to_wp'] = self.distance_to_target_wp
        self.fuzzy_simulator.input['wave_condition'] = wave_condition
        self.fuzzy_simulator.compute()

        speed = self.fuzzy_simulator.output['speed']
        yaw_rate = self.fuzzy_simulator.output['yaw_rate']

        return speed, yaw_rate
    
    def compute_heading_parameters(self):
        """Compute heading error in degrees using cosine rule, Desired path, actual path and waypoint path"""
        if len(self.waypoints) < 2:
            return
        
        if self.distance_from_prev_wp == 0.0 or self.distance_prev_target_wp == 0.0:
            return
        
        product = (self.distance_from_prev_wp**2 + self.distance_prev_target_wp**2 - self.distance_to_target_wp**2) / (2*self.distance_from_prev_wp*self.distance_prev_target_wp)
        product = max(-1.0, min(1.0, product))
        self.theta_actual_path = math.acos(product)
    
    def compute_cross_track_error(self, prev_lat, prev_lon, target_lat, target_lon):
        """ Compute cross-track error from the planned path line to the current position using sin rule."""
        x, y = math.radians(self.global_position.latitude), math.radians(self.global_position.longitude)

        # Determining sign (positive if vehicle is to the right of the path)
        dx, dy = math.radians(target_lat - prev_lat), math.radians(target_lon - prev_lon)

        # Calculating the sign based on which side of the line the vehicle is on
        cross_product = dx * (y - math.radians(prev_lon)) - dy * (x - math.radians(prev_lat))
        sign = 1 if cross_product > 0 else -1
        
        cte = sign * self.distance_from_prev_wp*math.sin(self.theta_actual_path)
        
        return cte
    
    def custom_backpropagation_step(self, inputs, speed, yaw_rate):
        """perform a single backpropagation step with the given inputs and targets"""

        # convert inputs to PyTorch tensors
        inputs = torch.FloatTensor(inputs).to(self.device)
        speed = torch.FloatTensor(speed).to(self.device)
        yaw_rate = torch.FloatTensor(yaw_rate).to(self.device)

        # zero the gradients
        self.optimizer.zero_grad()

        # forward pass
        pred_speed, pred_yaw_rate = self.model(inputs)

        # calculate losses
        speed_loss = self.criterion(pred_speed, speed)
        yaw_rate_loss = self.criterion(pred_yaw_rate, yaw_rate)
        combined_loss = speed_loss + yaw_rate_loss

        # backward pass and optimize
        combined_loss.backward()
        self.optimizer.step()

        return combined_loss.item(), speed_loss.item(), yaw_rate_loss.item()
    
    def train_neuro_fuzzy_model(self):
        """training the neuro-fuzzy model using collected data with backpropagation"""

        # skip training if vehicle is not armed or has not reached takeoff
        if not self.vehicle_state.armed and not self.current_wp_index >= 2:
            self.get_logger().info('Skipping training - vehicle is not armed')
            return False

        with self.lock:     # thread-safe access to training data
            if len(self.training_data['speed']) < 100:
                self.get_logger().info(f"Not enough training data yet (need at least 100 samples).")
                return False

        # create local copies to train on
        X = np.array(self.training_data['fuzzified_inputs'])
        speed = np.array(self.training_data['speed']).reshape(-1,1)
        yaw_rate = np.array(self.training_data['yaw_rate']).reshape(-1,1)

        # convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        speed_tensor = torch.FloatTensor(speed)
        yaw_rate_tensor = torch.FloatTensor(yaw_rate)

        # create tensorDataset
        dataset = TensorDataset(X_tensor, speed_tensor, yaw_rate_tensor)

        # split dataset into training and validation
        train_size = int((1 - self.validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=val_size)

        # initialize lists to track loss
        train_losses = []
        val_losses = []

        # initialize best validation loss for early stopping
        best_val_loss = float('inf')
        patience = 10        # lower patience for faster adaptation (frequent small trainings)
        patience_counter = 0

        self.get_logger().info(f"Starting neural network training with {train_size} training samples and {val_size} validation samples")
        self.training_iteration += 1

        # training loop with epochs
        for epoch in range(self.epochs):
            # check if vehicle  is still armed and has reached takeoff before each epoch
            if not self.vehicle_state.armed and not self.current_wp_index >= 2:
                self.get_logger().info("Vehicle disarmed, - pausing training process")
                return False
            
            start_time = time.time()

            # set model to training mode
            self.model.train()

            # initialize epoch loss
            epoch_loss = 0.0
            num_batches = 0

            # loop over batches
            for batch_X, batch_speed, batch_yaw_rate in train_loader:
                # move tensors to the configured device
                batch_X = batch_X.to(self.device)
                batch_speed = batch_speed.to(self.device)
                batch_yaw_rate = batch_yaw_rate.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward pass
                pred_speed, pred_yaw_rate = self.model(batch_X)

                # calculate losses
                speed_loss = self.criterion(pred_speed, batch_speed)
                yaw_rate_loss = self.criterion(pred_yaw_rate, batch_yaw_rate)
                combined_loss = speed_loss + yaw_rate_loss

                # backward pass and optimize
                combined_loss.backward()
                self.optimizer.step()

                epoch_loss += combined_loss.item()
                num_batches += 1  

                # checking if training should stop
                if not self.vehicle_state.armed:
                    self.get_logger().debug("Vehicle disarmed during training batch - interrupting training process")
                    return False

            # calculate average training loss for the epoch
            avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0
            train_losses.append(avg_train_loss)

            # validation
            self.model.eval()
            with torch.no_grad():
                # get validation data
                val_X, val_speed, val_yaw_rate = next(iter(val_loader))
                val_X = val_X.to(self.device)
                val_speed = val_speed.to(self.device)
                val_yaw_rate = val_yaw_rate.to(self.device)

                # forward pass
                val_pred_speed, val_pred_yaw_rate = self.model(val_X)

                # calculate losses
                val_speed_loss = self.criterion(val_pred_speed, val_speed).item()
                val_yaw_rate_loss = self.criterion(val_pred_yaw_rate, val_yaw_rate).item()
                val_loss = val_speed_loss + val_yaw_rate_loss

                val_losses.append(val_loss)

            # log epoch results
            epoch_time = time.time() - start_time
            self.get_logger().info(f"Epoch {epoch+1}/{self.epochs} - "
                            f"Train Loss: {avg_train_loss:.4f}, "
                            f"Val loss: {val_loss:.4f}, "
                            f"Speed Loss: {val_speed_loss:.4f}, "
                            f"Heading Loss: {val_yaw_rate_loss:.4f}, "
                            f"Time: {epoch_time:.2f}s")

            # early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # save best model weights
                torch.save(self.model.state_dict(), "/tmp/neuro_fuzzy_best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.get_logger().info(f"Early stopping at epoch {epoch+1}")
                    break

        # load best model weights
        try:
            self.model.load_state_dict(torch.load('/tmp/neuro_fuzzy_best_model.pt', weights_only=True))
            self.get_logger().info("Loaded best model weights.")
        except Exception as e:
            self.get_logger().warning(f"Could not load best model weights: {e}.")

        # store training history
        self.training_losses = train_losses
        self.validation_losses = val_losses

        self.get_logger().info(f"Model training complete with final val loss: {best_val_loss:.4f}, confidence: {self.model_confidence:.2f}")
        return True 
    
    def compute_navigation_parameters(self):
        """calculating mission speed and heading correction using neuro-fuzzy model."""
        # increment data collection counter
        self.data_collect_counter += 1

        self.max_samples = 2000

        # collect training data for the first 2000 iterattions
        if self.vehicle_state.mode in ['AUTO.MISSION', "GUIDED", "AUTO"] and self.vehicle_state.armed and self.data_collect_counter <= self.max_samples:
            # get fuzzy logic predictions first (always)
            speed, yaw_rate = self.collect_training_data_from_fuzzy()
            
            # log data collection progress
            if self.data_collect_counter % 100 == 0 and self.vehicle_state.armed:
                self.get_logger().info(f"Collecting training data: {self.data_collect_counter}/{self.max_samples} samples")

        # if model is trained, use a blend of fuzzy and neural predictions based on confidence
        elif self.model_trained:
            # get neural network predictions
            fuzzified_inputs = self.fuzzify_inputs()

            # get predictions from model
            self.model.eval()
            while torch.no_grad():
                fuzzified_tensor = torch.FloatTensor(fuzzified_inputs).to(self.device)
                # Model now predicts left and right motor speeds directs
                pred_speed, pred_yaw_rate = self.model(fuzzified_tensor)
                speed = float(pred_speed.detach().cpu().numpy()[0][0])
                yaw_rate = float(pred_yaw_rate.detach().cpu().numpy()[0][0])

            self.get_logger(f"Neuro-fuzzy preediction: Left/Right: ({speed:.2f}/{yaw_rate:.2f})")
            
            # collect data for online learning based on model performance - only if armed and at takeoff
            if self.vehicle_state.armed and self.current_wp_index >= 2:
                self.add_online_learning_sample(speed, yaw_rate)

        else:
            # use default fuzzy logic without collecting data
            speed, yaw_rate = self.collect_training_data_from_fuzzy()

        self.data_collect_counter = 0

        # ensure values are within valid ranges
        # max_speed = 2.5
        # left_motor_speed = max(0.0, min(max_speed, left_motor_speed))
        # right_motor_speed = max(0.0, min(max_speed, right_motor_speed))

        self.publish_motor_commands(speed, yaw_rate)       

        self.get_logger().debug(f"Navigation - Dist: {self.distance_to_target_wp:.2f}m, CTE: {self.cross_track_error:.2f}m, "
        f"Left/Right: {speed:.2f} / {yaw_rate:.2f} m/s")

    def publish_motor_commands(self, speed, yaw_rate):
        """Publish direct motor commands using MAVROS ActuatorControl."""
        # Import available MAVROS message types
        self.call_mavros_command_service(178, [0.0, speed, -1.0, 0.0, 0.0, 0.0, 0.0])  # Speed
        self.call_mavros_command_service(115, [yaw_rate, 20.0, 1.0, 0.0, 0.0, 0.0, 0.0])  # Heading

        self.get_logger().info(
            f"Distance from WP: {self.distance_from_prev_wp:.2f} | " + 
            f"Distance to WP: {self.distance_to_target_wp:.2f}m | " + 
            f"Waypoints distance: {self.distance_prev_target_wp:.2f} | " +
            f"Heading Error: {self.theta_actual_path:.2f}° | " + 
            f"CTE: {self.cross_track_error:.2f}m | " + 
            f"Speed/Yaw_rate: {speed:.2f}/{yaw_rate:.2f}")

    def call_mavros_command_service(self, command_id, params):
        """Call MAVROS command service."""
        from mavros_msgs.srv import CommandLong
        
        if not hasattr(self, 'command_client'):
            self.command_client = self.create_client(CommandLong, '/mavros/cmd/command')
        
        if not self.command_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('MAVROS command service not available')
            return
        
        request = CommandLong.Request()
        request.broadcast = False
        request.command = command_id
        request.confirmation = 0
        request.param1 = params[0] if len(params) > 0 else 0.0
        request.param2 = params[1] if len(params) > 1 else 0.0
        request.param3 = params[2] if len(params) > 2 else 0.0
        request.param4 = params[3] if len(params) > 3 else 0.0
        request.param5 = params[4] if len(params) > 4 else 0.0
        request.param6 = params[5] if len(params) > 5 else 0.0
        request.param7 = params[6] if len(params) > 6 else 0.0
        
        future = self.command_client.call_async(request)
        return future

    def add_waypoint(self, lat, lon, altitude):
        """Add a waypoint to the mission."""
        if not self.waypoint_push_service.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Waypoint push service not available')
            return False
        
        # Create a new waypoint
        wp = Waypoint()
        wp.frame = Waypoint.FRAME_GLOBAL_REL_ALT
        wp.command = 16  # MAV_CMD_NAV_WAYPOINT
        wp.is_current = False
        wp.autocontinue = True
        wp.param1 = 0.0  # Hold time
        wp.param2 = 2.0  # Acceptance radius
        wp.param3 = 0.0  # Pass through
        wp.param4 = float('nan')  # Yaw
        wp.x_lat = lat
        wp.y_long = lon
        wp.z_alt = altitude
        
        # Add to local waypoints list
        self.waypoints.append((lat, lon))
        
        # Create waypoint list for service call
        waypoints = [wp]
        
        # Send waypoint to vehicle
        request = WaypointPush.Request()
        request.start_index = 0
        request.waypoints = waypoints
        
        future = self.waypoint_push_service.call_async(request)
        future.add_done_callback(self.waypoint_push_callback)
        
        return True

    def waypoint_push_callback(self, future):
        """Callback for waypoint push service."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('Waypoint successfully uploaded')
            else:
                self.get_logger().error('Failed to upload waypoint')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    def set_mode(self, mode):
        """Set the flight mode of the vehicle."""
        if not self.set_mode_service.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Set mode service not available')
            return False
        
        request = SetMode.Request()
        request.custom_mode = mode
        
        future = self.set_mode_service.call_async(request)
        future.add_done_callback(self.set_mode_callback)
        
        return True

    def set_mode_callback(self, future):
        """Callback for set mode service."""
        try:
            response = future.result()
            if response.mode_sent:
                self.get_logger().info('Mode change request sent')
            else:
                self.get_logger().error('Failed to send mode change request')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    def arm(self, arm_state):
        """Arm or disarm the vehicle."""
        if not self.arm_service.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Arming service not available')
            return False
        
        request = CommandBool.Request()
        request.value = arm_state
        
        future = self.arm_service.call_async(request)
        future.add_done_callback(self.arm_callback)
        
        return True

    def arm_callback(self, future):
        """Callback for arm service."""
        try:
            response = future.result()
            if response.success:
                state = "armed" if future.request.value else "disarmed"
                self.get_logger().info(f'Vehicle successfully {state}')
            else:
                self.get_logger().error('Failed to change arm state')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    def navigation_timer_callback(self):
        """Timer triggered navigation computation."""
        # Check if we have a mission
        if not self.waypoints:
            self.get_logger().info("No waypoints available")
            return    
        
        # check if the mission is complete
        if self.current_wp_index >= len(self.waypoints):
            self.get_logger().info("Mission complete - all waypoints reached")
            return 
        
        # Execute navigation if we're in the correct mode and armed
        if self.vehicle_state.armed and self.vehicle_state.mode in ["AUTO.MISSION", "GUIDED", "AUTO"]:
            try:
                self.compute_navigation_parameters()
            except Exception as e:
                self.get_logger().error(f"Navigation computation failed: {e}") 
        else:
            self.get_logger().debug(f"Vehicle not ready - Mode: {self.vehicle_state.mode}, Armed: {self.vehicle_state.armed}")

    def training_timer_callback(self):
        """Periodic training callback that runs continuously throughout the mission"""

        # Only attempt training when armed AND has reached takeoff
        if not self.vehicle_state.armed:
            self.get_logger().debug("Training timer: vehicle not armed or not at takeoff")
            return

        # always attempt training when we have enough data
        if not self.model_trained and len(self.training_data['speed']) >= 100:
            self.get_logger().info("Starting initial model training...")
            self.model_trained = self.train_neuro_fuzzy_model()

        elif self.model_trained and len(self.training_data['speed']) % 100 == 0 and len(self.training_data['left_motor_speed']) > 0:
            self.get_logger().info(f"Retraining model with {len(self.training_data['speed'])} samples...")
            self.train_neuro_fuzzy_model()

    def add_online_learning_sample(self, left_motor_speed, right_motor_speed):
        """Add a sample to the training data for future online learning update"""
        if self.vehicle_state.armed:
            fuzzified_input = self.fuzzify_inputs().flatten()

            # add to training data
            with self.lock:     # thread-safe access to training data
                self.training_data['fuzzified_inputs'].append(fuzzified_input)
                self.training_data['left_motor_speed'].append(left_motor_speed)
                self.training_data['right_motor_speed'].append(right_motor_speed)

            # limit training data size to prevent memory issues
            max_samples = 1000      # bigger for more diverse training data
            if len(self.training_data['left_motor_speed']) > max_samples:
                self.training_data['fuzzified_inputs'] = self.training_data['fuzzified_inputs'][-max_samples:]
                self.training_data['left_motor_speed'] = self.training_data['left_motor_speed'][-max_samples:]
                self.training_data['right_motor_speed'] = self.training_data['right_motor_speed'][-max_samples:]

    def online_learning_update(self, actual_left_speed, actual_right_speed, expected_left_speed, expected_right_speed):
        """updating the model with online learning from performance feedback"""
        # Always allow online learning updates when the model is trained AND vehicle is armed AND has reached takeoff
        if not self.model_trained or not self.vehicle_state.armed:
            return

        # calculate errors
        left_speed_error = actual_left_speed - expected_left_speed
        right_speed_error = actual_right_speed - expected_right_speed

        # Only update if error is significant - reduced threshold for more frequent updates
        if abs(left_speed_error) > 0.2 or abs(right_speed_error) > 0.2:        
            # fuzzify the inputs
            fuzzified_inputs = self.fuzzify_inputs()

            # convert too tensors
            inputs_tensor = torch.FloatTensor(fuzzified_inputs).to(self.device)
            actual_left_speed_tensor = torch.FloatTensor([actual_left_speed]).to(self.device)
            actual_right_speed_tensor = torch.FloatTensor([actual_right_speed]).to(self.device)

            # set model to training mode
            self.model.train()
            
            # zero the gradient
            self.optimizer.zero_grad()

            # forward pass
            pred_left, pred_right= self.model(inputs_tensor)

            # calculate losses
            left_speed_loss = self.criterion(pred_left, actual_left_speed_tensor)
            right_speed_loss = self.criterion(pred_right, actual_right_speed_tensor)
            combined_loss = left_speed_loss + right_speed_loss

            # backward pass and optimize
            combined_loss.backward()
            self.optimizer.step()

            self.get_logger().debug(f"Online update - left speed error: {left_speed_error:.4f}, "
                            f"Right speed error: {right_speed_error:.4f}, "
                            f"Update loss: {combined_loss.item():.4f}")


def main(args=None):
    """main function to start the node."""
    rclpy.init(args=args)
    node = NeuroFuzzyNavigation()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Neuro-Fuzzy Navigation node terminated.")
    except Exception as e:
        print(f"Error: {e}")