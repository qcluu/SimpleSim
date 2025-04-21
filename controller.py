import numpy as np
import time
 
class Control():
    def send_to_XY(self, target_x, target_y, left_motor=0, right_motor=9):
        Kp_dist = 6
        Ki_dist = 0.01
        Kd_dist = 0.00  # 0.1
        Kp_theta = 10
        Ki_theta = 0.01
        Kd_theta = 0.000  # 0.1

        # Compute distance and angle to target
        dx = target_x - self.x
        dy = target_y - self.y
        distance_error = np.sqrt(dx2 + dy2)
        target_angle = np.degrees(np.arctan2(dy, dx))
        angle_error = target_angle - self.angle
        angle_error = (angle_error + 180) % 360 - 180  # Normalize to [-180, 180]
 
        # PID calculations for distance
        self.dist_integral += distance_error
        dist_derivative = distance_error - self.prev_dist_error
        self.prev_dist_error = distance_error
 
        P_dist = Kp_dist * distance_error
        I_dist = Ki_dist * self.dist_integral
        D_dist = Kd_dist * dist_derivative
 
        # PID calculations for angle
        self.theta_integral += angle_error
        theta_derivative = angle_error - self.prev_theta_error
        self.prev_theta_error = angle_error
 
        P_theta = Kp_theta * angle_error
        I_theta = Ki_theta * self.theta_integral
        D_theta = Kd_theta * theta_derivative
 
        # Compute motor outputs
        left_motor_speed = P_dist + I_dist + D_dist + P_theta + I_theta + D_theta
        right_motor_speed = P_dist + I_dist + D_dist - (P_theta + I_theta + D_theta)

        # Clip motor speeds to acceptable range
        left_motor_speed = np.clip(left_motor_speed, -127, 127)
        right_motor_speed = -np.clip(right_motor_speed, -127, 127)

        # Set motor speeds
        motor_values = self.robot.motor
        motor_values[left_motor] = int(left_motor_speed)
        motor_values[right_motor] = int(right_motor_speed)

        # Simulate sending motor values to robot (uncomment when integrating with real robot)
        self.robot.motors(motor_values)

        # Sleep for a short duration to simulate control loop timing
        time.sleep(0.05)  # Adjust the sleep time as needed

        self.stop_drive()
    
    