import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, output_max=1e6, integrator_max=1e6):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_max = output_max
        self.error_sum = 0
        self.last_error = 0
        self.integrator_max = integrator_max

    def output(self, error, dt=1.0):
        self.error_sum += error * dt
        self.error_sum = np.clip(self.error_sum, -self.integrator_max, self.integrator_max)
        error_diff = (error - self.last_error) / dt
        out = self.kp * error + self.ki * self.error_sum + self.kd * error_diff
        self.last_error = error
        return np.clip(out, -self.output_max, self.output_max)

    def output_d(self, error, d_error, dt=1.0):
        self.error_sum += error * dt
        self.error_sum = np.clip(self.error_sum, -self.integrator_max, self.integrator_max)
        out = self.kp * error + self.ki * self.error_sum + self.kd * d_error
        return np.clip(out, -self.output_max, self.output_max)
