import numpy as np
import mujoco

from discoverse.examples.mmk2.communication.mujoco_mmk2_receiver_api import MMK2_Receiver
from discoverse.examples.mmk2.communication.mujoco_mmk2_controller_api import MMK2_Controller


POSITIVE_DERECTION = [0.707, 0,  0, -0.707]
INPUT_KEY = ["x", "y", "theta"]


class MoveToPoint:
    def __init__(self, source: dict, target: dict, receiver: MMK2_Receiver, controller: MMK2_Controller):
        assert all([key in source for key in INPUT_KEY]
                   ), f"source should contain all keys in {INPUT_KEY}"
        self.soucre = source
        self.target = target
        self.receiver = receiver
        self.controller = controller

        self.x_1 = source["x"]
        self.x_2 = target["x"]
        self.y_1 = source["y"]
        self.y_2 = target["y"]
        self.theta_1 = source["theta"]
        self.theta_2 = target["theta"]

        self.turn_angle_1 = None

    def _turn_1(self):
        turn_angle_1 = 90 - self.theta_1 + \
            np.rad2deg(np.arctan((self.y_1 - self.y_2) / (self.x_1 - self.x_2)))
        self.turn_angle_1 = turn_angle_1
        self.controller.set_turn_left_angle(turn_angle_1)

    def _move(self):
        move_distance = np.sqrt((self.y_1 - self.y_2) **
                                2 + (self.x_1 - self.x_2)**2)
        self.controller.set_move_forward_pos(move_distance)

    def _turn_2(self):
        assert self.turn_angle_1 is not None, "_turn_1 should be called first"
        turn_angle_2 = self.theta_2 - self.theta_1 - self.turn_angle_1
        self.controller.set_turn_left_angle(turn_angle_2)

    def move_to_point(self):
        self._turn_1()
        self._move()
        self._turn_2()
        self.controller.stop_all()


def main():
    model = mujoco.MjModel.from_xml_path(
        "models/mjcf/s2r2025_env.xml")
    data = mujoco.MjData(model)

    receiver = MMK2_Receiver(model, data)
    controller = MMK2_Controller(model, data, receiver)
    move_to_point = MoveToPoint({"x": 0.0, "y": 0.0, "theta": 0.0}, {
                                "x": 0.4, "y": 0.4, "theta": 90.0}, receiver, controller)
    move_to_point.move_to_point()


if __name__ == "__main__":
    main()
