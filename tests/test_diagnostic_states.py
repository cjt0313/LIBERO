import numpy as np

from libero.libero.envs.diagnostic_states import (
    DiagnosticStateTracker,
    _drawer_state,
    _orientation_angle,
    _site_centered,
)


def test_orientation_ignores_yaw_but_detects_tipping():
    identity = [1, 0, 0, 0]
    yaw_half_turn = [0, 0, 0, 1]
    tipped = [np.sqrt(0.5), np.sqrt(0.5), 0, 0]
    assert np.isclose(_orientation_angle(identity, yaw_half_turn), 0)
    assert np.isclose(_orientation_angle(identity, tipped), 90)


def test_center_uses_site_local_frame():
    class Site:
        size = np.array([0.1, 0.2, 0.1])

    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert _site_centered(Site(), np.zeros(3), rotation, [-0.05, 0.02, 0], 0.5)
    assert not _site_centered(Site(), np.zeros(3), rotation, [-0.12, 0.02, 0], 0.5)


def test_drawer_bins_use_direction_of_joint_travel():
    assert _drawer_state(0, [-0.16, -0.14], [0, 0.005], True, 0.05, 0.9) == "closed"
    assert _drawer_state(0, [-0.16, -0.14], [0, 0.005], False, 0.05, 0.9) == "closed"
    assert _drawer_state(-0.08, [-0.16, -0.14], [0, 0.005], False, 0.05, 0.9) == "partially_open"
    assert _drawer_state(-0.15, [-0.16, -0.14], [0, 0.005], False, 0.05, 0.9) == "fully_open"
    assert _drawer_state(0.15, [0.10, 0.16], [-0.005, 0], False, 0.05, 0.9) == "fully_open"


def test_episode_reference_distinguishes_fall_from_held_object():
    class State:
        position = np.array([0.0, 0.0, 0.1])
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])

        def get_geom_state(self):
            return {"pos": self.position, "quat": self.quaternion}

    class Robot:
        gripper = object()

    class Env:
        objects_dict = {"bowl": object()}
        fixtures_dict = {"table": object()}
        object_states_dict = {"bowl": State()}
        robots = [Robot()]
        grasped = False

        def get_object(self, name):
            return self.objects_dict[name]

        def _check_grasp(self, gripper, obj):
            return self.grasped

        def check_contact(self, first, second):
            return True

    env = Env()
    tracker = DiagnosticStateTracker(env)
    tracker.reset()
    assert ("initial_orientation", "bowl") in tracker.sample([])

    env.object_states_dict["bowl"].position = np.array([0.0, 0.0, 0.06])
    env.object_states_dict["bowl"].quaternion = np.array([np.sqrt(0.5), np.sqrt(0.5), 0, 0])
    assert ("fallen", "bowl") in tracker.sample([])

    env.grasped = True
    tracker.sample([])
    labels = tracker.sample([])
    assert ("holding", "bowl") in labels
    assert ("fallen", "bowl") not in labels
