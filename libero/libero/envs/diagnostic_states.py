"""Simulator-grounded diagnostic predicates for LIBERO rollouts.

These labels supplement BDDL goal predicates; they do not change task success.
Call reset() after the environment has settled, then sample() after each action.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiagnosticThresholds:
    initial_orientation_degrees: float = 20.0
    fallen_orientation_degrees: float = 45.0
    fallen_height_drop: float = 0.02
    center_fraction: float = 0.5
    held_contact_steps: int = 2
    closed_fraction: float = 0.05
    fully_open_fraction: float = 0.9


def _up_axis(quaternion):
    """World direction of local +Z for a MuJoCo (w, x, y, z) quaternion."""
    w, x, y, z = np.asarray(quaternion, dtype=float) / np.linalg.norm(quaternion)
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def _orientation_angle(initial_quaternion, current_quaternion):
    cosine = np.clip(np.dot(_up_axis(initial_quaternion), _up_axis(current_quaternion)), -1, 1)
    return float(np.degrees(np.arccos(cosine)))


def _site_centered(site, site_position, site_matrix, object_position, fraction):
    local = np.asarray(site_matrix).T @ (np.asarray(object_position) - site_position)
    return bool(np.all(np.abs(local[:2]) < fraction * np.asarray(site.size[:2])))


def _drawer_state(qpos, open_range, close_range, closed, closed_fraction, fully_open_fraction):
    open_end = max(open_range, key=lambda value: abs(value - np.mean(close_range)))
    closed_end = np.mean(close_range)
    progress = (qpos - closed_end) / (open_end - closed_end)
    if closed or progress <= closed_fraction:
        return "closed"
    return "fully_open" if progress >= fully_open_fraction else "partially_open"


class DiagnosticStateTracker:
    """Grounded labels for objects and task-relevant placement/drawer relations."""

    def __init__(self, env, thresholds=None):
        self.env = env
        self.thresholds = thresholds or DiagnosticThresholds()
        self.initial = {}
        self.held_steps = {}

    def reset(self):
        self.initial = {
            name: (np.array(state.get_geom_state()["pos"], copy=True),
                   np.array(state.get_geom_state()["quat"], copy=True))
            for name, state in self.env.object_states_dict.items()
            if name in self.env.objects_dict
        }
        self.held_steps = {name: 0 for name in self.initial}

    def _holding(self, name):
        object_model = self.env.get_object(name)
        gripper = self.env.robots[0].gripper
        grasped = self.env._check_grasp(gripper, object_model)
        self.held_steps[name] = self.held_steps.get(name, 0) + 1 if grasped else 0
        return self.held_steps[name] >= self.thresholds.held_contact_steps

    def _supported(self, name):
        object_model = self.env.get_object(name)
        candidates = {**self.env.fixtures_dict, **self.env.objects_dict}
        return any(self.env.check_contact(object_model, other)
                   for other_name, other in candidates.items() if other_name != name)

    def _centered(self, predicate, name, target):
        source_state = self.env.object_states_dict[name]
        target_state = self.env.object_states_dict[target]
        if not self.env._eval_predicate((predicate, name, target)):
            return False
        source_position = source_state.get_geom_state()["pos"]
        if target_state.object_state_type == "site":
            site = self.env.object_sites_dict[target]
            return _site_centered(
                site, self.env.sim.data.get_site_xpos(target),
                self.env.sim.data.get_site_xmat(target), source_position,
                self.thresholds.center_fraction,
            )
        if predicate == "on":
            target_position = target_state.get_geom_state()["pos"]
            return np.linalg.norm(source_position[:2] - target_position[:2]) < 0.03 * self.thresholds.center_fraction
        return False

    def _drawer(self, name):
        state = self.env.object_states_dict[name]
        model = self.env.get_object(state.parent_name if state.object_state_type == "site" else name)
        properties = model.object_properties.get("articulation", {})
        open_range = properties.get("default_open_ranges", [])
        close_range = properties.get("default_close_ranges", [])
        joints = self.env.object_sites_dict[name].joints if state.object_state_type == "site" else model.joints
        if not open_range or not close_range or not joints:
            return None
        qpos = self.env.sim.data.qpos[self.env.sim.model.get_joint_qpos_addr(joints[0])]
        return _drawer_state(qpos, open_range, close_range, state.is_close(),
                             self.thresholds.closed_fraction, self.thresholds.fully_open_fraction)

    def sample(self, task_atoms):
        if not self.initial:
            raise RuntimeError("Call reset() after the environment has settled")
        labels = set()
        for name, (initial_position, initial_quaternion) in self.initial.items():
            state = self.env.object_states_dict[name].get_geom_state()
            angle = _orientation_angle(initial_quaternion, state["quat"])
            holding = self._holding(name)
            if holding:
                labels.add(("holding", name))
            if angle <= self.thresholds.initial_orientation_degrees:
                labels.add(("initial_orientation", name))
            if (angle >= self.thresholds.fallen_orientation_degrees
                    and state["pos"][2] <= initial_position[2] - self.thresholds.fallen_height_drop
                    and not holding and self._supported(name)):
                labels.add(("fallen", name))

        for atom in task_atoms:
            predicate, name, *rest = atom
            if predicate in ("on", "in") and self._centered(predicate, name, rest[0]):
                labels.add((predicate + "_center", name, rest[0]))
            elif predicate in ("open", "close"):
                drawer = self._drawer(name)
                if drawer:
                    labels.add((drawer, name))
        return labels
