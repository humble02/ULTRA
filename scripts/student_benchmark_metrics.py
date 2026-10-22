"""Shared task metrics for both engines; NumPy only."""
import numpy as np


def measure_trajectory(states, reference, hand_contact, support_contact, expected_controls):
    states, reference = np.asarray(states), np.asarray(reference)
    hand_contact, support_contact = np.asarray(hand_contact), np.asarray(support_contact)
    finite = bool(np.isfinite(states).all())
    complete = len(states) == expected_controls
    height = states[:, 2]
    position_error = np.linalg.norm(states[:, 71:74] - reference[:, 71:74], axis=-1)
    goal_error = np.linalg.norm(states[:, 71:74] - reference[-1, 71:74], axis=-1)
    fallen = bool(np.any(height < .25))
    success = finite and complete and not fallen and goal_error[-1] < .3
    count = min(15, len(states))
    tail = slice(-count, None)
    vertical = np.abs(states[:, 73] - reference[-1, 73])
    speed = np.linalg.norm(states[:, 78:81], axis=-1)
    settled = bool(count == 15 and np.all(goal_error[tail] < .3) and
                   np.all(vertical[tail] < .1) and np.all(speed[tail] < .2))
    released = bool(not np.any(hand_contact[tail]))
    supported = bool(np.all(support_contact[tail]))
    lifted = bool(states[:, 73].max() - reference[0, 73] >= .2 and hand_contact.sum() >= 10)
    return {"controls": len(states), "expected_controls": expected_controls, "all_finite": finite,
            "task_success": bool(success), "strict_placement_success": bool(success and settled and released and supported),
            "lifted_with_hand_contact": lifted,
            "carry_and_place_success": bool(success and lifted and settled and released and supported),
            "fell": fallen, "minimum_pelvis_height_m": float(height.min()),
            "final_box_position_error_m": float(goal_error[-1]),
            "final_box_xy_error_m": float(np.linalg.norm(states[-1, 71:73] - reference[-1, 71:73])),
            "box_trajectory_rmse_m": float(np.sqrt(np.mean(position_error ** 2))),
            "joint_trajectory_rmse_rad": float(np.sqrt(np.mean((states[:, 13:42] - reference[:, 13:42]) ** 2))),
            "pelvis_trajectory_rmse_m": float(np.sqrt(np.mean(np.sum((states[:, :3] - reference[:, :3]) ** 2, axis=-1)))),
            "box_lift_above_reference_start_m": float(states[:, 73].max() - reference[0, 73]),
            "box_travel_xy_m": float(np.linalg.norm(states[-1, 71:73] - reference[0, 71:73])),
            "hand_contact_controls": int(hand_contact.sum()), "tail_settled": settled,
            "tail_released": released, "tail_supported": supported,
            "final_box_speed_m_s": float(speed[-1])}
