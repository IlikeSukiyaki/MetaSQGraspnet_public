import warp as wp

from isaac_env.env_base.wp_cfg import *

from .element_cfg import *


@wp.func
def _classical_retreat_vector(rotation: wp.quat):
    return wp.quat_rotate(
        rotation,
        wp.vec3(
            CLASSICAL_APPROACH_AXIS_EE_X * APPROACH_DISTANCE,
            CLASSICAL_APPROACH_AXIS_EE_Y * APPROACH_DISTANCE,
            CLASSICAL_APPROACH_AXIS_EE_Z * APPROACH_DISTANCE,
        ),
    )


@wp.func
def approach_pose_from_classical_grasp_pose(grasp_pose: wp.transform):
    """Compute the pre-grasp pose by backing away along the configured EE axis."""
    retreat = _classical_retreat_vector(wp.transform_get_rotation(grasp_pose))
    return wp.transform(
        wp.transform_get_translation(grasp_pose) - retreat,
        wp.transform_get_rotation(grasp_pose),
    )


@wp.func
def interpolated_classical_approach_pose(grasp_pose: wp.transform, elapsed_time: float):
    """Move from pre-grasp to grasp at a controlled pace."""
    grasp_pos = wp.transform_get_translation(grasp_pose)
    grasp_rot = wp.transform_get_rotation(grasp_pose)
    retreat = _classical_retreat_vector(grasp_rot)

    approach_alpha = float(0.0)
    if elapsed_time > CLASSICAL_PREGRASP_HOLD_TIME:
        approach_alpha = (elapsed_time - CLASSICAL_PREGRASP_HOLD_TIME) / CLASSICAL_APPROACH_DURATION
        if approach_alpha > 1.0:
            approach_alpha = 1.0

    return wp.transform(
        grasp_pos - retreat * (1.0 - approach_alpha),
        grasp_rot,
    )


@wp.kernel
def infer_state_machine_disc(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    successive_grasp_failure: wp.array(dtype=wp.float32),
    epi_count_wp: wp.array(dtype=wp.int32),
    step_count_wp: wp.array(dtype=wp.int32),
    env_reachable: wp.array(dtype=bool),
    env_stable: wp.array(dtype=bool),
    ee_pose: wp.array(dtype=wp.transform),
    ee_vel: wp.array(dtype=float),
    des_ee_pose: wp.array(dtype=wp.transform),
    des_gripper_state: wp.array(dtype=float),
    ee_quat_default: wp.array(dtype=wp.quat),
    grasp_pose: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    state = sm_state[tid]

    if state == PickSmState.init_env:
        des_ee_pose[tid] = ee_pose[tid]
        des_gripper_state[tid] = GripperState.OPEN
        step_count_wp[tid] = 0
        epi_count_wp[tid] = wp.add(epi_count_wp[tid], 1)
        successive_grasp_failure[tid] = 0.0
        sm_state[tid] = PickSmState.start
        sm_wait_time[tid] = 0.0

    elif state == PickSmState.init:
        step_count_wp[tid] = wp.add(step_count_wp[tid], 1)
        successive_grasp_failure[tid] = wp.add(successive_grasp_failure[tid], 1.0)

        if (
            successive_grasp_failure[tid] == SUCCESSIVE_GRASP_FAILURE_LIMIT
            or env_reachable[tid] == False
            or step_count_wp[tid] == STEP_TOTAL
        ):
            sm_state[tid] = PickSmState.init_env
        else:
            sm_state[tid] = PickSmState.start

        des_ee_pose[tid] = ee_pose[tid]
        des_gripper_state[tid] = GripperState.OPEN
        sm_wait_time[tid] = 0.0

    elif state == PickSmState.start:
        des_ee_pose[tid] = ee_pose[tid]
        des_gripper_state[tid] = GripperState.OPEN
        if env_stable[tid] == True or sm_wait_time[tid] >= PickSmLimitTime.start:
            sm_state[tid] = PickSmState.choose_object
            sm_wait_time[tid] = 0.0

    elif state == PickSmState.choose_object:
        des_ee_pose[tid] = ee_pose[tid]
        des_gripper_state[tid] = GripperState.OPEN
        sm_state[tid] = PickSmState.reach
        sm_wait_time[tid] = 0.0

    elif state == PickSmState.reach:
        des_ee_pose[tid] = approach_pose_from_classical_grasp_pose(grasp_pose[tid])
        des_gripper_state[tid] = GripperState.OPEN
        if dist_transforms(ee_pose[tid], des_ee_pose[tid]) < DISTANCE_LIMIT and ee_vel[tid] < EE_VEL_LIMIT:
            sm_state[tid] = PickSmState.approach
        elif sm_wait_time[tid] >= PickSmLimitTime.reach:
            sm_state[tid] = PickSmState.init

    elif state == PickSmState.approach:
        des_ee_pose[tid] = interpolated_classical_approach_pose(grasp_pose[tid], sm_wait_time[tid])
        des_gripper_state[tid] = GripperState.OPEN
        if (
            sm_wait_time[tid] >= CLASSICAL_PREGRASP_HOLD_TIME + CLASSICAL_APPROACH_DURATION
            and dist_transforms(ee_pose[tid], grasp_pose[tid]) < DISTANCE_LIMIT
            and ee_vel[tid] < EE_VEL_LIMIT
            or sm_wait_time[tid] >= CLASSICAL_APPROACH_TIMEOUT
        ):
            sm_state[tid] = PickSmState.grasp
            sm_wait_time[tid] = 0.0

    elif state == PickSmState.grasp:
        des_ee_pose[tid] = grasp_pose[tid]
        des_gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= PickSmLimitTime.grasp:
            sm_state[tid] = PickSmState.lift
            sm_wait_time[tid] = 0.0

    elif state == PickSmState.lift:
        grasp_pos = wp.add(wp.transform_get_translation(grasp_pose[tid]), wp.vec3(0.0, 0.0, LIFT_HEIGHT))
        grasp_rot = wp.transform_get_rotation(grasp_pose[tid])
        des_ee_pose[tid] = wp.transform(grasp_pos, grasp_rot)
        des_gripper_state[tid] = GripperState.CLOSE
        if (
            dist_transforms(ee_pose[tid], des_ee_pose[tid]) < DISTANCE_LIMIT
            and ee_vel[tid] < EE_VEL_LIMIT
            or sm_wait_time[tid] >= PickSmLimitTime.lift
        ):
            sm_state[tid] = PickSmState.init

    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]
