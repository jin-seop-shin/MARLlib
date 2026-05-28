import numpy as np
from math import sqrt


def raycast(sim, geom1_id=None, geom2_id=None, pt1=None, pt2=None, geom_group=None):
    '''
        Given a mujoco sim, from a geom to a geom, point to a point
        Args:
            sim: Mujoco sim object
            geom1 (int), id of geom ray originates from
            geom2 (int), id of geom ray points to
            p1 (np.ndarray[3]): 3D point ray originates from
            p2 (np.ndarray[3]): 3D point ray points to
            geom_group: one-hot list determining which of the five geom groups
                        should be visible to the raycast
    '''
    assert (geom1_id is None) != (pt1 is None), "geom1_id or p1 must be specified"
    assert (geom2_id is None) != (pt2 is None), "geom2_id or p2 must be specified"
    if geom1_id is not None:
        pt1 = np.asarray(sim.data.geom_xpos[geom1_id], dtype=np.float64).flatten()
        body1 = int(sim.model.geom_bodyid[geom1_id])
    else:
        # Don't exclude any bodies if we originate ray from a point
        body1 = int(np.max(sim.model.geom_bodyid)) + 1
    if geom2_id is not None:
        pt2 = np.asarray(sim.data.geom_xpos[geom2_id], dtype=np.float64).flatten()

    ray_direction = np.asarray(pt2, dtype=np.float64) - np.asarray(pt1, dtype=np.float64)
    norm = sqrt(float(ray_direction[0]) ** 2 + float(ray_direction[1]) ** 2 + float(ray_direction[2]) ** 2)
    ray_direction /= norm

    if geom_group is not None:
        group_filter = np.asarray(geom_group, dtype=np.uint8).flatten()
    else:
        group_filter = np.array([1, 1, 1, 1, 1], dtype=np.uint8)

    pt1_c = np.ascontiguousarray(pt1, dtype=np.float64)
    vec_c = np.ascontiguousarray(ray_direction, dtype=np.float64)

    # Use mujoco_py's Python API to avoid raw C call + numpy 2.0 issues
    dist, geomid = sim.ray_fast_group(
        pnt=pt1_c,
        vec=vec_c,
        geomgroup=group_filter,
        flg_static=1,
        bodyexclude=body1,
    )
    collision_geom = geomid if geomid != -1 else None
    return dist, collision_geom
