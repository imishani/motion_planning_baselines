import torch
import time
import random
import matplotlib.pyplot as plt
import numpy as np

from examples.pointmass_3d_rectangles_rrt_star import fixed_rectangles
from experiment_launcher.utils import fix_random_seed
from mp_baselines.planners.rrt_connect import RRTConnect
from mp_baselines.planners.utils import elapsed_time, to_numpy
from robot_envs.base_envs.obstacle_map_env import ObstacleMapEnv
from torch_planning_objectives.fields.occupancy_map.map_generator import generate_obstacle_map, build_obstacle_map
from mp_baselines.planners.rrt_star import RRTStar
from torch_planning_objectives.fields.primitive_distance_fields import BoxField



if __name__ == "__main__":
    seed = 3
    fix_random_seed(seed)

    device = 'cpu'
    tensor_args = {'device': device, 'dtype': torch.float64}

    # -------------------------------- Environment ---------------------------------
    limits = torch.tensor([[-1, 1], [-1, 1], [-1, 1]], **tensor_args)

    ## Obstacle map
    cell_size = 0.05
    map_dim = [2, 2, 2]

    obst_list = fixed_rectangles()

    obst_params = dict(
        map_dim=map_dim,
        obst_list=obst_list,
        cell_size=cell_size,
        tensor_args=tensor_args,
    )
    obst_map = build_obstacle_map(**obst_params)

    env = ObstacleMapEnv(
        name='circles',
        q_n_dofs=3,
        q_min=limits[:, 0],
        q_max=limits[:, 1],
        obstacle_map=obst_map,
        tensor_args=tensor_args
    )

    # -------------------------------- Planner ---------------------------------
    n_iters = 30000
    max_best_cost_iters = 5000
    cost_eps = 1e-2
    step_size = 0.05
    n_radius = 0.3
    n_knn = 10
    goal_prob = 0.2
    max_time = 60.
    start_state = torch.tensor([-0.8, -0.8, -0.8], **tensor_args)
    goal_state = torch.tensor([0.8, 0.8, 0.8], **tensor_args)

    rrt_params = dict(
        env=env,
        n_iters=n_iters,
        start_state=start_state,
        step_size=step_size,
        n_radius=n_radius,
        goal_state=goal_state,
        tensor_args=tensor_args,
    )
    planner = RRTConnect(**rrt_params)

    # ---------------------------------------------------------------------------
    # Optimize
    start = time.time()
    traj = planner.optimize(debug=True, refill_samples_buffer=True)
    print(f"{elapsed_time(start)} seconds")

    # ---------------------------------------------------------------------------
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    planner.render(ax)
    obst_map.plot(ax)
    if traj is not None:
        traj = to_numpy(traj)
        ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2], 'b-')
        ax.scatter3D(traj[:, 0], traj[:, 1], traj[:, 2], color='b')
    start_state_np = to_numpy(start_state)
    goal_state_np = to_numpy(goal_state)
    ax.scatter3D(start_state_np[0], start_state_np[1], start_state_np[2], 'go', zorder=10, s=100)
    ax.scatter3D(goal_state_np[0], goal_state_np[1], goal_state_np[2], 'ro', zorder=10, s=100)
    ax.view_init(azim=0, elev=90)
    plt.show()
