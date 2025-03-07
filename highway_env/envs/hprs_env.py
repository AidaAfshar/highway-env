import gym
import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from rewards.reward_functions.hprs import hprs_utils
from rewards.reward_functions.hprs import constants as const


from gym import spaces
from gym.spaces import Box


class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        not implemented reward
        """
        reward = 0.0
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)


class HighwayEnvHPRS(HighwayEnv):
    """
    A variant of highway-v0 designed for HPRS:
    """


    def __init__(self) -> None:
        super(HighwayEnvHPRS, self).__init__()
        self.target_distance = const.TARGET_DISTANCE
        self.dist_target_tol = const.TARGET_DISTANCE_TOL
        self.soft_speed_limit = const.SOFT_SPEED_LIMIT
        self.hard_speed_limit = const.HARD_SPEED_LIMIT
        self.speed_lower_bound = const.SPEED_LOWER_BOUND
        self.step_count = 0

        self.observation_space = gym.spaces.Dict(dict(
            observation=super(HighwayEnvHPRS, self).observation_space,
            violated_safe_distance=Box(low=0.0, high=1.0, shape=(1,)),
            violated_hard_speed_limit=Box(low=0.0, high=1.0, shape=(1,)),
            step_count=Box(low=0.0, high=np.inf, shape=(1,)),
            road_progress=Box(low=0.0, high=np.inf, shape=(1,)),
            distance_to_target=Box(low=0.0, high=np.inf, shape=(1,)),
            max_velocity_difference_to_left=Box(low=0.0, high=np.inf, shape=(1,))
        ))



    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "lanes_count": 3,
            "vehicles_count": 15,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False


    def violated_safe_distance(self, obs, info):
        # assuming states are absolute
        assert len(obs) > 0
        assert len(obs[0]) == 7
        ego_obs = obs[0]
        for i in range(1, len(obs)):
            vehicle_obs = obs[i]
            if vehicle_obs[0] == 1:  # if vehicle is present
                d_lon = abs(vehicle_obs[1] - ego_obs[1])  # | x_ego - x_other |
                d_lat = abs(vehicle_obs[2] - ego_obs[2])  # | y_ego - y_other |
            else:
                d_lon = -float('inf')
                d_lat = -float('inf')
            d_lon_safe = hprs_utils.safe_long_dist(ego_obs, vehicle_obs)
            d_lat_safe = hprs_utils.safe_lat_dist(ego_obs, vehicle_obs)
            violated = bool(d_lon < d_lon_safe and d_lat < d_lat_safe)
            if violated:
                return True
        return False



    def violated_hard_speed_limit(self, obs, info):
        assert len(obs) > 0 and len(obs[0]) == 7
        assert 'HARD_SPEED_LIMIT' in info
        ego_v_lon = obs[0][3]
        return bool(ego_v_lon > info['HARD_SPEED_LIMIT'])


    def reached_target(self, obs, info):
        assert len(obs) > 0 and len(obs[0]) == 7
        assert 'TARGET_DISTANCE' in info and 'TARGET_DISTANCE_TOL' in info

        ego_x = obs[0][1]
        check_goal = bool(abs(ego_x - info['TARGET_DISTANCE']) <= info['TARGET_DISTANCE_TOL'])
        return True if check_goal else False


    def get_ego_road_progress(self, obs, info):
        assert len(obs) > 0
        assert len(obs[0]) == 7
        ego_obs = obs[0]
        ego_driven_distance = ego_obs[1]  # x
        return ego_driven_distance

    def get_distance_to_target(self, obs, info):
        assert 'TARGET_DISTANCE' in info
        ego_driven_distance = self.get_ego_road_progress(obs, info)
        if ego_driven_distance >= info['TARGET_DISTANCE']:
            return 0   # the distance is considered 0 if the ego passes the target
        return info['TARGET_DISTANCE'] - ego_driven_distance


    def get_max_velocity_difference_to_left(self, obs, info):
        assert len(obs) > 0 and len(obs[0]) == 7
        ego_obs = obs[0]
        dif_list = []

        for i in range(1, len(obs)):
            vehicle_obs = obs[i]
            if hprs_utils.left_lane(vehicle_obs, ego_obs) and \
               hprs_utils.in_vicinity(vehicle_obs, ego_obs) and \
               not hprs_utils.behind(vehicle_obs, ego_obs):

                vel_dif = ego_obs[3] - vehicle_obs[3]
                if vel_dif > 0:
                    dif_list.append(vel_dif)

        max_dif = max(dif_list) if len(dif_list) > 0 else 0
        return max_dif


    def reward(self, obs, info):
        assert 'done' in info
        if info['done']:
            if self.violated_safe_distance(obs, info):
                return -1.0
            elif self.reached_target(obs, info):
                return 1.0
        return 0


    def step(self, action: Action):

        self.step_count += 1
        obs, reward, done, info = super(HighwayEnvHPRS, self).step(action)

        reached_target = self.reached_target(obs, info)
        violated_safe_distance = self.violated_safe_distance(obs, info)
        violated_hard_speed_limit = self.violated_hard_speed_limit(obs, info)
        ego_road_progress = self.get_ego_road_progress(obs, info)
        distance_to_target = self.get_distance_to_target(obs, info)
        max_velocity_difference_to_left = self.get_max_velocity_difference_to_left(obs, info)

        state = {
            "observation": obs,
            "violated_safe_distance": violated_safe_distance,
            "violated_hard_speed_limit": violated_hard_speed_limit,
            "step_count": self.step_count,
            "road_progress": ego_road_progress,
            "distance_to_target": distance_to_target,
            "max_velocity_difference_to_left": max_velocity_difference_to_left
        }

        done = done or reached_target or violated_safe_distance or violated_hard_speed_limit

        info['TARGET_DISTANCE'] = self.target_distance
        info['TARGET_DISTANCE_TOL'] = self.target_distance_tol
        info['SOFT_SPEED_LIMIT'] = self.soft_speed_limit
        info['HARD_SPEED_LIMIT'] = self.hard_speed_limit
        info['SPEED_LOWER_BOUND'] = self.speed_lower_bound
        info['done'] = done


        reward = self.reward(state, info)


        return state, reward, done, info


    def reset(self):
        super.reset()
        self.step_count = 0


register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)

register(
    id='highway-hprs-v0',
    entry_point='highway_env.envs:HighwayEnvHPRS',
)
