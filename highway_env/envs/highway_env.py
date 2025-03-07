import statistics

import gym
import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.envs.common.observation import observation_factory
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
                # speed=21,
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
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        reward = utils.lmap(reward,
                          [self.config["collision_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        """
        reward = 0
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)


class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False




class HighwayEnvHPRS(HighwayEnvFast):
    """
    A variant of highway-v0 designed for HPRS:
    """


    def __init__(self):
        super(HighwayEnvHPRS, self).__init__()
        self.target_distance = const.TARGET_DISTANCE
        self.target_distance_tol = const.TARGET_DISTANCE_TOL
        self.soft_speed_limit = const.SOFT_SPEED_LIMIT
        self.hard_speed_limit = const.HARD_SPEED_LIMIT
        self.speed_lower_bound = const.SPEED_LOWER_BOUND
        self.x_limit = const.X_LIMIT
        self.y_limit = const.Y_LIMIT
        self.vx_limit = const.VX_LIMIT
        self.vy_limit = const.VY_LIMIT

        self.step_count = 0
        self.max_steps = self.config['duration']
        self.ego_avg_speed = []

        self.observation_space = gym.spaces.Dict(dict(
            observation=observation_factory(self, self.config["observation"]).space(),
            ego_x=Box(low=0.0, high=np.inf, shape=(1,)),
            ego_y=Box(low=0.0, high=np.inf, shape=(1,)),
            ego_vx=Box(low=0.0, high=np.inf, shape=(1,)),
            ego_vy=Box(low=-np.inf, high=np.inf, shape=(1,)),
            collision=Box(low=0.0, high=np.inf, shape=(1,)),
            violated_safe_distance=Box(low=0.0, high=1.0, shape=(1,)),
            violated_hard_speed_limit=Box(low=0.0, high=1.0, shape=(1,)),
            road_progress=Box(low=0.0, high=np.inf, shape=(1,)),
            distance_to_target=Box(low=0.0, high=np.inf, shape=(1,)),
            max_velocity_difference_to_left=Box(low=0.0, high=np.inf, shape=(1,)),
            step_count=Box(low=0.0, high=np.inf, shape=(1,))
        ))

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": const.LANES_COUNT,
            "vehicles_count": 20,
            "duration": 40,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg


    def violated_safe_distance(self, obs, info):
        # assuming states are absolute
        if self.vehicle.crashed:
            return 1
        assert (len(obs) > 0) and (len(obs[0]) == 7)
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
                # print('lon: ', d_lon, '     -    ', d_lon_safe)
                # print('lat:', d_lat, '     -    ', d_lat_safe)
                # print('----------------------------------')
                return 1
        return 0



    def violated_hard_speed_limit(self, obs, info):
        assert len(obs) > 0 and len(obs[0]) == 7
        assert 'HARD_SPEED_LIMIT' in info
        ego_v_lon = obs[0][3]
        return float(ego_v_lon > info['HARD_SPEED_LIMIT'])


    def reached_target(self, obs, info):
        assert len(obs) > 0 and len(obs[0]) == 7
        assert 'TARGET_DISTANCE' in info and 'TARGET_DISTANCE_TOL' in info

        ego_x = obs[0][1]
        check_goal = bool(abs(ego_x - info['TARGET_DISTANCE']) <= info['TARGET_DISTANCE_TOL'])
        return 1 if check_goal else 0


    def get_ego_road_progress(self, obs, info):
        assert (len(obs) > 0) and (len(obs[0]) == 7)
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
            if self.vehicle.crashed:
                return -1.0
            elif self.reached_target(obs, info):
                return 1.0
        return 0


    def set_info_constants(self, info):
        info['TARGET_DISTANCE'] = self.target_distance
        info['TARGET_DISTANCE_TOL'] = self.target_distance_tol
        info['SOFT_SPEED_LIMIT'] = self.soft_speed_limit
        info['HARD_SPEED_LIMIT'] = self.hard_speed_limit
        info['SPEED_LOWER_BOUND'] = self.speed_lower_bound
        info['X_LIMIT'] = self.x_limit
        info['Y_LIMIT'] = self.y_limit
        info['VX_LIMIT'] = self.vx_limit
        info['VY_LIMIT'] = self.vy_limit
        info['MAX_STEPS'] = self.max_steps
        return info

    def step(self, action: Action):
        self.step_count += 1
        obs, reward, done, info = super(HighwayEnvHPRS, self).step(action)

        self.ego_avg_speed.append(obs[0][3])

        # print(action)
        # print('v_lon: ', obs[0][3])
        # print('avg_v_lon: ', (obs[1][3]+obs[2][3]+obs[3][3]+obs[4][3])/4)
        # print('------------------------')
        info = self.set_info_constants(info)

        reached_target = self.reached_target(obs, info)
        collision = float(self.vehicle.crashed)
        violated_safe_distance = self.violated_safe_distance(obs, info)
        violated_hard_speed_limit = self.violated_hard_speed_limit(obs, info)
        ego_road_progress = self.get_ego_road_progress(obs, info)
        distance_to_target = self.get_distance_to_target(obs, info)
        max_velocity_difference_to_left = self.get_max_velocity_difference_to_left(obs, info)


        state = {
            "observation": obs,
            "ego_x": obs[0][1],
            "ego_y": obs[0][2],
            "ego_vx": obs[0][3],
            "ego_vy": obs[0][4],
            "collision": collision,
            "violated_safe_distance": violated_safe_distance,
            "violated_hard_speed_limit": violated_hard_speed_limit,
            "road_progress": ego_road_progress,
            "distance_to_target": distance_to_target,
            "max_velocity_difference_to_left": max_velocity_difference_to_left,
            "step_count": self.step_count
        }

        done = done or reached_target or violated_safe_distance or violated_hard_speed_limit

        info['done'] = done


        if info['done']:
            print(self.step_count)
            print(collision)
            print(state['violated_safe_distance'])
            print(state['violated_hard_speed_limit'])
            print(statistics.mean(self.ego_avg_speed))
            print(state['road_progress'])
            print(reached_target)
            print('_________________')

        reward = self.reward(obs, info)
        return state, reward, done, info


    def reset(self):
        obs = super(HighwayEnvHPRS, self).reset()

        self.step_count = 0
        self.ego_avg_speed = []
        state = {
            "observation": obs,
            "ego_x": 0,
            "ego_y": 0,
            "ego_vx": 0,
            "ego_vy": 0,
            "collision": 0,
            "violated_safe_distance": 0,
            "violated_hard_speed_limit": 0,
            "road_progress": 0,
            "distance_to_target": const.TARGET_DISTANCE,
            "max_velocity_difference_to_left": 0,
            "step_count": self.step_count
        }
        return state






class HighwayEnvBaseline(HighwayEnvFast):
    """
    A variant of highway-v0 designed to be used as a baseline to be compared with HPRS:
    """


    def __init__(self):
        super(HighwayEnvBaseline, self).__init__()
        self.target_distance = const.TARGET_DISTANCE
        self.target_distance_tol = const.TARGET_DISTANCE_TOL

        self.step_count = 0

        self.observation_space = gym.spaces.Dict(dict(
            observation=observation_factory(self, self.config["observation"]).space(),
            road_progress=Box(low=0.0, high=np.inf, shape=(1,)),
            distance_to_target=Box(low=0.0, high=np.inf, shape=(1,)),
            collision=Box(low=0.0, high=1, shape=(1,)),
            step_count=Box(low=0.0, high=np.inf, shape=(1,))
        ))

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 40,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg



    def reached_target(self, obs, info):
        assert len(obs) > 0 and len(obs[0]) == 7
        assert 'TARGET_DISTANCE' in info and 'TARGET_DISTANCE_TOL' in info

        ego_x = obs[0][1]
        check_goal = bool(abs(ego_x - info['TARGET_DISTANCE']) <= info['TARGET_DISTANCE_TOL'])
        return 1 if check_goal else 0


    def get_ego_road_progress(self, obs, info):
        assert (len(obs) > 0) and (len(obs[0]) == 7)
        ego_obs = obs[0]
        ego_driven_distance = ego_obs[1]  # x
        return ego_driven_distance


    def get_distance_to_target(self, obs, info):
        assert 'TARGET_DISTANCE' in info
        ego_driven_distance = self.get_ego_road_progress(obs, info)
        if ego_driven_distance >= info['TARGET_DISTANCE']:
            return 0   # the distance is considered 0 if the ego passes the target
        return info['TARGET_DISTANCE'] - ego_driven_distance



    def reward(self, obs, info):
        assert 'done' in info
        if info['done']:
            if self.vehicle.crashed:
                return -1.0
            elif self.reached_target(obs, info):
                return 1.0
        return 0


    def step(self, action: Action):
        self.step_count += 1
        obs, reward, done, info = super(HighwayEnvBaseline, self).step(action)

        # print('v_lon: ', obs[0][3])
        # print('avg_v_lon: ', (obs[1][3]+obs[2][3]+obs[3][3]+obs[4][3])/4)

        info['TARGET_DISTANCE'] = self.target_distance
        info['TARGET_DISTANCE_TOL'] = self.target_distance_tol

        reached_target = self.reached_target(obs, info)
        ego_road_progress = self.get_ego_road_progress(obs, info)
        distance_to_target = self.get_distance_to_target(obs, info)
        crashed = float(self.vehicle.crashed)



        state = {
            "observation": obs,
            "road_progress": ego_road_progress,
            "distance_to_target": distance_to_target,
            "collision": crashed,
            "step_count": self.step_count
        }

        # if done:
        #     print('end of duration')


        done = done or reached_target or crashed

        info['done'] = done



        if info['done']:
            print(state['violated_safe_distance'])
            print(state['violated_hard_speed_limit'])
            print(state['road_progress'])
            print(reached_target)
            print('_________________')

        reward = self.reward(obs, info)


        return state, reward, done, info


    def reset(self):
        obs = super(HighwayEnvBaseline, self).reset()

        self.step_count = 0

        state = {
            "observation": obs,
            "road_progress": 0,
            "distance_to_target": const.TARGET_DISTANCE,
            "collision": 0,
            "step_count": self.step_count,
        }
        return state



register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)

register(
    id='highway-fast-v0',
    entry_point='highway_env.envs:HighwayEnvFast',
)

register(
    id='highway-hprs-v0',
    entry_point='highway_env.envs:HighwayEnvHPRS',
)

register(
    id='highway-hprs-baseline-v0',
    entry_point='highway_env.envs:HighwayEnvBaseline',
)