import math

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


class HighwayADEXEnv(AbstractEnv):
    """
    A highway driving environment extended to handle EGO and SUT vehicles.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "features": ["is_sut", "presence", "x", "y", "vx", "vy", "cos_h", "sin_h"]
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
            "collision_reward": -1,  # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
            # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
            # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,  # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "offroad_terminal": False
        })
        return config

    @property
    def vehicle(self) -> Vehicle:
        """First (default) controlled vehicle."""
        return self.controlled_vehicles[0] if self.controlled_vehicles else None

    @property
    def sut_vehicle(self) -> Vehicle:
        """First (default) sut vehicle."""
        return self.sut_vehicles[0] if self.sut_vehicles else None

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
        self.sut_vehicles = []
        for i, others in enumerate(other_per_controlled):
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            # create controlled vehicle
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            vehicle.role = 'ego'

            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

        # mark one random vehicle as sut
        id_sut = np.random.choice([i for i, v in enumerate(self.road.vehicles) if v.role != 'ego'])
        self.road.vehicles[id_sut].role = 'sut'
        self.sut_vehicles.append(self.road.vehicles[id_sut])

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


class HighwayADEXEnvFast(HighwayADEXEnv):
    """
    A variant of highway-adex-v0 with faster execution:
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


class HighwayADEXEnvDebug(HighwayADEXEnvFast):
    """
    A variant of highway-adex-v0 with faster execution:
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
            "vehicles_count": 2,
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg


class HighwayADEXEnvHPRS(HighwayADEXEnvDebug):

    def __init__(self) -> None:
        super(HighwayADEXEnvHPRS, self).__init__()
        self.target_distance = 850
        self.dist_target_tol = 50
        self.soft_speed_limit = const.SOFT_SPEED_LIMIT
        self.hard_speed_limit = const.HARD_SPEED_LIMIT
        self.time_step = 0

    def violated_safe_distance(self, state, info):
        # assuming states are absolute
        assert len(state) > 0
        assert len(state[0]) == 8
        ego_state = state[0]
        for i in range(1, len(state)):
            vehicle_state = state[i]
            if vehicle_state[1] == 1:  # if vehicle is present
                d_lon = abs(vehicle_state[2] - ego_state[2])  # | x_ego - x_other |
                d_lat = abs(vehicle_state[3] - ego_state[3])  # | y_ego - y_other |
            else:
                d_lon = math.inf
                d_lat = math.inf
            d_lon_safe = hprs_utils.safe_long_dist(ego_state, vehicle_state)
            d_lat_safe = hprs_utils.safe_lat_dist(ego_state, vehicle_state)
            violated = bool(d_lon < d_lon_safe and d_lat < d_lat_safe)
            if violated:
                return True
        return False


    def violated_hard_speed_limit(self, state, info):
        assert len(state) > 0
        assert len(state[0]) == 8
        ego_state = state[0]
        ego_v_lon = ego_state[4]
        return int(ego_v_lon > info['HARD_SPEED_LIMIT'])


    def ego_drives_faster_than_left(self, state, info):
        assert len(state) > 0
        assert len(state[0]) == 8
        ego_state = state[0]
        for i in range(1, len(state)):
            vehicle_state = state[i]
            if vehicle_state[1] == 1:  # if vehicle is present
                if hprs_utils.left_lane(vehicle_state, ego_state) and hprs_utils.in_vicinity(vehicle_state, ego_state):
                    if ego_state[4] > vehicle_state[4]:
                        return True
        return False


    def reward(self, state, info):
        """
        Kinda-Vanilla Reward - Punishment for crash, bonus for reaching the target
        """
        if info['done']:
            if self.violated_safe_distance(state, info):
                return -1.0
            elif self.reached_target(state, info):
                return 1.0
        return 0

    def reached_target(self, state, info):
        assert len(state) > 0
        assert len(state[0]) == 8
        ego_x = state[0][2]
        check_goal = bool(abs(ego_x - info['target_distance']) <= info['dist_target_tol'])
        return True if check_goal else False


    def step(self, action: Action):
        state, reward, done, info = super(HighwayADEXEnvHPRS, self).step(action)
        print(state)
        '''
        state = {
            "observation": obs,
            "rss": ...,

        }
        '''
        info['target_distance'] = self.target_distance
        info['dist_target_tol'] = self.dist_target_tol
        info['SOFT_SPEED_LIMIT'] = self.soft_speed_limit
        info['HARD_SPEED_LIMIT'] = self.hard_speed_limit
        info['time_step'] = self.time_step
        done = done or \
               self.reached_target(state, info) or \
               self.violated_safe_distance(state, info) or \
               self.violated_hard_speed_limit(state, info)
               # self.ego_drives_faster_than_left(state, info)
        info['done'] = done
        reward = self.reward(state, info)
        return state, reward, done, info




register(
    id='highway-adex-v0',
    entry_point='highway_env.envs:HighwayADEXEnv',
)

register(
    id='highway-adex-fast-v0',
    entry_point='highway_env.envs:HighwayADEXEnvFast',
)

register(
    id='highway-adex-debug-v0',
    entry_point='highway_env.envs:HighwayADEXEnvDebug',
)

