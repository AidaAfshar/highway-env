from highway_env import utils
import imageio as imageio
import time
import gym
import pathlib
import json
import os
from rewards import reward_register
from rewards.reward_wrapper import HighwayRewardWrapper
from gym.wrappers import RecordVideo, Monitor, FlattenObservation
import highway_env
from rewards import reward_register


from stable_baselines3 import DQN

from rewards.reward_wrapper import HighwayRewardWrapper
import _pickle as pickle

LOGDIR = f"../logs/highway-adex-debug-v0/dqn_purely_adv_1659582155.2525618"


def get_modified_obs(number_of_vehicles, obs, previous_observation, dt, lane_width):
    new_obs = dict()
    for i in range(number_of_vehicles):
        new_obs[i] = dict()
        if i == 0:
            new_obs[i]['type'] = 'ego'
        elif obs[i][0] == 1:
            new_obs[i]['type'] = 'sut'
        else:
            new_obs[i]['type'] = 'npc'
        new_obs[i]['x'] = float(obs[i][2])
        new_obs[i]['y'] = float(obs[i][3])
        new_obs[i]['v_lon'], new_obs[i]['v_lat'] = float(obs[i][4]), float(obs[i][5])
        new_obs[i]['a_lon'], new_obs[i]['a_lat'] = calculate_acceleration(new_obs[i]['v_lon'], new_obs[i]['v_lat'], previous_observation[i], dt)
        new_obs[i]['j_lon'], new_obs[i]['j_lat'] = calculate_jerk(new_obs[i]['a_lon'], new_obs[i]['a_lat'], previous_observation[i], dt)
        new_obs[i]['lane_id'] = get_lane_id(new_obs[i]['y'], lane_width)
    return new_obs


def calculate_acceleration(v_lon, v_lat, previous_observation, dt):
    pre_v_lon, pre_v_lat = previous_observation['v_lon'], previous_observation['v_lat']
    a_lon = (v_lon - pre_v_lon) / dt
    a_lat = (v_lat - pre_v_lat) / dt
    return float(a_lon), float(a_lat)

def calculate_jerk(a_lon, a_lat, previous_observation, dt):
    pre_a_lon, pre_a_lat = previous_observation['a_lon'], previous_observation['a_lat']
    j_lon = (a_lon - pre_a_lon) / dt
    j_lat = (a_lat - pre_a_lat) / dt
    return float(j_lon), float(j_lat)


def get_lane_id(y, lane_width):
    return int(y/lane_width)

def get_initial_modified_obs(number_of_vehicles, obs, lane_width):
    new_obs = dict()
    for i in range(number_of_vehicles):
        new_obs[i] = dict()
        if i == 0:
            new_obs[i]['type'] = 'ego'
        elif obs[i][0] == 1:
            new_obs[i]['type'] = 'sut'
        else:
            new_obs[i]['type'] = 'npc'
        new_obs[i]['x'] = float(obs[i][2])
        new_obs[i]['y'] = float(obs[i][3])
        new_obs[i]['v_lon'], new_obs[i]['v_lat'] = float(obs[i][4]), float(obs[i][5])
        new_obs[i]['a_lon'], new_obs[i]['a_lat'] = 0, 0
        new_obs[i]['j_lon'], new_obs[i]['j_lat'] = 0, 0
        new_obs[i]['lane_id'] = get_lane_id(new_obs[i]['y'], lane_width)
    return new_obs



def random_policy_main(lane_width):
    env = gym.make(env_name)
    env.configure(config)
    number_of_vehicles = env.config['vehicles_count']
    number_of_lanes = env.config['lanes_count']
    env.configure({"simulation_frequency": 15})  # Higher FPS for rendering
    dt = 1 / env.config["simulation_frequency"]
    obs = env.reset()
    previous_observation = get_initial_modified_obs(number_of_vehicles, obs, lane_width)
    reward_fn = reward_register.make(reward_name, dt=dt)
    env = HighwayRewardWrapper(env, reward_fn)

    observation_log = dict()
    for videos in range(1):
        t = 0
        done = False
        s_t = env.reset()
        frames = []
        while not done:
            frame = env.render(mode="rgb_array")
            frames.append(frame)
            t += 1
            a_t = env.action_space.sample()
            s_t, r_t, done, info = env.step(a_t)
            modified_obs = get_modified_obs(number_of_vehicles, s_t, previous_observation, dt, lane_width)
            observation_log[t] = modified_obs
            #print(modified_obs)
            previous_observation = modified_obs

        name = time.time()
        with open(f"states_{name}.json", 'w') as fp:
            fp.write(json.dumps(observation_log, indent=6))
        writer = imageio.get_writer(f"video_{name}.mp4", fps=5)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
    env.close()


def trained_model_main():
    env_name = "highway-adex-debug-v0"
    reward_name = "purely_adv"

    env = gym.make(env_name)
    reward_fn = reward_register.make(reward_name, dt=1 / env.config["simulation_frequency"])
    env = HighwayRewardWrapper(env, reward_fn)
    env.configure(config)
    previous_observation = env.reset()
    number_of_vehicles = env.config['vehicles_count']
    dt = 1 / env.config["simulation_frequency"]

    logdir = pathlib.Path(LOGDIR)
    logdir.mkdir(exist_ok=True, parents=True)

    model = DQN.load(f"{logdir}/model", env=env)
    env = RecordVideo(env, video_folder=f"{logdir}/videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    env.configure({"simulation_frequency": 15})  # Higher FPS for rendering

    for videos in range(5):
        done = False
        s_t = env.reset()
        frames = []
        observation_log = dict()
        t = 0
        while not done:
            t += 1
            a_t, _states = model.predict(s_t, deterministic=True)
            s_t, r_t, done, info = env.step(a_t)
            modified_obs = get_modified_obs(number_of_vehicles, s_t, previous_observation, dt)
            observation_log[t] = modified_obs
            print(modified_obs)
            frame = env.render(mode="rgb_array")
            frames.append(frame)

        name = time.time()
        with open(f"states_{name}.json", 'w') as fp:
            fp.write(json.dumps(observation_log, indent=6))

        writer = imageio.get_writer(f"video_{name}.mp4", fps=5)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
    env.close()




if __name__ == "__main__":
    # Create the environment
    env_name = "highway-adex-debug-v0"
    reward_name = "purely_adv"
    config = {
        "observation": {
            "type": "Kinematics",
            "features": ["is_sut", "presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "vehicles_count": 5,
            "absolute": True,
            "normalize":False
        },
        "absolute": True,
        "vehicles_count": 5,
        "lanes_count": 4,
        "offscreen_rendering": os.environ.get("OFFSCREEN_RENDERING", "0") == "1",
        "real_time_rendering": True
    }

    lane_width = 4
    random_policy_main(lane_width)

