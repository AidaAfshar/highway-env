from highway_env import utils
import imageio as imageio
import time
import gym
import pathlib

from rewards import reward_register
from rewards.reward_wrapper import HighwayRewardWrapper
from gym.wrappers import RecordVideo, Monitor, FlattenObservation
import highway_env
from rewards import reward_register


from stable_baselines3 import DQN

from rewards.reward_wrapper import HighwayRewardWrapper

LOGDIR = f"../logs/highway-adex-debug-v0/dqn_purely_adv_1659556404.739288"


def get_modified_obs(number_of_vehicles, obs, action, previous_action, previous_observation, acceleration_range, dt):
    new_obs = dict()
    for i in range(number_of_vehicles):
        new_obs[i] = dict()
        if i == 1:
            new_obs[i]['type'] = 'ego'
        elif obs[i][0] == 1:
            new_obs[i]['type'] = 'sut'
        else:
            new_obs[i]['type'] = 'npc'
        new_obs[i]['x'] = obs[i][2]
        new_obs[i]['y'] = obs[i][3]
        new_obs[i]['v_lon'] = obs[i][4]
        new_obs[i]['v_lat'] = obs[i][5]
        new_obs[i]['a_lon'], new_obs[i]['a_lat'] = calculate_acceleration(action[0], obs[i][6], obs[i][7], acceleration_range)
        new_obs[i]['j_lon'], new_obs[i]['j_lat'] = calculate_jerk(action[0], obs[i][6], obs[i][7], previous_action,
                                                                       previous_observation[i], acceleration_range, dt)
    return new_obs


def calculate_acceleration(acceleration, cos_h, sin_h, acceleration_range):
    acceleration = utils.lmap(acceleration, [-1, 1], acceleration_range)
    a_lon = acceleration * cos_h
    a_lat = acceleration * sin_h
    return a_lon, a_lat

def calculate_jerk(acceleration, sin_h, cos_h, previous_action, previous_observation, acceleration_range, dt):
    acceleration = utils.lmap(acceleration, [-1, 1], acceleration_range)
    pre_acceleration = utils.lmap(previous_action[0], [-1, 1], acceleration_range)
    a_lon = acceleration * cos_h
    a_lat = acceleration * sin_h
    pre_a_lon = pre_acceleration * previous_observation[6]
    pre_a_lat = pre_acceleration * previous_observation[7]
    j_lon = (a_lon - pre_a_lon)/dt
    j_lat = (a_lat - pre_a_lat)/dt
    return j_lon, j_lat



def random_policy_main():
    env = gym.make(env_name)
    env.configure(config)
    previous_observation = env.reset()
    acceleration_range = [-5.0, 5.0]
    previous_action = [0, 0]
    number_of_vehicles = env.config['vehicles_count']
    dt = 1 / env.config["simulation_frequency"]
    reward_fn = reward_register.make(reward_name, dt=dt)
    env = HighwayRewardWrapper(env, reward_fn)
    env.configure({"simulation_frequency": 15})  # Higher FPS for rendering

    for videos in range(10):
        done = False
        s_t = env.reset()
        frames = []
        while not done:
            a_t = env.action_space.sample()
            s_t, r_t, done, info = env.step(a_t)
            modified_obs = get_modified_obs(number_of_vehicles, s_t, a_t, previous_action, previous_observation,
                                            acceleration_range, dt)
            print(modified_obs)
            previous_action = a_t
            previous_observation = s_t
            frame = env.render(mode="rgb_array")
            frames.append(frame)

        writer = imageio.get_writer(f"video_{time.time()}.mp4", fps=5)
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
    acceleration_range = [-5.0, 5.0]
    previous_action = [0, 0]
    number_of_vehicles = env.config['vehicles_count']
    dt = 1 / env.config["simulation_frequency"]

    logdir = pathlib.Path(LOGDIR)
    logdir.mkdir(exist_ok=True, parents=True)

    model = DQN.load(f"{logdir}/model", env=env)
    env = RecordVideo(env, video_folder=f"{logdir}/videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    # env.configure({"simulation_frequency": 15})  # Higher FPS for rendering

    for videos in range(10):
        done = False
        s_t = env.reset()
        frames = []
        while not done:
            a_t, _states = model.predict(s_t, deterministic=True)
            s_t, r_t, done, info = env.step(a_t)
            modified_obs = get_modified_obs(number_of_vehicles, s_t, a_t, previous_action, previous_observation,
                                            acceleration_range, dt)
            print(modified_obs)
            previous_action = a_t
            previous_observation = s_t
            frame = env.render(mode="rgb_array")
            frames.append(frame)

        writer = imageio.get_writer(f"video_{time.time()}.mp4", fps=5)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
    env.close()




if __name__ == "__main__":
    # Create the environment
    env_name = "highway-adex-debug-v0"
    reward_name = "purely_adv"
    config = {
        "action": {
            "type": "ContinuousAction"
        },
        "observation": {
            "type": "Kinematics",
            "features": ["is_sut", "presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "vehicles_count": 6,
            "absolute": True,
        },
        "vehicles_count": 6,
        "lanes_count": 4
    }

    trained_model_main()

