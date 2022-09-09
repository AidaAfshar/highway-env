import pathlib
import time

import gym
from gym.wrappers import RecordVideo
from rewards import hprs_reward_register

from stable_baselines3 import PPO

from rewards.reward_wrapper import HPRSHighwayRewardWrapper
import rewards.reward_functions.hprs.potential as potential




def setup_env():
    # Create the environment
    env_name = "highway-adex-hprs-v0"
    reward_name = "hprs"

    env = gym.make(env_name)
    config = {
        "observation": {
            "type": "Kinematics",
            "features": ["is_sut", "presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": {
                "x": [0, 1000],
                "y": [0, 50],
                "vx": [0, 50],
                "vy": [-5, 5]
            },
            "normalize": False,
            "absolute": True,
        },
        "absolute": True,
        "vehicles_count": 7,
        "lanes_count": 5,
    }
    env.configure(config)
    env.reset()

    reward_fn = hprs_reward_register.make(reward_name, dt=1 / env.config["simulation_frequency"])
    env = HPRSHighwayRewardWrapper(env, reward_fn)
    # env = FlattenObservation(env)
    obs = env.reset()

    return env


def setup_model(env):

    logdir = pathlib.Path(LOGDIR)
    logdir.mkdir(exist_ok=True, parents=True)

    # Run the trained model and record video
    model = PPO.load(f"{logdir}/model", env=env)
    env = RecordVideo(env, video_folder=f"{logdir}/videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)

    return env, model


def get_metrics_for_one_timestep(state, info):
    s1 = potential.safety_RSS_potential(state, info)
    s2 = potential.safety_hard_speed_limit_potential(state, info)
    c1 = potential.comfort_soft_speed_limit_potential(state, info)
    c2 = potential.comfort_soft_speed_limit_potential(state, info)
    t = potential.target_reach_destination_potential(state, info)
    return s1, s2, c1, c2, t

def get_metrics_for_one_episode(s1_list, s2_list, c1_list, c2_list, t_list):
    assert len(s1_list) == len(s2_list) and len(s2_list) == len(c1_list)
    assert len(c1_list) == len(c2_list) and len(s1_list) == len(t_list)

    episode_length = len(s1_list)
    s1_sum, s2_sum, c1_sum, c2_sum, t_sum = sum(s1_list), sum(s2_list), sum(c1_list), sum(c2_list), sum(t_list)
    s1_percent, s2_percent, c1_percent, c2_percent, t_percent = (s1_sum/episode_length)*100, \
                                                                (s2_sum/episode_length)*100, \
                                                                (c1_sum/episode_length)*100, \
                                                                (c2_sum/episode_length)*100, \
                                                                (t_sum/episode_length)*100

    return s1_percent, s2_percent, c1_percent, c2_percent, t_percent


def print_final_stats(number_of_episodes, s1_stat, s2_stat, c1_stat, c2_stat, t_stat):

    print(f'stats for {number_of_episodes} episodes: ')
    print()

    import statistics
    s1_avg = statistics.mean(s1_stat)
    s1_std = statistics.stdev(s1_stat)
    print('safety RSS distnace: ')
    print(f'mean: {s1_avg}      std:{s1_std}')
    print('_______________________________________')

    s2_avg = statistics.mean(s2_stat)
    s2_std = statistics.stdev(s2_stat)
    print('safety hard speed limit: ')
    print(f'mean: {s2_avg}      std:{s2_std}')
    print('_______________________________________')

    c1_avg = statistics.mean(c1_stat)
    c1_std = statistics.stdev(c1_stat)
    print('comfort no faster than left: ')
    print(f'mean: {c1_avg}      std:{c1_std}')
    print('_______________________________________')

    c2_avg = statistics.mean(c2_stat)
    c2_std = statistics.stdev(c2_stat)
    print('comfort soft speed limit: ')
    print(f'mean: {c2_avg}      std:{c2_std}')
    print('_______________________________________')

    t_avg = statistics.mean(t_stat)
    t_std = statistics.stdev(t_stat)
    print('target reach destination: ')
    print(f'mean: {t_avg}      std:{t_std}')
    print('_______________________________________')




LOGDIR = f"logs/highway-adex-hprs-v0/PPO_sparse_reward_1662475900.323082"


if __name__ == '__main__':

    env = setup_env()
    env, model = setup_model(env)

    number_of_episodes = 50

    s1_stat, s2_stat, c1_stat, c2_stat, t_stat = [], [], [], [], []

    for i in range(number_of_episodes):
        # print(i)
        done = False
        obs = env.reset()

        s1_list, s2_list, c1_list, c2_list, t_list = [], [], [], [], []
        steps = 0
        one_more = False
        while not done or one_more:
            steps += 1
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            s1, s2, c1, c2, t = get_metrics_for_one_timestep(obs, info)
            s1_list.append(s1)
            s2_list.append(s2)
            c1_list.append(c1)
            c2_list.append(c2)
            t_list.append(t)

            #env.render()
            #time.sleep(0.1)

            if done and not one_more:
                one_more = True
            elif done and one_more:
                one_more = False


        s1_percent, s2_percent, c1_percent, c2_percent, t_percent = get_metrics_for_one_episode(s1_list, s2_list,
                                                                                                c1_list, c2_list, t_list)
        s1_stat.append(s1_percent)
        s2_stat.append(s2_percent)
        c1_stat.append(c1_percent)
        c2_stat.append(c2_percent)
        t_stat.append(t_percent)

    env.close()

    print_final_stats(number_of_episodes, s1_stat, s2_stat, c1_stat, c2_stat, t_stat)