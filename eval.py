import pathlib
import time

import gym
from gym.wrappers import RecordVideo, Monitor, FlattenObservation
from rewards import reward_register, hprs_reward_register


from stable_baselines3 import DQN, PPO

from rewards.reward_wrapper import HighwayRewardWrapper, HPRSHighwayRewardWrapper
from training.callbacks import VideoRecorderCallback
from wrappers import HPRSFilterObservationWrapper

LOGDIR = f"logs/highway-hprs-v0/PPO_hprs_1662668191.051967"


if __name__ == '__main__':
    # Create the environment
    env_name = 'highway-hprs-v0'  # duration : 40
    reward_name = 'hprs'
    algo = 'PPO'
    env = gym.make(env_name)
    config = {
        "observation": {
            "type": "Kinematics",
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": {
                "x": [0, 2000],
                "y": [0, 50],
                "vx": [0, 50],
                "vy": [-10, 10]
            },
            "normalize": False,
            "absolute": True,
        },
        "vehicles_count": 20,
        "lanes_count": 3,
    }
    env.configure(config)
    env.reset()
    reward_fn = hprs_reward_register.make(reward_name, dt=1 / env.config["simulation_frequency"])
    env = HPRSHighwayRewardWrapper(env, reward_fn)
    env = HPRSFilterObservationWrapper(env, ['observation'])
    # env = FlattenObservation(env)
    obs = env.reset()
    print(env.config)

    logdir = pathlib.Path(LOGDIR)
    logdir.mkdir(exist_ok=True, parents=True)

    # Run the trained model and record video
    model = PPO.load(f"{logdir}/model", env=env)
    env = RecordVideo(env, video_folder=f"{logdir}/videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    #env.configure({"simulation_frequency": 15})  # Higher FPS for rendering


    for videos in range(20):
        done = False
        obs = env.reset()
        while not done:
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, info = env.step(action)
            # Render
            env.render()
            time.sleep(0.1)
    env.close()