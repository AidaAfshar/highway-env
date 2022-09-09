import pathlib
import time

import gym
from gym.wrappers import RecordVideo, Monitor, FlattenObservation
import highway_env
from rewards import reward_register


from stable_baselines3 import DQN

from rewards.reward_wrapper import HighwayRewardWrapper
from training.callbacks import VideoRecorderCallback


LOGDIR = f"logs/highway-adex-debug-v0/dqn_purely_adv_1659344723.722615"

if __name__ == '__main__':
    # Create the environment
    env_name = "highway-adex-debug-v0"
    reward_name = "purely_adv"

    env = gym.make(env_name)
    reward_fn = reward_register.make(reward_name, dt=1/env.config["simulation_frequency"])
    env = HighwayRewardWrapper(env, reward_fn)
    env = FlattenObservation(env)
    obs = env.reset()

    logdir = pathlib.Path(LOGDIR)
    logdir.mkdir(exist_ok=True, parents=True)

    # Run the trained model and record video
    model = DQN.load(f"{logdir}/model", env=env)
    env = RecordVideo(env, video_folder=f"{logdir}/videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    # env.configure({"simulation_frequency": 15})  # Higher FPS for rendering

    for videos in range(100):
        done = False
        obs = env.reset()

        frames = []
        while not done:
            # Predict
            agent_obs = obs[kinematics]
            action, _states = model.predict(agent_obs, deterministic=True)
            # Get reward
            obs, reward, done, info = env.step(action)
            frame = env.render(mode="rgb_array")
            frames.append(frame)
            # Render
            #env.render()
            time.sleep(0.1)

        writer = imageio.get_writer(f"video_{time.time()}.mp4", fps=5)
        for frame in frames:
            writer.append(frame)
        writer.close()
    env.close()
