import pathlib
import time
import pprint

import gym
from gym.wrappers import RecordVideo, Monitor, FlattenObservation
from stable_baselines3.common.callbacks import CheckpointCallback

from rewards import reward_register, hprs_reward_register

from stable_baselines3 import DQN, PPO

from rewards.reward_wrapper import HighwayRewardWrapper, HPRSHighwayRewardWrapper
from training.callbacks import VideoRecorderCallback
from wrappers import HPRSFilterObservationWrapper

TRAIN = True
TRAINING_STEPS = 1e6

if __name__ == '__main__':
    # Create the environment
    env_name = 'highway-hprs-baseline-v0'  # duration : 40
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
    env = HPRSFilterObservationWrapper(env, ['observation'])
    # env = FlattenObservation(env)
    obs = env.reset()
    print(env.config)

    path_name = f"logs/hprs_sparse_comparison/{env_name}/{algo}_sparse_reward_{time.time()}"
    logdir = pathlib.Path(path_name)
    logdir.mkdir(exist_ok=True, parents=True)

    # callback
    checkpoint_callback = CheckpointCallback(
        save_freq=1e4,
        save_path=path_name + "/callback",
        name_prefix="check_points"
    )

    # Create the model
    if algo == "DQN":
        model = DQN('MlpPolicy', env,
                    policy_kwargs=dict(net_arch=[256, 256]),
                    learning_rate=5e-4,
                    buffer_size=15000,
                    learning_starts=200,
                    batch_size=32,
                    gamma=0.8,
                    train_freq=1,
                    gradient_steps=1,
                    target_update_interval=50,
                    verbose=1,
                    tensorboard_log=logdir)
    elif algo == "PPO":
        model = PPO('MlpPolicy', env,
                    policy_kwargs=dict(net_arch=[256, 256]),
                    learning_rate=5e-4,
                    batch_size=32,
                    gamma=0.99,
                    verbose=1,
                    tensorboard_log=logdir)

    # Train the model
    if TRAIN:
        video_cb = VideoRecorderCallback(Monitor(env, logdir / "videos"), render_freq=1e3,
                                         n_eval_episodes=1)
        model.learn(total_timesteps=int(TRAINING_STEPS), callback=[video_cb],
                    eval_env=env, eval_freq=1e4, n_eval_episodes=10, eval_log_path=logdir)

        model.save(f"{logdir}/model")
        del model

    # Run the trained model and record video
    model = PPO.load(f"{logdir}/model", env=env)
    env = RecordVideo(env, video_folder=f"{logdir}/videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    env.configure({"simulation_frequency": 15})  # Higher FPS for rendering

    for videos in range(10):
        done = False
        obs = env.reset()
        while not done:
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, info = env.step(action)
            # Render
            env.render()
    env.close()

