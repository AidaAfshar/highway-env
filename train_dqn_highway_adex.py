import pathlib
import time

import gym
from gym.wrappers import RecordVideo, Monitor, FlattenObservation
import highway_env
from rewards import reward_register


from stable_baselines3 import DQN

from rewards.reward_wrapper import HighwayRewardWrapper
from training.callbacks import VideoRecorderCallback

TRAIN = True
TRAINING_STEPS = 5e5

if __name__ == '__main__':
    # Create the environment
    env_name = "highway-adex-debug-v0"
    reward_name = "hprs_cutin"

    env = gym.make(env_name)
    reward_fn = reward_register.make(reward_name, env=env)
    env = HighwayRewardWrapper(env, reward_fn)
    env = FlattenObservation(env)
    obs = env.reset()

    logdir = pathlib.Path(f"logs/{env_name}/dqn_{reward_name}_{time.time()}")
    logdir.mkdir(exist_ok=True, parents=True)

    # Create the model
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

    # Train the model
    if TRAIN:
        video_cb = VideoRecorderCallback(Monitor(env, logdir / "videos"), render_freq=1e3,
                                         n_eval_episodes=1)
        model.learn(total_timesteps=int(TRAINING_STEPS), callback=[video_cb])
        model.save(f"{logdir}/model")
        del model

    # Run the trained model and record video
    model = DQN.load(f"{logdir}/model", env=env)
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
