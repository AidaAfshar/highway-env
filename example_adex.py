import gym
import highway_env
from gym.wrappers import RecordVideo


if __name__ == '__main__':
    # Create the environment
    env = gym.make("highway-adex-debug-v0")
    obs = env.reset()

    # Run the trained model and record video
    env = RecordVideo(env, video_folder="videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    env.configure({"simulation_frequency": 15})  # Higher FPS for rendering

    for videos in range(10):
        tot_rewards = 0
        done = False
        obs = env.reset()
        while not done:
            # Predict
            action = env.action_space.sample()
            # Get reward
            obs, reward, done, info = env.step(action)
            # Render
            env.render()
            print(reward)
            tot_rewards += reward
        print(f"tot reward: {tot_rewards}\n")
        input()
    env.close()