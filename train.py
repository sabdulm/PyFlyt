import gymnasium as gym
import PyFlyt.gym_envs 
from PyFlyt.gym_envs import FlattenWaypointEnv
from stable_baselines3 import PPO, SAC  # Added SAC
from stable_baselines3.common.env_util import make_vec_env
import wandb
from wandb.integration.sb3 import WandbCallback
import argparse
import os

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="PyFlyt RL Training Script")
parser.add_argument("--algo", type=str, choices=["PPO", "SAC"], default="PPO", help="RL Algorithm to use")
parser.add_argument("--steps", type=int, default=2000000, help="Total training timesteps")
parser.add_argument("--save_path", type=str, default="fixedwing_agent", help="Path to save the trained model")
parser.add_argument("--load_path", type=str, default="fixedwing_agent", help="Path to load a model for testing")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments for training")
parser.add_argument("--test", action="store_true", help="Run in test mode (visualize) instead of training")
args = parser.parse_args()

def train_model():
    """Trains the selected RL model on the PyFlyt Fixedwing environment."""
    
    print(f"\n=== STARTING TRAINING ===")
    print(f"Algorithm:  {args.algo}")
    print(f"Timesteps:  {args.steps}")
    print(f"Envs:       {args.num_envs}")
    print(f"Save Path:  {args.save_path}")

    # Initialize WandB
    wandb.init(
        project="pyflyt-training",
        name=f"{args.algo}_FIXEDWING_FLATTENED",
        sync_tensorboard=True,
        monitor_gym=False,
    )

    # Create Vectorized Environment with Flatten Wrapper
    train_env = make_vec_env(
        "PyFlyt/Fixedwing-Waypoints-v4", 
        n_envs=args.num_envs, 
        wrapper_class=FlattenWaypointEnv,   
        wrapper_kwargs={"context_length": 2}
    )
    
    # Select the Model Class
    ModelClass = SAC if args.algo == "SAC" else PPO

    # Initialize Model
    # Note: SAC is off-policy and memory intensive, so we set a buffer_size that fits in RAM
    model_kwargs = {"verbose": 1, "tensorboard_log": "./ppo_tensorboard/", "device": "cuda"}
    if args.algo == "SAC":
        model_kwargs["buffer_size"] = 1_000_000  # Default buffer size for SAC
    
    model = ModelClass("MlpPolicy", train_env, **model_kwargs)

    # Train
    model.learn(total_timesteps=args.steps, callback=WandbCallback())
    
    # Save
    model.save(args.save_path)
    print(f"Training finished. Model saved to {args.save_path}.zip")
    
    train_env.close()
    wandb.finish()

def test_model():
    """Tests the trained model with rendering."""
    print(f"\n=== TESTING MODEL ===")
    print(f"Algorithm: {args.algo}")
    print(f"Loading:   {args.load_path}")

    if not os.path.exists(f"{args.load_path}.zip"):
        print(f"Error: Model file '{args.load_path}.zip' not found.")
        return

    # Create Single Environment for Human Viewing
    env = gym.make("PyFlyt/Fixedwing-Waypoints-v4", render_mode="human")
    env = FlattenWaypointEnv(env, context_length=2) 
    
    # Load Model
    ModelClass = SAC if args.algo == "SAC" else PPO
    model = ModelClass.load(args.load_path)

    for episode in range(10):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += float(reward)
            done = terminated or truncated

        print(f"Episode {episode + 1} - Total Reward: {episode_reward:.2f}")

    env.close()

if __name__ == "__main__":
    if args.test:
        test_model()
    else:
        train_model()