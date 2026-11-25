"""Train one policy-gradient run (REINFORCE, PPO or A2C).
    uv run python -m training.pg_training --algo ppo --steps 100000 --set clip_eps=0.1"""
from ._cli import run

if __name__ == "__main__":
    run(["reinforce", "ppo", "a2c"], "ppo", "Train a policy-gradient agent with TorchRL on AgriResilienceEnv")
