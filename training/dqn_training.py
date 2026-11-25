"""Train one DQN run.   uv run python -m training.dqn_training --steps 100000 --set lr=1e-3 gamma=0.95"""
from ._cli import run

if __name__ == "__main__":
    run(["dqn"], "dqn", "Train a DQN agent with TorchRL on AgriResilienceEnv")
