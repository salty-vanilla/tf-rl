import yaml
import sys
import shutil
from trainer import Trainer
from cartpole import env
from replay_memory import ReplayMemory
from pathlib import Path


yml_path = sys.argv[1]
with open(yml_path) as f:
    config = yaml.load(f)
logdir = Path(config['logdir'])
logdir.mkdir(parents=True, exist_ok=True)
shutil.copy(yml_path, logdir/'config.yml')

memory = ReplayMemory(**config['replay_memory'])

trainer = Trainer(input_shape=env.get_screen_shape(),
                  env=env,
                  memory=memory,
                  n_actions=env.action_space.n,
                  logdir=logdir)

trainer.fit(**config['fit_params'])

env.close()