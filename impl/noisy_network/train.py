import yaml
import sys
import shutil
from trainer import Trainer
from pathlib import Path
sys.path.append('../../')
from envs import CartPoleEnv
from model import CNN, MLP
from replay_memory import SimpleMemory, PrioritizedMemory


yml_path = sys.argv[1]
with open(yml_path) as f:
    config = yaml.load(f)
logdir = Path(config['logdir'])
logdir.mkdir(parents=True, exist_ok=True)
shutil.copy(yml_path, logdir/'config.yml')

env = CartPoleEnv(**config['env_params'])

memory = PrioritizedMemory(**config['replay_memory'])

if config['env_params']['state_mode'] == 'image':
    policy_model = CNN(env.action_space.n)
    target_model = CNN(env.action_space.n)
else:
    policy_model = MLP(env.action_space.n)
    target_model = MLP(env.action_space.n)

trainer = Trainer(input_shape=env.get_state_shape(),
                  policy_model=policy_model,
                  target_model=target_model,
                  env=env,
                  memory=memory,
                  n_actions=env.action_space.n,
                  logdir=logdir)

trainer.fit(**config['fit_params'])

env.close()