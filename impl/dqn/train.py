from trainer import Trainer
from cartpole import env
from replay_memory import ReplayMemory

memory = ReplayMemory(10000)

trainer = Trainer(input_shape=env.get_screen_shape(),
                  env=env,
                  memory=memory,
                  n_actions=env.action_space.n)

trainer.fit(batch_size=128,
            episodes=50000,
            target_update=100)

env.close()