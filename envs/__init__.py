import os
import sys
import pathlib

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent))

from cartpole import CartPoleEnv