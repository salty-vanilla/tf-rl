
import os
import sys
import pathlib
from collections import namedtuple

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent))

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

from simple_memory import SimpleMemory
from prioritzed_memory import PrioritizedMemory
