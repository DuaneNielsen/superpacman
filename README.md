# superpacman

A stateless, vectorized implementation of pacman, implemented in pytorchRL




https://github.com/DuaneNielsen/superpacman/assets/11070508/6ede32de-1917-4702-bd38-0a5b978302d8





Action space: (0, 1, 2, 3) -> N, E, S, W

### Features

  * it's stateless!  Supports Monte-carlo tree search
  * its vectorized by default, run batch sizes of 2048, 4096.. whatever your GPU can handle
  * outputs trajectories of pytorch tensors, ready to use
  * outputs highly detailed 11 layer channel (walls, food, energizers, 4 ghosts, players, etc)
  * ghost AI is implemented as per original game
  * can output absolute, egocentric, and partial egocentric observations
  * can output pixel observations (also absolute or partial egocentric)
  * comes with a fully working and tuned PPO implementation, so you can verify it works!
  * manual interface, so you can play the game yourself

After implementing a number of RL algorithms, I wanted an environment that allowed me to get test results fast, and was also satisfying to solve.

Pacman, with gigantic batch is a non-trivial, yet data-rich environment for developing RL algorithms.

The constant stream of rewards generated by pacman eating means you have plenty reward signal to work with.

As training progresses, ghosts are released, and rewards become more sparse, naturally challenging your agent in a very satsifying way!


### Installing

Tested under python 3.11 venv

clone the directory using git

```commandline
git clone https://github.com/duanenielsen/superpacman
```

minimal requirements

```commandline
pip install torch
pip install torchrl
pip install torchvision
pip install matplotlib
pip install tqdm
pip install av
```

optional requirements
```commandline
pip install wandb
```

### Running

basic demo, using pretrained agent
```commandline
python superpacman.py --demo
```
after running, check the logs/superpacman/videos directory

manual mode - play the environment yourself
```commandline
python manual.py
```

manual mode - play the environment with partial observation
```commandline
python manual.py --partial_size 4
```

help with parameters
```commandline
python gridworld --help
```

log data to wandb
```commandline
python gridworld.py --demo --wandb
```

### Parameter sweeps

run a parameter sweep using wandb
```commandline
wandb sweep sweep.yaml
```

you will see output like below..
```commandline
wandb: Creating sweep from: sweep.yaml
wandb: Creating sweep with ID: t1qjy41y
wandb: View sweep at: https://wandb.ai/duanenielsen/supergrid/sweeps/t1qjy41y
wandb: Run sweep agent with: wandb agent duanenielsen/supergrid/t1qjy41y
```

run generated agent command
```commandline
wandb agent duanenielsen/supergrid/t1qjy41y
```
