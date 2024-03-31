# Supergrid

A minimal, vectorized and stateless gridword implemented in pytorchRL

https://github.com/DuaneNielsen/supergrid/assets/11070508/98b7e1ad-6f90-4403-b727-e6d9c3d861d4

Action space: (0, 1, 2, 3) -> N, E, S, W

### Features

  * walls
  * 1 time pickup rewards
  * terminated tiles
  * fully observable state in tensordict format
  * fully observable colored RGB image
  * solves the environment using pytorchRL PPO
  * wandb logging and parameter sweeps

The purpose of this repo is to provide a minimal example to learn from

### Installing

Tested under python 3.11 venv

clone the directory using git

```commandline
git clone https://github.com/duanenielsen/supergrid
```

requirements

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

basic demo
```commandline
python gridworld.py --demo
```

help with parameters
```commandline
pythone gridworld --help
```

log data to wandb
```commandline
python gridworld.py --demo --wandb
```

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

### changing the world

Just look at the gen_params method in gridwold.py
