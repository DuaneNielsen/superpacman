# superpacman

A stateless, vectorized implementation of pacman, implemented in torchRL




https://github.com/DuaneNielsen/superpacman/assets/11070508/6ede32de-1917-4702-bd38-0a5b978302d8



After implementing a number of RL algorithms, I wanted an environment that allowed me to get test results fast, and was also satisfying to solve.

Pacman, with gigantic batch is a non-trivial, yet data-rich environment for developing RL algorithms.

The constant stream of rewards generated by pacman eating means you have plenty reward signal to work with.

As training progresses, ghosts are released, and rewards become more sparse, naturally challenging your agent in a very satsifying way!


Action space: (0, 1, 2, 3) -> N, E, S, W

### Features

  * it's stateless!  Supports Monte-carlo tree search
  * its vectorized by default, run batch sizes of 2048, 4096.. whatever your GPU can handle
  * outputs trajectories of pytorch tensors, ready to use
  * outputs highly detailed 11 channel image (walls, food, energizers, 4 ghosts, players, etc)
  * ghost AI is implemented as per original game
  * can output absolute, egocentric, and partial egocentric observations
  * can output pixel observations (also absolute or partial egocentric)
  * comes with a fully working and tuned PPO implementation, so you can verify it works!
  * manual interface, so you can play the game yourself


### Installing

Tested under python 3.11 venv

If you intend to run on gpu, install the gpu version of pytorch

```commandline
pip install superpacman 
```


### Running

basic demo, using pretrained agent
```commandline
superpacman train --enjoy_checkpoint demo_checkpoint.pt
```
after running, check the logs/superpacman/videos directory

manual mode - play the environment yourself
```commandline
superpacman play
```

manual mode - play the environment with partial observation
```commandline
superpacman play --partial_size 4
```

help with parameters
```commandline
superpacman --help
```

### Training and Logging

its possible to reproduce the PPO training baselines in your favourite ml tool

```commandline
superpacman train
```

default is csv, but wandb, mlflow and tensorboard are also supported

### wandb integration
log training data to wandb
```commandline
pip install wandb
superpacman train --logger wandb
```

### wandb parameter sweeps

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

### mlflow integration

```commandline
pip install mlflow
superpacman train --logger mlflow
mlflow ui --backend-store-uri file://$PWD/mlflow
```

you may need to modify the file uri for your OS

### tensorboard integration

```commandline
pip install tensorboard
superpacman train --logger tensorboard
tensorboard --logdir tensorboard
```
