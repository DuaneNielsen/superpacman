import torch
from torch import tensor
from tensordict import TensorDict
from torchrl.data import CompositeSpec, BoundedTensorSpec, UnboundedDiscreteTensorSpec, \
    DiscreteTensorSpec, \
    UnboundedContinuousTensorSpec
from torchrl.envs import (
    EnvBase,
    StepCounter,
    RewardSum,
    TransformedEnv,
    Resize,
    ToTensorImage,
    PermuteTransform
)
from torchrl.record import VideoRecorder, CSVLogger
from torchrl.envs.transforms.transforms import _apply_to_composite, ObservationTransform
from typing import Any, Dict, List, Optional, OrderedDict, Sequence, Tuple, Union
from enum import IntEnum
from math import prod

"""
A minimal stateless vectorized gridworld in pytorch rl
Action space: (0, 1, 2, 3) -> N, E, S, W

Features

  walls
  1 time pickup rewards or penalties
  terminated tiles
  outputs a fully observable RGB image 

look at the gen_params function to setup the world

PPO training code and visualization provided
"""


class Actions(IntEnum):
    N = 0,
    E = 1,
    S = 2,
    W = 3


class Ghost(IntEnum):
    BLINKY = 0,
    PINKY = 1,
    INKY = 2,
    CLAUDE = 3


class GhostMode(IntEnum):
    WAIT = 0,
    SCATTER = 1,
    CHASE = 2
    FRIGHTENED = 3


# N/S is reversed as y-axis in images is reversed
action_vec = torch.stack(
    [
        tensor([-1, 0]),  # N
        tensor([0, 1]),  # E
        tensor([1, 0]),  # S
        tensor([0, -1])  # W
    ]
)

tile_keys = ['player_tiles', 'wall_tiles', 'reward_tiles', 'energizer_tiles', 'pinky_tiles', 'blinky_tiles',
             'inky_tiles', 'claude_tiles', 'frightened_tiles']
egocentric_tile_keys = [f'c_{key}' for key in tile_keys]


def hex_to_rgb(hex_color):
    r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return tensor([r, g, b], dtype=torch.uint8)


# colors for RGB image
yellow = tensor([255, 255, 0], dtype=torch.uint8)
red = tensor([255, 0, 0], dtype=torch.uint8)
pink = tensor([255, 0, 255], dtype=torch.uint8)
violet = tensor([226, 43, 138], dtype=torch.uint8)
white = tensor([255, 255, 255], dtype=torch.uint8)
gray = tensor([128, 128, 128], dtype=torch.uint8)
wall_color = hex_to_rgb("6E7C91")
food = hex_to_rgb("91836E")
energizer_color = hex_to_rgb("FF6B6B")
blinky_color = tensor([255, 0, 0], dtype=torch.uint8)
pinky_color = tensor([255, 180, 255], dtype=torch.uint8)
inky_color = tensor([0, 255, 255], dtype=torch.uint8)
claude_color = tensor([255, 184, 81], dtype=torch.uint8)
blue = hex_to_rgb('0066FF')


def pos_to_grid(pos, H, W, device='cpu', dtype=torch.float32):
    """
    Converts positions to grid where 1 indicates a position
    :param pos: N, ..., 2 tensor of grid positions (x = H, y = W) or 2 tensor
    :param H: height
    :param W: width
    :param: device: device
    :param: dtype: type of tensor
    :return: N, H, W tensor or single H, W tensor
    """

    if len(pos.shape) == 1:
        grid = torch.zeros((H, W), dtype=dtype, device=device)
        grid[pos[0], pos[1]] = 1.
        return grid

    N = pos.shape[0]
    batch_shape = tuple([N] + [1] * (len(pos.shape) - 1))
    batch_range = torch.arange(N, device=device).view(*batch_shape)
    grid = torch.zeros((N, H, W), dtype=dtype, device=device)
    grid[batch_range, pos[..., 0].unsqueeze(-1), pos[..., 1].unsqueeze(-1)] = 1
    return grid


def _step(state):
    device = state.device
    N, H, W = state['wall_tiles'].shape
    _, G, _ = state['ghost_pos'].shape

    wall_tiles = state['wall_tiles']
    batch_range = torch.arange(N, device=device)
    ghost_range = torch.arange(G, device=device)
    dtype = state['wall_tiles'].dtype
    action = state['action'].squeeze(-1)
    direction = action_vec.to(device)
    big_distance = H ** 2 + W ** 2
    ghost_pos = state['ghost_pos']
    player_pos = state['player_pos']
    energized = state['energized_t'] > 0
    ghost_wait_t = state['ghost_wait_t']
    t = state['t']

    # 4 ghost candidate positions, N, G, A, xy
    candidate_tile_pos = direction.view(1, 1, 4, 2) + ghost_pos.view(N, G, 1, 2)
    candidate_tile_pos = candidate_tile_pos % H
    is_wall = wall_tiles[batch_range.view(N, 1, 1), candidate_tile_pos[..., 0], candidate_tile_pos[..., 1]]

    # scatter tiles are target tiles that cause the ghosts to go to their respective corners
    scatter_targets = tensor([
        [-3, 19],
        [-3, 1],
        [21, 20],
        [21, 0]
    ], dtype=torch.int64, device=device).expand(N, G, 2)

    # this is the core of the AI, each ghost has a different target tile
    chase_targets = torch.zeros(N, G, 2, dtype=torch.int64, device=device)
    chase_targets[batch_range, Ghost.BLINKY] = player_pos
    chase_targets[batch_range, Ghost.PINKY] = player_pos + direction[action] * 4
    pacman_pred_2 = player_pos + direction[action] * 2
    chase_targets[batch_range, Ghost.INKY] = (pacman_pred_2 + ghost_pos[batch_range, Ghost.BLINKY]) // 2
    claude_pacman_dist = (player_pos - ghost_pos[batch_range, Ghost.CLAUDE]) ** 2
    chase_targets[batch_range, Ghost.CLAUDE] = torch.where(claude_pacman_dist < 64,
                                                           scatter_targets[batch_range, Ghost.CLAUDE], player_pos)

    # on a schedule, set the target tile to the ghosts scatter tile for 7 turns
    scatter = (t % 27 <= 7) & (t < 27 * 3)
    targets = torch.where(scatter.view(N, 1, 1), scatter_targets, chase_targets)

    # get the distance from each candidate move to the target tile
    # later the ghost will move to tile with the smallest value
    candidate_tile_dist = torch.sum((targets.view(N, G, 1, 2) - candidate_tile_pos) ** 2, dim=-1)

    # if pacman is energized, ghosts are frightened, and go in random directions
    random_tile_weights = torch.randint_like(candidate_tile_dist, low=0, high=big_distance - 1)
    candidate_tile_dist = torch.where(energized.view(N, 1, 1), random_tile_weights, candidate_tile_dist)

    # add a big distance if the tile is a wall so the ghost won't consider it
    candidate_tile_dist = candidate_tile_dist + is_wall * big_distance

    # ghosts can't reverse the direction of travel, so add a big distance to the tile that would reverse direction
    rev_direction = (state['ghost_dir'] + 2) % 4
    candidate_tile_dist[batch_range.view(N, 1), ghost_range.view(1, G), rev_direction] = big_distance

    # now finally, we can pick the ghosts direction, and start to calculate his next position
    distance, ghost_direction = torch.min(candidate_tile_dist, dim=-1)
    next_ghost_pos = ghost_pos + direction[ghost_direction]
    next_ghost_pos = next_ghost_pos % H

    # frightened ghosts move at half speed
    skip_turn = (t % 2 == 0) & energized
    next_ghost_pos = torch.where(skip_turn.view(N, 1, 1), ghost_pos, next_ghost_pos)

    # the ghosts will be in the house if eaten or until a certain number of reward tiles are eaten
    wait_pos = tensor([[7, 10], [9, 10], [9, 9], [9, 11]], dtype=torch.int64, device=device).expand(N, G, 2)
    start_pos = tensor([[7, 10], [7, 10], [7, 10], [7, 10]], dtype=torch.int64, device=device).expand(N, G, 2)
    dots_remaining = state['reward_tiles'].sum(-1).sum(-1)
    ghost_wait_t[batch_range, Ghost.INKY] = (dots_remaining != (176 - 30)) * ghost_wait_t[batch_range, Ghost.INKY]
    ghost_wait_t[batch_range, Ghost.CLAUDE] = (dots_remaining != (176 - 60)) * ghost_wait_t[batch_range, Ghost.CLAUDE]
    next_ghost_pos = torch.where(ghost_wait_t.view(N, G, 1) > 0, wait_pos, next_ghost_pos)
    next_ghost_pos = torch.where(ghost_wait_t.view(N, G, 1) == 0, start_pos, next_ghost_pos)

    # write the grids for each ghost
    blinky_tiles = pos_to_grid(next_ghost_pos[:, Ghost.BLINKY], H, W, device=device, dtype=dtype)
    pinky_tiles = pos_to_grid(next_ghost_pos[:, Ghost.PINKY], H, W, device=device, dtype=dtype)
    inky_tiles = pos_to_grid(next_ghost_pos[:, Ghost.INKY], H, W, device=device, dtype=dtype)
    claude_tiles = pos_to_grid(next_ghost_pos[:, Ghost.CLAUDE], H, W, device=device, dtype=dtype)

    # now we can move the player position checking for collisions
    next_player_pos = player_pos + direction[action]
    next_player_pos = next_player_pos % H
    next_player_grid = pos_to_grid(next_player_pos, H, W, device=device, dtype=torch.bool)
    collide_wall = torch.logical_and(next_player_grid, state['wall_tiles'] == 1).any(-1).any(-1)
    player_pos = torch.where(collide_wall[..., None], player_pos, next_player_pos)
    player_tiles = pos_to_grid(player_pos, H, W, device=device, dtype=dtype)

    # pickup any rewards
    reward = state['reward_tiles'][batch_range, player_pos[:, 0], player_pos[:, 1]]
    state['reward_tiles'][batch_range, player_pos[:, 0], player_pos[:, 1]] = 0.

    # energized state
    ate_energizer = state['energizer_tiles'][batch_range, player_pos[:, 0], player_pos[:, 1]]
    state['energizer_tiles'][batch_range, player_pos[:, 0], player_pos[:, 1]] = 0.
    state['energized_t'][ate_energizer == 1] = 8
    state['energized_t'][state['energized_t'] < 0] = 0
    energized = state['energized_t'] > 0
    frightened_tiles = pos_to_grid(next_ghost_pos, H, W, device=device, dtype=dtype)
    frightened_tiles = frightened_tiles * (state['energized_t'] / 8).view(N, 1, 1)


    # check collisions
    collide = torch.logical_or((ghost_pos == player_pos.view(N, 1, 2)).all(-1),
                               (next_ghost_pos == player_pos.view(N, 1, 2)).all(-1))

    # terminate if we hit a ghost when not energized, or we got all the rewards
    terminated = collide.any(-1) & ~ energized.squeeze() | (state['reward_tiles'].sum(-1).sum(-1) == 0)
    ghost_wait_t[collide & energized] = 7
    reward += (collide & energized).any(-1) * 4

    out = {
        't': t + 1,
        'energized_t': state['energized_t'] - 1,
        'player_pos': player_pos,
        'player_tiles': player_tiles,
        'ghost_pos': next_ghost_pos,
        'ghost_dir': ghost_direction,
        "ghost_wait_t": ghost_wait_t - 1,
        'blinky_tiles': blinky_tiles,
        'pinky_tiles': pinky_tiles,
        'inky_tiles': inky_tiles,
        'claude_tiles': claude_tiles,
        'frightened_tiles': frightened_tiles,
        'wall_tiles': state['wall_tiles'],
        'reward_tiles': state['reward_tiles'],
        'energizer_tiles': state['energizer_tiles'],
        'reward': reward.unsqueeze(-1),
        'terminated': terminated.unsqueeze(-1)
    }
    return TensorDict(out, state.shape)


def _reset(self, tensordict=None):
    batch_size = tensordict.shape if tensordict is not None else self.batch_size
    if tensordict is None or tensordict.is_empty():
        tensordict = self.gen_params(batch_size).to(self.device)
    if '_reset' in tensordict.keys():
        reset_state = self.gen_params(batch_size).to(self.device)
        reset_mask = tensordict['_reset'].squeeze(-1)
        for key in reset_state.keys():
            tensordict[key][reset_mask] = reset_state[key][reset_mask]
    return tensordict


def gen_params(batch_size=None):
    """

    To change the layout of the gridworld, change these parameters

    walls: 1 indicates the position of a wall.  The boundary grid cells must have a wall.
    rewards: The amount of reward for entering the tile.
        Rewards are only received the first time the agent enters the tile
    terminal_states: Indicated by 1, when this tile is entered, the terminated flag is set true

    :param batch_size: the number of environments to run simultaneously
    :return: a batch_size tensordict, with the following entries

       "player_pos": N, 2 tensor indices that correspond to the players location
       "player_tiles": N, 21, 21 tensor, with a single tile set to 1 that indicates player position
       "wall_tiles": N, 21, 21 tensor, 1 indicates wall
       "reward_tiles": N, 21, 21 tensor, rewards remaining in environment
       "energizer_tiles": N, 21, 21 tensor, episode will terminate when tile with value True is entered

    """
    walls = tensor([
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    ], dtype=torch.float32)

    rewards = tensor([
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
        [0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0.],
        [0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
        [0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0.],
        [0., 0., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
        [0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0.],
        [0., 0., 1., 1., 0., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 0., 1., 1., 0., 0.],
        [0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0.],
        [0., 0., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    ], dtype=torch.float32)

    terminal_states = tensor([
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    ], dtype=torch.bool)

    H, W = walls.shape
    player_pos = tensor([15, 10], dtype=torch.int64)
    ghost_pos = tensor([[7, 10], [9, 10], [9, 9], [9, 11]], dtype=torch.int64)
    ghost_dir = tensor([Actions.W] * len(Ghost), dtype=torch.int64)
    ghost_wait_t = tensor([-1, 1, -1, -1], dtype=torch.int64)
    player_tiles = pos_to_grid(player_pos, H, W, dtype=walls.dtype)
    blinky_tiles = pos_to_grid(ghost_pos[Ghost.BLINKY], H, W, dtype=walls.dtype)
    pinky_tiles = pos_to_grid(ghost_pos[Ghost.PINKY], H, W, dtype=walls.dtype)
    inky_tiles = pos_to_grid(ghost_pos[Ghost.INKY], H, W, dtype=walls.dtype)
    claude_tiles = pos_to_grid(ghost_pos[Ghost.CLAUDE], H, W, dtype=walls.dtype)
    frightened_tiles = torch.zeros_like(walls)
    t = tensor([0], dtype=torch.int64)
    energized_t = tensor([0], dtype=torch.int64)

    state = {
        "t": t,
        "energized_t": energized_t,
        "player_pos": player_pos,
        "player_tiles": player_tiles,
        "ghost_pos": ghost_pos,
        "ghost_dir": ghost_dir,
        "ghost_wait_t": ghost_wait_t,
        "blinky_tiles": blinky_tiles,
        "pinky_tiles": pinky_tiles,
        "inky_tiles": inky_tiles,
        "claude_tiles": claude_tiles,
        "frightened_tiles": frightened_tiles,
        "wall_tiles": walls,
        "reward_tiles": rewards,
        "energizer_tiles": terminal_states
    }

    td = TensorDict(state, batch_size=[])

    if batch_size:
        td = td.expand(batch_size).contiguous()
    return td


def _make_spec(self, td_params):
    batch_size = td_params.shape
    self.observation_spec = CompositeSpec(
        t=UnboundedDiscreteTensorSpec(
            shape=torch.Size((*batch_size, 1)),
            dtype=torch.int64
        ),
        energized_t=UnboundedDiscreteTensorSpec(
            shape=torch.Size((*batch_size, 1)),
            dtype=torch.int64
        ),
        player_tiles=BoundedTensorSpec(
            minimum=0,
            maximum=1,
            shape=torch.Size(td_params['player_tiles'].shape),
            dtype=torch.float32,
        ),
        blinky_tiles=BoundedTensorSpec(
            minimum=0,
            maximum=1,
            shape=torch.Size(td_params['blinky_tiles'].shape),
            dtype=torch.float32,
        ),
        pinky_tiles=BoundedTensorSpec(
            minimum=0,
            maximum=1,
            shape=torch.Size(td_params['pinky_tiles'].shape),
            dtype=torch.float32,
        ),
        inky_tiles=BoundedTensorSpec(
            minimum=0,
            maximum=1,
            shape=torch.Size(td_params['inky_tiles'].shape),
            dtype=torch.float32,
        ),
        claude_tiles=BoundedTensorSpec(
            minimum=0,
            maximum=1,
            shape=torch.Size(td_params['claude_tiles'].shape),
            dtype=torch.float32,
        ),
        wall_tiles=BoundedTensorSpec(
            minimum=0,
            maximum=1,
            shape=torch.Size(td_params['wall_tiles'].shape),
            dtype=torch.float32,
        ),
        reward_tiles=UnboundedContinuousTensorSpec(
            shape=torch.Size(td_params['reward_tiles'].shape),
            dtype=torch.float32,
        ),
        energizer_tiles=BoundedTensorSpec(
            minimum=0,
            maximum=1,
            shape=torch.Size(td_params['energizer_tiles'].shape),
            dtype=torch.bool,
        ),
        frightened_tiles=BoundedTensorSpec(
            minimum=0,
            maximum=1,
            shape=torch.Size(td_params['frightened_tiles'].shape),
            dtype=torch.float32,
        ),
        player_pos=UnboundedDiscreteTensorSpec(
            shape=torch.Size((*batch_size, 2,)),
            dtype=torch.int64
        ),
        ghost_pos=UnboundedDiscreteTensorSpec(
            shape=torch.Size((*batch_size, len(Ghost), 2)),
            dtype=torch.int64
        ),
        ghost_dir=UnboundedDiscreteTensorSpec(
            shape=torch.Size((*batch_size, len(Ghost))),
            dtype=torch.int64
        ),
        ghost_wait_t=UnboundedDiscreteTensorSpec(
            shape=torch.Size((*batch_size, len(Ghost))),
            dtype=torch.int64
        ),
        shape=torch.Size((*batch_size,))
    )
    self.state_spec = self.observation_spec.clone()
    self.action_spec = DiscreteTensorSpec(4, shape=torch.Size((*batch_size, 1)))
    self.reward_spec = UnboundedContinuousTensorSpec(shape=torch.Size((*batch_size, 1)))


def _set_seed(self, seed: Optional[int]):
    rng = torch.manual_seed(seed)
    self.rng = rng


class SuperPacman(EnvBase):
    metadata = {
        "render_modes": ["human", ""],
        "render_fps": 30
    }
    batch_locked = False

    def __init__(self, td_params=None, device="cpu", batch_size=None):
        if td_params is None:
            td_params = self.gen_params(batch_size)
        super().__init__(device=device, batch_size=batch_size)
        self._make_spec(td_params)

    gen_params = staticmethod(gen_params)
    _make_spec = _make_spec
    _reset = _reset
    _step = staticmethod(_step)
    _set_seed = _set_seed


class CenterPlayerTransform(ObservationTransform):
    def __init__(self, patch_radius=2):
        super().__init__(
            in_keys=tile_keys,
            out_keys=[f'c_{key}' for key in tile_keys])
        self.size = patch_radius

    def forward(self, tensordict):
        return self._call(tensordict)

    def _reset(self, tensordict, tensordict_reset):
        return self._call(tensordict_reset)

    def _call(self, td):
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            N, H, W = td[in_key].shape
            device = td[in_key].device
            max_x, max_y = 2 * self.size + 1, 2 * self.size + 1

            x, y = torch.meshgrid(torch.arange(max_x, device=device), torch.arange(max_y, device=device), indexing='ij')
            grid = torch.stack([x, y], 0).view(1, 2, max_x, max_y)
            player_coords = td['player_pos'].view(N, 2, 1, 1)
            indexes = player_coords + grid - self.size
            indexes = indexes % H
            batch_range = torch.arange(N, device=device)
            td[out_key] = td[in_key][batch_range.view(N, 1), indexes[:, 0].flatten(-2), indexes[:, 1].flatten(-2)].view(
                N, max_x, max_y)
        return td

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        N, H, W = observation_spec.shape
        return BoundedTensorSpec(
            minimum=0,
            maximum=255,
            shape=torch.Size((N, self.size * 2 + 1, self.size * 2 + 1)),
            dtype=observation_spec.dtype,
            device=observation_spec.device
        )


class RGBPartialObsTransform(ObservationTransform):
    """
    Converts the state to a N, 3, H, W uint8 image tensor
    Adds it to the tensordict under the key [image]
    """

    def __init__(self):
        super().__init__(in_keys=['c_wall_tiles'], out_keys=['ego_pixels'])

    def forward(self, tensordict):
        return self._call(tensordict)

    def _reset(self, tensordict, tensordict_reset):
        return self._call(tensordict_reset)

    def _call(self, td):
        walls = td['c_wall_tiles']
        device = walls.device
        shape = *walls.shape, 3
        center_x, center_y = shape[1] // 2, shape[2] // 2

        td['ego_pixels'] = torch.zeros(shape, dtype=torch.uint8, device=device)
        td['ego_pixels'][td['c_wall_tiles'] == 1] = wall_color.to(device)
        td['ego_pixels'][td['c_reward_tiles'] > 0] = food.to(device)
        td['ego_pixels'][td['c_reward_tiles'] < 0] = red.to(device)
        td['ego_pixels'][td['c_energizer_tiles'] == 1] = energizer_color.to(device)
        td['ego_pixels'][:, center_x, center_y] = yellow.to(device)
        td['ego_pixels'][td['c_blinky_tiles'] == 1] = blinky_color.to(device)
        td['ego_pixels'][td['c_pinky_tiles'] == 1] = pinky_color.to(device)
        td['ego_pixels'][td['c_inky_tiles'] == 1] = inky_color.to(device)
        td['ego_pixels'][td['c_claude_tiles'] == 1] = claude_color.to(device)
        td['ego_pixels'][td['c_frightened_tiles'] > 0] = blue.to(device)
        return td

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        N, H, W = observation_spec.shape
        return BoundedTensorSpec(
            minimum=0,
            maximum=255,
            shape=torch.Size((N, H, W, 3)),
            dtype=torch.uint8,
            device=observation_spec.device
        )


class RGBFullObsTransform(ObservationTransform):
    """
    Converts the state to a N, 3, H, W uint8 image tensor
    Adds it to the tensordict under the key [image]
    """

    def __init__(self):
        super().__init__(in_keys=['wall_tiles'], out_keys=['pixels'])

    def forward(self, tensordict):
        return self._call(tensordict)

    def _reset(self, tensordict, tensordict_reset):
        return self._call(tensordict_reset)

    def _call(self, td):
        device = td['wall_tiles'].device
        shape = *td['wall_tiles'].shape, 3
        td['pixels'] = torch.zeros(shape, dtype=torch.uint8, device=device)
        td['pixels'][td['wall_tiles'] == 1] = wall_color.to(device)
        td['pixels'][td['reward_tiles'] > 0] = food.to(device)
        td['pixels'][td['reward_tiles'] < 0] = red.to(device)
        td['pixels'][td['energizer_tiles'] == 1] = energizer_color.to(device)
        td['pixels'][td['player_tiles'] == 1] = yellow.to(device)
        td['pixels'][td['blinky_tiles'] == 1] = blinky_color.to(device)
        td['pixels'][td['pinky_tiles'] == 1] = pinky_color.to(device)
        td['pixels'][td['inky_tiles'] == 1] = inky_color.to(device)
        td['pixels'][td['claude_tiles'] == 1] = claude_color.to(device)
        td['pixels'][td['frightened_tiles'] > 0] = blue.to(device)
        return td

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        N, H, W = observation_spec.shape
        return BoundedTensorSpec(
            minimum=0,
            maximum=255,
            shape=torch.Size((N, H, W, 3)),
            dtype=torch.uint8,
            device=observation_spec.device
        )


class FlatTileTransform(ObservationTransform):
    """
    takes all the input keys and outputs a single flat tensor
    in_keys: tensors to use
    out_keys: name of flat tensor, defautls to flat_obs
    """

    def __init__(self, in_keys, out_key='flat_obs'):
        super().__init__(in_keys=in_keys, out_keys=out_key)

    def forward(self, tensordict):
        return self._call(tensordict)

    def _reset(self, tensordict, tensordict_reset):
        return self._call(tensordict_reset)

    def _call(self, td):
        td[self.out_keys[0]] = torch.cat([td[key].flatten(start_dim=1) for key in self.in_keys], dim=-1)
        return td

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        N, H, W = observation_spec.shape
        S = 0
        for key in self.in_keys:
            S += prod(self.parent.full_observation_spec[key].shape[1:])
        return BoundedTensorSpec(
            minimum=0,
            maximum=1,
            shape=torch.Size((N, S)),
            dtype=observation_spec.dtype,
            device=observation_spec.device
        )


class StackTileTransform(ObservationTransform):
    """
    stacks all the in_keys into an N, C, H, W tensor and outputs it under the out_key
    in_keys: must be N, H, W
    out_key: string, default "image"
    """

    def __init__(self, in_keys, out_key='image'):
        super().__init__(in_keys=in_keys, out_keys=out_key)

    def forward(self, tensordict):
        return self._call(tensordict)

    def _reset(self, tensordict, tensordict_reset):
        return self._call(tensordict_reset)

    def _call(self, td):
        td[self.out_keys[0]] = torch.stack([td[key] for key in self.in_keys], dim=-3)
        return td

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        N, H, W = observation_spec.shape
        C = len(self.in_keys)
        return BoundedTensorSpec(
            minimum=0,
            maximum=1,
            shape=torch.Size((N, C, H, W)),
            dtype=observation_spec.dtype,
            device=observation_spec.device
        )


def has_transform(env, transform_class):
    if isinstance(env, TransformedEnv):
        for transform in env.transform.transforms:
            if isinstance(transform, transform_class):
                return True
    return False


def make_env(env_batch_size, obs_keys=None, device='cpu',
             ego_patch_radius=10, log_video=False,  log_stats=True, seed=None,
             logger=None):

    assert env_batch_size > 1, "sorry, batch size 1 is not supported yet, try batch size 2"

    env = SuperPacman(batch_size=torch.Size([env_batch_size]), device=device)
    env = TransformedEnv(env)

    obs_keys = ["image", "pixels"] if obs_keys is None else obs_keys
    obs_keys = [obs_keys] if isinstance(obs_keys, str) else obs_keys
    obs_keys = set(obs_keys)

    if "flat_obs" in obs_keys:
        env.append_transform(FlatTileTransform(tile_keys))

    if "image" in obs_keys:
        env.append_transform(StackTileTransform(in_keys=tile_keys))

    if "ego_image" in obs_keys:
        env.append_transform(CenterPlayerTransform(patch_radius=ego_patch_radius))
        env.append_transform(StackTileTransform(in_keys=egocentric_tile_keys, out_key='ego_image'))

    if "pixels" in obs_keys:
        env.append_transform(RGBFullObsTransform())

    if "ego_pixel" in obs_keys:
        if not has_transform(env, CenterPlayerTransform):
            env.append_transform(CenterPlayerTransform(patch_radius=ego_patch_radius))
        env.append_transform(RGBPartialObsTransform())

    if log_video:
        if not has_transform(env, RGBFullObsTransform):
            env.append_transform(RGBFullObsTransform())
        env.append_transform(ToTensorImage(from_int=False))
        env.append_transform(Resize(21 * 8, 21 * 8, in_keys=['pixels'], out_keys=['pixels'], interpolation='nearest'))
        env.append_transform(PermuteTransform([-2, -1, -3], in_keys=['pixels'], out_keys=['pixels']))
        if logger is None:
            logger = CSVLogger(exp_name='superpacman', log_dir="../logs", video_fps=3, video_format='mp4')
        recorder = VideoRecorder(logger=logger, tag='pacman', fps=3, skip=1)
        env.append_transform(recorder)
        env.video_recorder = recorder

    if log_stats:
        env.append_transform(StepCounter())
        env.append_transform(RewardSum(reset_keys=['_reset']))

    if seed is not None:
        env.set_seed(seed)

    return env

