import torch
from torch import tensor
from tensordict import TensorDict
from torchrl.data import CompositeSpec, BoundedTensorSpec, UnboundedDiscreteTensorSpec, \
    DiscreteTensorSpec, \
    UnboundedContinuousTensorSpec
from torchrl.envs import (
    EnvBase,
)
from torchrl.envs.transforms.transforms import _apply_to_composite, ObservationTransform
from typing import Any, Dict, List, Optional, OrderedDict, Sequence, Tuple, Union
from enum import IntEnum


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


def hex_to_rgb(hex_color):
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return tensor([r, g, b], dtype=torch.uint8)

# colors for RGB image

yellow = tensor([255, 255, 0], dtype=torch.uint8)
red = tensor([255, 0, 0], dtype=torch.uint8)
blue = tensor([0, 0, 255], dtype=torch.uint8)
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

    # 4 ghost candidate positions, N, G, Pos, xy
    candidate_tile_pos = direction.view(1, 1, 4, 2) + ghost_pos.view(N, G, 1, 2)
    candidate_tile_pos = candidate_tile_pos % H
    is_wall = wall_tiles[batch_range.view(N, 1, 1), candidate_tile_pos[..., 0], candidate_tile_pos[..., 1]]

    # scatter mode based on timer
    scatter_targets = tensor([
        [-3, 19],
        [-3, 1],
        [21, 20],
        [21, 0]
    ], dtype=torch.int64, device=device).expand(N, G, 2)

    chase_targets = torch.zeros(N, G, 2, dtype=torch.int64, device=device)
    chase_targets[batch_range, Ghost.BLINKY] = player_pos
    chase_targets[batch_range, Ghost.PINKY] = player_pos + direction[action] * 4
    pacman_pred_2 = player_pos + direction[action] * 2
    chase_targets[batch_range, Ghost.INKY] = (pacman_pred_2 + ghost_pos[batch_range, Ghost.BLINKY]) // 2
    claude_pacman_dist = (player_pos - ghost_pos[batch_range, Ghost.CLAUDE]) ** 2
    chase_targets[batch_range, Ghost.CLAUDE] = torch.where(claude_pacman_dist < 64, scatter_targets[batch_range, Ghost.CLAUDE], player_pos)

    scatter = (t % 27 <= 7) & (t < 27 * 3)
    targets = torch.where(scatter.view(N, 1, 1), scatter_targets, chase_targets)

    # get the distance from each candidate to the target tile
    candidate_tile_dist = torch.sum((targets.view(N, G, 1, 2) - candidate_tile_pos) ** 2, dim=-1)

    # if pacman is energized, ghosts are frightened, and go in random directions
    random_tile_weights = torch.randint_like(candidate_tile_dist, low=0, high=big_distance - 1)
    candidate_tile_dist = torch.where(energized.view(N, 1, 1), random_tile_weights, candidate_tile_dist)

    # add a big distance if it's a wall so it will be no bueuno
    candidate_tile_dist = candidate_tile_dist + is_wall * big_distance

    # ghosts cant reverse the direction of travel
    rev_direction = (state['ghost_dir'] + 2) % 4
    candidate_tile_dist[batch_range.view(N, 1), ghost_range.view(1, G), rev_direction] = big_distance

    # pick the best direction
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
    frightened_tiles = pos_to_grid(next_ghost_pos, H, W, device=device, dtype=dtype)
    frightened_tiles = frightened_tiles * energized.view(N, 1, 1)

    # move player position checking for collisions
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

    # resolve collisions
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
       "player_tiles": N, 5, 5 tensor, with a single tile set to 1 that indicates player position
       "wall_tiles": N, 5, 5 tensor, 1 indicates wall
       "reward_tiles": N, 5, 5 tensor, rewards remaining in environment
       "energizer_tiles": N, 5, 5 tensor, episode will terminate when tile with value True is entered

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


class Gridworld(EnvBase):
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
        in_keys = ['wall_tiles', 'reward_tiles', 'energizer_tiles', 'pinky_tiles', 'blinky_tiles', 'inky_tiles', 'claude_tiles', 'frightened_tiles']
        super().__init__(
            in_keys=in_keys,
            out_keys=[f'c_{key}' for key in in_keys])
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
        super().__init__(in_keys=['c_wall_tiles'], out_keys=['pixels'])

    def forward(self, tensordict):
        return self._call(tensordict)

    def _reset(self, tensordict, tensordict_reset):
        return self._call(tensordict_reset)

    def _call(self, td):
        walls = td['c_wall_tiles']
        device = walls.device
        shape = *walls.shape, 3
        center_x, center_y = shape[1] // 2, shape[2] // 2

        td['pixels'] = torch.zeros(shape, dtype=torch.uint8, device=device)
        td['pixels'][td['c_wall_tiles'] == 1] = wall_color.to(device)
        td['pixels'][td['c_reward_tiles'] > 0] = food.to(device)
        td['pixels'][td['c_reward_tiles'] < 0] = red.to(device)
        td['pixels'][td['c_energizer_tiles'] == 1] = energizer_color.to(device)
        td['pixels'][:, center_x, center_y] = yellow.to(device)
        td['pixels'][td['c_blinky_tiles'] == 1] = blinky_color.to(device)
        td['pixels'][td['c_pinky_tiles'] == 1] = pinky_color.to(device)
        td['pixels'][td['c_inky_tiles'] == 1] = inky_color.to(device)
        td['pixels'][td['c_claude_tiles'] == 1] = claude_color.to(device)
        td['pixels'][td['c_frightened_tiles'] == 1] = blue.to(device)
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
        td['pixels'][td['frightened_tiles'] == 1] = blue.to(device)
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


if __name__ == '__main__':

    """
    Optimize the agent using Proximal Policy Optimization (Actor - Critic)
    the Generalized Advantage Estimation module is used to compute Advantage
    """

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--device', default='cpu', help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=42, help="seed (defaults 42)")
    parser.add_argument('--env_batch_size', type=int, default=2048, help="number of environments")
    parser.add_argument('--steps_per_batch', type=int, default=8, help="number of steps to take in env per batch")
    parser.add_argument('--train_steps', type=int, default=1000, help="number of PPO updates to run")
    parser.add_argument('--clip_epsilon', type=float, default=0.1, help="PPO clipping parameter")
    parser.add_argument('--gamma', type=float, default=0.99, help="GAE gamma parameter")
    parser.add_argument('--lmbda', type=float, default=0.99, help="GAE lambda parameter")
    parser.add_argument('--entropy_eps', type=float, default=0.08, help="policy entropy bonus weight")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="gradient clipping")
    parser.add_argument('--hidden_dim', type=int, default=64, help="hidden dim size of MLP")
    parser.add_argument('--lr', type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument('--lr_sched_step_size', type=int, default=1e5, help="decay lr after this many steps")
    parser.add_argument('--lr_sched_gamma', type=int, default=0.7, help="decay lr after this many steps")
    parser.add_argument('--ppo_steps', type=int, default=6, help="number of ppo updates per batch")
    parser.add_argument('--eval_freq', type=int, default=100, help="run eval after this many training steps")
    parser.add_argument('--eval_len', type=int, default=400, help="run eval after this many training steps")
    parser.add_argument('--demo', action='store_true', help="command switch to visualize after training completes")
    parser.add_argument('--log_train_video', action='store_true', help='enable video logging')
    parser.add_argument('--log_eval_video', action='store_true', help='enable video logging during eval')
    parser.add_argument('--wandb', action='store_true', help='command switch to enable wandb logging')
    parser.add_argument('--warmup_steps', type=int, default=16, help='delay before starting to learn')
    parser.add_argument('--load_checkpoint')

    args = parser.parse_args()

    import tqdm
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch.optim.lr_scheduler import LambdaLR
    from tensordict.nn import TensorDictModule
    from torch.distributions import Categorical
    from torch.nn.functional import log_softmax
    from torchrl.modules import ProbabilisticActor, ValueOperator
    from torchrl.collectors import SyncDataCollector
    from torchrl.objectives import ClipPPOLoss
    from torchrl.objectives.value import GAE
    from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
    from torchrl.envs import (
        StepCounter,
        RewardSum,
        TransformedEnv,
        FlattenObservation,
        CatTensors,
        Resize,
        ToTensorImage,
        PermuteTransform
    )
    from torchrl.record import VideoRecorder, CSVLogger
    from pathlib import Path
    from time import time
    from warnings import warn
    from math import inf

    exp_name = 'superpacman'
    if args.wandb:
        import wandb

        wandb.init(project='superpacman', config=args)
        exp_name = f'{wandb.run.name}-{wandb.run.id}'

    frames_per_batch = args.env_batch_size * args.steps_per_batch
    total_frames = args.env_batch_size * args.steps_per_batch * args.train_steps


    def make_env(log_video, env_batch_size):
        env = Gridworld(batch_size=torch.Size([env_batch_size]), device=args.device)
        env = TransformedEnv(
            env
        )
        env.append_transform(FlattenObservation(
            -2, -1,
            in_keys=["player_tiles", "wall_tiles", "reward_tiles",
                     'energizer_tiles', 'pinky_tiles', 'blinky_tiles', 'frightened_tiles'],
            out_keys=["flat_player_tiles", "flat_wall_tiles", "flat_reward_tiles",
                      'flat_energizer_tiles', 'flat_pinky_tiles', 'flat_blinky_tiles', 'flat_frightened_tiles']
        ))
        env.append_transform(CatTensors(
            in_keys=["flat_player_tiles", "flat_wall_tiles", "flat_reward_tiles",
                     'flat_energizer_tiles', 'flat_pinky_tiles', 'flat_blinky_tiles', 'flat_frightened_tiles'],
            out_key='flat_obs'
        ))
        env.append_transform(CenterPlayerTransform(
            patch_radius=4
        ))
        env.append_transform(StepCounter())
        env.append_transform(RewardSum(reset_keys=['_reset']))

        recorder = None
        if log_video:
            env.append_transform(RGBFullObsTransform())
            env.append_transform(ToTensorImage(from_int=False))
            env.append_transform(Resize(21*8, 21*8, in_keys=['pixels'], out_keys=['pixels'], interpolation='nearest'))
            env.append_transform(PermuteTransform([-2, -1, -3], in_keys=['pixels'], out_keys=['pixels']))
            logger = CSVLogger(exp_name=exp_name, log_dir="logs", video_fps=3, video_format='mp4')
            recorder = VideoRecorder(logger=logger, tag='pacman', fps=3, skip=1)
            env.append_transform(recorder)
        check_env_specs(env)
        env.set_seed(args.seed)
        return env, recorder


    env, recorder = make_env(args.log_train_video, args.env_batch_size)
    eval_env, eval_recorder = make_env(args.log_eval_video, 32)

    in_features = env.observation_spec['flat_obs'].shape[-1]
    actions_n = env.action_spec.n


    class VGGConvBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels=8, out_channels=128, stride=1, kernel_size=3, padding=1),
                nn.SELU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=128, out_channels=256, stride=1, kernel_size=3, padding=1),
                nn.SELU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        def forward(self, wall_tiles, reward_tiles, player_tiles, pinky_tiles, blinky_tiles, inky_tiles, claude_tiles, frightened_tiles):
            vision = torch.stack([wall_tiles, reward_tiles, player_tiles, pinky_tiles, blinky_tiles, inky_tiles, claude_tiles, frightened_tiles], dim=-3)
            if len(vision.shape) == 5:
                N, L, C, H, W = vision.shape
                vision = vision.flatten(0, 1)
                vision = self.layers(vision)
                vision = vision.unflatten(0, (N, L))
            else:
                vision = self.layers(vision)
            return vision


    class Value(nn.Module):
        """
        MLP value function
        """

        def __init__(self, in_features, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features=in_features + 1024, out_features=hidden_dim),
                nn.ReLU(),
                nn.Linear(in_features=hidden_dim, out_features=1, bias=False)
            )
            self.convblock = VGGConvBlock()

        def forward(self, flat_obs, wall_tiles, reward_tiles, player_tiles, pinky_tiles, blinky_tiles, inky_tiles, claude_tiles, frightened_tiles):
            conv_values = self.convblock(wall_tiles, reward_tiles, player_tiles, pinky_tiles, blinky_tiles, inky_tiles, claude_tiles, frightened_tiles)
            features = torch.cat([flat_obs, conv_values.flatten(-3)], dim=-1)
            values = self.net(features)
            return values


    value_net = Value(in_features=in_features, hidden_dim=args.hidden_dim)

    value_module = ValueOperator(
        module=value_net,
        in_keys=['flat_obs', 'c_wall_tiles', 'c_reward_tiles', 'c_energizer_tiles', 'c_pinky_tiles', 'c_blinky_tiles', 'c_inky_tiles', 'c_claude_tiles', 'c_frightened_tiles']
    ).to(args.device)


    class Policy(nn.Module):
        """
        Policy network for flat observation
        """

        def __init__(self, in_features, hidden_dim, actions_n):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features=in_features + 1024, out_features=hidden_dim),
                nn.ReLU(),
                nn.Linear(in_features=hidden_dim, out_features=actions_n, bias=False)
            )
            self.convblock = VGGConvBlock()

        def forward(self, flat_obs, wall_tiles, reward_tiles, player_tiles, pinky_tiles, blinky_tiles, inky_tiles, claude_tiles, frightened_tiles):
            conv_values = self.convblock(wall_tiles, reward_tiles, player_tiles, pinky_tiles, blinky_tiles, inky_tiles, claude_tiles, frightened_tiles)
            features = torch.cat([flat_obs, conv_values.flatten(-3)], dim=-1)
            return log_softmax(self.net(features), dim=-1)


    policy_net = Policy(in_features, args.hidden_dim, actions_n)

    policy_module = TensorDictModule(
        policy_net,
        in_keys=['flat_obs', 'c_wall_tiles', 'c_reward_tiles', 'c_energizer_tiles', 'c_pinky_tiles', 'c_blinky_tiles', 'c_inky_tiles', 'c_claude_tiles', 'c_frightened_tiles'],
        out_keys=['logits'],
    )

    policy_module = ProbabilisticActor(
        policy_module,
        in_keys=['logits'],
        out_keys=['action'],
        distribution_class=Categorical,
        return_log_prob=True
    ).to(args.device)

    # no need to reuse data for PPO as it's an online algo
    # so we will go with datacollector only and collect fresh batches each time

    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=args.device,
    )

    advantage_module = GAE(
        gamma=args.gamma, lmbda=args.lmbda, value_network=value_module, average_gae=True
    )

    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=args.clip_epsilon,
        entropy_bonus=bool(args.entropy_eps),
        entropy_coef=args.entropy_eps,
        critic_coef=1.0,
        loss_critic_type="smooth_l1"
    )

    optim = Adam(loss_module.parameters(), lr=args.lr)


    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optim, total_frames // frames_per_batch, 0.0
    # )

    def warmup(current_step: int):
        if current_step < args.warmup_steps:
            return 0.
        else:
            return 1.


    scheduler = LambdaLR(optim, lr_lambda=warmup)


    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optim, step_size=args.lr_sched_step_size, gamma=args.lr_sched_gamma
    # )


    def save_checkpoint(filename):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "value_net_state_dict": value_net.state_dict(),
            "policy_net_state_dict": policy_net.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
        }, filename)


    def load_checkpoint(filename):
        chkpt = torch.load(filename)
        value_net.load_state_dict(chkpt["value_net_state_dict"])
        policy_net.load_state_dict(chkpt["policy_net_state_dict"])
        optim.load_state_dict(chkpt["optim_state_dict"])
        scheduler.load_state_dict(chkpt["scheduler_state_dict"])


    if args.load_checkpoint:
        load_checkpoint(args.load_checkpoint)

    pbar = tqdm.tqdm(total=total_frames)
    train_reward_mean, train_reward_max, eval_reward_mean = 0., 0., 0.
    after_update = time()

    for i, tensordict_data in enumerate(collector):
        after_collect = time()
        env_time = after_collect - after_update
        for _ in range(args.ppo_steps):
            advantage_module(tensordict_data)
            loss_vals = loss_module(tensordict_data)
            loss_value = (loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"])
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), args.max_grad_norm)
            optim.step()
            optim.zero_grad()

        scheduler.step()
        after_update = time()
        update_time = after_update - after_collect

        def retrieve_episode_stats(tensordict_data, loss_value, prefix=None):
            with torch.no_grad():
                prefix = '' if prefix is None else f"{prefix}_"
                episode_reward = tensordict_data["next", "episode_reward"]
                step_count = tensordict_data["step_count"].float()
                state_value = tensordict_data['state_value']
                entropy = loss_value['entropy']
                return {
                    f"{prefix}episode_reward_mean": episode_reward.mean().item(),
                    f"{prefix}episode_reward_max": episode_reward.max().item(),
                    f"{prefix}episode_reward_stdev": torch.std(episode_reward).item(),
                    f"{prefix}step_count_max": step_count.max().item(),
                    f"{prefix}step_count_mean": step_count.mean().item(),
                    f"{prefix}state_value_max": state_value.max().item(),
                    f"{prefix}state_value_mean": state_value.mean().item(),
                    f"{prefix}state_value_min": state_value.min().item(),
                    f"{prefix}policy_entropy_value_max": entropy.max().item(),
                    f"{prefix}policy_entropy_value_mean": entropy.mean().item(),
                    f"{prefix}policy_entropy_value_min": entropy.min().item(),
                }

        if i % 10 == 0 and args.wandb:
            epi_stats = retrieve_episode_stats(tensordict_data, loss_vals, 'train')
            wandb.log(epi_stats, step=i)
            wandb.log({
                "learning rate": scheduler.get_last_lr()[0],
                "env_time": env_time,
                "update_time": update_time,
            }, step=i)

            train_reward_mean = epi_stats['train_episode_reward_mean']
            train_reward_max = epi_stats['train_episode_reward_max']

        pbar.set_description(f'train reward mean/max {train_reward_mean:.2f}/{train_reward_max:.2f} eval reward mean: {eval_reward_mean:.2f}')
        pbar.update(tensordict_data.numel())

        if i % args.eval_freq == 0:
            with set_exploration_type(ExplorationType.RANDOM), torch.no_grad():
                pbar.set_description('starting eval rollout')
                eval_rollout = eval_env.rollout(args.eval_len, policy_module, break_when_any_done=False)
                pbar.set_description('computing stats')
                advantage_module(eval_rollout)
                loss = loss_module(eval_rollout)
                epi_stats = retrieve_episode_stats(eval_rollout, loss, prefix='eval')
                if args.wandb:
                    wandb.log(epi_stats, step=i)

                eval_reward_mean = epi_stats['eval_episode_reward_mean']

                if args.log_eval_video:
                    pbar.set_description('writing video')
                    eval_recorder.dump(suffix=f'eval_{eval_reward_mean:.2f}')

                pbar.set_description(f'saving checkpoint {eval_reward_mean:.2f}')
                save_checkpoint(f'models/{exp_name}/checkpoint_{i // args.eval_freq}_{eval_reward_mean:.2f}.pt')


    def best_checkpt(directory):
        best_rew = -inf
        best_chkpt = None
        for chkp in Path(directory).glob('*.pt'):
            reward_str = chkp.name.split('_')[-1][:-3]
            try:
                rew = float(reward_str)
                if rew > best_rew:
                    best_rew = rew
                    best_chkpt = str(chkp)
            except ValueError:
                warn(f'{reward_str} is not a valid float')

        return best_chkpt, best_rew


    best_chkpt, best_reward = best_checkpt(f'models/{exp_name}')
    if best_chkpt is not None:
        with set_exploration_type(ExplorationType.RANDOM), torch.no_grad():
            eval_env, eval_recorder = make_env(True, 32)
            load_checkpoint(best_chkpt)
            pbar.set_description('rolling out best policy found')
            eval_rollout = eval_env.rollout(args.eval_len, policy_module, break_when_any_done=False)
            pbar.set_description('writing video')
            eval_recorder.dump(suffix=f'eval_{best_reward:.2f}')

    # if args.demo:
    #   from matplotlib import pyplot as plt
    #   import matplotlib.animation as animation
    #   from torchvision.utils import make_grid
    #     with set_exploration_type(ExplorationType.RANDOM), torch.no_grad():
    #         eval_rollout = env.rollout(args.eval_len, policy_module, break_when_any_done=False)
    #         recorder.dump()
    #
    #         """
    #         The data dict layout is transitions
    #
    #         {(S, A), next: {R_next, S_next, A_next}}
    #
    #         [
    #           { state_t0, reward_t0, terminal_t0, action_t0 next: { state_t2, reward_t2:1.0, terminal_t2:False } },
    #           { state_t1, reward_t1, terminal_t1, action_t1 next: { state_t3, reward_t2:-1.0, terminal_t3:True } }
    #           { state_t0, reward_t0, terminal_t0, action_t0 next: { state_t3, reward_t2:1.0, terminal_t3:False } }
    #         ]
    #
    #         But which R to use, R or next: R?
    #
    #         recall: Q(S, A) = R + Q(S_next, A_next)
    #
    #         Observe that reward_t0 is always zero, reward is a consequence for taking an action in a state, therefore...
    #
    #         reward = data['next']['reward'][timestep]
    #
    #         Which terminal to use?
    #
    #         Recall that the value of a state is the expectation of future reward.
    #         Thus the terminal state has no value, therefore...
    #
    #         Q(S, A) = R_next + Q(S_next, A_next) * terminal_next
    #
    #         terminal =  data['next']['terminal'][timestep]
    #         """
    #
    #         eval_rollout = eval_rollout.cpu()
    #         observation = eval_rollout['pixels']
    #
    #         fig, ax = plt.subplots(1)
    #         img_plt = ax.imshow(make_grid(observation[:, 0].permute(0, 3, 1, 2)).permute(1, 2, 0))
    #
    #
    #         def animate(i):
    #             x = make_grid(observation[:, i].permute(0, 3, 1, 2)).permute(1, 2, 0)
    #             img_plt.set_data(x)
    #
    #
    #         myAnimation = animation.FuncAnimation(fig, animate, frames=90, interval=500, blit=False, repeat=False)
    #
    #         # uncomment if you want to save the output to mp4
    #         # you will need ffmpeg in your path, or in the directory where you run the script
    #         FFwriter = animation.FFMpegWriter(fps=2)
    #         myAnimation.save('animation.mp4', writer=FFwriter)
    #         plt.close(fig)
