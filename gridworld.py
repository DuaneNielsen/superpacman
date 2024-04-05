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
from torchvision.utils import draw_segmentation_masks

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


# N/S is reversed as y-axis in images is reversed
action_vec = torch.stack(
    [
        tensor([-1, 0]),  # N
        tensor([0, 1]),  # E
        tensor([1, 0]),  # S
        tensor([0, -1])  # W
    ]
)

# colors for RGB image
yellow = tensor([255, 255, 0], dtype=torch.uint8)
red = tensor([255, 0, 0], dtype=torch.uint8)
green = tensor([0, 255, 0], dtype=torch.uint8)
pink = tensor([255, 0, 255], dtype=torch.uint8)
violet = tensor([226, 43, 138], dtype=torch.uint8)
white = tensor([255, 255, 255], dtype=torch.uint8)
gray = tensor([128, 128, 128], dtype=torch.uint8)
light_gray = tensor([211, 211, 211], dtype=torch.uint8)
blue = tensor([0, 0, 255], dtype=torch.uint8)



def pos_to_grid(pos, H, W, device='cpu', dtype=torch.float32):
    """
    Converts positions to grid where 1 indicates a position
    :param pos: N, 2 tensor of grid positions (x = H, y = W) or 2 tensor
    :param H: height
    :param W: width
    :param: device: device
    :param: dtype: type of tensor
    :return: N, H, W tensor or single H, W tensor
    """

    if len(pos.shape) == 2:
        N = pos.size(0)
        batch_range = torch.arange(N, device=device)
        grid = torch.zeros((N, H, W), dtype=dtype, device=device)
        grid[batch_range, pos[:, 0], pos[:, 1]] = 1.
    else:
        grid = torch.zeros((H, W), dtype=dtype, device=device)
        grid[pos[0], pos[1]] = 1.
    return grid


def l2_distance_to_target(tile_pos, target_pos):
    N, G, _, _ = tile_pos.shape
    return torch.sum((target_pos.view(N, 1, 1, 2) - tile_pos) ** 2, dim=-1)


def _step(state):
    device = state.device
    N, H, W = state['wall_tiles'].shape
    _, G, _ = state['ghost_pos'].shape

    wall_tiles = state['wall_tiles']
    batch_range = torch.arange(N, device=device)
    dtype = state['wall_tiles'].dtype
    action = state['action'].squeeze(-1)
    direction = action_vec.to(device)
    big_distance = H ** 2 + W ** 2
    ghost_pos = state['ghost_pos']

    # update ghost positions
    candidate_tile_pos = direction.view(1, 1, 4, 2) + ghost_pos.view(N, G, 1, 2)
    candidate_tile_pos = candidate_tile_pos % H

    # get the distance from each candidate to the target tile
    candidate_tile_dist = torch.sum((state['player_pos'].view(N, 1, 1, 2) - candidate_tile_pos) ** 2, dim=-1)

    # add a big distance if it's a wall so it will be no bueano
    is_wall = wall_tiles[batch_range.view(N, 1, 1), candidate_tile_pos[..., 0], candidate_tile_pos[..., 1]]
    candidate_tile_dist = candidate_tile_dist + is_wall * big_distance

    # ghosts cant reverse the direction of travel
    rev_direction = (state['ghost_dir'] + 2) % 4
    candidate_tile_dist[batch_range, torch.arange(candidate_tile_dist.size(1)), rev_direction.squeeze()] = big_distance

    # pick the best direction
    distance, ghost_direction = torch.min(candidate_tile_dist, dim=-1)
    next_ghost_pos = ghost_pos + direction[ghost_direction]
    next_ghost_pos = next_ghost_pos % H
    pinky_tiles = pos_to_grid(next_ghost_pos[:, 0], H, W, device=device, dtype=dtype)

    # move player position checking for collisions
    next_player_pos = state['player_pos'] + direction[action]
    next_player_pos = next_player_pos % H
    next_player_grid = pos_to_grid(next_player_pos, H, W, device=device, dtype=torch.bool)
    collide_wall = torch.logical_and(next_player_grid, state['wall_tiles'] == 1).any(-1).any(-1)
    player_pos = torch.where(collide_wall[..., None], state['player_pos'], next_player_pos)
    player_tiles = pos_to_grid(player_pos, H, W, device=device, dtype=dtype)

    # pickup any rewards
    reward = state['reward_tiles'][batch_range, player_pos[:, 0], player_pos[:, 1]]
    state['reward_tiles'][batch_range, player_pos[:, 0], player_pos[:, 1]] = 0.

    # set terminated flag if hit terminal tile
    # terminated = state['terminal_tiles'][batch_range, player_pos[:, 0], player_pos[:, 1]]
    terminated = (next_ghost_pos == player_pos.view(N, 1, 2)).all(-1).any(-1)
    terminated = (ghost_pos == player_pos.view(N, 1, 2)).all(-1).any(-1) | terminated


    out = {
        'player_pos': player_pos,
        'player_tiles': player_tiles,
        'ghost_pos': next_ghost_pos,
        'ghost_dir': ghost_direction,
        'pinky_tiles': pinky_tiles,
        'wall_tiles': state['wall_tiles'],
        'reward_tiles': state['reward_tiles'],
        'terminal_tiles': state['terminal_tiles'],
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
       "terminal_tiles": N, 5, 5 tensor, episode will terminate when tile with value True is entered

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
    ghost_pos = tensor([[7, 10]], dtype=torch.int64)
    ghost_dir = tensor([Actions.W], dtype=torch.int64)
    player_tiles = pos_to_grid(player_pos, H, W, dtype=walls.dtype)
    pinky_tiles = pos_to_grid(ghost_pos[0], H, W, dtype=walls.dtype)

    state = {
        "player_pos": player_pos,
        "player_tiles": player_tiles,
        "ghost_pos": ghost_pos,
        "ghost_dir": ghost_dir,
        "pinky_tiles": pinky_tiles,
        "wall_tiles": walls,
        "reward_tiles": rewards,
        "terminal_tiles": terminal_states
    }

    td = TensorDict(state, batch_size=[])

    if batch_size:
        td = td.expand(batch_size).contiguous()
    return td


def _make_spec(self, td_params):
    batch_size = td_params.shape
    self.observation_spec = CompositeSpec(
        player_tiles=BoundedTensorSpec(
            minimum=0,
            maximum=1,
            shape=torch.Size(td_params['player_tiles'].shape),
            dtype=torch.float32,
        ),
        pinky_tiles=BoundedTensorSpec(
            minimum=0,
            maximum=1,
            shape=torch.Size(td_params['pinky_tiles'].shape),
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
        terminal_tiles=BoundedTensorSpec(
            minimum=0,
            maximum=1,
            shape=torch.Size(td_params['terminal_tiles'].shape),
            dtype=torch.bool,
        ),
        player_pos=UnboundedDiscreteTensorSpec(
            shape=torch.Size((*batch_size, 2,)),
            dtype=torch.int64
        ),
        ghost_pos=UnboundedDiscreteTensorSpec(
            shape=torch.Size((*batch_size, 1, 2)),
            dtype=torch.int64
        ),
        ghost_dir=UnboundedDiscreteTensorSpec(
            shape=torch.Size((*batch_size, 1)),
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
        super().__init__(
            in_keys=['wall_tiles', 'reward_tiles', 'terminal_tiles', 'pinky_tiles'],
            out_keys=['c_wall_tiles', 'c_reward_tiles', 'c_terminal_tiles', 'c_pinky_tiles'])
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
            td[out_key] = td[in_key][batch_range.view(N, 1), indexes[:, 0].flatten(-2), indexes[:, 1].flatten(-2)].view(N, max_x, max_y)
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
        rewards = td['c_reward_tiles']
        terminal = td['c_terminal_tiles']
        pinky_tiles = td['c_pinky_tiles']
        device = walls.device

        shape = *walls.shape, 3
        center_x, center_y = shape[1] // 2, shape[2] // 2
        td['pixels'] = torch.zeros(shape, dtype=torch.uint8, device=device)
        td['pixels'][walls == 1] = light_gray.to(device)
        td['pixels'][rewards > 0] = green.to(device)
        td['pixels'][rewards < 0] = red.to(device)
        td['pixels'][terminal == 1] = blue.to(device)
        td['pixels'][:, center_x, center_y] = yellow.to(device)
        td['pixels'][pinky_tiles == 1] = red.to(device)
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
        player_tiles = td['player_tiles']
        pinky_tiles = td['pinky_tiles']
        walls = td['wall_tiles']
        rewards = td['reward_tiles']
        terminal = td['terminal_tiles']
        device = walls.device

        shape = *walls.shape, 3
        td['pixels'] = torch.zeros(shape, dtype=torch.uint8, device=device)
        td['pixels'][walls == 1] = light_gray.to(device)
        td['pixels'][rewards > 0] = green.to(device)
        td['pixels'][rewards < 0] = red.to(device)
        td['pixels'][terminal == 1] = blue.to(device)
        td['pixels'][player_tiles == 1] = yellow.to(device)
        td['pixels'][pinky_tiles == 1] = red.to(device)
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
    parser.add_argument('--entropy_eps', type=float, default=0.001, help="policy entropy bonus weight")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="gradient clipping")
    parser.add_argument('--hidden_dim', type=int, default=64, help="hidden dim size of MLP")
    parser.add_argument('--lr', type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument('--lr_sched_step_size', type=int, default=1e5, help="decay lr after this many steps")
    parser.add_argument('--lr_sched_gamma', type=int, default=0.7, help="decay lr after this many steps")
    parser.add_argument('--ppo_steps', type=int, default=6, help="number of ppo updates per batch")
    parser.add_argument('--eval_freq', type=int, default=100, help="run eval after this many training steps")
    parser.add_argument('--eval_len', type=int, default=1000, help="run eval after this many training steps")
    parser.add_argument('--demo', action='store_true', help="command switch to visualize after training completes")
    parser.add_argument('--log_train_video', action='store_true', help='enable video logging')
    parser.add_argument('--log_eval_video', action='store_true', help='enable video logging during eval')
    parser.add_argument('--wandb', action='store_true', help='command switch to enable wandb logging')
    parser.add_argument('--warmup_steps', type=int, default=3, help='delay before starting to learn')
    parser.add_argument('--load_checkpoint')

    args = parser.parse_args()

    import tqdm
    import torch
    from collections import defaultdict
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
        CatTensors
    )
    from torchrl.record import VideoRecorder
    from csv import CSVLogger
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
                     'terminal_tiles'],
            out_keys=["flat_player_tiles", "flat_wall_tiles", "flat_reward_tiles",
                      'flat_terminal_tiles']
        ))
        env.append_transform(CatTensors(
            in_keys=["flat_player_tiles", "flat_wall_tiles", "flat_reward_tiles", 'flat_terminal_tiles'],
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
                nn.Conv2d(in_channels=3, out_channels=128, stride=1, kernel_size=3, padding=1),
                nn.SELU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=128, out_channels=256, stride=1, kernel_size=3, padding=1),
                nn.SELU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        def forward(self, wall_tiles, reward_tiles, player_tiles):
            vision = torch.stack([wall_tiles, reward_tiles, player_tiles], dim=-3)
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

        def forward(self, flat_obs, wall_tiles, reward_tiles, player_tiles):
            conv_values = self.convblock(wall_tiles, reward_tiles, player_tiles)
            features = torch.cat([flat_obs, conv_values.flatten(-3)], dim=-1)
            values = self.net(features)
            return values


    value_net = Value(in_features=in_features, hidden_dim=args.hidden_dim)

    value_module = ValueOperator(
        module=value_net,
        in_keys=['flat_obs', 'c_wall_tiles', 'c_reward_tiles', 'c_terminal_tiles']
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

        def forward(self, flat_obs, wall_tiles, reward_tiles, player_tiles):
            conv_values = self.convblock(wall_tiles, reward_tiles, player_tiles)
            features = torch.cat([flat_obs, conv_values.flatten(-3)], dim=-1)
            return log_softmax(self.net(features), dim=-1)


    policy_net = Policy(in_features, args.hidden_dim, actions_n)

    policy_module = TensorDictModule(
        policy_net,
        in_keys=['flat_obs', 'c_wall_tiles', 'c_reward_tiles', 'c_terminal_tiles'],
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

    logs = defaultdict(list)
    pbar = tqdm.tqdm(total=total_frames)
    eval_str = ""
    after_update = time()

    for i, tensordict_data in enumerate(collector):
        after_collect = time()
        env_time = after_collect - after_update
        for _ in range(args.ppo_steps):
            advantage_module(tensordict_data)
            loss_vals = loss_module(tensordict_data)
            loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
            )

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
                step_count = tensordict_data["step_count"]
                state_value = tensordict_data['state_value']
                entropy = loss_value['entropy']
                return {
                    f"{prefix}episode_reward_mean": episode_reward.mean().item(),
                    f"{prefix}episode_reward_max": episode_reward.max().item(),
                    f"{prefix}episode_reward_stdev": torch.std(episode_reward).item(),
                    f"{prefix}step_count_max": step_count.max().item(),
                    f"{prefix}state_value_max": state_value.max().item(),
                    f"{prefix}state_value_mean": state_value.mean().item(),
                    f"{prefix}state_value_min": state_value.min().item(),
                    f"{prefix}policy_entropy_value_max": entropy.max().item(),
                    f"{prefix}policy_entropy_value_mean": entropy.mean().item(),
                    f"{prefix}policy_entropy_value_min": entropy.min().item(),
                }


        epi_stats = retrieve_episode_stats(tensordict_data, loss_vals, 'train')
        if i % 10 == 0 and args.wandb:
            wandb.log(epi_stats, step=i)
            wandb.log({
                "learning rate": scheduler.get_last_lr()[0],
                "env_time": env_time,
                "update_time": update_time,
            }, step=i)

        if args.log_train_video and i % 1000 == 0:
            recorder.dump(suffix='train')

        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        logs["episode_reward"].append(epi_stats['train_episode_reward_mean'])
        pbar.update(tensordict_data.numel())

        cum_reward_str = (
            f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        )

        logs["step_count"].append(epi_stats['train_step_count_max'])

        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"

        pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

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

                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(
                    eval_rollout["next", "reward"].sum().item()
                )
                logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                    f"eval step-count: {logs['eval step_count'][-1]}"
                )
                del eval_rollout
                reward_mean = epi_stats['eval_episode_reward_mean']

                if args.log_eval_video:
                    pbar.set_description('writing video')
                    eval_recorder.dump(suffix=f'eval_{reward_mean:.2f}')

                pbar.set_description('saving checkpoint')
                save_checkpoint(f'models/{exp_name}/checkpoint_{i // args.eval_freq}_{reward_mean:.2f}.pt')


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
