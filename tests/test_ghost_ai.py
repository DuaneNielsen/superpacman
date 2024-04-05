from gridworld import l2_distance_to_target, action_vec, Actions
import torch


def test_update_pos():
    ghost_pos = torch.tensor([[[2, 3]]])
    player_pos = torch.tensor([[0, 0]])

    tile_pos = action_vec.view(1, 1, 4, 2) + ghost_pos.view(1, 1, 1, 2)

    ghost_d = l2_distance_to_target(tile_pos, player_pos)

    exp_candidate_tile_pos = torch.tensor([[
        [1, 2, 3, 2],
        [3, 4, 3, 2]
    ]]).swapdims(-2, -1)
    exp_ghost_d = torch.tensor([[
        [10., 20., 18, 8.]
    ]])

    assert torch.all(tile_pos == exp_candidate_tile_pos)
    assert torch.all(ghost_d == exp_ghost_d)
