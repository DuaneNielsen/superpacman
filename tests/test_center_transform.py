from superpacman.superpacman import CenterPlayerTransform
from tensordict import TensorDict
import torch


def test_simple_center_transform():

    input_image = torch.arange(1, 17).reshape(1, 1, 4, 4)
    player_pos = torch.tensor([1, 1]).view(1, 2)
    td = TensorDict({"image": input_image, "player_pos": player_pos}, batch_size=[1])
    center_trans = CenterPlayerTransform(patch_radius=1, in_keys='image', out_keys='ego_image')
    td = center_trans(td)
    expected = input_image[:, :, 0:3, 0:3]
    ego_image = td['ego_image']
    assert ego_image.equal(expected)


def test_simple_batched_center_transform():

    image = torch.arange(1, 17).reshape(1, 4, 4)
    input_image = torch.stack([image.clone(), image.clone()], dim=0)
    player_pos = torch.tensor([[1, 1], [1, 1]])
    td = TensorDict({"image": input_image, "player_pos": player_pos}, batch_size=[2])
    center_trans = CenterPlayerTransform(patch_radius=1, in_keys='image', out_keys='ego_image')
    td = center_trans(td)
    expected = image[:, 0:3, 0:3]
    expected = torch.stack([expected.clone(), expected.clone()], dim=0)
    ego_image = td['ego_image']
    assert ego_image.equal(expected)


def test_edge_center_transform():

    input_image = torch.arange(1, 17).reshape(1, 1, 4, 4)
    player_pos = torch.tensor([0, 0]).view(1, 2)
    td = TensorDict({"image": input_image, "player_pos": player_pos}, batch_size=[1])
    center_trans = CenterPlayerTransform(patch_radius=1, in_keys='image', out_keys='ego_image')
    td = center_trans(td)
    expected = torch.tensor([
        [0, 0, 0],
        [0, 1, 2],
        [0, 5, 6]
    ]).view(1, 1, 3, 3)
    ego_image = td['ego_image']

    assert ego_image.equal(expected)


def test_edge_center_transform_with_fill():

    input_image = torch.arange(1, 17).reshape(1, 1, 4, 4)
    player_pos = torch.tensor([0, 0]).view(1, 2)
    td = TensorDict({"image": input_image, "player_pos": player_pos}, batch_size=[1])
    center_trans = CenterPlayerTransform(patch_radius=1, in_keys='image', out_keys='ego_image', fill_value=8)
    td = center_trans(td)
    expected = torch.tensor([
        [8, 8, 8],
        [8, 1, 2],
        [8, 5, 6]
    ]).view(1, 1, 3, 3)
    ego_image = td['ego_image']

    assert ego_image.equal(expected)


def test_batched_channel_center_transform():

    image = torch.arange(1, 17).reshape(4, 4)
    input_channel = torch.stack([image.clone()]*2, dim=0)
    input_image = torch.stack([input_channel.clone()] * 2, dim=0)
    player_pos = torch.tensor([[1, 1], [1, 1]])
    td = TensorDict({"image": input_image, "player_pos": player_pos}, batch_size=[2])
    center_trans = CenterPlayerTransform(patch_radius=1, in_keys='image', out_keys='ego_image')
    td = center_trans(td)
    expected_image = image[0:3, 0:3]
    expected_channel = torch.stack([expected_image.clone(), expected_image.clone()], dim=0)
    expected = torch.stack([expected_channel.clone(), expected_channel.clone()], dim=0)
    ego_image = td['ego_image']
    assert ego_image.equal(expected)