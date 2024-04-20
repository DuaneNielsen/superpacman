from superpacman import make_env, Actions
from torchrl.envs import check_env_specs
from matplotlib import pyplot as plt

def test_distance_image():
    env = make_env(2,obs_keys="distance_image")
    check_env_specs(env)
    state = env.reset()
    plt.imshow(state['distance_image'][0, 0])
    plt.show()

def test_ego_dist_image():
    env = make_env(2, obs_keys="ego_distance_image")
    state = env.reset()
    plt.imshow(state['ego_distance_image'][0, 0])
    plt.show()

def test_pixel_image():
    env = make_env(2, obs_keys="ego_pixels")
    state = env.reset()
    image = state['ego_pixels']
    plt.imshow(image[0])
    plt.show()