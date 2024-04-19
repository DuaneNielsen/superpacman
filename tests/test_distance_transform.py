from superpacman import make_env, Actions
from torchrl.envs import check_env_specs
from matplotlib import pyplot as plt

def test_distance_image():
    env = make_env(2,obs_keys="distance_image")
    check_env_specs(env)
    state = env.reset()
    plt.imshow(state['distance_reward_tiles'][0])
    plt.show()
    plt.imshow(state['distance_wall_tiles'][0])
    plt.show()
    plt.imshow(state['distance_inky_tiles'][0])
    plt.show()

def test_distance_image():
    env = make_env(2,obs_keys=['flat_distance_obs', 'ego_distance_image'])
    check_env_specs(env)
    state = env.reset()
    plt.imshow(state['ego_distance_reward_tiles'][0])
    plt.show()
    plt.imshow(state['ego_distance_wall_tiles'][0])
    plt.show()
    plt.imshow(state['ego_distance_inky_tiles'][0])
    plt.show()

