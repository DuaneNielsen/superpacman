import sys
import matplotlib.pyplot as plt
from superpacman import make_env, Actions
import torch
from torchrl.envs import check_env_specs, step_mdp
from torchvision.utils import make_grid
import matplotlib


def play(args):
    matplotlib.use('QtAgg')

    keymap = {"up": Actions.N, "right": Actions.E, "down": Actions.S, "left": Actions.W}

    if args.partial_radius > 0:
        env = make_env(2, "ego_pixels", 'cpu', ego_patch_radius=args.partial_radius)
        pixel_key = 'ego_pixels'
    else:
        env = make_env(2, "pixels", 'cpu')
        pixel_key = 'pixels'

    check_env_specs(env)

    global td
    td = env.reset()

    def on_press(event):
        global td
        try:
            print('press', event.key, keymap[event.key])
            sys.stdout.flush()

            td['action'] = torch.tensor([keymap[event.key], keymap[event.key]])
            td = env.step(td)
            td = step_mdp(td)
            img_plt.set_data(td[pixel_key][0])
            fig.canvas.draw()

            if td['terminated'][0]:
                plt.pause(0.5)
                td = env.reset()
                img_plt.set_data(td[pixel_key][0])
                fig.canvas.draw()
        except KeyError:
            pass

    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', on_press)
    observation = td[pixel_key]
    img_plt = ax.imshow(make_grid(observation[0]))

    ax.set_title('SUPERPACMAN')
    plt.show()
