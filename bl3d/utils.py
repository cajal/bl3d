""" Some utility functions. """

def list_hash(items):
    """ Compute the MD5 digest hash for a list."""
    import hashlib

    hashed = hashlib.md5()
    for item in items:
        hashed.update(str(item).encode())

    return hashed.hexdigest()


def create_video(stack, interval=100, repeat_delay=1000):
    """ Create an animation of the stack. """
    import matplotlib.pyplot as plt
    from matplotlib import animation

    fig = plt.figure()
    num_slices = stack.shape[0]
    im = fig.gca().imshow(stack[int(num_slices / 2)])
    def update_img(i):
        im.set_data(stack[i])
    video = animation.FuncAnimation(fig, update_img, num_slices, interval=interval,
                                    repeat_delay=repeat_delay)
    # video.save('my_video.mp4', dpi=250)
    return fig, video # if video is garbage collected, the animation stops


def colorize_label(label, num_colors=100):
    """ Transform single int labels into RGB with random colors."""
    from skimage import color
    import numpy as np

    colors = np.random.randint(0, 256, size=(num_colors, 3))
    rgb = color.label2rgb(label, colors=colors, bg_label=0)
    rgb = rgb.astype(np.uint8)

    return rgb


def memusage():
    """ Report all Tensor objects in memory and its size. """
    # Source: https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/3
    import torch, gc

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or torch.is_tensor(obj.data):
                print(type(obj), obj.shape)
        except:
            pass # other non-tensor objects