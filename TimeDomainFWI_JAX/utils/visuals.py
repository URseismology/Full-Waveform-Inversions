###############################################################################################
# Author: Sayan Kumar Swar
# Published: 05/08/2025
# University of Rochester
###############################################################################################

import glob
from PIL import Image


def make_gif(readpath = '../results/4fwd_model/', savepath='../results/4fwd_model/', dur=50):
    # cite: https://www.blog.pythonlibrary.org/2021/06/23/creating-an-animated-gif-with-python/
    # cite: https://pythonspot.com/matplotlib-save-figure-to-image-file/
    frames = [Image.open(image) for image in glob.glob(readpath)]
    frame_one = frames[0]
    frame_one.save(savepath, format="GIF", append_images=frames[1:],
                   save_all=True, duration=dur, loop=0)