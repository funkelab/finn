"""
Add image
=========

Display one image using the :func:`view_image` API.

.. tags:: visualization-basic
"""

from skimage import data

import napari

# create the viewer with an image
viewer = finn.view_image(data.astronaut(), rgb=True)

if __name__ == '__main__':
    finn.run()
