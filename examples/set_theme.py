"""
Set theme
=========

Displays an image and sets the theme to 'light'.

.. tags:: gui
"""

from skimage import data

import napari

# create the viewer with an image
viewer = finn.view_image(data.astronaut(), rgb=True, name='astronaut')

# set the theme to 'light'
viewer.theme = 'light'

if __name__ == '__main__':
    finn.run()
