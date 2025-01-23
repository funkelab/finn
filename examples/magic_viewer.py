"""
magicgui viewer
===============

Example showing how to access the current viewer from a function widget.

.. tags:: gui
"""

import napari


# annotating a parameter as `finn.Viewer` will automatically provide
# the viewer that the function is embedded in, when the function is added to
# the viewer with add_function_widget.
def my_function(viewer: finn.Viewer):
    print(viewer, f'with {len(viewer.layers)} layers')


viewer = finn.Viewer()
# Add our magic function to napari
viewer.window.add_function_widget(my_function)

if __name__ == '__main__':
    finn.run()
