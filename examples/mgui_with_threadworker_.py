"""
magicgui with threadworker
==========================

An example of calling a threaded function from a magicgui ``dock_widget``.
Note: this example requires python >= 3.9

.. tags:: gui
"""
from typing import Annotated

from magicgui import magic_factory, widgets
from skimage import data
from skimage.feature import blob_log

import napari
from finn.qt.threading import FunctionWorker, thread_worker
from finn.types import ImageData, LayerDataTuple


@magic_factory(pbar={'visible': False, 'max': 0, 'label': 'working...'})
def make_widget(
    pbar: widgets.ProgressBar,
    image: ImageData,
    min_sigma: Annotated[float, {'min': 0.5, 'max': 15, 'step': 0.5}] = 5,
    max_sigma: Annotated[float, {'min': 1, 'max': 200, 'step': 0.5}] = 30,
    num_sigma: Annotated[int, {'min': 1, 'max': 20}] = 10,
    threshold: Annotated[float, {'min': 0, 'max': 1000, 'step': 0.1}] = 6,
) -> FunctionWorker[LayerDataTuple]:

    # @thread_worker creates a worker that runs a function in another thread
    # we connect the "returned" signal to the ProgressBar.hide method
    @thread_worker(connect={'returned': pbar.hide})
    def detect_blobs() -> LayerDataTuple:
        # this is the potentially long-running function
        blobs = blob_log(image, min_sigma, max_sigma, num_sigma, threshold)
        points = blobs[:, : image.ndim]
        meta = {
            'size': blobs[:, -1],
            'border_color': 'red',
            'border_width': 2,
            'border_width_is_relative': False,
            'face_color': 'transparent',
        }
        # return a "LayerDataTuple"
        return (points, meta, 'points')

    # show progress bar and return worker
    pbar.show()
    return detect_blobs()


viewer = finn.Viewer()
viewer.window.add_dock_widget(make_widget(), area='right')
viewer.add_image(data.hubble_deep_field().mean(-1))

finn.run()
