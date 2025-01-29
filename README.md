# finn is (not) napari

The [motile tracker](https://github.com/funkelab/motile_tracker) started as a [napari](https://github.com/napari/napari) plugin. However, as we developed more advanced features, we found ourselves wanting to adapt the core code of napari, remove features that were not compatible with our applicaton, change layer controls, etc. finn started as a fork of napari, then a complete copy, and now it is an independent repo optimized for displaying cell tracking data.

Many thanks to the napari community for providing an excellent starting point and tons of great feedback and assistance!
> napari contributors (2019). napari: a multi-dimensional image viewer for python. [doi:10.5281/zenodo.3555620](https://zenodo.org/record/3555620)

## installation

finn is not yet on pypi - you must install from the github url or source code

``pip install git+https://github.com/funkelab/finn.git``
