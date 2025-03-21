# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
# or the napari documentation on benchmarking
# https://github.com/napari/napari/blob/main/docs/BENCHMARKS.md
import finn


class QtViewerSuite:
    """Benchmarks for viewing images in the viewer."""

    def setup(self):
        self.viewer = None

    def teardown(self):
        self.viewer.window.close()

    def time_create_viewer(self):
        """Time to create the viewer."""
        self.viewer = finn.Viewer()


if __name__ == "__main__":
    from utils import run_benchmark

    run_benchmark()
