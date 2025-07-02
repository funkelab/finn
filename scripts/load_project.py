import finn
from finn.track_data_views import ProjectViewer
from finn_builtins.example_projects import Fluo_N2DL_HeLa

if __name__ == "__main__":
    project = Fluo_N2DL_HeLa()
    project_viewer = ProjectViewer(project)
    finn.run()
