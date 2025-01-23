import finn


def test_top_level_availability(make_napari_viewer):
    """Current viewer should be available at finn.current_viewer."""
    viewer = make_napari_viewer()
    assert viewer == finn.current_viewer()
