from finn.layers.points import _points_key_bindings
from finn.layers.points.points import Points

# Note that importing _points_key_bindings is needed as the Points layer gets
# decorated with keybindings during that process, but it is not directly needed
# by our users and so is deleted below
del _points_key_bindings


__all__ = ["Points"]
