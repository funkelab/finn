import warnings

from finn.settings import *  # noqa: F403
from finn.utils.translations import trans

warnings.warn(
    trans._(
        "'finn.utils.settings' has moved to 'finn.settings' in 0.4.11. This will raise an ImportError in a future version",
        deferred=True,
    ),
    FutureWarning,
    stacklevel=2,
)
