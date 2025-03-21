"""ExperimentalNamespace and CommandProcessor classes."""

HELP_STR = """
Available Commands
------------------
experimental.cmds.loader
"""


class CommandProcessor:
    """Container for the LoaderCommand.

    Implements the console command "viewer.experimental.cmds.loader".

    Parameters
    ----------
    layers
        The viewer's layers.
    """

    def __init__(self, layers) -> None:
        self.layers = layers

    @property
    def loader(self):
        """The loader related commands."""
        from finn.components.experimental.chunk._commands import (
            LoaderCommands,
        )

        return LoaderCommands(self.layers)

    def __repr__(self):
        return "Available Commands:\nexperimental.cmds.loader"


class ExperimentalNamespace:
    """Container for the CommandProcessor.

    Implements the console command "viewer.experimental.cmds".

    Parameters
    ----------
    layers
        The viewer's layers.
    """

    def __init__(self, layers) -> None:
        self.layers = layers

    @property
    def cmds(self):
        """All experimental commands."""
        return CommandProcessor(self.layers)

    def __repr__(self):
        return "Available Commands:\nexperimental.cmds.loader"
