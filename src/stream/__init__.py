# see LICENSE.rst

"""This module contains class and functions for working with stellar streams."""

# LOCAL
from stream import utils  # noqa: F401, TC002
from stream.stream.core import StreamArm
from stream.stream.stream import Stream

__all__ = ["StreamArm", "Stream"]


# ===================================================================

# Fill in attrs, etc.
# isort: split
# LOCAL
from stream import frame  # noqa: F401, TC002
from stream import setup_package  # noqa: F401, TC002

# ===================================================================
# Register I/O

# isort: split
# LOCAL
from stream.io.register import UnifiedIOEntryPointRegistrar

UnifiedIOEntryPointRegistrar(data_class=StreamArm, group="stream.io.StreamArm.from_format", which="reader").run()
# clean up
del UnifiedIOEntryPointRegistrar
