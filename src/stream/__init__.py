# see LICENSE.rst

"""This module contains class and functions for working with stellar streams."""

# LOCAL
from stream.core import StreamArm
from stream.plural import Stream, StreamArms

__all__ = ["StreamArm", "Stream", "StreamArms"]


# ===================================================================
# Register I/O

# isort: split
# LOCAL
from stream.io.register import UnifiedIOEntryPointRegistrar

UnifiedIOEntryPointRegistrar(data_class=StreamArm, group="stream.io.StreamArm.from_format", which="reader").run()
# clean up
del UnifiedIOEntryPointRegistrar
