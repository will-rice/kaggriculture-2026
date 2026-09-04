"""The engine library builds, exports the ABI, and reports the pinned engine."""

import ctypes

from kaggriculture.campaign import config
from kaggriculture.campaign.engine import build


def test_build_produces_a_loadable_library_with_abi_version_1() -> None:
    """The compiled bridge loads and reports ABI version 1 and the pinned engine."""
    path = build.build()
    assert path == config.ENGINE_LIBRARY and path.exists()
    library = ctypes.CDLL(str(path))
    library.kag_abi_version.restype = ctypes.c_uint32
    assert library.kag_abi_version() == 1
    library.kag_engine_version.restype = ctypes.c_char_p
    assert library.kag_engine_version() == b"1.32.7"
