"""Compatibility setup for the MMCV 1.7.2 Python APIs.

DeepLabV3+ and ISDNet use standard PyTorch operations. MMSeg's eager registry
also imports unrelated models that require compiled MMCV extensions. With the
official pure-Python mmcv package, defer *only* these absent extension errors
until use. There is no replacement kernel or numerical fallback: calling any
such extension raises immediately.
"""
import importlib.util
import os
import types


def prepare_mmcv():
    import mmcv
    if mmcv.__version__ != "1.7.2":
        raise RuntimeError(f"Expected MMCV 1.7.2, got {mmcv.__version__}")
    force = os.environ.get("UHR_MMCV_NO_EXT") == "1"
    if not force and importlib.util.find_spec("mmcv._ext") is not None:
        return "compiled_extensions_available"
    from mmcv.utils import ext_loader
    if not getattr(ext_loader.load_ext, "uhr_fail_closed", False):
        original = ext_loader.load_ext

        def fail_closed(name, funcs):
            if name != "_ext":
                return original(name, funcs)
            def unavailable(*args, **kwargs):
                raise RuntimeError("This operation requires a compiled MMCV extension. Install the matching mmcv-full package.")
            return types.SimpleNamespace(**{fn: unavailable for fn in funcs})

        fail_closed.uhr_fail_closed = True
        ext_loader.load_ext = fail_closed
    return "python_mmcv_extensions_unavailable"
