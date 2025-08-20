import pydevd_pycharm, inspect, sys
print("python:", sys.executable)
print("module:", getattr(pydevd_pycharm, "__version__", "<no __version__>"))
print("settrace obj:", pydevd_pycharm.settrace)
try:
    print("signature:", inspect.signature(pydevd_pycharm.settrace))
except Exception as e:
    print("inspect.signature failed:", e)
print("doc:", (pydevd_pycharm.settrace.__doc__ or "")[:800])


