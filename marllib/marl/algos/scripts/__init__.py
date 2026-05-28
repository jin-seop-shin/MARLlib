# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

def _try_import(module_path, symbol):
    try:
        mod = __import__(module_path, fromlist=[symbol])
        return getattr(mod, symbol, None)
    except Exception:
        return None

run_vda2c   = _try_import("marllib.marl.algos.scripts.vda2c",        "run_vda2c")
run_vdppo   = _try_import("marllib.marl.algos.scripts.vdppo",        "run_vdppo")
run_joint_q = _try_import("marllib.marl.algos.scripts.vdn_qmix_iql", "run_joint_q")
run_maa2c   = _try_import("marllib.marl.algos.scripts.maa2c",        "run_maa2c")
run_coma    = _try_import("marllib.marl.algos.scripts.coma",          "run_coma")
run_ia2c    = _try_import("marllib.marl.algos.scripts.ia2c",          "run_ia2c")
run_iddpg   = _try_import("marllib.marl.algos.scripts.iddpg",         "run_iddpg")
run_maddpg  = _try_import("marllib.marl.algos.scripts.maddpg",        "run_maddpg")
run_facmac  = _try_import("marllib.marl.algos.scripts.facmac",        "run_facmac")
run_happo   = _try_import("marllib.marl.algos.scripts.happo",         "run_happo")
run_itrpo   = _try_import("marllib.marl.algos.scripts.itrpo",         "run_itrpo")
run_hatrpo  = _try_import("marllib.marl.algos.scripts.hatrpo",        "run_hatrpo")
run_matrpo  = _try_import("marllib.marl.algos.scripts.matrpo",        "run_matrpo")

from .ippo  import run_ippo   # noqa: E402
from .mappo import run_mappo  # noqa: E402


POlICY_REGISTRY = {
    "ia2c":   run_ia2c,
    "ippo":   run_ippo,
    "iql":    run_joint_q,
    "qmix":   run_joint_q,
    "vdn":    run_joint_q,
    "vda2c":  run_vda2c,
    "vdppo":  run_vdppo,
    "maa2c":  run_maa2c,
    "mappo":  run_mappo,
    "coma":   run_coma,
    "iddpg":  run_iddpg,
    "maddpg": run_maddpg,
    "facmac": run_facmac,
    "happo":  run_happo,
    "itrpo":  run_itrpo,
    "hatrpo": run_hatrpo,
    "matrpo": run_matrpo,
}



