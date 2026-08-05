"""HarmScope.

The OpenMP setting below is here rather than in a shell wrapper because it must
be set before libomp initializes, and libomp is pulled in transitively by numba
(via UMAP), faiss, and torch — whichever of those imports first. `Info #276`
announces that `omp_set_nested` is deprecated, once per worker pool, from
library code we do not control. It says nothing about this project and it
repeats often enough to bury output that does.

`setdefault`, not assignment: an operator who exports KMP_WARNINGS to debug a
threading problem gets to keep their setting.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_WARNINGS", "0")
