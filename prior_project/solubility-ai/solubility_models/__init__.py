# Import order matters on macOS: xgboost, catboost, and lightgbm each bundle
# their own OpenMP runtime. If torch/torch_geometric loads its copy first,
# the process segfaults the first time a tree model actually fits (confirmed
# via isolated testing -- crash is 100% reproducible import-order-dependent,
# not a threading race). Importing the tree libraries here, before any
# submodule gets a chance to `import torch`, fixes it package-wide.
import xgboost as _xgboost  # noqa: F401
import catboost as _catboost  # noqa: F401
import lightgbm as _lightgbm  # noqa: F401
