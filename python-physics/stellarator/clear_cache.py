import os
import shutil
from pathlib import Path
import jax

# Locate the default cache path (or replace with your custom path)
if "JAX_COMPILATION_CACHE_DIR" in os.environ:
    cache_path = Path(os.environ["JAX_COMPILATION_CACHE_DIR"])

else:
    raise RuntimeError("No cache dir set")

if cache_path.exists():
    shutil.rmtree(cache_path)
    print("Persistent disk cache cleared.")

# Optional: Clear the RAM/memory compilation caches as well
jax.clear_caches()
