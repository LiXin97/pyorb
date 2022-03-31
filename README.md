# PyORB

This repository exposes to Python Uniform ORB feature point extraction.

---

## Getting Started

### Building from source

Alternatively, we explain below how to compile PyCOLMAP from source. COLMAP should first be installed as a library
following [the instructions](https://colmap.github.io/install.html). We require the latest commit of the
COLMAP [`dev` branch](https://github.com/colmap/colmap/tree/dev). Using a previous COLMAP build might not work. Then
clone the repository and its submodules:

```
git clone --recursive git@github.com/LiXin97/pyorb.git
```

And finally build PyCOLMAP:

```bash
cd pyorb
pip install .
```

## Extractor uniform ORB feature

We can extractor orb feature from img path. featrue sorting as x, y, response

```python
import pyorb
import numpy as np

result = pyorb.extract({YOUR_IMG_PATH_STR_HERE})
feature = np.array(result['KeyPoints'])

print(feature.shape)
```

