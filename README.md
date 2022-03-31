# PyORB

This repository exposes to Python Uniform ORB feature point extraction.

---

## Getting Started

### Building from source

Clone the repository and its submodules:

```
git clone --recursive git@github.com/LiXin97/pyorb.git
```

And finally build PyORB:

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
