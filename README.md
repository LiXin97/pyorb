# PyORB

This repository exposes to Python Uniform ORB feature point extraction.

---

## Getting Started

Wheels for Linux can be install using pip:
```bash
pip install pyorbfeature
```

### Building from source

CPP OpenCV3 and Eigen3 should be installed as a library first. If not please install first. Then clone the repository
and its submodules:

```
git clone --recursive https://github.com/LiXin97/pyorb.git
```

And finally build PyORB:

```bash
cd pyorb
pip install .
```

## Extractor uniform ORB feature

We can extractor orb feature from img path. 

feature [num_features, 2]
scores [num_features, 1]

```python
import pyorb
import cv2

img = cv2.imread({YOUR_IMG_PATH_STR_HERE})
features, scores = pyorb.extract_frompath({YOUR_IMG_PATH_STR_HERE})
features, scores = pyorb.extract(img)
```

