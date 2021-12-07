# ASM

This repository contains data generated with a Liersch 506 equipped with a lower thread sensor system and neural network base classes as used in the publication (Reference here).

The data is split into two numpy arrays, one for normal sewing operations and one for anomalous sewing operations. Each row represents a completed stitching and each column a single stitch.

With the data folder being the active directory, the data files can be loaded using numpy as follows.

```
import numpy as np

normal = np.load('normal-3436x86.npy')
anomalous = np.load('anomalous-206x86.npy')
```