# Optimization_practice
Accuracy comparison using Gradient Descent with mini-batch, Momentum, and Adam.
Learning Rate Dacay and Scheduling included. 

# Used Packages
Numpy, Matplotlib, Scipy, Sklearn, etc.

## Installation
Install the required Python packages with:

```bash
pip install -r requirements.txt
```

# Assessment
Higher accuracy, better result.
The results of this project are as follows.

# Result
Momentum 95.6% > Gradient Descent 94.6% > Adam 94%

# Dependencies and Data
The notebook `Optimization_methods.ipynb` depends on a few external
modules that are **not** included in this repository: `opt_utils_v1a`,
`public_tests`, and `testCases`. It also expects images located in an
`images/` folder. These modules and images can typically be obtained
from the course or assignment materials where the notebook was
provided, and must be added by the user to run the examples.

## Running the notebook

The original notebook depends on helper modules `autils`, `public_tests` and
`lab_utils_softmax` from the DeepLearning.AI course material. Simplified
stand‑alone versions of these modules are provided in this repository so the
notebook can run without the original files. They generate synthetic data and
implement the minimal plotting utilities used in the notebook.

If you prefer to use the original course utilities or the full MNIST data
set, replace the local modules with the versions available from the course
repository or adjust `autils.load_data` accordingly.

# Personal Lesson
Gradient Descent is an effective tool, but it is not always the best solution for optimization. 
Consider other optimization methods if results are unsatisfactory.
