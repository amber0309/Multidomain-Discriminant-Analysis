# Multidomain Discriminant Analysis (MDA)

Python code of paper  
[Domain Generalization via Multidomain Discriminant Analysis](http://auai.org/uai2019/proceedings/papers/101.pdf)  
Shoubo Hu, Kun Zhang, Zhitang Chen, Laiwan Chan.  
*Conference on Uncertainty in Artificial Intelligence* (**UAI**) 2019.

## Prerequisites
- numpy
- scipy
- sklearn

We test the code using **Anaconda 4.3.30 64-bit for python 2.7** on Windows 10.

## Running the tests

After installing all required packages, you can run *demo.py* to see whether `MDA()` could work normally.

The test code does the following:
1. loads the synthetic data, where there are 3 domains and 3 classes in each domain;
2. uses source domains and validation set to learn and validate the invariant transformation;
3. applies the optimal transformation on the target domain.

## Apply `MDA()` on your data

### Usage

Import `MDA()` using

```python
from MDA import MDA
```

Apply `MDA()` on your data

```python
mdl = MDA(X_s_list, y_s_list, params) # initialize the MDA object
mdl.learn() # learn the optimal transformation
mdl.predict(X_t, y_t) # apply the learned transformation on the target data
```

### Description

Input of `MDA()`

| Argument  | Description  |
|---|---|
|X_s_list | list of `(n_s, d)` matrices, each matrix corresponds to the instance features of a source domain|
|y_s_list | list of `(n_s, 1)` matrices, each matrix corresponds to the instance labels of a source domain |
|params |dictionary containing hyperparameters and validation data (details in *MDA.py*) |

Input of `MDA.predict()`

| Argument  | Description  |
|---|---|
|X_t | `(n_t, d)` matrix, rows correspond to instances and columns correspond to features |
|y_t | `(n_t, 1)` matrix, each row is the class label of corresponding instances in `X_t` |

Output of `MDA.predict()`

| Argument  | Description  |
|---|---|
|acc_final | test accuracy of target instances |
|labels_final | predicted labels of target instances |

## Authors

* **Shoubo Hu** - shoubo [dot] sub [at] gmail [dot] com

See also the list of [contributors](https://github.com/amber0309/Multidomain-Discriminant-Analysis/graphs/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
