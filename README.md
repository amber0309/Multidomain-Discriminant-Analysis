# Multidomain Discriminant Analysis (MDA)

MATLAB implementation of paper

[Domain Generalization via Multidomain Discriminant Analysis](http://papers.nips.cc/paper/7767-causal-inference-and-mechanism-clustering-of-a-mixture-of-additive-noise-models)  
Shoubo Hu, Kun Zhang, Zhitang Chen, Laiwan Chan.  
In *Proceedings of Conference on Uncertainty in Artificial Intelligence* (**UAI**) 2019.

### Prerequisites

We test the code using **MATLAB R2017b** on Windows 10. Any later version should still work perfectly.

## Running the test

In MATLAB, you can directly run the file *demo.m* to see whether it could work normally.

The test does the following:

1. load the synthetic data (data/data2.m);
1. put source domain data in MATLAB *cell arrays*;
1. put data for testing and validation in matrix;
1. Apply MDA on this data set.


## Apply **MDA** on your data

### Usage

Apply **ANM-MM** on your data

```Matlab
[test_accuracy, predicted_labels, Zs, Zt] = MDA(X_s_cell, Y_s_cell, X_t, Y_t, params)
```

### Description

Input of function **MDA()**

| Argument  | Description  |
|---|---|
|X_s_cell | cell of (n_s*d) matrices, each matrix corresponds to the instance features of a source domain|
|Y_s_cell | cell of (n_s*1) matrices, each matrix corresponds to the instance labels of a source domain |
|X_t |(n_t*d) matrix, rows correspond to instances and columns correspond to features |
|Y_t|(n_t*1) matrix, each row is the class label of corresponding instances in X_t |
|params (optional)|struct containing hyperparameters and validation data (details in MDA.m)|

Output of function **MDA()**

| Argument  | Description  |
|---|---|
|test_accuracy | test accuracy of target instances |
|predicted_labels|predicted labels of target instances|
|Zs|projected source domain instances|
|Zt|projected target domain instances|

## Author

* **Shoubo Hu** - shoubo [dot] sub [at] gmail [dot] com

See also the list of [contributors](https://github.com/amber0309/Multidomain-Discriminant-Analysis/graphs/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* Hat tip to Ya Li for his [CIDG code](https://mingming-gong.github.io/papers/CIDG.zip).