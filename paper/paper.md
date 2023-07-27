---
title: 'sparse-lm: Sparse Linear Regression Models in Python'
tags:
  - Python
  - scikit-learn
  - cvxpy
  - linear regression
  - regularization
  - structured sparsity
authors:
  - name: Luis Barroso-Luque^[corresponding author]
    orcid: 0000-0002-6453-9545
    affiliation: "1"
  - name: Fengyu Xie
    affiliation: "1, 2"
    orcid: 0000-0002-1169-1690
affiliations:
 - name: Materials Sciences Division, Lawrence Berkeley National Laboratory, Berkeley CA, 94720, USA
   index: 1
 - name: Department of Materials Science and Engineering, University of California Berkeley, Berkeley CA, 94720, USA
   index: 2
date: 10 July 2023
bibliography: paper.bib
---

# Summary

# Statement of need

A variety of linear regression models with flexible regularization and feature selection
options are implemented in `sparse-lm`:

# Implemented regression models

- Lasso & Adaptive Lasso
- Group Lasso & Adaptive Group Lasso
- Sparse Group Lasso & Adaptive Sparse Group Lasso
- Ridged Group Lasso & Adaptive Ridge Group Lasso
- Best Subset Selection
- Ridged Best Subset Selection
- MIQP $\ell_0$ regularized regression
- MIQP $\ell_0\ell_2$ regularized regression

# Implemented model selection and composition tools
- One standard deviation rule grid search cross-validation
- Line search cross-validation
- Stepwise composite estimator

The package can be downloaded through the [Python Package Index](https://pypi.org/project/sparse-lm/).
Documentation, including an API reference and examples, can be found in the
[online documentation](https://cedergrouphub.github.io/sparse-lm).

# Acknowledgements

The first author (L.B.L.) is the lead developer of `sparse-lm`, and the lead and corresponding author.
The second author (F.X.) is a main contributor to the package. Both authors drafted, reviewed and edited the manuscript.

L.B.L. and F.X. would like acknowledge the contributions of the following individuals to the development of `sparse-lm`:
Peichen Zhong, Ronald L. Kam, and Tina Chen.

The development of `sparse-lm` was primarily funded by the U.S. Department of Energy, Office
of Science, Office of Basic Energy Sciences, Materials Sciences and Engineering Division
under Contract No. DE-AC02-05-CH11231 (Materials Project program KC23MP).
L.B.L gratefully acknowledges support from the National Science Foundation Graduate Research Fellowship
under Grant No. DGE 1752814.

# References
