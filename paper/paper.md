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

Sparse linear regression models are a powerful tool to capture linear relationships
in high dimensional spaces. Sparse models have only a small number of nonzero parameters
(even if the number of covariates used in estimation is large), and as a result can be
easier to estimate and interpret than dense models [@Hastie:2015]. Regression objectives
resulting in sparse linear models such as the Lasso [@Tibshirani:1996; @Zou:2006] and
Best Subset Selection [@Hocking:1967] have been widely used in a variety of fields.
However, many regression problems involve covariates that have natural underlying
structure such as group or hierarchical relationships. A common example of sparse 
regression problems with structure occurs in the context of multi-body expansion methods,
used in chemistry and materials science, that involve a hierarchy among main effects and
higher order corrections [@Leong:2019; @Barroso-Luque:2022]. Several generalizations of
the Lasso [@Yuan:2006; @Friedman:2010; @Simon:2013; @Wang:2019] and Best Subset Selection
[@Bertsimas:2016-a; @Bertsimas:2016-b] have been developed to effectively exploit
additional structure in linear regression.

(schematic of different forms of structured sparsity)

Structured sparsity can be introduced into regression problems in one of two ways. The
first method to obtain structured sparsity is by using regularization as in
generalizations of the Lasso such as the Group Lasso  and Sparse Group
Lasso [@Yuan:2006; @Friedman:2010; @Simon:2013; @Wang:2019]. The Sparse Group Lasso
regression problem can be expressed as follows,

\begin{equation}
    \bm{\beta}^* = \underset{\bm{\beta}}{\text{argmin}}\;||\bm{X} \bm{\beta} - \bm{Y}||^2_2 + (1-\alpha)\lambda\sum_{\bm{g}\in G}\sqrt{|\bm{g}
    }||\bm{\beta}_{\bm{g}}||_2 + \alpha\lambda||\bm{X}||_1
\end{equation}

where $\bm{X}$ is the design matrix, $\bm{Y}$ is the response vector, and $\bm{\beta}$ are
the regression coefficients. $\bm{g}$ are the groups of covariate indices, $G$ is the
set of all such groups being considered, and $\bm{\beta}_{\bm{g}}\in\mathbb{R}^{|\bm{g}|}$
are the covariate coefficients in group $g$. $\lambda \in \R_{+}$ and $\alpha\in[0,1]$
are regularization hyperparameters. The parameter $\alpha\in[0,1]$ controls the relative
weight of the single covariate $\ell_1$ and group regularization terms, i.e. when
$\alpha=0$ the Sparse Group Lasso is equivalent to the Group Lasso, and when $\alpha=1$
the Sparse Group Lasso is equivalent to the Lasso.

The (Sparse) Group Lasso can be directly used to obtain a grouped sparsity pattern, and
can be extended to obtain hierarchical sparsity patterns by using the Overlap Group
Lasso to introduce overlap between groups [@Hastie:2015].

The second method to obtain structured sparsity is by way of linear constraints
introduced into the regression objective as is done in mixed integer quadratic
programming (MIQP) formulations of the Best Subset Selection
[@Bertsimas:2016-a; @Bertsimas:2016-b]. The general MIQP formulation of Best Subset
Selection with group and hierarchical structure can be expressed as follows,

\begin{align}
    \bm{\beta}^* = \underset{\bm{\beta}}{\text{argmin}}\; & \bm{\beta}^{\top}\left(\bm{X}^{\top}\bm{X} + \lambda\bm{I}\right)\bm{\beta} - 2\bm{Y}^{\top}\bm{X}\bm{\beta} +  \lambda_0 \\
    &\begin{array}{r@{~}l@{}l@{\quad}l}
    \text{subject to} \quad z_{\bm{g}} &\in \{0,1\} \\
    -Mz_{\bm{g}}\bm{1} &\leq \bm{\beta}_{\bm{g}} \leq Mz_{\bm{g}}\bm{1} \\
    \sum_{\bm{g}\in G} z_{\bm{g}} \le k
    z_\bm{g} &\le z_\bm{h}
    \end{array} \nonumber
\end{align}

where $z_\bm{g}$ is a binary slack variable that indicates whether the covariates in group
$\bm{g}$ are included in the model. The first set of inequality constraints ensure that
coefficients $\bm{\beta}_{\bm{g}}$ are nonzero if and only if their corresponding slack
variable $z_{\bm{g}} = 1$. The second inequality constraint introduces general
sparsity by ensuring that at most $k$ coefficients are nonzero. If $G$ includes only
singleton groups of covariates then the MIQP formulation is equivalent to the Best
Subset Selection problem. The final inequality constraint can be used to introduce
hierarchical structure into the model. Finally, we have also included an $\ell_2$
regularization term controlled by the hyperparameter $\lambda$.

# Statement of need

The `sparse-lm` Python package implements a variety of sparse linear regression models
based on convex objectives (generalizations of the Lasso) and mixed integer quadratic
programming objectives (generalizations of Best Subset Selection) that support a
flexible range of ways to introduce structured sparsity. The linear models in
`sparse-lm` are implemented to be compatible with `scikit-learn` [@Pedregosa:2011; @Buitinck:2013],
in order to interoperability with the wide range of tools and workflows available. The
objective problems in `sparse-lm` are implemented and solved using `cvxpy` [@Diamond:2016],
which permits a flexible use of a variety of open-source and proprietary solvers.

A handful of pre-existing Python libraries implement a subset of `scikit-learn`
compatible sparse linear regression models---some of which are included in `sparse-lm`.
`celer` [@Massias:2018] includes efficient implementations of the Lasso and Group Lasso,
among other linear models. `group-lasso` [@Moe:2020] is another `scikit-learn`
compatible implementation of the Group Lasso. `skglm` [@Bertrand:2022] includes several
implementations of sparse linear models based on regularization using combinations of
$\ell_p$ ($p\in\{1/2,2/3,1,2\}$) norms and pseudo-norms. `abess` [@Zhu:2022] includes an
implementation of Best Subset Selection and $\ell_0$ pseudo-norm regularization.

The pre-existing packages mentioned include highly performant implementations of the
specific models they implement; however, none of these packages implement the full range
of sparse linear models  available in `sparse-lm`, nore do they support the flexibility
to modify the optimization objective and choose among different available solvers.
`sparse-lm` therefore satisfies the need for a flexible and comprehensive library that
enables easy experimentation and comparisons of different sparse linear regression
algorithms.

# Usage

The linear regression models in `sparse-lm` are implemented to be compatible with
`scikit-learn` [@Pedregosa:2011; @Buitinck:2013], and so can be used independently, in
a hyperparameter selection class, or as part of a pipeline in the same way as any one
of the available models in the `sklearn.linear_model` module.

A variety of linear regression models with flexible regularization and feature selection
options are implemented in `sparse-lm`. The implemented models are listed below:

## Implemented regression models

- Lasso & Adaptive Lasso
- Group Lasso & Adaptive Group Lasso
- Sparse Group Lasso & Adaptive Sparse Group Lasso
- Ridged Group Lasso & Adaptive Ridge Group Lasso
- Best Subset Selection
- Ridged Best Subset Selection
- MIQP $\ell_0$ regularized regression
- MIQP $\ell_0\ell_2$ regularized regression

## Implemented model selection and composition tools
- One standard deviation rule grid search cross-validation
- Line search cross-validation
- Stepwise composite estimator

The package can be downloaded through the [Python Package Index](https://pypi.org/project/sparse-lm/).
Documentation, including an API reference and examples, can be found in the
[online documentation](https://cedergrouphub.github.io/sparse-lm).

# Acknowledgements

The first author (L.B.L.) is the lead developer of `sparse-lm`, and the lead and
corresponding author. The second author (F.X.) is a main contributor to the package.
Both authors drafted, reviewed and edited the manuscript.

L.B.L. and F.X. would like acknowledge the contributions from Peichen Zhong, Ronald L.
Kam, and Tina Chen to the development of `sparse-lm`. L.B.L gratefully acknowledges
support from the National Science Foundation Graduate Research Fellowship under Grant
No. DGE 1752814.

# References
