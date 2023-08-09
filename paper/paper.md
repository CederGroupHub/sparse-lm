---
title: 'sparse-lm: Sparse linear regression models in Python'
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

Sparse linear regression models are a powerful tool for capturing linear relationships
in high dimensional spaces. Sparse models have only a small number of nonzero parameters
(even if the number of covariates used in estimation is large), and as a result can be
easier to fit and interpret compared to dense models [@Hastie:2015]. Regression objectives
resulting in sparse linear models such as the Lasso [@Tibshirani:1996; @Zou:2006] and
Best Subset Selection [@Hocking:1967] have been widely used in a variety of fields.
However, many regression problems involve covariates that have a natural underlying
structure, such as group or hierarchical relationships between covariates, that can be
leveraged to obtain improved model performance and interpretability. A common example of
linear regression problems with sparsity structure occurs in chemistry and materials
science when fitting multi-body expansions that involve a hierarchy among the main
effects from chemical composition and higher order corrections
aiming to capture the effects of chemical interactions [@Leong:2019; @Barroso-Luque:2022].
Several generalizations of the Lasso [@Yuan:2006; @Friedman:2010; @Simon:2013; @Wang:2019]
and Best Subset Selection [@Bertsimas:2016-a; @Bertsimas:2016-b] have been developed to
effectively exploit additional structure in linear regression.

![Schematic of a linear model with grouped covariates with hierarchical relations.
Groups of covariates are represented with different colors and hierarchical
relationships are represented with arrows (i.e. group 3 depends on group 1). The figure
was inspired by Ref. [@Richie-Halford:2021].](linear-model.pdf){ width=55% }

# Statement of need

The `sparse-lm` Python package implements a variety of sparse linear regression models
based on convex objectives (generalizations of the Lasso) and mixed integer quadratic
programming objectives (generalizations of Best Subset Selection) that support a
flexible range of ways to introduce structured sparsity. The linear models in
`sparse-lm` are implemented to be compatible with `scikit-learn` [@Pedregosa:2011; @Buitinck:2013],
in order to enable interoperability with the wide range of tools and workflows available.
The regression optimization problems in `sparse-lm` are implemented and solved using
`cvxpy` [@Diamond:2016], which allows users to choose from a variety of well-established
open-source and proprietary solvers. In particular, for regression problems with mixed
integer programming objectives, access to state-of-the-art proprietary solvers enables
solving larger problems that would otherwise be unsolvable within reasonable time limits.

A handful of pre-existing Python libraries implement a handful of sparse linear
regression models that are also `scikit-learn` compatible. `celer` [@Massias:2018] and
`groupyr` [@Richie-Halford:2021] include efficient implementations of the Lasso and
Group Lasso, among other linear models. `group-lasso` [@Moe:2020] is another
`scikit-learn` compatible implementation of the Group Lasso. `skglm` [@Bertrand:2022]
includes several implementations of sparse linear models based on regularization using
combinations of $\ell_p$ ($p\in\{1/2,2/3,1,2\}$) norms and pseudo-norms.
`abess` [@Zhu:2022] includes an implementation of Best Subset Selection and $\ell_0$
pseudo-norm regularization.

The pre-existing packages mentioned include highly performant implementations of the
specific models they implement. However, none of these packages implement the full range
of sparse linear models  available in `sparse-lm`, nore do they support the flexibility
to modify the optimization objective and choose among many open-source and commerically
available solvers. `sparse-lm` satisfies the need for a flexible and comprehensive
library that  enables easy experimentation and comparisons of different sparse
linear regression algorithms within a single package.

# Background

Structured sparsity can be introduced into regression problems in one of two ways. The
first method to obtain structured sparsity is by using regularization by way of
generalizations of the Lasso, such as the Group Lasso and the Sparse Group
Lasso [@Yuan:2006; @Friedman:2010; @Simon:2013; @Wang:2019]. The Sparse Group Lasso
regression problem can be expressed as follows,

\begin{equation}
    \mathbf{\beta}^* = \underset{\mathbf{\beta}}{\text{argmin}}\;||\mathbf{X}
    \mathbf{\beta} - \mathbf{y}||^2_2 + (1-\alpha)\lambda\sum_{\mathbf{g}\in G}\sqrt{|\mathbf{g}
    }||\mathbf{\beta}_{\mathbf{g}}||_2 + \alpha\lambda||\mathbf{\beta}||_1
\end{equation}

where $\mathbf{X}$ is the design matrix, $\mathbf{y}$ is the response vector, and
$\mathbf{\beta}$ are the regression coefficients. $\mathbf{g}$ are groups of
covariate indices, $G$ is the set of all such groups being considered, and
$\mathbf{\beta}_{\mathbf{g}}\in\mathbb{R}^{|\mathbf{g}|}$ are the covariate coefficients
in group $\mathbf{g}$. $\lambda \in \mathbb{R}_{+}$ and $\alpha\in[0,1]$  are regularization
hyperparameters. The parameter $\alpha\in[0,1]$ controls the relative weight between the
single covariate $\ell_1$ regularization and the group regularization term. When
$\alpha=0$, the regression problem reduces to the Group Lasso objective, and when $\alpha=1$,
the problem reduces to the Lasso objective.

The (Sparse) Group Lasso can be directly used to obtain a grouped sparsity pattern.
Hierarchical sparsity patterns can be obtained by extending the Group Lasso to allow
overlapping groups, which is referred to as the Overlap Group Lasso [@Hastie:2015].

The second method to obtain structured sparsity is by introducing linear constraints
into the regression objective. Introducing linear constraints is straight-forward in
mixed integer quadratic programming (MIQP) formulations of the Best Subset Selection
[@Bertsimas:2016-a; @Bertsimas:2016-b]. The general MIQP formulation of Best Subset
Selection with group and hierarchical structure can be expressed as follows,

\begin{align}
    \mathbf{\beta}^* = \underset{\mathbf{\beta}}{\text{argmin}}\;
    & \mathbf{\beta}^{\top}\left(\mathbf{X}^{\top}\mathbf{X} +
    \lambda\mathbf{I}\right)\mathbf{\beta} - 2\mathbf{y}^{\top}\mathbf{X}\mathbf{\beta}\\
    &\begin{array}{r@{~}l@{}l@{\quad}l}
    \text{subject to} \quad &z_{\mathbf{g}} \in \{0,1\} \\
    &-Mz_{\mathbf{g}}\mathbf{1} \leq \mathbf{\beta}_{\mathbf{g}} \leq Mz_{\mathbf{g}}\mathbf{1} \\
    &\sum_{\mathbf{g}\in G} z_{\mathbf{g}} \le k \\
    &z_\mathbf{g} \le z_\mathbf{h}
    \end{array} \nonumber
\end{align}

where $z_\mathbf{g}$ are binary slack variables that indicate whether the covariates in
each group $\mathbf{g}$ are included in the model. The first set of inequality constraints
ensure that coefficients $\mathbf{\beta}_{\mathbf{g}}$ are nonzero if and only if their
corresponding slack variable $z_{\mathbf{g}} = 1$. $M$ is a fixed parameter that can be
estimated from the data [@Bertsimas:2016-a]. The second inequality constraint
introduces general sparsity by ensuring that at most $k$ coefficients are nonzero. If
$G$ includes only singleton groups of covariates then the MIQP formulation is equivalent
to the Best Subset Selection problem; otherwise it is a generalization that enables
groups-level sparsity structure. The final inequality constraint can be used to
introduce hierarchical structure into the model. Finally, we have also included an
$\ell_2$ regularization term controlled by the hyperparameter $\lambda$, which is useful
when dealing with poorly conditioned design matrices.

# Usage

Since the linear regression models in `sparse-lm` are implemented to be compatible with
`scikit-learn` [@Pedregosa:2011; @Buitinck:2013], they can be used independently or as
part of a workflow---such as in a hyperparameter selection class or a pipeline---
in similar fashion to any of the available models in the `sklearn.linear_model` module.

A variety of linear regression models with flexible regularization and feature selection
options are implemented. The implemented models are listed below:

## Implemented regression models

- Lasso & Adaptive Lasso
- Group Lasso & Adaptive Group Lasso
- Sparse Group Lasso & Adaptive Sparse Group Lasso
- Ridged Group Lasso & Adaptive Ridge Group Lasso
- Best Subset Selection
- Ridged Best Subset Selection
- $\ell_0$ pseudo-norm regularized regression
- $\ell_0\ell_2$ mixed-norm regularized regression

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
