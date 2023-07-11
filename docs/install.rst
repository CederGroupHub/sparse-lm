Install
=======

**sparse-lm** can be installed from PyPI or from source using pip.

PyPI
----

You can install **sparse-lm** using pip::

   pip install sparse-lm


Install from source
-------------------

To install **sparse-lm** from source, (fork and) clone the repository from `github
<https://github.com/CederGroupHub/sparse-lm>`_::

    git clone https://github.com/CederGroupHub/sparse-lm
    cd sparselm
    pip install .

Installing MIQP solvers
-----------------------

Since **cvxpy** is used to specify and solve regression optimization problems, any of
`supported solvers <https://www.cvxpy.org/tutorial/advanced/index.html#solve-method-options>`_
can be used with **sparse-lm** estimators. **cvxpy** is shipped with open source solvers
(OSQP, SCS, and ECOS) which are usually enough to solve most convex regression problems.

However, for the mixed integer quadratic programming (MIQP) formulations used in
:class:`BestSubsetSelection` and :class:`RegularizedL0` based classes we highly
recommend installing an MIQP capable solver. ECOS_BB can be used to solve MIQP problems,
but it can be very slow and more importantly has recurring correctness issues. See the
`mixed-integer program section <https://www.cvxpy.org/version/1.2/tutorial/advanced/index.html#mixed-integer-programs>`_
in the cvxpy documentation for more details.

Gurobi
^^^^^^

For using **sparse-lm** with MIQP solvers, we highly recommend installing **Gurobi**.
It can be installed directly from PyPi::

    pip install gurobipy

Without a license, a free trial **Gurobi** can be used to solve small problems. For
larger problems a license is required. **Gurobi** grants
`free academic licenses <https://www.gurobi.com/academia/academic-program-and-licenses/>`_
to students and academic researchers.

SCIP
^^^^

If installing a licensed solver is not an option, **SCIP** can be used as a free
alternative. To use **SCIP**, the python interface **PySCIPOpt** must also be installed.
**PySCIPOpt** can be installed from PyPi, however this requires building SCIP from
source. See installation details `here <https://github.com/scipopt/PySCIPOpt>`_.

If you use conda, we recommend installing **SCIP** and **PySCIPOpt** using their
conda-forge channel::

    conda install -c conda-forge scipopt pyscipopt

The above command will install **PySCIPOpt** with a pre-built version of **SCIP**, and
so you will not need to build it from source.

Testing
-------

Unit tests can be run from the source folder using ``pytest``. First, the requirements
to run tests must be installed::

    pip install .[tests]

Then run the tests using::

    pytest tests
