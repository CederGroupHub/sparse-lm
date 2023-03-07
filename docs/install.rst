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

Testing
-------

Unit tests can be run from the source folder using ``pytest``. First, the requirements
to run tests must be installed::

    pip install .[tests]

Then run the tests using::

    pytest tests
