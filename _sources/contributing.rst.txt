Contributing
============

We welcome all forms of contribution, please consider contributing in any way you can!

Bugs, issues, input, and questions
----------------------------------
Please use the
`issue tracker <https://github.com/CederGroupHub/sparse-lm/issues>`_ to share any
of the following:

-   Bugs
-   Issues
-   Questions
-   Feature requests
-   Ideas
-   Input

Having these reported and saved in the issue tracker is very helpful to make
sure that they are properly addressed. Please make sure to be as descriptive
and neat as possible when opening up an issue.

Developing guidelines
---------------------
If you have written code or want to start writing new code that you think will improve **sparse-lm** then please follow
the steps below to make a contribution.

* All code should have unit tests.
* Code should be well documented following `google style <https://google.github.io/styleguide/pyguide.html>`_  docstrings.
* All code should pass the pre-commit hook. The code follows the `black code style <https://black.readthedocs.io/en/stable/>`_.
* Estimators should follow scikit-learn's `developing estimator guidelines <https://scikit-learn.org/stable/developers/develop.html>`_.

Adding code contributions
-------------------------

#.  If you are contributing for the first time:

    * *Fork* the repository and then *clone* your fork to your local workspace.
    * Make sure to add the *upstream* repository as a remote::

        git remote add upstream https://github.com/CederGroupHub/sparse-lm.git

    * You should always keep your ``main`` branch or any feature branch up to date
      with the upstream repository ``main`` branch. Be good about doing *fast forward*
      merges of the upstream ``main`` into your fork branches while developing.

#.  In order to have changes available without having to re-install the package:

    * Install the package in *editable* mode::

         pip install -e .

#.  To develop your contributions you are free to do so in your *main* branch or any feature
    branch in your fork.

    * We recommend to only your forks *main* branch for short/easy fixes and additions.
    * For more complex features, try to use a feature branch with a descriptive name.
    * For very complex feautres feel free to open up a PR even before your contribution is finished with
      [WIP] in its name, and optionally mark it as a *draft*.

#.  While developing we recommend you use the pre-commit hook that is setup to ensure that your
    code will satisfy all lint, documentation and black requirements. To do so install pre-commit, and run
    in your clones top directory::

        pre-commit install

    *  All code should use `google style <https://google.github.io/styleguide/pyguide.html>`_ docstrings
       and `black <https://black.readthedocs.io/en/stable/?badge=stable>`_ style formatting.

#.  Make sure to test your contribution and write unit tests for any new features. All tests should go in the
    ``sparse-lm\tests`` directory. The CI will run tests upon opening a PR, but running them locally will help find
    problems before::

        pytests tests
