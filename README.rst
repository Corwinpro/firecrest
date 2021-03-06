.. raw:: html

    <img src="firecrest_emblem3002.png" width="60">

Installing the **firecrest** package
------------------------------------

.. code-block:: bash

    git clone git@github.com:Corwinpro/firecrest.git
    cd firecrest
    pip install --user -e .


Setting up the **dolfin** environment
-------------------------------------

Install the dolfin and necessary packages and activate:

.. code-block:: bash

    conda create -n fenicsproject -c conda-forge fenics matplotlib meshio
    source activate fenicsproject


Deactivate:

.. code-block:: bash

    source deactivate fenicsproject


Additional packages
-------------------

Install **pysplines** `from here`_.

Logo by `Titov Fedor`_.

__
.. _from here:
    https://github.com/Corwinpro/PySplines
.. _Titov Fedor:
    https://www.artstation.com/quietvictories