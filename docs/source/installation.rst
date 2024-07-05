.. _installation:

============
Installation
============

At the moment we have verified installation for GreedyFHist only on linux system. Once we get ahold of Mac and Windows, we will test this as well.

------------
Requirements
------------

GreedyFHist is dependent on several Python packages which are all listed in ``pyproject.toml``. Otherwise, the only external dependency is the greedy software ``greedy``. 

--------------------
Installing using pip
--------------------

GreedyFHist can be installed by running the following command:

.. code-block:: bash

    pip install --user git+https://github.com/mwess/greedyfhist@master


----------------------------
Building GreedyFHist locally
----------------------------

GreedyFHist can be build locally using ``poetry``.

.. code-block:: bash

    poetry build
    pip install dist/greedyfhist-*.tar.gz

------------
Using Docker
------------

We also provided a Dockerfile for installing ``GreedyFHist``.


-----------------
Installing Greedy
-----------------

GreedyFHist has one external dependency, which is ``greedy`` (https://sites.google.com/view/greedyreg/about).

``greedy`` can be installed using the official instructions. It also comes with ITK-SNAP (http://www.itksnap.org). 

Installing Greedy via Docker
============================

If ``greedy`` cannot be installed using the official documentation, we have also provided a Dockerfile. 

Either build the Dockerfile locally and set an alias (for instance in your ``~/.bash.rc`` file).

.. code-block:: bash

    cd docker
    docker build -f Dockerfile_greedy -t greedy .

Setting the alias:

.. code-block:: bash

    alias greedy='docker run greedy'

Alternatively, a docker-image for ``greedyfhist`` can be downloaded::

.. code-block:: bash

    docker pull mwess89/greedyfhist:0.0.1


Or built:

.. code-block:: bash

    cd docker
    docker build -t greedyfhist -f Dockerfile_greedyfhist .
