==================
compas_rpc_example
==================

.. start-badges

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
    :target: https://github.com/yijiangh/compas_rpc_example/blob/master/LICENSE
    :alt: License MIT

.. end-badges

.. Write project description

*compas_rpc_example*: example scripts for calling cpython functions from 
Grasshopper/Rhino via `compas_rpc <https://compas-dev.github.io/main/api/compas.rpc.html>`_
and `compas_xfunc <https://compas-dev.github.io/main/api/generated/compas.utilities.XFunc.html#compas.utilities.XFunc>`_.

Installation
------------

.. Write installation instructions here

Prerequisites
^^^^^^^^^^^^^

0. Operating System: **Windows 10**
1. `Rhinoceros 3D 6.0 <https://www.rhino3d.com/>`_
    We will use Rhino / Grasshopper as a frontend for inputting
    geometric and numeric paramters, and use various python packages as the 
    computational backends. The tight integration between Grasshopper and python
    environments is made possible by the ``xfunc`` and ``rpc`` functions
    in `COMPAS <https://compas-dev.github.io/>`_.
2. `Git <https://git-scm.com/>`_
    We need ``Git`` for fetching required packages from github.
3. `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
    We will install all the required python packages using 
    `Miniconda` (a light version of Anaconda). Miniconda uses 
    **environments** to create isolated spaces for projects' 
    depedencies.

Working in a conda environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is recommended to set up a conda environment to create a clean, isolated space for
installing all the required python packages.

Have an existing conda environment with COMPAS (>=0.7.2)?
=========================================================

Activate your own conda environment and install the following depedencies:

::

    pip install pycddlib

We use these external libraries to demonstrate our examples. Now jump to 
`Package installaton & compas_rhino install`_ to install ``compas_rpc_example``
to your environment and connnect it to GH/Rhino.

First time working with a conda environment?
============================================

Type in the following commands in your Anaconda terminal 
(search for ``Anaconda Prompt`` in the Windows search bar):

::

    git clone https://github.com/yijiangh/compas_rpc_examples.git
    cd compas_rpc_example/test_envs
    conda env create -f rpc_example_ws.yml

Wait for the building process to finish, the command above will
fetch and build all the required packages, which will take some time
(1~2 mins).

Then, activate the newly created conda environment (with all the needed packages installed):

::

    conda activate rpc_example_ws

Package installaton & compas_rhino install
==========================================

Now, install ``compas_rpc_example`` to your conda environment
by:

::

    cd <path to where setup.py lives>
    pip install -e .

(Go the `Developing a package`_ session to learn more about
the difference between using with and without the ``-e``
flag here.)

Great - we are almost there! Now type `python` in your Anaconda Prompt, and test if the installation went well:

::

    >>> import compas

If that doesn't fail, you're good to go! Exit the python interpreter (either typing `exit()` or pressing `CTRL+Z` followed by `Enter`).

Now let's make all the installed packages available inside Rhino. Still from the Anaconda Prompt, type the following:

In order to make ``compas_rpc_example`` accessible in Rhino/Grasshopper,
we need the run the following commands in the Anaconda prompt first 
and then **restart Rhino**:

::

    python -m compas_rhino.install
    python -m compas_rhino.install -p compas_rpc_example

And you should be able to see outputs like:

::

   Installing COMPAS packages to Rhino 6.0 IronPython lib:
   IronPython location: C:\Users\<User Name>\AppData\Roaming\McNeel\Rhinoceros\6.0\Plug-ins\IronPython (814d908a-e25c-493d-97e9-ee3861957f49)\settings\lib

   compas               OK
   compas_rhino         OK
   compas_ghpython      OK
   compas_bootstrapper  OK
   # second time when you type with -p compas_rpc_example
   compas_rpc_example   OK

   Completed.

Congrats! 🎉 You are all set! 

Grasshopper examples can be found in ``gh_examples``.

Developing a package 
--------------------
If you are developing the python package and want to test
it in the GH/Rhino environment, you might want to avoid doing
``pip install .`` every time you make a change in the scripts.

In order to have this live editting - test workflow work, you
have to do the following **once**:

1. Activate your conda environment (bad habit if you work in the base!)
2. ``cd`` to the folder of your package (where ``setup.py`` lives) and
   type in ``pip install -e .``. This will install the package
   to your conda environment using the ``develop`` mode,
   which allows your change to be used immediately.
3. Run ``python -m compas_rhino.install -p <package_name>``

Restart Rhino, and you should be good to go!

**Important:**

The ``compas_rhino.install`` will try to find the package you specify
in ``-p <package name>`` by looking at all *installed* packages
in your python (conda) environment. So it is crucial to
have a ``setup.py`` file in your python package, which makes
your package ``pip``-installable.

Also, the return data from ``compas_rpc`` call needs to be
in a json serializable format (list, dictionary of primitive data types, etc), which
means that you cannot return class or other composed data types. Read more in the `compas_rpc_doc <https://compas-dev.github.io/main/api/compas.rpc.html>`_.

Troubleshooting 
---------------

Sometimes things don't go as expected. Here are some of answers to the most common issues you might bump into:

------------

..

    Q: When trying to install the framework in Rhino, it fails indicating the lib folder of IronPython does not exist.

Make sure you have opened Rhino 6 and Grasshopper at least once, so that it finishes setting up all its internal folder structure.

------------

..

    Q: Windows error in the Grasshopper rpc call.

Make sure you enter the correct python path in the GH file. An easy way to obtain
the path is to type ``where python`` in your conda prompt after you activate your conda environment.

------------

..

    Q: Runtime error: Fault in the Grasshopper rpc call.

Try the following:

1. If you have V-Ray installed, uninstall V-Ray for Rhinoceros and 
   restart your computer.
2. If the problem persists after retrying, first open your Task Manager and
   end all ``Python`` processes.

   Then in your activated conda environment, run:

    ::

        cd gh_examples
        python rpc_test.py

    It should print the following:

    ::

        Starting a new proxy server...
        New proxy server started.

    Then, retry opening the Grasshopper file.

------------

..

    Q: In Xfunc call, error message "Cannot find DLL specified. (_Arpack ...)"

This happens because some previous calls blocked the ``scipy`` complied libraries.
For a temporal fix, in your conda environment, uninstall ``pip install scipy`` and
then ``pip install scipy=1.3.1`` works.