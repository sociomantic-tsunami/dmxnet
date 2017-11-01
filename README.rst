Description
===========

``dmxnet`` provides D bindings for the `MXNet <http://mxnet.io/>`_ deep learning
library, together with a friendly D wrapper library to allow more idiomatic use
within D libraries and applications.

The currently implemented bindings and wrapper library cover only a small core
collection of the full MXNet functionality: we are not aiming for comprehensive
support for everything MXNet can do, only good support for the main essentials.
The current D API is written in D1, with the possibility to auto-convert to D2.

Further design constraints include:

* ``dmxnet`` is not cross-platform: we are only supporting Linux at present

* ``dmxnet`` is not written to support multithreading D code, and has only
  been tested with a single-threaded ``libmxnet``


Build/Use
=========

Dependencies
------------

Packages
********

``libmxnet`` (``-lmxnet``) v0.10.x is required to build, and can be installed
from the package provided in our `deb repository
<https://bintray.com/sociomantic-tsunami/mxnet/libmxnet>`_.

``zlib`` (``-lz``) is required in order to run integration tests, and can be
installed (on Debian/Ubuntu systems) via the ``zlib1g-dev`` package.

Submodules
**********

========== =======
Dependency Version
========== =======
makd       v2.0.x
ocean      v3.5.x
========== =======

``beaver`` 0.2.x is used by the Travis CI setup, but this submodule can be
ignored for general development or use of the library.

If you plan to use the provided ``Makefile`` (you will need it to convert code
to D2 or to run integration tests), you will also need to check out submodules
with ``git submodule update --init``.  This will fetch source code for `Makd
<https://github.com/sociomantic-tsunami/makd>`_ (used by the ``Makefile``)
into ``submodules/makd``.  It will also check out the source code for ``ocean``
and ``beaver``.

Conversion to D2
----------------

By default all ``dmxnet`` development is done in D1, but using a subset that is
almost D2 compatible, and can be auto-converted to D2 using `d1to2fix
<https://github.com/sociomantic-tsunami/d1to2fix>_`.  If you are using ``Makd``
there is a handy built-in target you can use::

  make d2conv

You will need to use the `dmd-transitional
<https://github.com/sociomantic-tsunami/dmd-transitional>`_ compiler to build
the converted D2 code.


Versioning
==========

``dmxnet`` follows the `Neptune
<https://github.com/sociomantic-tsunami/neptune/blob/master/doc/library-user.rst>`_
semantic versioning scheme.  Note that while it is in early ``0.x`` development,
increases to the minor version number may include both new features and breaking
changes.  Patch releases are unlikely at this stage but should contain only
bugfixes.

Once a ``1.0.0`` version has been released, breaking changes will only be made
when increasing the major version number.

Support Guarantees
------------------

Given the early ``0.x`` state of development, only the latest minor release will
be supported.  This policy will be updated once a ``1.0.0`` release has been
made.


Releases
========

`Latest release notes
<https://github.com/sociomantic-tsunami/dmxnet/releases/latest>`_ | `All
releases <https://github.com/sociomantic-tsunami/dmxnet/releases>`_

Releases are handled using GitHub releases.  The notes associated with a major
or minor github release are designed to help developers to migrate from one
version to another. The changes listed are the steps you need to take to move
from the previous version to the one listed.

The release notes are structured in 3 sections, a **Migration Instructions**,
which are the mandatory steps that users have to do to update to a new version,
**Deprecated** which contains deprecated functions that are recommended not to
use but will not break any old code, and the **New Features** which are optional
new features available in the new version that users might find interesting.
Using them is optional, but encouraged.


Contributing
============

See the guide for `contributing to Neptune-versioned libraries
<https://github.com/sociomantic-tsunami/neptune/blob/master/doc/library-contributor.rst>`_.
