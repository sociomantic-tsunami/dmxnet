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


D2 compatibility
================

By default all ``dmxnet`` development is done in D1, but using a subset that is
almost D2 compatible, and can be auto-converted to D2 using ``d1to2fix`` (e.g.
via makd's ``make d2conv`` target).  The ``dmd-transitional`` compiler provided
in <https://github.com/sociomantic-tsunami/dmd-transitional> must be used to
build the converted D2 code.


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
