* ``-debug=MXNetHandleManualFree``

  When this new debug option is enabled, the library will report to
  ``stderr`` whenever a MXNet handle is freed by the ``Handle`` class
  destructor instead of by an explicit call to the ``freeHandle``
  method.  This can be used to track down leaks of MXNet-allocated
  resources in an application using dmxnet.
