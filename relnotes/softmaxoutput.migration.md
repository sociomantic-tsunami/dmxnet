* `mxnet.Symbol.SoftmaxOutput`

  A new nested `Config` class has been added in order to store the various
  optional settings that can be used with this symbol.  The constructor has
  been rewritten to use an instance of this struct (initialized to default
  values) rather than to require each parameter separately.  This should
  make it easier to instantiate `SoftmaxOutput` instances with only one or
  a few custom settings, for example:

  ```D
  SoftmaxOutput.Config config = {
      grad_scale: 1.5,
      preserve_shape: true
  };
  auto softmax_custom = new SoftmaxOutput(input, label, config);
  ```

  The config parameter is entirely optional: instances created using the
  default configuration via

  ```D
  auto softmax_default = new SoftmaxOutput(input, label);
  ```

  will behave exactly the same as before.
