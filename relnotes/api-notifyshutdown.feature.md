### Provide function to trigger shutdown of MXNet engine

`mxnet.API`

The function `notifyShutdown` is added to notify MXNet to shut down its
execution engine. You may consider shutting down the engine manually to
avoid/reduce races on process exit.
