## `NDarray` must have `null` handles when calling `Executor.outputs`

`mxnet.Executor`

When obtaining the outputs of an `Executor` all `NDarray` instances passed to
the method `outputs` must have a `null` handle. For migration you can call
`freeHandle` on `NDArray` instances with non-`null` handles
(`array.handle !is null`). This will release all data associated with the
`NDArray` instance. This freeing used to happen implicitly under the hood when
executing the `outputs` method. The migration requires this release of
resources to be explicit in calling code. This reflects a design principle that
the user should control the lifetime of resources while `dmxnet` states what it
assumes about those.

If the handles of the passed `NDArray` instances are `null` no changes need to
be made.
