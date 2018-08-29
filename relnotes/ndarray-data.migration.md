### Synchronized read-only data slice for `NDArray.data()`

`mxnet.NDArray`

The `data` method of `NDArray` is changed to return a read-only slice (i.e., a
slice of `const` elements). The added `const` implies that code writing
elements of the returned slice has to be changed. There is no direct
replacement. The closest option is to use the `copyFrom` method to copy all
elements of a slice into an `NDArray`.

Also the returned slice is synchronized. That means the method waits for all
pending writes to finish prior to returning. Code performing `waitToRead()`
(prior to calling `data`) manually should be removed.
