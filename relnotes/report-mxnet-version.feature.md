### Provide used MXNet version for reporting

`mxnet.API`

The function `mxnetVersion` is added which allows querying for the used MXNet
version, i.e., the MXNet version as reported by the linked MXNet library. This
is useful to verify which version of MXNet is being used. This information can
be used when migrating from one MXNet version to another. We also add a
unittest which reports the used MXNet version.
