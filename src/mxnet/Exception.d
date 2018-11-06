/*******************************************************************************

    Provides common functionality for handling errors encountered when using
    MXNet

    Copyright:
        Copyright (c) 2017 dunnhumby Germany GmbH.

    License:
        Boost Software License Version 1.0.  See accompanying LICENSE.txt for
        details, or <https://www.boost.org/LICENSE_1_0.txt>

*******************************************************************************/

module mxnet.Exception;

import mxnet.c.c_api;

import ocean.core.Exception;
import ocean.text.util.StringC;
import ocean.transition;

version (unittest)
{
    import ocean.core.Test;
}


/*******************************************************************************

    Base class for exceptions to be thrown if any MXNet operation fails

    All `MXNetException` instances are potentially reusable, so when catching
    thrown exceptions derived from this type, take care never to store shallow
    copies, as the exception data may be rewritten in place by later throws of
    the same instance.

*******************************************************************************/

public class MXNetException : Exception
{
    mixin ReusableExceptionImplementation!();
}
