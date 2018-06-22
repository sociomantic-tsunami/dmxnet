/*******************************************************************************

    Provides functionality for interacting with the underlying C API

    It is recommended that all calls to the MXNet C API be managed using the
    `invoke` function provided in this module.  This method will ensure that
    the C API is invoked correctly and that any failures in API functions are
    translated appropriately into D exceptions.

    Copyright:
        Copyright (c) 2017 sociomantic labs GmbH.

    License:
        Boost Software License Version 1.0.  See accompanying LICENSE.txt for
        details, or <https://www.boost.org/LICENSE_1_0.txt>

*******************************************************************************/

module mxnet.API;

import mxnet.c.c_api;
import mxnet.Exception;

import ocean.text.util.StringC;
import ocean.transition;

version(UnitTest)
{
    import ocean.core.Test;
}


/*******************************************************************************

    Type of exception that will be thrown in the event that a call to a MXNet
    C API function fails

    In the typical use-case only one global reusable instance of this exception
    will ever exist per thread, and it will only ever be thrown from within the
    `invoke` method defined elsewhere in this module.  When catching thrown
    exceptions of this type, take care to not store shallow copies, since the
    exception data may be rewritten in place by other API failure events.

*******************************************************************************/

public class MXNetAPIException : MXNetException
{
}


/*******************************************************************************

    Exception instance to be thrown in the event that a call to a MXNet C API
    function fails

    Since this is a single reusable instance, take care to not store shallow
    copies of it, since the exception data will be rewritten in-place by other
    API failure events.

*******************************************************************************/

private MXNetAPIException mxnet_api_exception;

private static this ()
{
    mxnet_api_exception = new MXNetAPIException;
}


/*******************************************************************************

    Invokes a function from the MXNet C API, passing the specified arguments
    and handling any errors that result

    Params:
        MXNetAPIFunction = function from the MXNet C API to call
        file = file from which the call originates
        line = line from which the call originates
        Args = variadic list of the types of arguments to be passed when
               calling `MXNetAPIFunction`
        args = arguments to pass to `MXNetAPIFunction`

    Throws:
        unique `mxnet_api_exception` instance of `MXNetAPIException` if the
        call to `MXNetAPIFunction` fails (i.e. returns any value other than 0).

        Take care when catching the thrown exception not to store any shallow
        copies, since the reusable exception data will be overwritten in-place
        the next time a call to this function fails.

*******************************************************************************/

public void invoke (alias MXNetAPIFunction,
                    istring file = __FILE__, int line = __LINE__,
                    Args...) (Args args)
{
    // using a single reusable exception instance allows us
    // to avoid generating garbage when throwing and when
    // copying the output of `MXGetLastError()`
    mxnet_api_exception.enforce(MXNetAPIFunction(args) == 0,
                                StringC.toDString(MXGetLastError()),
                                file, line);
}

unittest
{
    int mxNetFails () { return 1; }

    testThrown!(MXNetAPIException)(invoke!(mxNetFails)());

    int mxNetSucceeds (int* n)
    {
        assert(n !is null);
        *n = 42;
        return 0;
    }

    int answer = 6 * 9;
    invoke!(mxNetSucceeds)(&answer);
    test!("==")(answer, 42);
}


/*******************************************************************************

    Notifies MXNet to shutdown

    By calling this function MXNet is notified to shut down its engine. When
    the engine is shut down MXNet will not execute pending calculations.

*******************************************************************************/

public void notifyShutdown ()
{
    invoke!(MXNotifyShutdown)();
}
