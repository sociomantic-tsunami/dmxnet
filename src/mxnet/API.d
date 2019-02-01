/*******************************************************************************

    Provides functionality for interacting with the underlying C API

    It is recommended that all calls to the MXNet C API be managed using the
    `invoke` function provided in this module.  This method will ensure that
    the C API is invoked correctly and that any failures in API functions are
    translated appropriately into D exceptions.

    Copyright:
        Copyright (c) 2017 dunnhumby Germany GmbH.

    License:
        Boost Software License Version 1.0.  See accompanying LICENSE.txt for
        details, or <https://www.boost.org/LICENSE_1_0.txt>

*******************************************************************************/

module mxnet.API;

import mxnet.c.c_api;
import mxnet.Exception;

import ocean.meta.types.Qualifiers;
import ocean.text.convert.Formatter;
import ocean.text.util.StringC;

version (unittest)
{
    import ocean.core.Test;
    import ocean.io.Stdout;
}

version (unittest)
{
    /***************************************************************************

        Notifies the MXNet engine to shutdown after leaving `main` for
        unittests

        Shutting down the engine limits a race when interacting with MXNet
        during process exit. The race happens when MXNet is cleaning up (on
        process exit).

        Note:
           This only reduces the probability of the race but it does not seem
           to eliminate them.

    ***************************************************************************/

    static ~this ()
    {
        notifyShutdown();
    }
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


/*******************************************************************************

    Reports the MXNet version

    This will report the used MXNet version, i.e., the version that is reported
    by the linked library.

    Returns:
        the used MXNet version

*******************************************************************************/

public MXNetVersion mxnetVersion ()
{
    int version_;
    invoke!(MXGetVersion)(&version_);
    assert(version_ >= 0);
    return mxnetVersion(version_);
}

unittest
{
    Stdout.formatln("Using MXNet version {}", mxnetVersion());
}


/// To represent MXNet versions
public struct MXNetVersion
{
    /// Major version
    uint major;

    /// Minor version
    ubyte minor;

    /// Patch version
    ubyte patch;


    /***************************************************************************

        Formats the MXNet version

        This method is used by `ocean.text.convert.Formatter` when this needs
        to formatted.

        Params:
            sink = delegate to write the formatting to

    ***************************************************************************/

    public void toString (scope FormatterSink sink)
    {
        sink("{}.{}.{}".format(this.major, this.minor, this.patch));
    }
}


/*******************************************************************************

    Converts the given MXNet version and returns it

    Params:
        mxnet_version = version encoded in `major * 10000 + minor * 100 + patch`

    Returns:
        the used MXNet version

*******************************************************************************/

private MXNetVersion mxnetVersion (uint mxnet_version)
{
    // major * 10_000 + minor * 100 + patch
    // see `MXNET_VERSION` is defined in `include/mxnet/base.h`
    MXNetVersion version_ =
    {
        major : mxnet_version / 10_000,
        minor : mxnet_version % 10_000 / 100,
        patch : mxnet_version % 100,
    };
    return version_;
}

unittest
{
    test!("==")(mxnetVersion(0), MXNetVersion(0, 0, 0));
    test!("==")(mxnetVersion(1), MXNetVersion(0, 0, 1));
    test!("==")(mxnetVersion(10000), MXNetVersion(1, 0, 0));
    test!("==")(mxnetVersion(10203), MXNetVersion(1, 2, 3));
    test!("==")(mxnetVersion(10209), MXNetVersion(1, 2, 9));

    // testing boundaries
    test!("==")(mxnetVersion(99), MXNetVersion(0, 0, 99));
    test!("==")(mxnetVersion(100), MXNetVersion(0, 1, 0));
    test!("==")(mxnetVersion(9900), MXNetVersion(0, 99, 0));
    test!("==")(mxnetVersion(10000), MXNetVersion(1, 0, 0));
    test!("==")(mxnetVersion(19999), MXNetVersion(1, 99, 99));
    test!("==")(mxnetVersion(99999), MXNetVersion(9, 99, 99));
    test!("==")(mxnetVersion(999999), MXNetVersion(99, 99, 99));
    test!("==")(mxnetVersion(4294960000), MXNetVersion(429496, 0, 0));
}
