/******************************************************************************

    Defines a class to encapsulate and manage MXNet handles

    Copyright:
        Copyright (c) 2017 sociomantic labs GmbH.

    License:
        Boost Software License Version 1.0.  See accompanying LICENSE.txt for
        details, or <https://www.boost.org/LICENSE_1_0.txt>

******************************************************************************/

module mxnet.Handle;

import mxnet.Exception;

import core.sys.posix.unistd;

debug (MXNetHandle) import ocean.io.Stdout;
import ocean.text.convert.Formatter;
import ocean.transition;

version (UnitTest)
{
    import core.stdc.stdlib;

    import ocean.core.Test;
}


/******************************************************************************

    Exception to be thrown in case of an error involving an MXNet handle
    instance (e.g. if a handle is null when it should not be)

    In the typical use-case only one global reusable instance of this exception
    will ever exist per thread, and it will only ever be thrown from within the
    `MXNetHandle` class defined in this module.  When catching exceptions of
    this type, take care to not store shallow copies, since the exception data
    may later be rewritten in place by other handle-related failures.

******************************************************************************/

public class MXNetHandleException : MXNetException
{
}


/******************************************************************************

    Exception instance to be thrown in case of an error involving an MXNet
    handle instance (e.g. if a handle is null when it should not be)

    Since this is a reusable instance, take care not to store shallow copies
    of it, since the exception data will be rewritten in place by other later
    handle-related failures.

******************************************************************************/

package MXNetHandleException mxnet_handle_exception;

private static this ()
{
    mxnet_handle_exception = new MXNetHandleException;
}


/******************************************************************************

    Count of the total number of MXNetHandle instances (of any type) wrapping
    a non-null handle pointer.  This should match the total number of handles
    allocated by the D library.  Should be >= 0; if its value ever becomes
    negative, this indicates a bug in the D library code where some handles
    are being wrapped without that being tracked.

******************************************************************************/

private long mxnet_handle_count = 0;


/******************************************************************************

    Module destructor that outputs an error message if `mxnet_handle_count`
    is non-zero when it is invoked.  This offers an extra safety check to
    confirm that all MXNet C API handles are being freed correctly before
    a program exits.

    Note that it is entirely possible for this module destructor to be called
    before the destructors of individual `MXNetHandle` class instances.  This
    means that the error message may trigger even if a program is cleaning up
    after itself adequately.  Build with `-debug=MXNetHandle` to track down
    any apparently unfreed handles.

******************************************************************************/

static ~this ()
{
    if (mxnet_handle_count != 0)
    {
        char[60] buf; // should be enough for message + any `long` value
        auto msg = snformat(buf, "{} MXNet handle(s) in use at shutdown!\n",
                            mxnet_handle_count);
        write(STDERR_FILENO, msg.ptr, msg.length);
    }
}


/******************************************************************************

    Returns:
        Current number of MXNet handles allocated by the D library.

        This should be >= 0; if it is negative, this indicates a bug in the
        D library code where some handle allocation is not being tracked.

******************************************************************************/

public long handleCount ()
{
    return mxnet_handle_count;
}


/******************************************************************************

    Manages an MXNet handle

    The C API of MXNet defines handles which reference MXNet objects. A single
    handle corresponds to a specific instance. For example when creating an
    n-dimensional array a handle is returned to reference the created array.
    For each kind of MXNet object the C API defines functions expecting
    an handle to communicate which object the function is applied to. Handles
    should be freed to clean up their resources. This class implements a
    function for freeing the underlying resources manually. This function is
    called on destruction to enable automatic release of the resources.

    Requesting MXNet resources quickly but not releasing them in a timely
    manner may result in resource over-consumption. Making it may seem as if
    there are not enough resources left when in fact they only haven't been
    reclaimed yet. To reclaim the MXNet resource (handled by an MXNetHandle)
    quickly after its usage consider manually freeing the handle. You can also
    use `scope (exit)` here. Another approach is to `scope` allocate the handle
    thereby calling its destructor when the handle goes out of scope.  Note
    when relying on the GC for destruction you have no control when the
    destructor is actually called, hence when the resource is freed.

    Params:
        HandleType = type of the underlying handle
        FreeHandleFunction = function to free the handle

******************************************************************************/

public class MXNetHandle (HandleType, alias FreeHandleFunction)
{
    import core.stdc.stdlib : abort;

    import mxnet.API;

    import ocean.core.Traits;

    /***************************************************************************

        Underlying handle used to interact with C library

        The underlying handle is not allowed to be shared between different
        instances, otherwise the handle could be freed more than once.

        The handle is allowed to be null, in which case the instance will be
        considered to be empty, i.e., it does not reference any object.

    ***************************************************************************/

    private HandleType c_api_handle;


    /***************************************************************************

        Constructs an MXNetHandle from a given handle

        You should never construct different instances with the same handle,
        because freeing a handle of one instance would leave the other
        instances with a dangling handle.

        Params:
            mxnet_handle = C API handle; the handle can be null to pre-allocate
                           a class instance whose handle will be set later

    ***************************************************************************/

    public this (HandleType mxnet_handle)
    out
    {
        assert(this.handle() is mxnet_handle);
    }
    body
    {
        if (mxnet_handle !is null)
        {
            this.handle(mxnet_handle);
        }
    }


    /***************************************************************************

        Applies a function to the underlying handle, together with the provided
        function arguments

        Params:
            MXNetFunction = MXNet function to apply to the wrapped handle;
                            it is assumed that the handle will be the first
                            function parameter
            file = file from which the call originates
            line = line from which the call originates
            Args = variadic list of the types of arguments to be passed when
                   calling `MXNetFunction` in addition to the handle itself
            args = arguments to pass to `MXNetFunction` in addition to the
                   wrapped handle

        Throws:
            `MXNetHandleException` if the underlying handle is null;
            `MXNetAPIException` if the call to `MXNetFunction` fails

            Take care not to store shallow copies of the `MXNetAPIException`
            instance, as it is a unique reusable instance whose data may be
            rewritten in-place by later API failures.

    ***************************************************************************/

    public void apply (alias MXNetFunction,
                       istring file = __FILE__, int line = __LINE__,
                       Args...) (Args args)
    {
        mxnet_handle_exception.enforce(this.exists(),
            "Cannot apply " ~ identifier!(MXNetFunction) ~ " to null handle!",
            file, line);

        invoke!(MXNetFunction, file, line)(this.c_api_handle, args);
    }

    unittest
    {
        scope null_handle = new MXNetHandle!(int*, freeMXNetHandleMemory)(null);
        testThrown!(MXNetHandleException)(null_handle.apply!(mxNetHandleFails)());
        testThrown!(MXNetHandleException)(null_handle.apply!(mxNetHandleSucceeds)(5));

        int* h = cast(int*) malloc(int.sizeof);
        scope valid_handle = new MXNetHandle!(int*, freeMXNetHandleMemory)(h);
        testThrown!(MXNetAPIException)(valid_handle.apply!(mxNetHandleFails)());

        valid_handle.apply!(mxNetHandleSucceeds)(6);
        test!("==")(*h, 6);

        valid_handle.apply!(mxNetHandleSucceeds)(7);
        test!("==")(*h, 7);
    }


    /***************************************************************************

        Gives access to the underlying handle

        Use this handle with care. This is provided as a mechanism to use the C
        API directly when needed.

        Returns:
            the underlying handle used with the C library

    ***************************************************************************/

    public HandleType handle ()
    {
        return this.c_api_handle;
    }


    /***************************************************************************

        Returns true if this handle references an underlying objects

        When no underlying object is referenced the handle is null.

        Returns:
            true if the handle is non-null, otherwise false

    ***************************************************************************/

    public bool exists () /* d1to2fix_inject: const */
    {
        return this.c_api_handle !is null;
    }


    /***************************************************************************

        Sets the underlying handle

        This frees any existing underlying handle and replaces it with the
        provided handle.

        Note that you must ensure that no two instances share the same handle
        as this would result in double freeing of the same handle.

        Params:
            mxnet_handle = the C API handle this instance's handle is to be
                           set to
            file = file name where this function is called from; defaults to
                   __FILE__
            line = line number in file where this function is called from;
                   defaults to __LINE__

    ***************************************************************************/

    public void handle (HandleType mxnet_handle,
                        istring file = __FILE__, uint line = __LINE__)
    {
        // if the 'new' handle is the same as the existing one, then the latter
        // should not be freed, as it would leave us with a dangling reference
        if (this.c_api_handle is mxnet_handle)
        {
            debug (MXNetHandle)
            {
                Stderr.formatln("Attempting to overwrite handle {} with itself "
                                    ~ "at {}:{}",
                                this.c_api_handle, file, line);
            }
            return;
        }

        this.freeHandle();
        this.c_api_handle = mxnet_handle;

        // increment the handle count if the new handle is non-null
        if (this.c_api_handle !is null)
        {
            ++mxnet_handle_count;
            debug (MXNetHandle)
            {
                Stderr.formatln("New handle: {} ({} allocated in total)",
                                this.c_api_handle, handleCount());
            }
        }
    }

    unittest
    {
        alias void* HandleType;
        alias MXNetHandle!(HandleType, freeMXNetHandleMemory) Memory;
        {
            HandleType some_handle = malloc(1);
            scope h = new Memory(some_handle);
            test(h.handle !is null);
            h.handle = null;
            test(h.handle is null);
        }
        // change the handle to itself testing the issue of double freeing
        {
            HandleType some_handle = malloc(1);
            scope h = new Memory(some_handle);
            test(h.handle !is null);
            h.handle = some_handle;
            test(h.handle is some_handle);
        }
    }


    /***************************************************************************

        Frees the underlying handle

    ***************************************************************************/

    public void freeHandle ()
    {
        // failure to free will abort, so we can decrement
        // the handle count before the free attempt
        if (this.c_api_handle !is null)
        {
            --mxnet_handle_count;
            debug (MXNetHandle)
            {
                // using fixed stack buffer, `snformat` and `write` should
                // avoid allocation, which (whether GC or not) might cause
                // problems if this method is called from a destructor
                char[80] buf; // enough space for message, `hash_t` and `long`
                auto free_handle_msg =
                    snformat(buf, "Freeing handle: {:x} ({} still allocated)\n",
                             this.c_api_handle, handleCount());
                write(STDERR_FILENO,
                      free_handle_msg.ptr, free_handle_msg.length);
            }

            if (handleCount() < 0)
            {
                // we cannot throw (even an Error) from here, since this
                // method may be called by the destructor, so we use the
                // C stdlib to output an error message before aborting
                const istring negative_handles_msg =
                    "Allocated handle count is negative!\n";
                write(STDERR_FILENO,
                      negative_handles_msg.ptr, negative_handles_msg.length);
                abort();
            }
        }

        if (FreeHandleFunction(this.c_api_handle) != 0)
        {
            // we cannot throw (even an Error) from here, since this
            // method may be called by the destructor, so we use C
            // stdlib to output an error message before aborting
            const istring msg = "Could not free MXNet handle using " ~
                                identifier!(FreeHandleFunction) ~ "@" ~
                                __FILE__ ~ ":" ~ __LINE__.stringof ~ "\n";
            write(STDERR_FILENO, msg.ptr, msg.length);
            abort();
        }

        this.c_api_handle = null;
    }


    /***************************************************************************

        Destructs this handle freeing its underlying resources

    ***************************************************************************/

    public ~this ()
    {
        debug (MXNetHandleManualFree)
        {
            if (this.exists())
            {
                // Relying on the GC calling this destructor to free the handle
                // is problematic since it may not be in-time. For this reason,
                // we recommend users of `MXNetHandle` to free MXNet resources
                // by calling `freeHandle` manually, and never relying on the
                // destructor.
                //
                // If this policy is followed, then the handle should already
                // have been freed by the time the destructor is initiated.
                // This debug check should help in tracking down any handles
                // for which this has not been done.
                sformat((cstring str) { write(STDERR_FILENO, str.ptr, str.length); },
                        "Non-null handle {:x} in MXNetHandle destructor!\n",
                        this.c_api_handle);
            }
        }
        this.freeHandle();
    }
}


version (UnitTest)
{
    /***************************************************************************

        Frees memory using `free`

        This function is used in testing as an example for a
        `FreeHandleFunction`. This allows testing `MXNetHandle` with the handle
        representing a pointer to `malloc`ed memory.

        The function returns an int to match the return type of the freeing
        functions in the MXNet C API. In particular it always returns zero
        since C's free function does not indicate any status. Note though
        freeing an already freed pointer is undefined.

        Returns:
            always zero to indicate successful freeing

    ***************************************************************************/

    // NOTE: public only so that `ocean.core.Traits.identifier` can access it
    public int freeMXNetHandleMemory (void* ptr)
    {
        free(ptr);
        return 0;
    }


    /***************************************************************************

        Fake MXNet handle function that returns a failure value

        Params:
            h = fake MXNet handle (a pointer to integer); is not used

        Returns:
            1 (MXNet failure value)

    ***************************************************************************/

    // NOTE: public only so that `ocean.core.Traits.identifier` can access it
    public int mxNetHandleFails (int* h) { return 1; }


    /***************************************************************************

        Fake MXNet handle function that takes an `int*` handle and sets the
        value it points to to the provided integer value

        Params:
            h = fake MXNet handle (a pointer to integer); must be non-null
            n = value to which to set the integer pointed to by h

        Returns:
            0 (MXNet success value)

    ***************************************************************************/

    // NOTE: public only so that `ocean.core.Traits.identifier` can access it
    public int mxNetHandleSucceeds (int* h, int n)
    {
        assert(h !is null);
        *h = n;
        return 0;
    }
}
