/*******************************************************************************

    Defines the context specifying the environment (devices like CPUs or GPUs)
    to perform the computations in

    Copyright:
        Copyright (c) 2017 sociomantic labs GmbH.

    License:
        Boost Software License Version 1.0.  See accompanying LICENSE.txt for
        details, or <https://www.boost.org/LICENSE_1_0.txt>

*******************************************************************************/

module mxnet.Context;

version(UnitTest)
{
    import ocean.core.Test;
}

/*******************************************************************************

    A Context defines the environment for NDArray and Executor

*******************************************************************************/

public struct Context
{
    /***************************************************************************

        Enum of the possible devices in a context

        List all types of devices computations can be performed on

    ***************************************************************************/

    // FIXME
    // not yet exposed by the C API
    public enum DeviceType : int
    {
        cpu = 1 << 0,   /// CPU
        gpu = 1 << 1,   /// GPU
        cpu_pinned = 3, /// CPU with pinned memory allocation
    }


    /***************************************************************************

        Device type of this context

    ***************************************************************************/

    private DeviceType type_;


    /***************************************************************************

        Device id of this context

        This is used to address a specific GPU in the system. For CPU devices
        it is unused.

    ***************************************************************************/

    private int id_;


    /***************************************************************************

        Returns:
            the device type of this context

    ***************************************************************************/

    public DeviceType type ()
    {
        return this.type_;
    }

    unittest
    {
        test!("==")(cpuContext.type(), Context.DeviceType.cpu);
    }


    /***************************************************************************

        Returns:
            the device id of this context

    ***************************************************************************/

    public int id ()
    {
        return this.id_;
    }

    unittest
    {
        test!("==")(gpuContext(2).id, 2);
    }
}


/*******************************************************************************

    Constructs a context for CPU execution

    The memory allocated by the context will be pinned or page-locked if
    `pinned_memory` is true. Otherwise the memory is allocated as usual.

    In a pinned CPU context all memory is allocated in a pinned (a.k.a.
    (page-)locked) fashion. That is the memory cannot be paged out by the
    operating system. This allows direct copying of memory from RAM to
    GPU and vice versa via DMA. It is more efficient by avoiding paging and
    intermediate copies to pinned memory.

    > [...] locked memory is stored in the physical memory (RAM), so the device
      can fetch it without the help of the host (synchronous copy).

    from <https://en.wikipedia.org/wiki/CUDA_Pinned_memory>

    Params:
        pinned_memory = if true use page-locked memory for allocation;
                        defaults to false

    Returns:
        a context for CPU execution

*******************************************************************************/

public Context cpuContext (bool pinned_memory = false)
{
    if (pinned_memory) return context(Context.DeviceType.cpu_pinned);
    return context(Context.DeviceType.cpu);
}

unittest
{
    test!("==")(cpuContext(), context(Context.DeviceType.cpu));
    test!("==")(cpuContext().id, 0);
    test!("==")(cpuContext(false), context(Context.DeviceType.cpu));
    test!("==")(cpuContext(false).id, 0);
    test!("==")(cpuContext(true), context(Context.DeviceType.cpu_pinned));
    test!("==")(cpuContext(true).id, 0);
}


/*******************************************************************************

    Constructs a context for GPU execution on device with id

    Params:
        id = device id of the returned GPU context

    Returns:
        a context for GPU execution on device with id

*******************************************************************************/

public Context gpuContext (int id)
{
    return context(Context.DeviceType.gpu, id);
}

unittest
{
    test!("==")(gpuContext(4), context(Context.DeviceType.gpu, 4));
    test!("==")(gpuContext(4).id, 4);
}


/*******************************************************************************

    Constructs a context with given device type and device id

    Params:
        type = device type of the context to be constructed
        id = device id of the context to be constructed; defaults to zero

    Returns:
        the context of given device type and device id

*******************************************************************************/

public Context context (Context.DeviceType type, int id = 0)
{
    Context context;
    context.type_ = type;
    context.id_ = id;
    return context;
}

unittest
{
    test!("==")(context(Context.DeviceType.cpu, 0).type, Context.DeviceType.cpu);
    test!("==")(context(Context.DeviceType.cpu, 0).id, 0);
}
