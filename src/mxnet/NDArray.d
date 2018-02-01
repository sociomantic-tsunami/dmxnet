/*******************************************************************************

    Defines an n-dimensional array (based on MXNet) to perform computations

    Copyright:
        Copyright (c) 2017 sociomantic labs GmbH.

    License:
        Boost Software License Version 1.0.  See accompanying LICENSE.txt for
        details, or <https://www.boost.org/LICENSE_1_0.txt>

*******************************************************************************/

module mxnet.NDArray;

import mxnet.c.c_api;
import mxnet.API;
import mxnet.Atomic;
import mxnet.Context;
import mxnet.Handle;
import mxnet.Util;

import ocean.core.array.Search;
import ocean.core.Tuple;
import ocean.util.Convert;
import ocean.text.util.StringC;
import ocean.transition;

version(UnitTest)
{
    import ocean.core.Tuple;
    import ocean.core.Test;
    import ocean.core.Traits;
    import ocean.text.convert.Formatter;
}


/*******************************************************************************

    An NDArray represents an n-dimensional array that is used to perform
    calculations

    An NDArray allocates MXNet resources accessed through an `MXNetHandle`. Its
    resources should be freed by calling `freeHandle` when done with the
    NDArray. This should happen in a timely manner to avoid resource
    depletion. Note that scope allocating an object or manually calling
    `delete` won't free the resources since an NDArray's destructor does not
    free the resources. To reclaim resources `freeHandle` must be called
    manually.

    Params:
        T = the element type of the n-dimensional array;
            supported types are float, double, int and ubyte

*******************************************************************************/

public class NDArray (T)
{
    /***************************************************************************

        Alias of the element type of this NDArray

    ***************************************************************************/

    public alias T ElementType;

    static assert(isSupportedElementType!(ElementType));


    /***************************************************************************

        Underlying managed NDArray handle used to interact with C library

        The underlying handle is not allowed to be shared between different
        instances, otherwise the handle could be freed more than once.

        The handle is allowed to be null, in which case the array will be
        considered to be empty (i.e. no dimension and no elements). You cannot
        call any public member functions on empty n-dimensional arrays besides
        for obtaining the value of the handle or setting a new value of the
        handle.

    ***************************************************************************/

    private MXNetHandle!(NDArrayHandle, MXNDArrayFree) mxnet_ndarray;


    /***************************************************************************

        Constructs an n-dimensional array of given shape for given context

        Note that its data is not initialized.

        Params:
            context = context in which the n-dimensional array is defined
            shape = the shape of the returned n-dimensional array
            lazy_allocation = delay allocation until first access; defaults to
                              false

    ***************************************************************************/

    public this (Context context,
                 Const!(uint)[] shape,
                 bool lazy_allocation = false)
    {
        NDArrayHandle ndarray_handle;
        mxnet.API.invoke!(MXNDArrayCreateEx)
                         (shape.ptr,
                          to!(uint)(shape.length),
                          context.type,
                          context.id,
                          lazy_allocation,
                          dataTypeOf!(T),
                          &ndarray_handle);
        this(ndarray_handle);
        assert(this.dataType() == dataTypeOf!(T));
    }

    unittest
    {
        scope empty_array = new NDArray(cpuContext(), []);
        scope (exit) empty_array.freeHandle();
        test!("==")(empty_array.context(), cpuContext());
        test!("==")(empty_array.shape().length, 0);
        test!("==")(empty_array.length(), 0);

        scope array = new NDArray(cpuContext(), [2, 2, 2]);
        scope (exit) array.freeHandle();
        test!("==")(array.context(), cpuContext());
        test!("==")(array.shape(), [2u, 2u, 2u]);
        test!("==")(array.length(), 2u * 2u * 2u);
    }


    /***************************************************************************

        Constructs an n-dimensional array of given shape for given context with
        each element initialized to value

        Params:
            context = context in which the n-dimensional array is defined
            shape = the shape of the returned n-dimensional array
            value = value to initialize each element of this n-dimensional
                    array to
            lazy_allocation = delay allocation until first access; defaults to
                              false

    ***************************************************************************/

    public this (Context context,
                 Const!(uint)[] shape,
                 T value,
                 bool lazy_allocation = false)
    {
        this(context, shape, lazy_allocation);
        assert(this.dataType() == dataTypeOf!(T));
        assert(this.length);
        this = value;
    }

    unittest
    {
        scope array = new NDArray(cpuContext(), [2, 2], to!(T)(1));
        scope (exit) array.freeHandle();
        test!("==")(array.context(), cpuContext());
        test!("==")(array.shape(), [2u, 2u]);
        test!("==")(array.length(), 2u * 2u);
        T[] data = [1, 1, 1, 1];
        test!("==")(array.data(), data);
    }


    /***************************************************************************

        Constructs an n-dimensional array from a given handle

        You should never construct different n-dimensional arrays with the same
        handle, because freeing a handle of one n-dimensional array would leave
        the other instances with a dangling handle.

        Params:
            handle = C API n-dimensional array handle; the handle can be null
                     to pre-allocate a class instance whose handle will be set
                     later

    ***************************************************************************/

    public this (NDArrayHandle handle)
    {
        this.mxnet_ndarray = new MXNetHandle!(NDArrayHandle, MXNDArrayFree)(handle);
    }


    /***************************************************************************

        Sets the underlying handle

        This sets the underlying handle to the provided handle, freeing a
        previous non-null handle.

        Note that you must ensure that no two n-dimensional array instances
        share the same handle as this would result in double freeing of the
        same handle.

        Params:
            handle = the handle this n-dimensional array's handle is to be set
                     to

    ***************************************************************************/

    public void handle (NDArrayHandle handle)
    {
        this.mxnet_ndarray.handle = handle;
    }


    /***************************************************************************

        Returns:
            the shape (an array of dimensions) of this n-dimensional array

    ***************************************************************************/

    public Const!(uint[]) shape ()
    in
    {
        assert(this.mxnet_ndarray.exists());
    }
    body
    {
        uint num_dims;
        Const!(uint*) dims_ptr;
        this.mxnet_ndarray.apply!(MXNDArrayGetShape)(&num_dims, &dims_ptr);
        return dims_ptr[0 .. num_dims];
    }

    unittest
    {
        scope array = new NDArray(cpuContext(), [2, 3]);
        scope (exit) array.freeHandle();
        test!("==")(array.shape(), [2u, 3u]);
    }


    /***************************************************************************

        Returns:
            the length of this n-dimensional array, i.e. the total number of
            elements

    ***************************************************************************/

    public uint length ()
    in
    {
        assert(this.mxnet_ndarray.exists());
    }
    body
    {
        if (this.shape().length == 0) return 0;

        uint length = 1;
        foreach (dim_length; this.shape())
        {
            length *= dim_length;
        }

        return length;
    }

    unittest
    {
        {
            scope array = new NDArray(cpuContext(), []);
            scope (exit) array.freeHandle();
            test!("==")(array.length, 0);
        }
        {
            scope array = new NDArray(cpuContext(), [8]);
            scope (exit) array.freeHandle();
            test!("==")(array.length, 8);
        }
        {
            scope array = new NDArray(cpuContext(), [2, 3]);
            scope (exit) array.freeHandle();
            test!("==")(array.length, 2 * 3);
        }
        {
            scope array = new NDArray(cpuContext(), [2, 3, 5]);
            scope (exit) array.freeHandle();
            test!("==")(array.length, 2 * 3 *5);
        }
    }


    /***************************************************************************

        Returns the data type of this n-dimensional array

        Note the returned data type always matches this NDArray's ElementType
        for initialized n-dimensional arrays.

        This function is only used to verify the run-time data type of this
        n-dimensional array with its compile-time defined type.

        Returns:
            the data type of this n-dimensional array

    ***************************************************************************/

    private DataType dataType ()
    in
    {
        assert(this.mxnet_ndarray.exists());
    }
    out (data_type)
    {
        assert(data_type == dataTypeOf!(T));
    }
    body
    {
        int data_type;
        this.mxnet_ndarray.apply!(MXNDArrayGetDType)(&data_type);
        assert(DataType.min <= data_type && data_type <= DataType.max);
        return cast(DataType) data_type;
    }

    unittest
    {
        scope array = new NDArray(cpuContext(), [2, 3]);
        scope (exit) array.freeHandle();
        test!("==")(array.dataType(), dataTypeOf!(T));
    }


    /***************************************************************************

        Returns the slice to the data of this n-dimensional array

        The underlying data is mapped to shape of this n-dimensional array by
        varying the last index the fastest. In the special case of matrices
        this is called row-major storage order. The array `[1, 2, 3, 4, 5, 6]`
        represents e.g. 2 by 3 shaped array
        `[1, 2, 3,
          4, 5, 6]`.

        This function calls `waitToRead` before returning the data slice. This
        means all pending writes (when calling this method) have finished. But
        note that it is not synchronized for later writes. Call `waitToRead` as
        needed to synchronize. Alternatively, instead of using this method you
        can perform a copy via `copyTo`.

        Note that you cannot change elements of the data slice: use `copyFrom`
        if you wish to write new values for the n-dimensional array data.

        Returns:
            a read-only data slice of this n-dimensional array

    ***************************************************************************/

    public Const!(T)[] data ()
    in
    {
        assert(this.mxnet_ndarray.exists());
    }
    body
    {
        // all pending writes must have been finished
        this.waitToRead();
        T* ptr;
        this.mxnet_ndarray.apply!(MXNDArrayGetData)(cast(void**) &ptr);
        return ptr[0 .. this.length];
    }

    unittest
    {
        scope array = new NDArray(cpuContext(), [2, 3], to!(T)(1));
        scope (exit) array.freeHandle();
        T[] data = [1, 1, 1,
                    1, 1, 1];
        test!("==")(array.data(), data);
    }


    /***************************************************************************

        Gives access to the underlying handle

        Use this.handle with care. This is provided as a mechanism to use the C
        API directly when needed.

        Returns:
            the underlying handle used with the C library

    ***************************************************************************/

    public NDArrayHandle handle ()
    {
        return this.mxnet_ndarray.handle;
    }

    unittest
    {
        scope a = new NDArray!(float)(cpuContext(), [2, 3]);
        scope (exit) a.freeHandle();
        test!("!is")(a.handle, null);
    }


    /***************************************************************************

        Reshapes this n-dimensional array to given shape

        This n-dimensional array's length must larger or equal to the number of
        elements addressable bu the passed shaped.

        Params:
            shape = the new shape of this n-dimensional array; the number of
                    elements in shape must be less or equal the number of
                    elements of this this array

    ***************************************************************************/

    public void reshape (int[] shape)
    in
    {
        assert(this.mxnet_ndarray.exists());
        assert(shape.length <= this.length);
    }
    body
    {
        NDArrayHandle reshaped_handle;
        this.mxnet_ndarray.apply!(MXNDArrayReshape)(to!(int)(shape.length),
                                                    shape.ptr,
                                                    &reshaped_handle);
        this.handle = reshaped_handle;
    }

    unittest
    {
        scope a = new NDArray(cpuContext(), [2, 3]);
        scope (exit) a.freeHandle();
        test!("==")(a.shape(), [2u, 3u]);
        a.reshape([3, 2]);
        test!("==")(a.shape(), [3u, 2u]);
    }


    /***************************************************************************

        Copies the elements of an array over the elements of this n-dimensional
        array

        The underlying data is mapped to shape by varying the last index the
        fastest. In the special case of matrices this is called row-major. The
        array `[1, 2, 3, 4, 5, 6]` represents e.g. 2 by 3 shaped array
        `[1, 2, 3,
          4, 5, 6]`.

        This function calls waitToWrite() implicitly before performing the copy.

        Params:
            src = source of copy; must be same length as this n-dimensional
                  array

    ***************************************************************************/

    public void copyFrom (Const!(T)[] src)
    in
    {
        assert(this.mxnet_ndarray.exists());
        assert(this.length == src.length);
    }
    body
    {
        this.mxnet_ndarray.apply!(MXNDArraySyncCopyFromCPU)(src.ptr, src.length);
    }

    unittest
    {
        scope a = new NDArray(cpuContext(), [2, 3]);
        scope (exit) a.freeHandle();
        T[] data = [1, 2, 3,
                    4, 5, 6];
        a.copyFrom(data);
        test!("==")(a.data(), data);
    }


    /***************************************************************************

        Copies the elements of this n-dimensional array's data to the elements
        of the provided array

        The elements are copied in the order obtained by varying the last index
        the fastest. For matrices this coincides with row-major order.

        This function calls waiToRead() implicitly before performing the copy.

        Params:
            dst = destination of copy; must be same length as this
                  n-dimensional array

    ***************************************************************************/

    public void copyTo (T[] dst)
    in
    {
        assert(this.mxnet_ndarray.exists());
        assert(this.length == dst.length);
    }
    body
    {
        this.mxnet_ndarray.apply!(MXNDArraySyncCopyToCPU)(dst.ptr, dst.length);
    }

    unittest
    {
        scope a = new NDArray(cpuContext(), [2, 3]);
        scope (exit) a.freeHandle();
        T[] src = [1, 2, 3,
                   4, 5, 6];
        a.copyFrom(src);
        T[] dst = new T[a.length];
        a.copyTo(dst);
        test!("==")(src, dst);
    }


    /***************************************************************************

        Blocks until all pending write operations for this n-dimensional array
        are finished

        You must call this function before any synchronous reading of this
        n-dimensional array's data.

    ***************************************************************************/

    public void waitToRead ()
    in
    {
        assert(this.mxnet_ndarray.exists());
    }
    body
    {
        this.mxnet_ndarray.apply!(MXNDArrayWaitToRead)();
    }


    /***************************************************************************

        Blocks until all pending read/write operations for this n-dimensional
        array are finished

        You must call this function before synchronous writing of this
        n-dimensional array's data.

    ***************************************************************************/

    public void waitToWrite ()
    in
    {
        assert(this.mxnet_ndarray.exists());
    }
    body
    {
        this.mxnet_ndarray.apply!(MXNDArrayWaitToWrite)();
    }


    /***************************************************************************

        Returns:
            the context of this n-dimensional array

    ***************************************************************************/

    public Context context ()
    in
    {
        assert(this.mxnet_ndarray.exists());
    }
    body
    {
        Context.DeviceType device_type;
        int device_id;
        static assert(typeof(device_type).sizeof == int.sizeof);
        static assert(is(typeof(device_type) : int));

        this.mxnet_ndarray.apply!(MXNDArrayGetContext)(cast(int*) &device_type,
                                                       &device_id);
        return mxnet.Context.context(device_type, device_id);
    }

    unittest
    {
        scope array = new NDArray(cpuContext(), [1]);
        scope (exit) array.freeHandle();
        test!("==")(array.context(), cpuContext());
    }


    /***************************************************************************

        Assigns all elements of this n-dimensional array to `scalar`

        Params:
            scalar = scalar to assign to each element of this n-dimensional
                     array

    ***************************************************************************/

    public void opAssign (T scalar)
    in
    {
        assert(this.mxnet_ndarray.exists());
    }
    body
    {
        istring key = "src";
        char[16] value = void;
        auto value_len = toNoLossString(scalar, value).length;
        value[value_len] = '\0';

        Immut!(char)*[1] keys;
        keys[0] = key.ptr;
        Const!(char)*[1] values;
        values[0] = value.ptr;

        NDArrayHandle[] inputs = null;
        NDArrayHandle[1] outputs;
        outputs[0] = this.handle();

        imperativeInvoke("_set_value", inputs, outputs, keys, values);
    }

    unittest
    {
        scope a = new NDArray(cpuContext(), [2, 3]);
        scope (exit) a.freeHandle();
        testNoAlloc(a = 1);
        T[] data = [1, 1, 1,
                    1, 1, 1];
        test!("==")(a.data(), data);
        data = [4, 4, 4,
                4, 4, 4];
        testNoAlloc(a = 4);
        test!("==")(a.data(), data);
    }


    /***************************************************************************

        Subtracts `rhs` element wise from this n-dimensional array

        `rhs` is broadcast if necessary to match the shape of this
        n-dimensional array. All dimensions of the right hand side not equal to
        one must match the corresponding dimension of this n-dimensional array.

        For an explanation of broadcasting see `applyBroadcastOp`.

        Params:
            rhs = the n-dimensional array to be subtracted;
                  all dimensions not equal to 1 must match the corresponding
                  dimension of this n-dimensional array

        Returns:
            this n-dimensional array after the subtraction

    ***************************************************************************/

    public NDArray opSubAssign (NDArray rhs)
    in
    {
        assert(this.mxnet_ndarray.exists());
    }
    body
    {
        return applyBroadcastOp("broadcast_sub", this, this, rhs);
    }

    unittest
    {
        // matrix subtraction
        {
            scope a = new NDArray(cpuContext(), [2, 3], to!(T)(10));
            scope (exit) a.freeHandle();
            scope b = new NDArray(cpuContext(), [2, 3], to!(T)(2));
            scope (exit) b.freeHandle();
            a -= b;
            T[] a_data = [8, 8, 8,
                          8, 8, 8];
            test!("==")(a.data(), a_data);
        }
        // subtract scalar b
        {
            scope a = new NDArray(cpuContext(), [2, 3], to!(T)(10));
            scope (exit) a.freeHandle();
            scope b = new NDArray(cpuContext(), [1], to!(T)(3));
            scope (exit) b.freeHandle();
            a -= b;
            T[] a_data = [7, 7, 7,
                          7, 7, 7];
            test!("==")(a.data(), a_data);
        }
        // column-wise subtract column vector b
        {
            scope a = new NDArray(cpuContext(), [2, 3], to!(T)(2));
            scope (exit) a.freeHandle();
            scope b = new NDArray(cpuContext(), [2, 1]);
            scope (exit) b.freeHandle();
            b.copyFrom([1, 2]);
            a -= b;
            T[] a_data = [1, 1, 1,
                          0, 0, 0];
            test!("==")(a.data(), a_data);
        }
        // row-wise subtract row vector b
        {
            scope a = new NDArray(cpuContext(), [2, 3], to!(T)(3));
            scope (exit) a.freeHandle();
            scope b = new NDArray(cpuContext(), [1, 3]);
            scope (exit) b.freeHandle();
            b.copyFrom([1, 2, 3]);
            a -= b;
            T[] a_data = [2, 1, 0,
                          2, 1, 0];
            test!("==")(a.data(), a_data);
        }
    }


    /***************************************************************************

        Multiplies each element of this n-dimensional array by a value

        Each element of this n-dimensional array is multiplied by the passed
        scalar and updated to this result.

        Params:
            rhs = scalar to be multiplied with each element of this
                  n-dimensional array

        Returns:
            this n-dimensional array after the element wise multiplications

    ***************************************************************************/

    public NDArray opMulAssign (T rhs)
    in
    {
        assert(this.mxnet_ndarray.exists());
    }
    body
    {
        return applyScalarOp("_mul_scalar", this, this, rhs);
    }

    unittest
    {
        scope a = new NDArray(cpuContext(), [2, 3], to!(T)(1));
        scope (exit) a.freeHandle();
        T[] a_data = [1, 1, 1,
                      1, 1, 1];
        test!("==")(a.data(), a_data);
        a_data = [2, 2, 2,
                  2, 2, 2];
        a *= to!(T)(2);
        test!("==")(a.data(), a_data);
        a_data = [6, 6, 6,
                  6, 6, 6];
        a *= to!(T)(3);
        test!("==")(a.data(), a_data);
    }


    /***************************************************************************

        Frees the underlying handle

    ***************************************************************************/

    public void freeHandle ()
    {
        this.mxnet_ndarray.freeHandle();
    }
}

unittest
{
    foreach (T; Tuple!(float))
    {
        scope array = new NDArray!(T)(cpuContext(), [2, 3]);
        scope (exit) array.freeHandle();
    }
}


/*******************************************************************************

    Template list of the element types supported by `NDArray`.

*******************************************************************************/

public alias Tuple!(float, double, int, ubyte) SupportedElementTypes;


/*******************************************************************************

    Template check to confirm if a given element type is supported by NDArray

    Params:
        T = element type to check

    Returns:
        `true` if the element type can be used with an `NDArray`
        (i.e. if `NDArray!(T)` is a valid type),
        `false` otherwise

*******************************************************************************/

public template isSupportedElementType (T)
{
    const bool isSupportedElementType =
        IndexOf!(T, SupportedElementTypes) < SupportedElementTypes.length;
}

unittest
{
    static assert(isSupportedElementType!(float));
    static assert(isSupportedElementType!(double));
    static assert(isSupportedElementType!(int));
    static assert(isSupportedElementType!(ubyte));

    static assert(!isSupportedElementType!(uint));
    static assert(!isSupportedElementType!(long));
    static assert(!isSupportedElementType!(real));
    static assert(!isSupportedElementType!(byte));
}


/*******************************************************************************

    Broadcast a binary operation across a pair of n-dimensional arrays

    Broadcasting is a technique for allowing binary operations to be performed
    on n-dimensional arrays of different shapes.  Subject to some constraints,
    any dimension of size 1 ('singleton dimension') on one side of the operation
    will be 'broadcast' across the size of corresponding dimension on the other
    side, until we have a view over the data where both have the same shape and
    it is then possible to apply the desired operation.  This is done without
    any adjustment to or copying of the underlying input data, meaning that the
    result can usually be calculated with great efficiency compared to what we
    would have to do if we had to directly create compatible n-dimensional array
    instances.

    Where the two n-dimensional arrays concerned already have the same shape,
    this of course means that the broadcast is a no-op: no dimensions need to
    be expanded and the operation is performed using the arrays as-is.  For
    arrays with different shapes, some examples may help to clarify what the
    broadcasting process entails.

    The simplest example is to apply the operation to an n-dimensional array
    and a scalar (meaning here an `NDArray` whose shape is `[1]`).  In this
    case the broadcast treats the scalar as if it were expanded into an
    n-dimensional array of the same shape as the other side of the operation,
    but with every element equal to the original scalar, e.g.

    ---------------------------------------------------------------------------
        [1, 2, 3,  .*  [2]  =  [1, 2, 3,  .*  [2, 2, 2,   (.* == elementwise
         4, 5, 6]               4, 5, 6]       2, 2, 2]    multiplication)

                            =  [2,  4,  6,
                                8, 10, 12]
    ---------------------------------------------------------------------------

    where, as we see, the scalar of shape `[1]` is broadcast across the two
    dimensions of the left-hand side array to match its `[2, 3]` shape, at
    which point the elementwise multiplication can be applied.  The effect here
    is what we instinctively expect if we multiply an n-dimensional array by a
    scalar, i.e. each element of the array being multiplied by the scalar value.

    If both sides of the operation are scalars, the broadcast is of course a
    no-op, i.e. `[1] + [1] == [2]`.

    Alternatively, consider a row vector (shape `[1, 2]`) and a column vector
    (shape `[3, 1]`):

    ---------------------------------------------------------------------------
        [2, 4]  +  [1,  =  [2, 4,  +  [1, 1,
                    2,      2, 4,      2, 2,
                    3]      2, 4]      3, 3]

                        =  [3, 5,
                            4, 6,
                            5, 7]
    ---------------------------------------------------------------------------

    where here we can see that the row vector is broadcast across its singleton
    first dimension to form a matrix of shape `[2, 3]` in which every row is
    identical to the original row vector, while the column vector is similarly
    broadcast across its singleton second dimension to form another `[2, 3]`
    matrix where each column is identical to the original column vector.  The
    operation of addition can then be applied to these two matrices as normal.

    These few examples give the flavour of the constraints that are required
    on array shape in order for the broadcast to be possible:

      * if both the left- and right-hand side of the operation are not
        scalars (`shape == [1]`), then they must have the same number
        of dimensions (i.e. `lhs.shape.length == rhs.shape.length`)

          - strictly speaking this is more a practical constraint of
            MXNet than a constraint that comes from broadcasting per
            se, since in some other n-dimensional array libraries,
            missing higher dimensions are just inferred to be singleton
            (for example, technically any array of shape `[1, ..., 1]`
            can be considered a scalar, but for `shape != [1]` MXNet
            requires the number of dimensions to match)

      * if the size of a dimension is non-singleton (i.e. > 1) on both
        the left and right hand side, it must be the same size (i.e.
        if `lhs.shape[i] > 1` and `rhs.shape[i] > 1` then we must have
        `lhs.shape[i] == rhs.shape[i]`

    The shape of the result will match the shapes of the left- and right-hand
    sides of the operation after broadcasting is applied to them.  Where this
    function is concerned, the user is required to provide a `result` `NDArray`
    whose shape is already set correctly, so that the results can be written
    to it in-place without any resizing or reallocation required.

    A nice more extended overview of n-dimensional array broadcasting is given
    in <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html> and
    <https://www.gnu.org/software/octave/doc/interpreter/Broadcasting.html>.

    Params:
        op_name = name of the operation to perform (from the available
                  `broadcast_` atomic symbol types defined by MXNet)
        result = `NDArray` instance into which to write the result of the
                 operation.  Its dimensions must be set such that
                 `return.shape[i] == max(lhs.shape[i], rhs.shape[i])`.
                 May be the same `NDArray` instance as either (or both)
                 of `lhs` and `rhs`, in which case the calculation will
                 be performed in-place with respect to that input
        lhs = `NDArray` instance to place on the left-hand side of the
              broadcast operation; any non-singleton dimensions must match
              the size of any corresponding non-singleton dimensions of `rhs`
        rhs = `NDArray` instance to place on the right-hand side of the
              broadcast operation; any non-singleton dimensions must match
              the size of any corresponding non-singleton dimensions of `lhs`

    Returns:
        `result`, filled with values containing the result of `lhs op rhs`

*******************************************************************************/

private NDArray!(T) applyBroadcastOp (T) (istring op_name,
                                          NDArray!(T) result,
                                          NDArray!(T) lhs,
                                          NDArray!(T) rhs)
in
{
    assert(startsWith(op_name, "broadcast_"));

    assert(result !is null);
    assert(result.mxnet_ndarray.exists());

    assert(lhs !is null);
    assert(lhs.mxnet_ndarray.exists());
    assert(rhs !is null);
    assert(rhs.mxnet_ndarray.exists());
}
body
{
    NDArrayHandle[2] inputs;
    inputs[0] = lhs.handle();
    inputs[1] = rhs.handle();

    NDArrayHandle[1] outputs;
    outputs[0] = result.handle();

    imperativeInvoke(op_name, inputs, outputs, null, null);

    return result;
}

unittest
{
    istring broadcastOperator (istring op)
    {
        switch (op)
        {
            case "+": return "broadcast_add";
            case "-": return "broadcast_sub";
            case "*": return "broadcast_mul";
            case "/": return "broadcast_div";
            default: assert(0, "Unexpected operator: " ~ op);
        }
    }

    // MXNet only implements floating-point broadcast
    // operations, so we test only with the supported
    // floating-point element types
    foreach (T; Tuple!(float, double))
    {
        foreach (op; Tuple!("+", "-", "*", "/"))
        with (new NamedTest(T.stringof ~ " (" ~ op ~ ")"))
        {
            // identical shape of LHS and RHS NDArrays results
            // in element-wise application of the operation
            {
                T[] lhs_data = [1, 2, 3,
                                4, 5, 6];

                T[] rhs_data = [2,  4,  6,
                                8, 10, 12];

                scope ret = new NDArray!(T)(cpuContext(), [2, 3]);
                scope (exit) ret.freeHandle();

                scope lhs = new NDArray!(T)(cpuContext(), [2, 3]);
                lhs.copyFrom(lhs_data);
                scope (exit) lhs.freeHandle();

                scope rhs = new NDArray!(T)(cpuContext(), [2, 3]);
                rhs.copyFrom(rhs_data);
                scope (exit) rhs.freeHandle();
                assert(lhs.shape() == rhs.shape());

                testNoAlloc(applyBroadcastOp(broadcastOperator(op),
                                             ret, lhs, rhs));
                test!("==")(ret.data(),
                            elementWiseTestOp!(op)(lhs_data, rhs_data));
            }

            // different case of same shape
            {
                T[] lhs_data = [6, 5,
                                4, 3,
                                2, 1];

                T[] rhs_data = [12, 10,
                                 8,  6,
                                 4,  2];

                scope ret = new NDArray!(T)(cpuContext(), [3, 2]);
                scope (exit) ret.freeHandle();

                scope lhs = new NDArray!(T)(cpuContext(), [3, 2]);
                lhs.copyFrom(lhs_data);
                scope (exit) lhs.freeHandle();

                scope rhs = new NDArray!(T)(cpuContext(), [3, 2]);
                rhs.copyFrom(rhs_data);
                scope (exit) rhs.freeHandle();
                assert(lhs.shape() == rhs.shape());

                testNoAlloc(applyBroadcastOp(broadcastOperator(op),
                                             ret, lhs, rhs));
                test!("==")(ret.data(),
                            elementWiseTestOp!(op)(lhs_data, rhs_data));
            }

            // NDArray op scalar
            {
                T[] lhs_data = [1, 2, 3,
                                4, 5, 6];

                T rhs_scalar = 2;

                scope ret = new NDArray!(T)(cpuContext(), [2, 3]);
                scope (exit) ret.freeHandle();

                scope lhs = new NDArray!(T)(cpuContext(), [2, 3]);
                lhs.copyFrom(lhs_data);
                scope (exit) lhs.freeHandle();

                scope rhs = new NDArray!(T)(cpuContext(), [1], rhs_scalar);
                scope (exit) rhs.freeHandle();

                testNoAlloc(applyBroadcastOp(broadcastOperator(op),
                                             ret, lhs, rhs));

                // compare to result of manually-performed broadcast
                // of the scalar
                T[] rhs_broadcast_data = new T[lhs_data.length];
                rhs_broadcast_data[] = rhs_scalar;

                auto expected_data = elementWiseTestOp!(op)(lhs_data,
                                                            rhs_broadcast_data);

                test!("==")(ret.data(), expected_data);

                // validate consistency of `applyBroadcastOp` results
                scope ret_b = new NDArray!(T)(cpuContext(), [2, 3]);
                scope (exit) ret_b.freeHandle();

                scope rhs_b = new NDArray!(T)(cpuContext(), [2, 3], rhs_scalar);
                scope (exit) rhs_b.freeHandle();

                testNoAlloc(applyBroadcastOp(broadcastOperator(op),
                                             ret_b, lhs, rhs_b));

                test!("==")(ret_b.data(), expected_data);
            }

            // scalar op NDArray
            {
                T lhs_scalar = 2;

                T[] rhs_data = [12, 10,
                                 8,  6,
                                 4,  2];

                scope ret = new NDArray!(T)(cpuContext(), [3, 2]);
                scope (exit) ret.freeHandle();

                scope lhs = new NDArray!(T)(cpuContext(), [1], lhs_scalar);
                scope (exit) lhs.freeHandle();

                scope rhs = new NDArray!(T)(cpuContext(), [3, 2]);
                rhs.copyFrom(rhs_data);
                scope (exit) rhs.freeHandle();

                testNoAlloc(applyBroadcastOp(broadcastOperator(op),
                                             ret, lhs, rhs));

                // compare to result of manually-performed broadcast
                // of the scalar
                T[] lhs_broadcast_data = new T[rhs_data.length];
                lhs_broadcast_data[] = lhs_scalar;

                auto expected_data = elementWiseTestOp!(op)(lhs_broadcast_data,
                                                            rhs_data);

                test!("==")(ret.data(), expected_data);

                // validate consistency of `applyBroadcastOp` results
                scope ret_b = new NDArray!(T)(cpuContext(), [3, 2]);
                scope (exit) ret_b.freeHandle();

                scope lhs_b = new NDArray!(T)(cpuContext(), [3, 2], lhs_scalar);
                scope (exit) lhs_b.freeHandle();

                testNoAlloc(applyBroadcastOp(broadcastOperator(op),
                                             ret_b, lhs_b, rhs));

                test!("==")(ret_b.data(), expected_data);

            }

            // row vector op column vector
            {
                T[] lhs_data = [1, 2];

                T[] rhs_data = [2,
                                4,
                                6];

                scope ret = new NDArray!(T)(cpuContext(), [3, 2]);
                scope (exit) ret.freeHandle();

                scope lhs = new NDArray!(T)(cpuContext(), [1, 2]);
                lhs.copyFrom(lhs_data);
                scope (exit) lhs.freeHandle();

                scope rhs = new NDArray!(T)(cpuContext(), [3, 1]);
                rhs.copyFrom(rhs_data);
                scope (exit) rhs.freeHandle();

                testNoAlloc(applyBroadcastOp(broadcastOperator(op),
                                             ret, lhs, rhs));

                // compare to result of manually-performed broadcast
                // of row- and column-vector elements
                T[] lhs_broadcast_data = [1, 2,
                                          1, 2,
                                          1, 2];

                T[] rhs_broadcast_data = [2, 2,
                                          4, 4,
                                          6, 6];

                auto expected_data = elementWiseTestOp!(op)(lhs_broadcast_data,
                                                            rhs_broadcast_data);

                test!("==")(ret.data(), expected_data);

                // validate consistency of `applyBroadcastOp` results
                scope ret_b = new NDArray!(T)(cpuContext(), [3, 2]);
                scope (exit) ret_b.freeHandle();

                scope lhs_b = new NDArray!(T)(cpuContext(), [3, 2]);
                lhs_b.copyFrom(lhs_broadcast_data);
                scope (exit) lhs_b.freeHandle();

                scope rhs_b = new NDArray!(T)(cpuContext(), [3, 2]);
                rhs_b.copyFrom(rhs_broadcast_data);
                scope (exit) rhs_b.freeHandle();

                testNoAlloc(applyBroadcastOp(broadcastOperator(op),
                                             ret_b, lhs_b, rhs_b));

                test!("==")(ret_b.data(), expected_data);
            }

            // column vector op row vector
            {
                T[] lhs_data = [1,
                                2,
                                3];

                T[] rhs_data = [2, 4];

                scope ret = new NDArray!(T)(cpuContext(), [3, 2]);
                scope (exit) ret.freeHandle();

                scope lhs = new NDArray!(T)(cpuContext(), [3, 1]);
                lhs.copyFrom(lhs_data);
                scope (exit) lhs.freeHandle();

                scope rhs = new NDArray!(T)(cpuContext(), [1, 2]);
                rhs.copyFrom(rhs_data);
                scope (exit) rhs.freeHandle();

                testNoAlloc(applyBroadcastOp(broadcastOperator(op),
                                             ret, lhs, rhs));

                // compare to result of manually-performed broadcast
                // of row- and column-vector elements
                T[] lhs_broadcast_data = [1, 1,
                                          2, 2,
                                          3, 3];

                T[] rhs_broadcast_data = [2, 4,
                                          2, 4,
                                          2, 4];

                auto expected_data = elementWiseTestOp!(op)(lhs_broadcast_data,
                                                            rhs_broadcast_data);

                test!("==")(ret.data(), expected_data);

                // validate consistency of `applyBroadcastOp` results
                scope ret_b = new NDArray!(T)(cpuContext(), [3, 2]);
                scope (exit) ret_b.freeHandle();

                scope lhs_b = new NDArray!(T)(cpuContext(), [3, 2]);
                lhs_b.copyFrom(lhs_broadcast_data);
                scope (exit) lhs_b.freeHandle();

                scope rhs_b = new NDArray!(T)(cpuContext(), [3, 2]);
                rhs_b.copyFrom(rhs_broadcast_data);
                scope (exit) rhs_b.freeHandle();

                testNoAlloc(applyBroadcastOp(broadcastOperator(op),
                                             ret_b, lhs_b, rhs_b));

                test!("==")(ret_b.data(), expected_data);
            }

            // multiple different dimensions on both sides
            {
                // we use identical values for all LHS elements and
                // all RHS elements just to simplify validation of
                // the result: the real point of this test is to
                // confirm the dimension of the result NDArray

                T lhs_val = 1;
                T rhs_val = 2;

                scope ret = new NDArray!(T)(cpuContext(), [5, 6, 7, 8]);
                scope (exit) ret.freeHandle();

                scope lhs = new NDArray!(T)(cpuContext(), [1, 6, 1, 8],
                                            lhs_val);
                scope (exit) lhs.freeHandle();

                scope rhs = new NDArray!(T)(cpuContext(), [5, 1, 7, 1],
                                            rhs_val);
                scope (exit) rhs.freeHandle();

                testNoAlloc(applyBroadcastOp(broadcastOperator(op),
                                             ret, lhs, rhs));

                T expected_val = scalarTestOp!(op)(lhs_val, rhs_val);

                foreach (val; ret.data())
                {
                    test!("==")(val, expected_val);
                }
            }

            // calls where the `NDArray` instance into which the
            // results are written is the same as one or both of
            // the `lhs`/`rhs` input `NDArray` instances
            {
                T[] lhs_data = [1, 2, 3,
                                4, 5, 6];

                T[] rhs_data = [2,  4,  6,
                                8, 10, 12];

                scope ret = new NDArray!(T)(cpuContext(), [2, 3]);
                scope (exit) ret.freeHandle();

                scope other = new NDArray!(T)(cpuContext(), [2, 3]);
                scope (exit) other.freeHandle();

                // completely in-place operation (return NDArray is lhs and rhs)
                ret.copyFrom(lhs_data);
                assert(ret.data() == lhs_data);
                assert(ret.data().ptr !is lhs_data.ptr);

                auto expected_data = elementWiseTestOp!(op)(lhs_data, lhs_data);
                assert(ret.data() != expected_data);

                applyBroadcastOp(broadcastOperator(op), ret, ret, ret);
                test!("==")(ret.data(), expected_data);
                test!("!=")(ret.data(), lhs_data);

                // in-place w.r.t. lhs only
                ret.copyFrom(lhs_data);
                other.copyFrom(rhs_data);

                expected_data = elementWiseTestOp!(op)(lhs_data, rhs_data);

                assert(ret.data() != expected_data);
                assert(ret.data() == lhs_data);
                assert(other.data() == rhs_data);

                applyBroadcastOp(broadcastOperator(op), ret, ret, other);
                test!("==")(ret.data(), expected_data);
                test!("!=")(ret.data(), lhs_data);
                test!("==")(other.data(), rhs_data);

                // in-place w.r.t rhs only
                other.copyFrom(lhs_data);
                ret.copyFrom(rhs_data);

                assert(ret.data() != expected_data);
                assert(ret.data() == rhs_data);
                assert(other.data() == lhs_data);

                applyBroadcastOp(broadcastOperator(op), ret, other, ret);
                test!("==")(ret.data(), expected_data);
                test!("!=")(ret.data(), rhs_data);
                test!("==")(other.data(), lhs_data);
            }
        }
    }
}


/*******************************************************************************

    Apply a binary operation with a scalar elementwise to an n-dimensional
    array

    Params:
        op_name = name of the operation to apply (from the available
                  `_scalar` atomic symbol types defined by MXNet)
        result = `NDArray` instance into which to write the results of
                 the operation.  Must be the same shape as `lhs`, and
                 may be the same `NDArray` instance as `lhs`, in which
                 case the operation will be performed in-place with
                 respect to that input
         lhs = `NDArray` instance to whose elements the operation with
               the scalar `rhs` will be applied
         rhs = scalar value of the same type as the elements of `lhs`,
               which will form the right-hand side of the elementwise
               binary operation

    Returns:
        `result`, with each of its elements set to the value produced
        by applying the binary operation to the corresponding element
        of `lhs` and the scalar `rhs`

*******************************************************************************/

private NDArray!(T) applyScalarOp (T) (istring op_name,
                                       NDArray!(T) result,
                                       NDArray!(T) lhs,
                                       T rhs)
in
{
    assert(endsWith(op_name, "_scalar"));

    assert(result !is null);
    assert(result.mxnet_ndarray.exists());
    assert(lhs !is null);
    assert(lhs.mxnet_ndarray.exists());
}
body
{
    istring key = "scalar";
    char[16] value = void;
    auto value_len = toNoLossString(rhs, value).length;
    value[value_len] = '\0';

    Immut!(char)*[1] keys;
    keys[0] = key.ptr;
    Const!(char)*[1] values;
    values[0] = value.ptr;

    NDArrayHandle[1] inputs;
    inputs[0] = lhs.handle();
    NDArrayHandle[1] outputs;
    outputs[0] = result.handle();

    imperativeInvoke(op_name, inputs, outputs, keys, values);

    return result;
}

unittest
{
    istring broadcastOperator (istring op)
    {
        switch (op)
        {
            case "_plus_scalar": return "broadcast_add";
            case "_minus_scalar": return "broadcast_sub";
            case "_mul_scalar": return "broadcast_mul";
            case "_div_scalar": return "broadcast_div";
            default: assert(0);
        }
    }

    // MXNet only implements floating-point scalar
    // operations, so we test only with supported
    // floating-point element types
    foreach (T; Tuple!(float, double))
    {
        foreach (op; ["_plus_scalar", "_minus_scalar", "_mul_scalar", "_div_scalar"])
        {
            // identity of the operation such that x op identity == x
            // (== 1 for mul/div, 0 for plus/minus
            T op_identity = (op == "_mul_scalar" || op == "_div_scalar");

            // scalar values with which to test the operation
            // (including identity and non-identity values)
            T[] scalars = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5];

            foreach (rhs; scalars)
            with (new NamedTest(format("{} ({} {})", T.stringof, op, rhs)))
            {
                T[] lhs_data = [1, 2, 3,
                                4, 5, 6];

                scope ret = new NDArray!(T)(cpuContext(), [2, 3]);
                scope (exit) ret.freeHandle();

                scope lhs = new NDArray!(T)(cpuContext(), [2, 3]);
                lhs.copyFrom(lhs_data);
                scope (exit) lhs.freeHandle();

                scope ret_broadcast = new NDArray!(T)(cpuContext(), [2, 3]);
                scope (exit) ret_broadcast.freeHandle();
                scope rhs_broadcast = new NDArray!(T)(cpuContext(), [1], rhs);
                scope (exit) rhs_broadcast.freeHandle();

                testNoAlloc(applyScalarOp(op, ret, lhs, rhs));

                // compare to results of already tested broadcast operation
                testNoAlloc(applyBroadcastOp(broadcastOperator(op),
                                             ret_broadcast, lhs, rhs_broadcast));
                test!("==")(ret.data(), ret_broadcast.data());
                test!("==")(lhs.data(), lhs_data);

                if (rhs == op_identity)
                {
                    test!("==")(ret.data(), lhs_data);
                }
                else
                {
                    test!("!=")(ret.data(), lhs_data);
                }

                // test in-place operation where `NDArray` instance into
                // which the results are written is the same as the `lhs`
                // array instance
                scope ret_in_place = new NDArray!(T)(cpuContext(), [2, 3]);
                scope (exit) ret_in_place.freeHandle();

                ret_in_place.copyFrom(lhs_data);
                assert(ret_in_place.data() == lhs_data);
                assert(ret_in_place.data().ptr !is lhs_data.ptr);

                testNoAlloc(applyScalarOp(op, ret_in_place, ret_in_place, rhs));
                test!("==")(ret_in_place.data(), ret_broadcast.data());

                if (rhs == op_identity)
                {
                    test!("==")(ret_in_place.data(), lhs_data);
                }
                else
                {
                    test!("!=")(ret_in_place.data(), lhs_data);
                }
            }
        }
    }
}


/*******************************************************************************

    Invoke an imperative operation on a collection of n-dimensional arrays

    Params:
        op_name = name of the operation to perform
        inputs = handles of the n-dimensional arrays to use as input
        outputs = handles of the n-dimensional arrays to which output will
                  be written
        keys = array of C strings representing the keys (names) of the keyword
               parameters required by the operation (should be empty or `null`
               if no keyword parameters are required)
        values = array of C strings representing the values of the keyword
                 parameters required by the operation (must be the same length
                 as `keys`; should be empty or `null` if no keyword parameters
                 are required)

*******************************************************************************/

private void imperativeInvoke (istring op_name,
                               NDArrayHandle[] inputs,
                               NDArrayHandle[] outputs,
                               in char*[] keys,
                               in char*[] values)
in
{
    assert(keys.length == values.length);
}
body
{
    auto outputs_len = to!(int)(outputs.length);
    auto outputs_ptr = outputs.ptr;

    invoke!(MXImperativeInvoke)(atomicSymbolCreator(op_name),
                                to!(int)(inputs.length), inputs.ptr,
                                &outputs_len, &outputs_ptr,
                                to!(int)(keys.length), keys.ptr, values.ptr);
    assert(outputs_len == outputs.length);
}


/*******************************************************************************

    Blocks until all pending operations on all n-dimensional arrays are
    finished

*******************************************************************************/

public void waitAll()
{
    invoke!(MXNDArrayWaitAll)();
}


/*******************************************************************************

    Enum of the possible types of the elements of an NDArray

*******************************************************************************/

// FIXME
// not yet exposed by the C API
public enum DataType
{
    float32 = 0, // default
    float64,
    float16,
    uint8,
    int32,
}


/*******************************************************************************

    Returns the data type of `U`

    Params:
        U = type to return the data type of

    Returns:
        the appropriate data type of given type

*******************************************************************************/

private template dataTypeOf (U)
{
    static if (is(U == float)) alias DataType.float32 dataTypeOf;
    else static if (is(U == double)) alias DataType.float64 dataTypeOf;
    else static if (is(U == int)) alias DataType.int32 dataTypeOf;
    else static if (is(U == ubyte)) alias DataType.uint8 dataTypeOf;
    else static assert(false, "dataTypeOf: unsupported type " ~ U.stringof);
}

unittest
{
    static assert(dataTypeOf!(float) == DataType.float32);
    static assert(dataTypeOf!(double) == DataType.float64);
    static assert(dataTypeOf!(int) == DataType.int32);
    static assert(dataTypeOf!(ubyte) == DataType.uint8);
}


version (UnitTest)
{
    /***************************************************************************

        Helper function that generates expected output of broadcast operations
        in the case where we have two n-dimensional arrays of exact same shape,
        in which case the broadcast essentially amounts to performing the
        operation elementwise.

        For simplicity, this helper method does not bother with n-dimensional
        arrays, but just uses raw arrays that can be mapped to the underlying
        data.

        Params:
            op = element-wise operation to perform ("+", "-", "*", "/")
            T = type of elements of the n-dimensional arrays
            lhs = array whose elements correspond to the elements of the
                  left-hand n-dimensional array in the operation; must
                  be the same length as `rhs`
            rhs = array whose elements correspond to the elements of the
                  right-hand n-dimensional array in the operation; must
                  be the same length as `lhs`

        Returns:
            array of the same length as `lhs` and `rhs`, whose i'th element
            is equal to `lhs[i] op rhs[i]`

    ***************************************************************************/

    private T[] elementWiseTestOp (istring op, T) (in T[] lhs, in T[] rhs)
    in
    {
        assert(lhs.length == rhs.length);
    }
    body
    {
        static assert(op == "+" || op == "-" || op == "*" || op == "/");
        static assert(isFloatingPointType!(T));

        auto res = new T[lhs.length];

        for (size_t i = 0; i < res.length; ++i)
        {
            res[i] = scalarTestOp!(op)(lhs[i], rhs[i]);
        }

        return res;
    }


    /***************************************************************************

        Helper function that generates the result of a binary operation applied
        to two scalars

        Params:
            op = element-wise operation to perform ("+", "-", "*", "/")
            T = type of scalars to perform the operation on
            lhs = value of the left hand side of the binary operation
            rhs = value of the right hand side of the binary operation

        Returns:
            `lhs op rhs`

    ***************************************************************************/

    private T scalarTestOp (istring op, T) (in T lhs, in T rhs)
    {
        static assert(op == "+" || op == "-" || op == "*" || op == "/");
        static assert(isFloatingPointType!(T));

        mixin("return lhs " ~ op ~ " rhs;");
    }
}
