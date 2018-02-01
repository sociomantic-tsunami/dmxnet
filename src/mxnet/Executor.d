/*******************************************************************************

    Defines an executor to perform passes over a symbolic network given
    n-dimensional array inputs

    An executor makes forward and backward passes over a network given
    n-dimensional array inputs. The forward pass computes the outputs of a
    network. The backward pass computes gradients.

    Copyright:
        Copyright (c) 2017 sociomantic labs GmbH.

    License:
        Boost Software License Version 1.0.  See accompanying LICENSE.txt for
        details, or <https://www.boost.org/LICENSE_1_0.txt>

*******************************************************************************/

module mxnet.Executor;

import mxnet.c.c_api;
import mxnet.Context;
import mxnet.Exception;
import mxnet.Handle;
import mxnet.NDArray;
import mxnet.Symbol;

import ocean.text.util.StringC;
import ocean.transition;
import ocean.util.Convert;

version(UnitTest)
{
    import ocean.core.Test;
}


/*******************************************************************************

    Exception to be thrown in case of an error involving an MXNet executor
    instance (e.g. if a forward or backward pass fails)

    In the typical use-case only one global reusable instance of this exception
    will ever exist per thread, and it will only ever be thrown from within the
    `MXNetExecutor` class defined in this module.  When catching exceptions of
    this type, take care to not store shallow copies, since the exception data
    may later be rewritten in place by other executor-related failures.

*******************************************************************************/

public class MXNetExecutorException : MXNetException
{
}


/*******************************************************************************

    Exception instance to be thrown in case of an error involving an MXNet
    executor instance (e.g. if a forward or backward pass fails)

    Since this is a reusable instance, take care not to store shallow copies
    of it, since the exception data will be rewritten in place by other later
    executor-related failures.

*******************************************************************************/

private MXNetExecutorException mxnet_executor_exception;

private static this ()
{
    mxnet_executor_exception = new MXNetExecutorException;
}


/*******************************************************************************

    An Executor is used to perform operations on a network composed of symbols

    An Executor allocates MXNet resources accessed through an `MXNetHandle`.
    Its resources should be freed by calling `freeHandle` when done with the
    Executor. This should happen in a timely manner to avoid resource
    depletion. Note that scope allocating an object or manually calling
    `delete` won't free the resources since an Executor's destructor does not
    free the resources. To reclaim resources `freeHandle` must be called
    manually.

    Params:
        T = element type of the n-dimensional arrays computed as outputs

*******************************************************************************/

public class Executor (T)
{
    import mxnet.API;

    /***************************************************************************

        Underlying managed executor handle used to interact with C library

    ***************************************************************************/

    private MXNetHandle!(ExecutorHandle, MXExecutorFree) mxnet_executor;


    /***************************************************************************

        Stores the kind of the last forward pass which is used for verification
        in the backward pass

    ***************************************************************************/

    private ForwardPass last_forward_pass = ForwardPass.outputs;


    /***************************************************************************

        Enum of possible operations to perform when computing gradients

    ***************************************************************************/

    // FIXME
    // not yet exposed by the C API
    public enum OpReq
    {
        null_op,        /// no operation
        write,          /// write gradient
        write_in_place, /// write gradient in place avoiding a temporary
        add,            /// add computed result to existing
    }


    /***************************************************************************

        Constructs an executor through binding the network defined by model
        with its inputs and additional storage

        An Executor glues the symbolic network (the model) together with input
        values (the input data). It additionally requires a context to perform
        the computations in and space to store the computed gradients.

        Params:
            context = context in which the executor is defined
            model = symbol representing a (symbolic) network to perform passes
                    over
            inputs = inputs used when performing a forward pass
            gradients = outputs updated when performing a backward pass
            gradient_ops = defines how to update the gradients
            aux_states = auxiliary states of the model

    ***************************************************************************/

    public this (Context context, Symbol model,
                 NDArray!(T)[] inputs, NDArray!(T)[] gradients,
                 OpReq[] gradient_ops, NDArray!(T)[] aux_states)
    in
    {
        assert(inputs.length == gradients.length);
        assert(inputs.length == gradient_ops.length);
        assert(model.auxiliaryStates().length == aux_states.length);
    }
    body
    {
        void extractHandles (NDArray!(T)[] ndarrays, NDArrayHandle[] handles)
        {
            foreach (i, ndarray; ndarrays)
            {
                handles[i] = (ndarray is null) ? null : ndarray.handle();
            }
        }

        ExecutorHandle executor_handle;
        scope NDArrayHandle[] input_handles = new NDArrayHandle[inputs.length];
        extractHandles(inputs, input_handles);
        scope NDArrayHandle[] gradient_handles = new NDArrayHandle[gradients.length];
        extractHandles(gradients, gradient_handles);
        scope NDArrayHandle[] aux_states_handles = new NDArrayHandle[aux_states.length];
        extractHandles(aux_states, aux_states_handles);

        invoke!(MXExecutorBind)
               (model.handle(),
                context.type, context.id,
                to!(uint)(inputs.length), input_handles.ptr,
                gradient_handles.ptr, cast(uint*) gradient_ops.ptr,
                to!(uint)(aux_states.length), aux_states_handles.ptr,
                &executor_handle);

        this.mxnet_executor =
            new MXNetHandle!(ExecutorHandle, MXExecutorFree)(executor_handle);
    }


    /***************************************************************************

        Gives access to the underlying handle

        Use this handle with care. This is provided as a mechanism to use the C
        API directly when needed.

        Returns:
            the underlying handle used with the C library

    ***************************************************************************/

    private ExecutorHandle handle ()
    {
        return this.mxnet_executor.handle;
    }


    /***************************************************************************

        Performs a forward pass (from the inputs to the outputs)

        During training `pass` must be set to `ForwardPass.gradients` to
        perform additional calculations needed during the backward pass to
        calculate gradients. `ForwardPass.outputs` computes the outputs for the
        inputs provided during construction only.

        Params:
            pass = kind of forward pass to perform; defaults to a pass enabling
                   gradient calculations

    ***************************************************************************/

    public void forward (ForwardPass pass = ForwardPass.gradients)
    {
        this.mxnet_executor.apply!(MXExecutorForward)(pass);
        this.last_forward_pass = pass;
    }

    unittest
    {
        // model for least squares 1/2 * \|A*w' - b\|_2^2
        scope matrix_a = new Variable("A");
        scope (exit) matrix_a.freeHandle();
        scope vector_w = new Variable("w");
        scope (exit) vector_w.freeHandle();
        scope fc = new FullyConnected(matrix_a, 1, vector_w);
        scope (exit) fc.freeHandle();
        scope vector_b = new Variable("b");
        scope (exit) vector_b.freeHandle();
        scope model = new LinearRegressionOutput(fc, vector_b);
        scope (exit) model.freeHandle();

        auto context = cpuContext();

        // initialize two dimensional input vector
        scope input = new NDArray!(float)(context, [1, 2]);
        scope (exit) input.freeHandle();
        input.copyFrom([1, 2]);
        // and label to 1
        scope label = new NDArray!(float)(context, [1], 1f);
        scope (exit) label.freeHandle();
        // and weights
        scope weights = new NDArray!(float)(context, [1, 2]);
        scope (exit) weights.freeHandle();
        weights.copyFrom([0, 0]);
        auto inputs = [input, weights, label];

        // initialize gradient w.r.t. weights to zero
        scope gradient_weights = new NDArray!(float)(context, weights.shape(), 0f);
        scope (exit) gradient_weights.freeHandle();
        test!("==")(gradient_weights.data(), [0f, 0f]);
        auto gradients = [NDArray!(float).init, gradient_weights, NDArray!(float).init];
        auto gradients_ops = [OpReq.null_op, OpReq.write, OpReq.null_op];

        scope executor = new Executor(context, model, inputs, gradients, gradients_ops, []);
        scope (exit) executor.freeHandle();

        scope output = new NDArray!(float)(null);
        scope (exit) output.freeHandle();
        scope outputs = [output];
        // output is A*w'
        executor.outputs(outputs);

        // update outputs only (gradients are updated only in backward pass)
        executor.forward(ForwardPass.outputs);
        test!("==")(output.data(), [input.data[0] * weights.data[0] +
                                    input.data[1] * weights.data[1]]);

        // test with different inputs
        input.copyFrom([3, 4]);
        label.copyFrom([3]);
        executor.forward(ForwardPass.outputs);
        test!("==")(output.data(), [input.data[0] * weights.data[0] +
                                    input.data[1] * weights.data[1]]);

        // test with different weights
        weights.copyFrom([3, 4]);
        executor.forward(ForwardPass.outputs);
        test!("==")(output.data(), [input.data[0] * weights.data[0] +
                                    input.data[1] * weights.data[1]]);
    }


    /***************************************************************************

        Performs a backward pass (from the outputs to the inputs)

        Prior to calling `backward`, `forward(ForwardPass.gradients)` must have
        been called to allow computing gradients in the backward pass.

        After performing a backward pass the gradient array passed to the
        constructor holds the computed gradients.

        Throws:
            `MXNetException` if last forward pass computed only outputs

    ***************************************************************************/

    public void backward ()
    {
        mxnet_executor_exception.enforce(
            this.last_forward_pass == ForwardPass.gradients,
            "last forward pass has to be `ForwardPass.gradients`");

        uint gradients_length = 0;
        NDArrayHandle* gradients_ptr = null;
        this.mxnet_executor.apply!(MXExecutorBackward)(gradients_length,
                                                       gradients_ptr);
    }

    unittest
    {
        // model for least squares 1/2 * \|A*w' - b\|_2^2
        scope matrix_a = new Variable("A");
        scope (exit) matrix_a.freeHandle();
        scope vector_w = new Variable("w");
        scope (exit) vector_w.freeHandle();
        scope fc = new FullyConnected(matrix_a, 1, vector_w);
        scope (exit) fc.freeHandle();
        scope vector_b = new Variable("b");
        scope (exit) vector_b.freeHandle();
        scope model = new LinearRegressionOutput(fc, vector_b);
        scope(exit) model.freeHandle();

        auto context = cpuContext();

        // initialize two dimensional input vector
        scope input = new NDArray!(float)(context, [1, 2]);
        scope (exit) input.freeHandle();
        input.copyFrom([1, 2]);
        // and label to 1
        scope label = new NDArray!(float)(context, [1], 1f);
        scope (exit) label.freeHandle();
        // and weights
        scope weights = new NDArray!(float)(context, [1, 2]);
        scope (exit) weights.freeHandle();
        weights.copyFrom([0, 0]);
        auto inputs = [input, weights, label];

        // initialize gradient w.r.t. weights to zero
        scope gradient_weights = new NDArray!(float)(context, weights.shape(), 0f);
        scope (exit) gradient_weights.freeHandle();
        test!("==")(gradient_weights.data(), [0f, 0f]);
        auto gradients = [NDArray!(float).init, gradient_weights, NDArray!(float).init];
        auto gradients_ops = [OpReq.null_op, OpReq.write, OpReq.null_op];

        scope executor = new Executor(context, model, inputs, gradients, gradients_ops, []);
        scope (exit) executor.freeHandle();

        scope output = new NDArray!(float)(null);
        scope (exit) output.freeHandle();
        scope outputs = [output];
        executor.outputs(outputs);

        // update outputs and gradients
        executor.forward();
        executor.backward();
        // gradient w.r.t. to w is (A*w'-b) * A
        test!("==")(gradient_weights.data(), [(output.data[0] - label.data[0]) * input.data[0],
                                              (output.data[0] - label.data[0]) * input.data[1]]);

        // test with different inputs
        input.copyFrom([3, 4]);
        label.copyFrom([3]);
        executor.forward();
        executor.backward();
        // gradient w.r.t. to w
        test!("==")(gradient_weights.data(), [(output.data[0] - label.data[0]) * input.data[0],
                                              (output.data[0] - label.data[0]) * input.data[1]]);

        // test with different weights
        weights.copyFrom([3, 4]);
        executor.forward();
        executor.backward();
        // gradient w.r.t. to w
        test!("==")(gradient_weights.data(), [(output.data[0] - label.data[0]) * input.data[0],
                                              (output.data[0] - label.data[0]) * input.data[1]]);
    }


    /***************************************************************************

        Makes output data of the executor available via the provided `NDArray`
        instances

        The underlying handles of the provided `NDArray` instances are set to
        point to the output data generated by the executor's forward passes.
        As a result, after each forward pass, the results can be accessed via
        these `NDArray` instances.

        The handles of the passed `NDArray` instances must be `null` (free the
        handles explicitly if needed). This is to avoid the surprising behavior
        of this method freeing the handles under the hood (because after
        freeing the caller may have no `NDArray` instances able to access some
        data).

        Each time this method is called the handles of the passed instances are
        set. When calling this method more than once with different instances
        you end up with different `NDArray` instances with different handles
        but referencing the same underlying elements.

        Params:
            outputs = array of `NDArray` instances with `null` handles (must be
                      exactly one per output calculated by the executor) whose
                      handles will be updated such that the instances will
                      refer to the executor's underlying output data

        Throws:
            `MXNetExecutorException` if `outputs.length` is not equal to the
            number of computed outputs, or if any element of `outputs` is null
            or its handle is non-null

    ***************************************************************************/

    public void outputs (NDArray!(T)[] outputs)
    {
        NDArrayHandle* outputs_ptr;
        uint num_outputs;
        this.mxnet_executor.apply!(MXExecutorOutputs)(&num_outputs, &outputs_ptr);

        mxnet_executor_exception.enforce(outputs.length == num_outputs,
            "Insufficient NDArrays to contain output data!");

        foreach (i, handle; outputs_ptr[0 .. num_outputs])
        {
            if (handle is null)
            {
                throw mxnet_executor_exception.set(
                    "Output NDArrayHandle instance is null at index ").append(i);
            }

            if (outputs[i] is null)
            {
                throw mxnet_executor_exception.set(
                    "Output NDArray instance is null at index ").append(i);
            }

            if (outputs[i].handle !is null)
            {
                throw mxnet_executor_exception
                    .set("Output NDArray instance handle is non-null at index ")
                    .append(i);
            }

            outputs[i].handle = handle;
        }
    }

    unittest
    {
        scope model = new Variable("test");
        scope (exit) model.freeHandle();
        auto context = cpuContext();
        scope input = new NDArray!(float)(context, [2]);
        scope (exit) input.freeHandle();
        input.copyFrom([1, 2]);
        auto gradients = [NDArray!(float).init];
        auto gradients_ops = [OpReq.null_op];
        scope executor = new Executor(context, model, [input], gradients, gradients_ops, []);
        scope (exit) executor.freeHandle();
        scope output = new NDArray!(float)(null);
        scope (exit) output.freeHandle();
        scope outputs = [output];
        executor.outputs(outputs);
        test!("==")(output.data, input.data);

        scope output2 = new NDArray!(float)(null);
        executor.outputs([output2]);

        // different instances
        test!("!is")(output, output2);
        // with different handles
        test!("!is")(output.handle, output2.handle);
        // but sharing the same data
        test!("is")(output.data, output2.data);

        output2.freeHandle();
        // output still valid after freeing output2
        test!("==")(output.data, input.data);
    }


    /***************************************************************************

        Frees the underlying handle including its resources

    ***************************************************************************/

    public void freeHandle ()
    {
        this.mxnet_executor.freeHandle();
    }
}


/*******************************************************************************

    Kind of the forward pass

*******************************************************************************/

public enum ForwardPass
{
    gradients = true, /// forward pass to enable calculating gradients in the backward pass
    outputs = false   /// calculate only the outputs
}
