/*******************************************************************************

    Combines the higher layer MXNet API in one module

    The API is separated into

      * n-dimensional arrays to perform calculations (see NDArray)

      * symbolic definition of a network (see Symbol)

      * execution of a network (see Executor)

    This separation follows MXNet's C API. The functions to perform IO (for
    reading data sets) and interaction with a key value store (for distributed
    setups) are not covered by the D API implemented here. Resort to the C API
    for using this functionality.

    To carry out the computations it defines a context that is used to
    construct n-dimensional arrays and executors.

    This module publicly imports Symbol, Context, NDArray and Executor to ease
    usage since these are commonly used together.

    Copyright:
        Copyright (c) 2017 dunnhumby Germany GmbH.

    License:
        Boost Software License Version 1.0.  See accompanying LICENSE.txt for
        details, or <https://www.boost.org/LICENSE_1_0.txt>

*******************************************************************************/

module mxnet.MXNet;

public import mxnet.Atomic;
public import mxnet.API;
public import mxnet.Context;
public import mxnet.Exception;
public import mxnet.Executor;
public import mxnet.Handle;
public import mxnet.NDArray;
public import mxnet.Symbol;

///
unittest
{
    // number of classes per example
    auto num_classes = 10;
    // number of features per example
    auto num_features = 20;
    // number of examples per training batch
    auto batch_size = 100;

    // define multi class logistic regression
    // according to the formula
    // sum_i -log(softmax(x_i*W' + b, y_i))
    //
    // where i runs over the number of training examples and softmax(x, j) =
    // exp(x_j) / sum(exp(x))
    //
    // this formula combines the cross entropy loss with the softmax function
    //
    // a label is predicted for some unseen example x
    // by returning the index for which x*W' + b is maximal
    scope feature_matrix = new Variable("X");
    scope (exit) feature_matrix.freeHandle();
    scope label_vector = new Variable("y");
    scope (exit) label_vector.freeHandle();
    scope weights = new Variable("W");
    scope (exit) weights.freeHandle();
    scope biases = new Variable("b");
    scope (exit) biases.freeHandle();
    scope prediction_vector = new FullyConnected(feature_matrix, num_classes,
                                                 weights, biases);
    scope (exit) prediction_vector.freeHandle();
    scope model = new SoftmaxOutput(prediction_vector, label_vector);
    scope (exit) model.freeHandle();

    // define context for carrying out the computations
    auto context = cpuContext();

    // provide data for the parameters
    scope matrix_x = new NDArray!(float)(context, [batch_size, num_features]);
    scope (exit) matrix_x.freeHandle();
    matrix_x = 1f;
    scope vector_y = new NDArray!(float)(context, [batch_size]);
    scope (exit) vector_y.freeHandle();
    vector_y = 2f;

    // initialize variables
    scope matrix_w = new NDArray!(float)(context, [num_classes, num_features]);
    scope (exit) matrix_w.freeHandle();
    matrix_w = 0f;
    scope vector_b = new NDArray!(float)(context, [num_classes]);
    scope (exit) vector_b.freeHandle();
    vector_b = 0f;

    // storage for gradients
    scope gradient_w = new NDArray!(float)(context, matrix_w.shape());
    scope (exit) gradient_w.freeHandle();
    scope gradient_b = new NDArray!(float)(context, vector_b.shape());
    scope (exit) gradient_b.freeHandle();

    // setup executor
    // the inputs
    auto inputs = [matrix_x, matrix_w, vector_b, vector_y];
    // the gradient storage
    auto gradients = [NDArray!(float).init, gradient_w, gradient_b, NDArray!(float).init];
    // the operations to perform on gradients
    auto gradients_req_type = [Executor!(float).OpReq.null_op,
                               Executor!(float).OpReq.write,
                               Executor!(float).OpReq.write,
                               Executor!(float).OpReq.null_op];
    // bind all together
    scope executor = new Executor!(float)(context, model, inputs, gradients, gradients_req_type, []);
    scope (exit) executor.freeHandle();

    // make a forward and a backward pass over the defined model using the
    // provided inputs and computing the gradients
    executor.forward();
    executor.backward();
}
