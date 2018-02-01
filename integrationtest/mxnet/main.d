/*******************************************************************************

    Performs multi-class logistic regression on the MNIST dataset using MXNet

    Copyright:
        Copyright (c) 2017 sociomantic labs GmbH.

    License:
        Boost Software License Version 1.0.  See accompanying LICENSE.txt for
        details, or <https://www.boost.org/LICENSE_1_0.txt>

*******************************************************************************/

module integrationtest.mxnet.main;

import MNIST = integrationtest.mxnet.MNIST;

import mxnet.MXNet;

import ocean.core.Test;
import ocean.io.Stdout;
import ocean.time.StopWatch;
import ocean.transition;
import ocean.util.Convert;


version (UnitTest) {} else
void main (istring[] args)
{
    auto mnist_dataset_dir = MNIST.datasetPath(args);

    // mnist dataset dimensions
    uint num_pixels = 784;
    uint num_classes = 10;
    uint training_size = 60000;
    uint testing_size = 10000;

    Stdout.formatln("MXNet handles in use at start: {}", handleCount());
    scope (exit)
    {
        Stdout.formatln("MXNet handles in use at exit: {}", handleCount());
    }

    // load and prepare MNIST dataset
    auto mnist_training = MNIST.trainingSet(mnist_dataset_dir);
    auto mnist_training_images = to!(float[])(mnist_training.images);
    test!("==")(mnist_training_images.length, training_size * num_pixels);
    mnist_training_images[] /= 255;
    auto mnist_training_labels = to!(float[])(mnist_training.labels);
    test!("==")(mnist_training_labels.length, training_size);

    auto mnist_testing = MNIST.testingSet(mnist_dataset_dir);
    auto mnist_testing_images = to!(float[])(mnist_testing.images);
    test!("==")(mnist_testing_images.length, testing_size * num_pixels);
    mnist_testing_images[] /= 255;
    auto mnist_testing_labels = to!(size_t[])(mnist_testing.labels);
    test!("==")(mnist_testing_labels.length, testing_size);

    // creating the network
    // inputs
    scope x_symbol = new Variable("X"); // feature matrix
    scope (exit) x_symbol.freeHandle();
    scope y_symbol = new Variable("y"); // label vector
    scope (exit) y_symbol.freeHandle();

    // network architecture including variables
    scope w_symbol = new Variable("W");
    scope (exit) w_symbol.freeHandle();
    scope fc = new FullyConnected(x_symbol, num_classes, w_symbol);
    scope (exit) fc.freeHandle();
    scope softmax = new SoftmaxOutput(fc, y_symbol);
    scope (exit) softmax.freeHandle();

    // setup context to carry out computations
    auto context = cpuContext();
    // size of a training batch
    auto batch_size = 100;
    // parameter X
    scope matrix_x = new NDArray!(float)(context, [batch_size, num_pixels]);
    scope (exit) matrix_x.freeHandle();
    // parameter y
    scope vector_y = new NDArray!(float)(context, [batch_size]);
    scope (exit) vector_y.freeHandle();

    // variable W initialized to zero
    scope matrix_w = new NDArray!(float)(context, [num_classes, num_pixels], 0f);
    scope (exit) matrix_w.freeHandle();

    NDArray!(float)[] variables = [matrix_x,
                                   matrix_w,
                                   vector_y];

    // gradient w.r.t. W
    scope gradient_w = new NDArray!(float)(context, matrix_w.shape());
    scope (exit) gradient_w.freeHandle();

    NDArray!(float)[] gradients = [NDArray!(float).init,  // no gradient for inputs
                                   gradient_w,
                                   NDArray!(float).init]; // no gradient for outputs

    auto gradients_req_type = [Executor!(float).OpReq.null_op,
                               Executor!(float).OpReq.write,
                               Executor!(float).OpReq.null_op];

    // verify that the all variables are provided in proper order
    assert(softmax.arguments == ["X", "W", "y"]);
    // define the executor binding the model with the parameters (data) and variables
    scope executor = new Executor!(float)(context, softmax, variables, gradients,
                                          gradients_req_type, []);
    scope (exit) executor.freeHandle();

    // training
    auto num_iterations = 4000;
    StopWatch watch;
    watch.start();
    for (size_t i = 0; i < num_iterations; ++i)
    {
        assert(training_size % batch_size == 0);
        auto num_batches_per_epoch = training_size / batch_size;
        auto ii = i % num_batches_per_epoch;

        // extract batch from training set
        auto images_batch = mnist_training_images[ii * batch_size * num_pixels .. (ii + 1) * batch_size * num_pixels];
        auto labels_batch = mnist_training_labels[ii * batch_size .. (ii + 1) * batch_size];

        assert(images_batch.length == batch_size * num_pixels);
        assert(labels_batch.length == batch_size);

        // set batch and ...
        matrix_x.copyFrom(images_batch);
        vector_y.copyFrom(labels_batch);

        // make a forward and a backward pass
        executor.forward();
        executor.backward();
        auto step_length = 5e-1f;
        // and update variable in direction of negative gradient
        gradient_w *= step_length;
        matrix_w -= gradient_w;
    }
    auto iteration_time_sec = watch.sec();

    auto predicted_testing_labels = predict(softmax, matrix_w, testing_size, mnist_testing_images);
    auto testing_accuracy = accuracy(predicted_testing_labels, mnist_testing_labels);
    // prediction on testing
    test!(">=")(testing_accuracy, 0.915);
    Stdout.formatln("Percentage of correctly predicted digits on MNIST testing: {:f4}%", testing_accuracy);
    Stdout.formatln("Calculated in {:f6} seconds", iteration_time_sec);
    Stdout.formatln("MXNet handles in use at finish: {}", handleCount());
}


/*******************************************************************************

    Predicts the labels for MNIST digit images

    Params:
        model = the model used for prediction
        matrix_w = learned parameters of model
        num_images = number of images
        images = digit images data

    Returns:
        the predicted label for each image

*******************************************************************************/

size_t[] predict (Symbol model, NDArray!(float) matrix_w,
                  uint num_images, Const!(float)[] images)
{
    assert(images.length % num_images == 0);
    uint num_pixels = to!(uint)(images.length) / num_images;

    scope matrix_x = new NDArray!(float)(cpuContext(), [num_images, num_pixels]);
    scope (exit) matrix_x.freeHandle();
    matrix_x.copyFrom(images);
    scope vector_y = new NDArray!(float)(cpuContext(), [num_images]);
    scope (exit) vector_y.freeHandle();

    NDArray!(float)[] variables = [matrix_x,
                                   matrix_w,
                                   vector_y];

    NDArray!(float)[] gradients = [NDArray!(float).init,
                                   NDArray!(float).init,
                                   NDArray!(float).init];

    auto gradients_req_type = [Executor!(float).OpReq.null_op,
                               Executor!(float).OpReq.null_op,
                               Executor!(float).OpReq.null_op];

    scope executor = new Executor!(float)(cpuContext(), model, variables, gradients,
                                          gradients_req_type, []);
    scope (exit) executor.freeHandle();

    executor.forward(ForwardPass.outputs);
    scope output = new NDArray!(float)(null);
    scope (exit) output.freeHandle();
    auto outputs = [output];
    executor.outputs(outputs);
    auto predictions_vector = to!(float[])(output.data());

    auto predictions = new size_t[num_images];
    for (uint i = 0; i < num_images; ++i)
    {
        uint num_classes = 10;
        auto prediction = predictions_vector[i * num_classes .. (i + 1) * num_classes];

        // search for label with largest prediction probability
        // (this is basically a `foldr` operation on the index
        // values, so could be replaced by that in future if a
        // `fold` implementation becomes available)
        size_t predicted_label = 0;
        for (size_t label = 1; label < num_classes; ++label)
        {
            assert(prediction[label] >= 0);
            if (prediction[label] > prediction[predicted_label])
            {
                predicted_label = label;
            }
        }
        assert(predicted_label < num_classes);

        predictions[i] = predicted_label;
    }

    return predictions;
}


/*******************************************************************************

    Calculates the percentages of correctly predicted labels

    Params:
        predicted_labels = predicted labels
        expected_labels = expected labels

    Returns:
        the percentage of labels in predicted_labels that was predicted
        correctly w.r.t. expected_labels

*******************************************************************************/

double accuracy (Const!(size_t)[] predicted_labels, Const!(size_t)[] expected_labels)
{
    assert(expected_labels.length == predicted_labels.length);

    size_t correct_predicted = 0;
    for (size_t i = 0; i < predicted_labels.length; ++i)
    {
        correct_predicted += predicted_labels[i] == expected_labels[i];
    }

    return (1.0 * correct_predicted) / predicted_labels.length;
}
