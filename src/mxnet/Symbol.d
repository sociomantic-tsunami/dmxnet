/*******************************************************************************

    Defines a symbol and various predefined symbols which can be combined to
    specify a model

    Copyright:
        Copyright (c) 2017 dunnhumby Germany GmbH.

    License:
        Boost Software License Version 1.0.  See accompanying LICENSE.txt for
        details, or <https://www.boost.org/LICENSE_1_0.txt>

*******************************************************************************/

module mxnet.Symbol;

import mxnet.c.c_api;
import mxnet.API;
import mxnet.Exception;
import mxnet.Handle;
import mxnet.Util;

import core.exception;

import ocean.core.Enforce;
import ocean.text.convert.Formatter;
import ocean.text.util.StringC;
import ocean.transition;
import ocean.util.Convert;

version(UnitTest)
{
    import ocean.core.Test;
}


/*******************************************************************************

    A Symbol is the basic unit for constructing networks

    Symbols are used to build networks which describe a model.

    A Symbol allocates MXNet resources accessed through an `MXNetHandle`. Its
    resources should be freed by calling `freeHandle` when done with the
    Symbol. This should happen in a timely manner to avoid resource depletion.
    Note that scope allocating an object or manually calling `delete` won't
    free the resources since a Symbols's destructor does not free the
    resources. To reclaim resources `freeHandle` must be called manually.

*******************************************************************************/

public class Symbol
{
    import mxnet.Atomic;

    /***************************************************************************

        Underlying managed symbol handle used to interact with C library

    ***************************************************************************/

    private MXNetHandle!(SymbolHandle, MXSymbolFree) mxnet_symbol;


    /**************************************************************************

        Array holding the names of the auxiliary states of this symbol

        The array is resized as necessary to hold all the names of the
        auxiliary states of this symbol.

    ***************************************************************************/

    private cstring[] auxiliary_state_names;


    /**************************************************************************

        Array holding the names of the arguments of this symbol

        The array is resized as necessary to hold all the names of the
        arguments of this symbol.

    ***************************************************************************/

    private cstring[] argument_names;


    /***************************************************************************

        Constructor for atomic symbol instances

        Atomic symbols are symbols predefined by the MXNet library.  A full
        list can be obtained using the `mxnet.Atomic.atomicSymbolList` helper
        function, while detailed information on an individual atomic symbol
        type can be obtained by using `mxnet.Atomic.atomicSymbolInfo`.

        Params:
            name = name of the type of atomic symbol to create
            keys = array of C strings defining the keys (names) of the
                   input parameters required to create the atomic symbol
                   (should be empty or `null` if no input parameters are
                   required)
            values = array of C strings encoding the values of the input
                     parameters required to create the atomic symbol
                     (must be the same length as `keys`; should be empty
                     or `null` if no input parameters are required)

    ***************************************************************************/

    this (cstring name, in char*[] keys, in char*[] values)
    in
    {
        assert(keys.length == values.length);
    }
    body
    {
        SymbolHandle symbol_handle;
        invoke!(MXSymbolCreateAtomicSymbol)
               (atomicSymbolCreator(name), to!(uint)(keys.length),
                keys.ptr, values.ptr, &symbol_handle);
        this(symbol_handle);
    }

    unittest
    {
        scope broadcast_sub = new Symbol("broadcast_sub", null, null);
        scope (exit) broadcast_sub.freeHandle();
        test!("==")(broadcast_sub.toStringDebug(),
                    "AtomicFunctor  Op:broadcast_sub\n");

        // these naughty, naughty MXNet people, putting formatting
        // indications into the debug string ... :-)

        scope softmax_output = new Symbol("SoftmaxOutput", null, null);
        scope (exit) softmax_output.freeHandle();
        test!("==")(softmax_output.toStringDebug(),
                    "AtomicFunctor  Op:SoftmaxOutput\n");
    }


    /***************************************************************************

        Constructs a Symbol from a given symbol handle

        Params:
            handle = C API symbol handle

    ***************************************************************************/

    public this (SymbolHandle handle)
    in
    {
        assert(handle !is null);
    }
    body
    {
        this.mxnet_symbol = new MXNetHandle!(SymbolHandle, MXSymbolFree)(handle);
    }


    /***************************************************************************

        Returns:
            MXNet's internal debugging string for this symbol

    ***************************************************************************/

    public cstring toStringDebug ()
    {
        Const!(char*) str;
        this.mxnet_symbol.apply!(MXSymbolPrint)(&str);
        return StringC.toDString(str);
    }

    unittest
    {
        SymbolHandle symbol_handle;
        invoke!(MXSymbolCreateVariable)("".ptr, &symbol_handle);
        scope symbol = new Symbol(symbol_handle);
        scope (exit) symbol.freeHandle();
        test(symbol.toStringDebug().length);
    }


    /***************************************************************************

        Gives access to the underlying handle

        Use this handle with care. This is provided as a mechanism to use the C
        API directly when needed.

        Returns:
            the underlying handle used with the C library

    ***************************************************************************/

    public SymbolHandle handle ()
    {
        return this.mxnet_symbol.handle();
    }

    unittest
    {
        SymbolHandle symbol_handle;
        invoke!(MXSymbolCreateVariable)("".ptr, &symbol_handle);
        scope symbol = new Symbol(symbol_handle);
        scope (exit) symbol.freeHandle();
        test!("is")(symbol.handle(), symbol_handle);
    }


    /**************************************************************************

        Determines the names of auxiliary states of this symbol and returns
        them

        Auxiliary states cover additional information of a symbol. No
        information on the gradients of auxiliary states is calculated. This is
        the primary difference to normal arguments. It can be used for tracking
        purposes. Most symbols have no auxiliary states.

        When binding an executor to this symbol an array of n-dimensional
        arrays must be provided for the auxiliary states in the same order as
        returned by this function.

        Returns:
            an array of the names of auxiliary states of this symbol;
            the returned array is an internal buffer that is reused in later
            invocations, thus if necessary, perform a deep copy

    ***************************************************************************/

    public Const!(cstring[]) auxiliaryStates ()
    {
        mx_uint states_length;
        Const!(char**) states_ptr;
        this.mxnet_symbol.apply!(MXSymbolListAuxiliaryStates)(&states_length, &states_ptr);

        this.auxiliary_state_names.length = states_length;
        enableStomping(this.auxiliary_state_names);
        foreach (i, str; states_ptr[0 .. states_length])
        {
            this.auxiliary_state_names[i] = StringC.toDString(str);
        }

        return this.auxiliary_state_names;
    }

    unittest
    {
        scope norm_symbol = new Symbol("norm", null, null);
        scope (exit) norm_symbol.freeHandle();
        test!("==")(norm_symbol.auxiliaryStates().length, 0);

        // symbol with auxiliary states
        scope batch_norm_symbol = new Symbol("BatchNorm", null, null);
        scope (exit) batch_norm_symbol.freeHandle();
        scope input = new Variable("input");
        scope (exit) input.freeHandle();
        batch_norm_symbol.compose([input.handle], null);
        test!("==")(batch_norm_symbol.auxiliaryStates(), ["moving_mean", "moving_var"]);
    }


    /***************************************************************************

        Determines and returns the names of all arguments of this symbol

        When executing a model the inputs must be provided in the same order as
        returned by this function to properly match up.

        Returns:
            an array containing the names of all arguments;
            the returned array is an internal buffer that is reused in later
            invocations, thus if necessary, perform a deep copy

    ***************************************************************************/

    public cstring[] arguments ()
    {
        mx_uint arguments_length;
        Const!(char**) arguments_ptr;
        this.mxnet_symbol.apply!(MXSymbolListArguments)
                                (&arguments_length, &arguments_ptr);

        this.argument_names.length = arguments_length;
        enableStomping(this.argument_names);
        foreach (i, str; arguments_ptr[0 .. arguments_length])
        {
            this.argument_names[i] = StringC.toDString(str);
        }

        return this.argument_names;
    }

    unittest
    {
        scope s = new Symbol("broadcast_sub", null, null);
        scope (exit) s.freeHandle();
        test!("==")(s.arguments().length, 0);

        scope feature_matrix = new Variable("X");
        scope (exit) feature_matrix.freeHandle();
        scope label_vector = new Variable("y");
        scope (exit) label_vector.freeHandle();
        scope weight_matrix = new Variable("W");
        scope (exit) weight_matrix.freeHandle();
        scope bias_vector = new Variable("b");
        scope (exit) bias_vector.freeHandle();
        scope prediction_vector = new FullyConnected(feature_matrix, 10,
                                                     weight_matrix, bias_vector);
        scope (exit) prediction_vector.freeHandle();
        scope model = new SoftmaxOutput(prediction_vector, label_vector);
        scope (exit) model.freeHandle();
        test!("==")(model.arguments, ["X", "W", "b", "y"]);
    }


    /***************************************************************************

        Composes this symbol with input symbols which updates this symbol to be
        applied to the inputs

        With symbol composition you take an symbol expecting inputs and
        appropriate input symbols and combine them. Any input symbol can be
        destructed after this symbol has been composed with the inputs. There
        is no dependency on the input symbols after performing a composition.

        Params:
            inputs = symbols that are inputs to s
            keys = keys for keyword arguments;
                   pass `null` for no keyword arguments, i.e., input symbol are
                   passed by position
            file = file from which the call originates
            line = line from which the call originates

        Throws:
            `MXNetHandleException` if any input SymbolHandle is null

    ***************************************************************************/

    private void compose (SymbolHandle[] inputs, Const!(char*)[] keys,
                          istring file = __FILE__, int line = __LINE__)
    {
        foreach (i, input; inputs)
        {
            if (input is null)
            {
                throw mxnet_handle_exception.set(
                    "Null input SymbolHandle at index ", file, line).append(i);
            }
        }

        this.mxnet_symbol.apply!(MXSymbolCompose)("".ptr,
                                                  to!(uint)(inputs.length),
                                                  keys.ptr,
                                                  inputs.ptr);
    }

    unittest
    {
        scope s = new Symbol("norm", null, null);
        scope (exit) s.freeHandle();
        SymbolHandle symbol_handle;
        invoke!(MXSymbolCreateVariable)("".ptr, &symbol_handle);
        s.compose([symbol_handle], null);
        test!("!is")(s.handle(), null);
        // safe to free any input symbol handle
        invoke!(MXSymbolFree)(symbol_handle);
        test!("!is")(s.handle(), null);
        test(s.toStringDebug().length);

        scope symbol = new Symbol("norm", null, null);
        scope (exit) symbol.freeHandle();
        // input handles must be non-null
        testThrown!(MXNetHandleException)(symbol.compose([null], null));
    }


    /***************************************************************************

        Frees the underlying handle including its resources

    ***************************************************************************/

    public void freeHandle ()
    {
        this.mxnet_symbol.freeHandle();
    }
}


/*******************************************************************************

    A symbol representing a variable

    In MXNet variables are used to represent both the input variables provided
    to a model and the parameters of the model itself. For example given a
    model `f(x; a) = a*x^2`, `x` and `a` are both MXNet variables, where `x` is
    the input variable of `f` and `a` is the parameter of `f`.

    A Variable allocates MXNet resources accessed through an `MXNetHandle`. Its
    resources should be freed by calling `freeHandle` when done with the
    Variable. This should happen in a timely manner to avoid resource
    depletion. Note that scope allocating an object or manually calling
    `delete` won't free the resources since a Variable's destructor does not
    free the resources. To reclaim resources `freeHandle` must be called
    manually.

*******************************************************************************/

public class Variable : Symbol
{
    /***************************************************************************

        Null terminated name of this variable

        The null terminator is excluded via slicing but nevertheless created
        during construction. This allows direct usage of its pointer when a
        null terminated string (a.k.a. C string) has to be passed to a C
        function.

        This is also used to provide the symbol name back to the user, to
        ensure that any shallow copies will remain valid even after this
        `Variable` instance (or its underlying symbol handle) is freed.

    ***************************************************************************/

    private mstring name_;


    /***************************************************************************

        Constructs a symbol representing a variable

        Params:
            variable_name = name of the variable

    ***************************************************************************/

    public this (cstring variable_name)
    {
        this.name = variable_name;
        SymbolHandle s;
        invoke!(MXSymbolCreateVariable)(this.name_.ptr, &s);
        super(s);
    }

    unittest
    {
        scope v = new Variable("test");
        scope (exit) v.freeHandle();
        test!("!is")(v.handle(), null);
    }


    /***************************************************************************

        Sets the name of this variable

        Params:
            variable_name = the name this variable will be set to

    ***************************************************************************/

    private void name (cstring variable_name)
    {
        this.name_.length = variable_name.length + 1;
        this.name_[0 .. $ - 1] = variable_name[];
        this.name_[$ - 1] = '\0';
        this.name_.length = this.name_.length - 1;
        enableStomping(this.name_);
    }


    /***************************************************************************

        Returns:
            the name of this variable

        Throws:
            an MXNetException if the underlying MXNet symbol name could not be
            obtained or if the variable name could not be validated against the
            underlying MXNet symbol name

    ***************************************************************************/

    public cstring name ()
    {
        Const!(char*) mxnet_symbol_name;
        int failed_name;
        this.mxnet_symbol.apply!(MXSymbolGetName)
                                (&mxnet_symbol_name, &failed_name);
        enforce!(MXNetException)
                (failed_name != 0,
                 "Failed to get the underlying MXNet symbol name");
        enforce!(MXNetException)
                (this.name_ == StringC.toDString(mxnet_symbol_name),
                 "Name is not consistent with underlying MXNet symbol name");

        return this.name_;
    }

    unittest
    {
        scope v = new Variable("test");
        scope(exit) v.freeHandle();
        test!("==")(v.name(), "test");
    }
}


/*******************************************************************************

    Activation functions

    This enum is used to specify which activation function to use when
    constructing an Activation symbol.

*******************************************************************************/

public enum ActivationFunction
{
    relu = 0,
    sigmoid,
    softrelu,
    tanh,
}


/*******************************************************************************

    A symbol representing an activation function

    Activation functions are applied to obtain an node's output in a neural
    network. That is, the node computes the linear combination of its inputs
    and its weights and applies an activation function (typically non-linear)
    on this computed value.
    Common choices are:

        * Hyperbolic tangent <https://en.wikipedia.org/wiki/Hyperbolic_function#Hyperbolic_tangent>

        * Sigmoid function <https://en.wikipedia.org/wiki/Sigmoid_function>

        * Rectified linear unit (ReLU) <https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>

    An Activation allocates MXNet resources accessed through an `MXNetHandle`.
    Its resources should be freed by calling `freeHandle` when done with the
    Activation. This should happen in a timely manner to avoid resource
    depletion. Note that scope allocating an object or manually calling
    `delete` won't free the resources since a Activation's destructor does not
    free the resources. To reclaim resources `freeHandle` must be called
    manually.

*******************************************************************************/

public class Activation : Symbol
{
    /***************************************************************************

        Constructs a symbol representing performing the `activation` on
        the `input` symbol

        Params:
            input = input symbol to apply the activation to
            activation = activation function to perform

    ***************************************************************************/

    public this (Symbol input, ActivationFunction activation)
    in
    {
        assert(input !is null);
    }
    body
    {
        const istring[] activations = ["relu", "sigmoid", "softrelu", "tanh"];

        istring key = "act_type";

        Immut!(char)*[1] keys;
        keys[0] = key.ptr;
        Immut!(char)*[1] values;
        values[0] = activations[activation].ptr;

        super("Activation", keys, values);

        SymbolHandle[1] input_handle;
        input_handle[0] = input.handle();
        this.compose(input_handle, null);
    }

    unittest
    {
        scope v = new Variable("data");
        scope (exit) v.freeHandle();
        scope a = new Activation(v, ActivationFunction.relu);
        scope (exit) a.freeHandle();
        test!("!is")(a.handle(), null);
    }
}


/*******************************************************************************

    A symbol representing a fully connected layer

    In a fully connected layer each input is connected to each output. Assuming
    there n inputs and m outputs the number of connections is n * m. Each
    of the m output values is computed as a linear combination of the input x
    and n weight values.

    The operation performed by fully connected layer can be represented by a
    matrix vector product, W*x. This yields an m-dimensional (output) vector
    for W an m by n matrix.

    An FullyConnected allocates MXNet resources accessed through an
    `MXNetHandle`. Its resources should be freed by calling `freeHandle` when
    done with the FullyConnected. This should happen in a timely manner to
    avoid resource depletion. Note that scope allocating an object or manually
    calling `delete` won't free the resources since a FullyConnected's
    destructor does not free the resources. To reclaim resources `freeHandle`
    must be called manually.

*******************************************************************************/

public class FullyConnected : Symbol
{
    /***************************************************************************

        Constructs a symbol representing a fully connected layer from
        `input` to `num_hidden`

        In more formal notation the returned symbol encodes `input * weights +
        biases`.

        Params:
            input = input symbol to fully connect to
            num_hidden = number of outputs of the returned symbol
            weights = symbol representing the weights
            biases = optional symbol representing the bias terms

    ***************************************************************************/

    public this (Symbol input,
                 uint num_hidden,
                 Symbol weights,
                 Symbol biases = null)
    in
    {
        assert(input !is null);
        assert(num_hidden > 0);
        assert(weights !is null);
    }
    body
    {
        istring[2] keys;
        keys[0] = "num_hidden";
        keys[1] = "no_bias";

        // 12 chars should be enough to store the largest uint plus `\0`
        char[12] num_hidden_str;
        snformat(num_hidden_str, "{}\0", num_hidden);

        istring no_bias = (biases is null) ? "true" : "false";

        Immut!(char)*[2] c_keys;
        foreach (i, ref key; keys) c_keys[i] = key.ptr;

        Const!(char)*[2] c_values;
        c_values[0] = num_hidden_str.ptr;
        c_values[1] = no_bias.ptr;

        super("FullyConnected", c_keys, c_values);

        SymbolHandle[3] args;
        args[0] = input.handle();
        args[1] = weights.handle();

        if (biases is null)
        {
            this.compose(args[0 .. 2], null);
        }
        else
        {
            args[2] = biases.handle();
            this.compose(args, null);
        }
    }

    unittest
    {
        scope inputs = new Variable("X");
        scope (exit) inputs.freeHandle();
        scope weights = new Variable("W");
        scope (exit) weights.freeHandle();
        scope biases = new Variable("b");
        scope (exit) biases.freeHandle();
        scope fc = new FullyConnected(inputs, 100, weights, biases);
        scope (exit) fc.freeHandle();
        test!("!is")(fc.handle(), null);
    }
}


/*******************************************************************************

    Normalization for softmax outputs

*******************************************************************************/

public enum SoftmaxOutputNormalization
{
    batch = 0, /// divide by batch size
    off   = 1, /// do nothing
    valid = 2, /// divide by the number of examples which are not ignored
}


/*******************************************************************************

    A symbol representing softmax (and cross entropy loss for calculating
    objective and gradient information)

    This symbol represents the application of the softmax function to the
    input. For a backward pass the cross entropy is calculated using the label
    as reference for obtaining objective and gradient information.

    A label is either already a probability vector or treated internally as
    such by one-hot encoding.

    An input is transformed to a probability vector by the softmax function
    which is defined as `softmax(x) = exp(x) / sum(exp(x))`. The softmax
    function is a generalization of the logistic function allowing for more
    than two classes.

    The cross entropy measures the similarity of two probability distributions
    over the same set of events. It is calculated between two probability
    vectors as `-y' * log(softmax(x))` where `'` denotes transposition. It is
    used as an objective function in this symbol.

    In case you have more than one label-input pair, this function sums the
    cross entropy losses over all pairs, formally
    `-sum(y_i' * log(softmax(x_i)))` where the summation runs over the index
    `i` and all vectors `y_i` and `x_i` have the same length, that is, the
    number of label classes.

    For prediction purposes (i.e., on a forward pass) only the softmax is
    calculated, turning the input into a probability vector. In training (i.e.,
    a forward pass with objective followed by a backward pass) the cross
    entropy is used for calculating objective and gradient information.

    An SoftmaxOutput allocates MXNet resources accessed through an
    `MXNetHandle`. Its resources should be freed by calling `freeHandle` when
    done with the SoftmaxOutput. This should happen in a timely manner to
    avoid resource depletion. Note that scope allocating an object or manually
    calling `delete` won't free the resources since a SoftmaxOutput's
    destructor does not free the resources. To reclaim resources `freeHandle`
    must be called manually.

*******************************************************************************/

public class SoftmaxOutput : Symbol
{
    import ocean.core.Traits;

    /***************************************************************************

        Struct containing configuration options for the `SoftmaxOutput`
        symbol

        By default, all values will match the defaults used by the upstream
        MXNet library.  This struct only needs to be used directly if one or
        more of these parameters is to be set to a non-default value.

    ***************************************************************************/

    public struct Config
    {
        /***********************************************************************

            Scale factor for scaling the gradient

        ***********************************************************************/

        public float grad_scale = 1;

        /***********************************************************************

            Label to ignore during the backward pass, if `use_ignore` is `true`

        ***********************************************************************/

        public float ignore_label = -1;

        /***********************************************************************

            If `true`, the softmax function will be computed along the second
            axis, i.e. `axis == 1`

        ***********************************************************************/

        public bool multi_output = false;

        /***********************************************************************

            If `true`, the `ignore_label` field can be used to specify a label
            to ignore during the backward pass

        ***********************************************************************/

        public bool use_ignore = false;

        /***********************************************************************

            If `true`, the softmax function will be computed along the last
            axis, i.e. `axis == -1`

        ***********************************************************************/

        public bool preserve_shape = false;

        /***********************************************************************

            Normalization applied to the gradient

        ***********************************************************************/

        public SoftmaxOutputNormalization normalization =
            SoftmaxOutputNormalization.batch;

        /***********************************************************************

            If `true`, the gradient will be multiplied elementwise by the
            output gradient

        ***********************************************************************/

        public bool out_grad = false;
    }

    ///
    unittest
    {
        // initialize a `Config` struct instance with only a couple of
        // non-default values
        Config config = {
            ignore_label: 1,
            use_ignore: true
        };

        // check values of custom fields
        assert(config.ignore_label == 1.0f);
        assert(config.ignore_label != Config.init.ignore_label);

        assert(config.use_ignore);
        assert(config.use_ignore != Config.init.use_ignore);

        // check values of non-customized fields
        assert(config.grad_scale == Config.init.grad_scale);
        assert(config.multi_output == Config.init.multi_output);
        assert(config.preserve_shape == Config.init.preserve_shape);
        assert(config.normalization == Config.init.normalization);
        assert(config.out_grad == Config.init.out_grad);
    }


    /***************************************************************************

        Constructs `SoftmaxOutput` symbol

        Params:
            input = input symbol to apply softmax to
            label = ground truth to compare against the output of softmax
            config = optional instance of Config struct containing extra
                     configuration parameters (if not specified, all will
                     be set to their default values)

    ***************************************************************************/

    public this (Symbol input,
                 Symbol label,
                 Config config = Config.init)
    in
    {
        assert(input !is null);
        assert(label !is null);
    }
    body
    {
        const istring[] softmax_normalizations = ["batch", "null", "valid"];

        alias typeof(Config.tupleof) ConfigTuple;

        istring[ConfigTuple.length] keys;
        Const!(char)*[ConfigTuple.length] c_keys;
        Const!(char)*[ConfigTuple.length] c_values;

        foreach (i, T; ConfigTuple)
        {
            keys[i] = FieldName!(i, Config);
            c_keys[i] = keys[i].ptr;

            auto field_value = config.tupleof[i];

            static if (isFloatingPointType!(T))
            {
                char[16] float_buf = void;
                cstring float_string = toNoLossString(field_value, float_buf);
                float_buf[float_string.length] = '\0';
                c_values[i] = float_string.ptr;

            }
            else static if (is(T == bool))
            {
                istring bool_string = field_value ? "true" : "false";
                c_values[i] = bool_string.ptr;
            }
            else static if (is(T == SoftmaxOutputNormalization))
            {
                istring norm_string = softmax_normalizations[field_value];
                c_values[i] = norm_string.ptr;
            }
            else
            {
                static assert("Unsupported SoftmaxOutput config type "
                              ~ T.stringof);
            }
        }

        super("SoftmaxOutput", c_keys, c_values);

        SymbolHandle[2] args;
        args[0] = input.handle();
        args[1] = label.handle();

        this.compose(args, null);
    }

    unittest
    {
        scope input = new Variable("y hat");
        scope (exit) input.freeHandle();
        scope label = new Variable("y");
        scope (exit) label.freeHandle();
        scope s = new SoftmaxOutput(input, label);
        scope (exit) s.freeHandle();
        test!("!is")(s.handle(), null);
    }
}

///
unittest
{
    // set up input and label variables
    scope input = new Variable("x");
    scope (exit) input.freeHandle();
    scope label = new Variable("y");
    scope (exit) label.freeHandle();

    // create SoftmaxOutput symbol using default configuration
    scope softmax_default = new SoftmaxOutput(input, label);
    scope (exit) softmax_default.freeHandle();

    // create SoftmaxOutput symbol with custom configuration
    SoftmaxOutput.Config config = {
        grad_scale: 1.5,
        preserve_shape: true
    };
    scope softmax_custom = new SoftmaxOutput(input, label, config);
    scope (exit) softmax_custom.freeHandle();
}


/*******************************************************************************

    A symbol representing linear regression using least squares

    The symbol represents the operations `1/2 * \|input - label\|_2^2` which
    effectively computes the squared error for each input-label pair.

    An LinearRegressionOutput allocates MXNet resources accessed through an
    `MXNetHandle`. Its resources should be freed by calling `freeHandle` when
    done with the LinearRegressionOutput. This should happen in a timely manner
    to avoid resource depletion. Note that scope allocating an object or
    manually calling `delete` won't free the resources since a
    LinearRegressionOutput's destructor does not free the resources. To reclaim
    resources `freeHandle` must be called manually.

*******************************************************************************/

public class LinearRegressionOutput : Symbol
{
    /***************************************************************************

        Constructs a symbol representing the result of performing the linear
        regression on `input` and `label`.

        Params:
            input = input symbol to apply linear regression to
            label = label symbol to apply linear regression to
            grad_scale = scale factor for scaling the gradient; defaults to 1

    ***************************************************************************/

    public this (Symbol input,
                 Symbol label,
                 float grad_scale = 1)
    in
    {
        assert(input !is null);
        assert(label !is null);
    }
    body
    {
        istring key = "grad_scale";
        char[16] value = void;
        auto value_len = toNoLossString(grad_scale, value).length;
        value[value_len] = '\0';

        Immut!(char)*[1] c_keys;
        c_keys[0] = key.ptr;

        Const!(char)*[1] c_values;
        c_values[0] = value.ptr;

        super("LinearRegressionOutput", c_keys, c_values);

        SymbolHandle[2] args;
        args[0] = input.handle();
        args[1] = label.handle();

        this.compose(args, null);
    }

    unittest
    {
        scope input = new Variable("y hat");
        scope (exit) input.freeHandle();
        scope label = new Variable("y");
        scope (exit) label.freeHandle();
        scope lr = new LinearRegressionOutput(input, label);
        scope (exit) lr.freeHandle();
        test!("!is")(lr.handle(), null);
    }
}


/*******************************************************************************

    Dot symbol representing a generalized dot product

    A sum is performed over the last dimension of the first operand and the
    first dimension of the second operand. These dimensions have to be of same
    length as they are collapsed by summing over the pair-wise products. The
    result is an n-dimensional array whose shape is the shape of the input
    n-dimensional arrays concatenated excluding the two dimensions used when
    summing.

    This generalized dot product recovers the (vector) dot product and also the
    matrix matrix product as special cases.

    For example, for two vectors `x` and `y` of same length we sum over the
    products `x[i] * y[i]`. For each vector there is only one dimension, hence
    we end up with a 0-dimensional array, i.e., a scalar. For two matrices `A`
    of shape `[m,n]` and `B` of shape `[n,o]` we sum over the inner dimensions
    of `A` and `B` (which again have to have same length, here, `n`). The
    result is a matrix of shape `[m,o]`.

    More formally given two n-dimensional arrays `x` and `y`. The length of
    the last dimension of `x` must match the length of the first dimension of
    `y` (`x.shape[$ - 1] == y.shape[0]`) as a requirement since values along
    these dimensions are multiplied and summed. The resulting n-dimensional
    array has shape `x.shape[0 .. $ - 1] ~ y.shape[1 .. $]`. The inner
    dimensions are collapsed. The element at `[i_1, ..., i_m, j_1, ..., j_m]`
    of the result is calculated by summing over the products

        `x[i_1, ..., i_m, k] * y[k, j_1, ..., j_n]`

    for `k` in `[0, ..., y.shape[0] - 1]`.

*******************************************************************************/

public class Dot : Symbol
{
    /***************************************************************************

        Constructs a `Dot` symbol

        Params:
            x = symbol whose last dimension's length must match `y` first
                dimension's length
            y = symbol whose first dimension's length must match the length of
                the last dimension of `x`
            transpose_x = transpose `x` (reversing its shape) prior to
                          performing the dot; defaults to false
            transpose_y = transpose `y` (reversing its shape) prior to
                          performing the dot; defaults to false

    ***************************************************************************/

    public this (Symbol x,
                 Symbol y,
                 bool transpose_x = false,
                 bool transpose_y = false)
    in
    {
        assert(x !is null);
        assert(y !is null);
    }
    body
    {
        istring[2] keys;
        keys[0] = "transpose_a";
        keys[1] = "transpose_b";

        Immut!(char)*[2] c_keys;
        c_keys[0] = keys[0].ptr;
        c_keys[1] = keys[1].ptr;

        Immut!(char)*[2] c_values;
        const istring[] true_and_false = ["false", "true"];
        c_values[0] = true_and_false[transpose_x].ptr;
        c_values[1] = true_and_false[transpose_y].ptr;

        super("dot", c_keys, c_values);

        SymbolHandle[2] args;
        args[0] = x.handle();
        args[1] = y.handle();

        this.compose(args, null);
    }

    unittest
    {
        scope x = new Variable("x");
        scope (exit) x.freeHandle();
        scope y = new Variable("y");
        scope (exit) y.freeHandle();
        scope dot = new Dot(x, y);
        scope (exit) dot.freeHandle();
        test!("!is")(dot.handle(), null);
    }
}
