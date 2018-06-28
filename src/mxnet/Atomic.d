/*******************************************************************************

    Helper functionality for working with MXNet atomic symbols

    Atomic symbols are symbol types predefined by the MXNet library.  In the
    underlying C API, individual atomic symbol instances are generated using
    the corresponding `AtomicSymbolCreator`.  This module provides functions
    to allow the rest of the wrapper code to instantiate atomic symbols, but
    without exposing `AtomicSymbolCreator` instances directly to the user of
    the wrapper.

    Atomic symbols can be broadly divided into two groups: those used in
    symbolic programming (exposed by the `mxnet.Symbol` module), and those
    used in imperative programming (exposed in the `mxnet.NDArray module).
    The present module defines the lowest-level common internal functions
    needed in order to define those higher-level public APIs.

    As a convenience, besides the `package`-level functions defined for
    internal use only, this module also exposes some public help queries
    that can be used to request information on the atomic symbols currently
    defined by MXNet.

    Copyright:
        Copyright (c) 2017 dunnhumby Germany GmbH.

    License:
        Boost Software License Version 1.0.  See accompanying LICENSE.txt for
        details, or <https://www.boost.org/LICENSE_1_0.txt>

*******************************************************************************/

module mxnet.Atomic;

import mxnet.c.c_api;
import mxnet.API;

import ocean.text.convert.Formatter;
import ocean.text.util.StringC;
import ocean.transition;

version (UnitTest)
{
    import ocean.core.Test;
}


/*******************************************************************************

    Associative array mapping the name of each type of atomic symbol to the
    `AtomicSymbolCreator` instance needed to create it.  To access the results,
    use the `atomicSymbolCreator` function rather than accessing `atomic_cache`
    directly.

*******************************************************************************/

private AtomicSymbolCreator[istring] atomic_cache;

static this ()
{
    foreach (creator; atomicSymbolCreatorList())
    {
        atomic_cache[idup(nameOf(creator))] = creator;
    }
}


/*******************************************************************************

    Get the `AtomicSymbolCreator` instance needed to create a particular
    predefined (atomic) symbol type

    This function is intended for internal use only, and is expected to
    only ever be called from `mxnet.NDArray` and `mxnet.Symbol`.

    Params:
        name = name of the atomic symbol creator you wish to access

    Returns:
        `AtomicSymbolCreator` corresponding to the provided name

*******************************************************************************/

package AtomicSymbolCreator atomicSymbolCreator (cstring name)
out (returned_creator)
{
    assert(returned_creator !is null);
}
body
{
    if (auto creator = name in atomic_cache)
    {
        assert(nameOf(*creator) == name);
        return *creator;
    }

    assert(false, "Unknown AtomicSymbolCreator: " ~ name);
}

unittest
{
    test!("!is")(atomicSymbolCreator("_set_value"), null);
    test!("!is")(atomicSymbolCreator("softmax"), null);
    test!("!is")(atomicSymbolCreator("Dropout"), null);
    test!("!is")(atomicSymbolCreator("RNN"), null);
}


/*******************************************************************************

    Get all defined `AtomicSymbolCreator` instances

    Returns:
        array of all `AtomicSymbolCreator` instances defined by MXNet

*******************************************************************************/

private AtomicSymbolCreator[] atomicSymbolCreatorList ()
{
    uint len;
    AtomicSymbolCreator* ptr;
    invoke!(MXSymbolListAtomicSymbolCreators)(&len, &ptr);
    return ptr[0 .. len];
}


/*******************************************************************************

    Get the name of an `AtomicSymbolCreator` instance

    Params:
        creator = atomic symbol creator whose name to get

    Returns:
        name of the provided atomic symbol creator

*******************************************************************************/

private cstring nameOf (AtomicSymbolCreator creator)
in
{
    assert(creator !is null);
}
body
{
    char* name;
    invoke!(MXSymbolGetAtomicSymbolName)(creator, &name);
    return StringC.toDString(name);
}


/*******************************************************************************

    Provides detailed information on a single named atomic symbol type

    Params:
        name = name of the type of atomic symbol

    Returns:
        data structure that can be lazily formatted into a string containing
        information on the atomic symbol

    Example:
    --------
    Stdout(atomicSymbolInfo("SoftmaxOutput"));
    --------

*******************************************************************************/

public AtomicSymbolInfo atomicSymbolInfo (cstring name)
{
    return AtomicSymbolInfo(atomicSymbolCreator(name));
}


/*******************************************************************************

    Provides information on all available atomic symbol types

    Params:
        detailed_info = if false (the default), only symbol names will
                        be provided; if true, detailed information will
                        be provided for each atomic symbol

    Returns:
        data structure that can be lazily formatted into a string containing
        information on every atomic symbol type defined by MXNet

    Example:
    --------
    Stdout(atomicSymbolList());     // symbol names only
    Stdout(atomicSymbolList(true)); // detailed info on every symbol
    --------

*******************************************************************************/

public AtomicSymbolList atomicSymbolList (bool detailed_info = false)
{
    return AtomicSymbolList(detailed_info);
}


/*******************************************************************************

    Helper struct to obtain detailed info on an individual atomic symbol type

*******************************************************************************/

private struct AtomicSymbolInfo
{
    /***************************************************************************

        Creator of the atomic symbol whose information to fetch

    ***************************************************************************/

    private AtomicSymbolCreator creator;


    /***************************************************************************

        Helper method to generate a string representation of information on
        an atomic symbol

        Returns:
            formatted string containing information on the type of atomic
            symbol created by `this.creator`

    ***************************************************************************/

    public cstring toString ()
    {
        return format("{}", *this);
    }


    /***************************************************************************

        Helper method to allow `ocean.text.convert.Formatter` functionality
        to generate a string representation of atomic symbol information

        Params:
            sink = delegate that can be called by the formatting process to
                   handle portions of the resulting string

    ***************************************************************************/

    public void toString (FormatterSink sink)
    in
    {
        assert(this.creator !is null);
    }
    body
    {
        Const!(char)* name;
        Const!(char)* description;

        uint num_args;
        Const!(char)** arg_names;
        Const!(char)** arg_type_infos;
        Const!(char)** arg_descriptions;

        Const!(char)* key_var_num_args;
        Const!(char)* return_type;

        invoke!(MXSymbolGetAtomicSymbolInfo)
               (this.creator, &name, &description,
                &num_args, &arg_names, &arg_type_infos, &arg_descriptions,
                &key_var_num_args, &return_type);

        sformat(sink, "{}\n\n", StringC.toDString(name));

        auto desc = StringC.toDString(description);
        if (desc.length) sformat(sink, "{}\n\n", desc);

        if (num_args > 0)
        {
            sformat(sink, "{} parameters:\n", num_args);
            foreach (i, arg_name_c; arg_names[0 .. num_args])
            {
                auto arg_name = StringC.toDString(arg_name_c);
                auto arg_type = StringC.toDString(arg_type_infos[i]);
                auto arg_desc = StringC.toDString(arg_descriptions[i]);
                sformat(sink, "  * {} : {}\n", arg_name, arg_type);
                sformat(sink, "      - {}\n", arg_desc);
            }
            sformat(sink, "\n");
        }

        auto varargs_key = StringC.toDString(key_var_num_args);
        if (varargs_key.length)
        {
            sformat(sink, "Variable arguments key: '{}'\n\n", varargs_key);
        }

        if (return_type !is null)
        {
            sformat(sink, "Returns: {}", StringC.toDString(return_type));
        }
    }
}


/*******************************************************************************

    Helper struct to obtain information on all atomic symbol types currently
    defined by MXNet

*******************************************************************************/

private struct AtomicSymbolList
{
    /***************************************************************************

        If true, detailed information will be provided on every atomic symbol;
        if false, only the name of each symbol will be output

    ***************************************************************************/

    private bool detailed_info;


    /***************************************************************************

        Returns:
            formatted string containing information on all the available
            atomic symbol types

    ***************************************************************************/

    public cstring toString ()
    {
        return format("{}", *this);
    }


    /***************************************************************************

        Helper method to allow `ocean.text.convert.Formatter` functionality
        to generate a string representation of information on all the atomic
        symbol types defined by MXNet

        Params:
            sink = delegate that can be called by the formatting process to
                   handle portions of the resulting string

    ***************************************************************************/

    public void toString (FormatterSink sink)
    {
        foreach (creator; atomicSymbolCreatorList())
        {
            assert(creator !is null);

            if (this.detailed_info)
            {
                sformat(sink, "----\n");
                sformat(sink, "{}", AtomicSymbolInfo(creator));
            }
            else
            {
                sformat(sink, "{}\n", nameOf(creator));
            }
        }
    }
}
