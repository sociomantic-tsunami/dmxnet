/******************************************************************************

    Defines common utility functions

    Copyright:
        Copyright (c) 2017 dunnhumby Germany GmbH.

    License:
        Boost Software License Version 1.0.  See accompanying LICENSE.txt for
        details, or <https://www.boost.org/LICENSE_1_0.txt>

******************************************************************************/

module mxnet.Util;

import ocean.core.Enforce;
import ocean.meta.types.Qualifiers;
import ocean.text.convert.Formatter;

version (unittest)
{
    import ocean.core.Test;
    import Float = ocean.text.convert.Float;
}

/*******************************************************************************

    Returns a string representation of a floating point value with no loss of
    information

    The returned string is of the format `[-]d.dddddddde<+/->dd`, where d
    denotes a decimal digit.

    Any float value is converted such that it can be converted back to the
    original input value, i.e., `parse(toNoLossString(x)) == x` for all values.
    Hence, no information is lost when using this conversion to a string.

    Params:
        x = floating point value to convert
        buf = buffer to store the converted float; must have length at least 15

    Returns:
        slice of `buf` containing the converted string

    Throws:
        if `buf` has less than 15 elements

*******************************************************************************/

public mstring toNoLossString (in float x, mstring buf)
{
    enum max_length = 1 + // optional minus sign
                      1 + // leading decimal digit
                      1 + // .
                      8 + // 8 decimal digits
                      1 + // e
                      1 + // sign
                      2;  // decimal exponent
    enum format_string = "{:e8}";

    enforce(buf.length >= max_length);
    return snformat(buf, format_string, x);
}

unittest
{
    char[15] buf = void;
    test!("==")(toNoLossString(0f, buf), "0.00000000e+00");
    test!("==")(toNoLossString(-0f, buf), "-0.00000000e+00");
    test!("==")(toNoLossString(0.5f, buf), "5.00000000e-01");
    test!("==")(toNoLossString(-1f, buf), "-1.00000000e+00");

    // special values
    test!("==")(toNoLossString(float.min_normal, buf), "1.17549435e-38");
    test!("==")(toNoLossString(float.infinity, buf), "inf");
    test!("==")(toNoLossString(-float.infinity, buf), "-inf");
    test!("==")(toNoLossString(float.nan, buf), "nan");
    test!("==")(toNoLossString(-float.nan, buf), "-nan");

    // conversion to string and backward with no loss of information
    foreach (x; [0f, 1f, float.min_normal, float.infinity])
    {
        float x_parsed = Float.parse(toNoLossString(x, buf));
        test!("==")(x_parsed, x);
        float minus_x_parsed = Float.parse(toNoLossString(-x, buf));
        test!("==")(minus_x_parsed, -x);
    }
}
