/*******************************************************************************

    Provides functionality to load the images and labels of the MNIST dataset

    The MNIST dataset consists of labeled hand written digits. Each example is
    a hand written digit `0` to `9` that is labeled with `0` to `9`. The hand
    written digits are provided as 28x28 black and white images where a pixel's
    intensity is encoded from 0 (white) to 255 (black).

    The MNIST dataset provides a training and a testing dataset. The training
    dataset consists of 60000 labeled images. The testing dataset consists of
    10000 labeled images.

    The dataset can be obtained from <http://yann.lecun.com/exdb/mnist/>.

    Copyright:
        Copyright (c) 2017 dunnhumby Germany GmbH.

    License:
        Boost Software License Version 1.0.  See accompanying LICENSE.txt for
        details, or <https://www.boost.org/LICENSE_1_0.txt>

*******************************************************************************/

module integrationtest.mxnet.MNIST;

import core.stdc.string;
import core.sys.posix.arpa.inet;

import ocean.core.Enforce;
import ocean.io.compress.ZlibStream;
import ocean.io.device.File;
import ocean.io.Path;
import ocean.math.Math;
import ocean.sys.Environment;
import ocean.transition;


// we use a nested function in this unittest example to ensure that
// the code compiles without actually running it, to avoid the need
// to really load the dataset
///
unittest
{
    void useMNIST ()
    {
        // load the MNIST training dataset
        auto mnist_training = trainingSet(datasetPath([]));
        auto images = mnist_training.images;
        auto labels = mnist_training.labels;

        // dimensions of an image
        auto image_rows = 28;
        auto image_cols = 28;
        assert(labels.length == images.length / (image_rows * image_cols));

        // access the first image and its label
        auto first_training_image = images[0 .. image_rows * image_cols];
        auto first_training_label = labels[0];

        // image is stored row wise with values between 0 (white) and 255 (black)
        auto first_row = first_training_image[0 .. image_rows];
    }
}


/*******************************************************************************

    A Dataset consisting of images with their respective labels

*******************************************************************************/

struct Dataset
{
    /***************************************************************************

        Images representing hand-written digits

        Each image is of size 28 x 28 with its (pixel) values between 0 (white)
        and 255 (black). The pixels are stored row wise and all the images'
        data are flattened into this array. The i-th image corresponds to the
        slice `[i * 28 * 28 .. (i + 1) * 28 * 28]` of this array.

    ***************************************************************************/

    ubyte[] images;


    /***************************************************************************

        Labels representing the numbers from 0 to 9

        Each label indicates the corresponding digit, i.e., the label `0`
        encodes the digit 0 and so on.

        The i-th label corresponds to the i-th element of this array.

    ***************************************************************************/

    ubyte[] labels;
}


/*******************************************************************************

    Returns the MNIST training dataset consisting of 60000 examples

    Params:
        dataset_directory = name of the directory to load the training dataset
                            from

    Returns:
        the MNIST training dataset of 60000 images and labels

*******************************************************************************/

public Dataset trainingSet (cstring dataset_directory)
{
    auto dataset = Dataset(mnistImages(join(dataset_directory, "train-images-idx3-ubyte.gz")),
                           mnistLabels(join(dataset_directory, "train-labels-idx1-ubyte.gz")));
    assert(dataset.images.length == 60 * 1000 * 28 * 28);
    assert(dataset.labels.length == 60 * 1000);
    return dataset;
}


/*******************************************************************************

    Returns the MNIST testing dataset consisting of 10000 examples

    From <http://yann.lecun.com/exdb/mnist/>:

    > The first 5000 examples of the test set are taken from the original NIST
    > training set. The last 5000 are taken from the original NIST test set.
    > The first 5000 are cleaner and easier than the last 5000.

    Params:
        dataset_directory = name of the directory to load the training dataset
                            from

    Returns:
        the MNIST testing dataset of 10000 images and labels

*******************************************************************************/

public Dataset testingSet (cstring dataset_directory)
{
    auto dataset = Dataset(mnistImages(join(dataset_directory, "t10k-images-idx3-ubyte.gz")),
                           mnistLabels(join(dataset_directory, "t10k-labels-idx1-ubyte.gz")));
    assert(dataset.images.length == 10 * 1000 * 28 * 28);
    assert(dataset.labels.length == 10 * 1000);
    return dataset;
}


/*******************************************************************************

    Determines the path of the MNIST dataset

    Params:
        args = command line arguments whose second argument, if it exists, is
               the returned path to the dataset

    Returns:
        args[1], if args.length == 2;
        else `MNIST_DATA_DIR` environment variable, if set;
        else, $HOME/.cache/mnist

*******************************************************************************/

public cstring datasetPath (istring[] args)
{
    if (args.length == 2) return args[1];

    auto env_mnist_data_dir = Environment.get("MNIST_DATA_DIR");
    if (env_mnist_data_dir !is null) return env_mnist_data_dir;

    return join(Environment.get("HOME"), ".cache", "mnist");
}


/*******************************************************************************

    Loads MNIST images from a compressed data file, validating and stripping
    the header data

    The images are returned in an array. Each image is 28x28 `ubyte`s long.

    Params:
        filename = name of the file providing gzipped image data

    Throws:
        if filename cannot be opened or the decompressed file's header is
        ill-formed

    Returns:
        array of MNIST image data where each image is represented by a 28 * 28
        sequence of `ubyte` values (the i-th image corresponds to the slice
        `[i * 28 * 28 .. (i + 1) * 28 * 28]` of this array)

*******************************************************************************/

private ubyte[] mnistImages (cstring filename)
{
    auto decompressed_file_data = decompress(filename);

    ImageFileHeader image_header;
    memcpy(&image_header, decompressed_file_data.ptr, image_header.sizeof);
    // must encode a 3-dimensional array of ubyte
    enforce(ntohl(image_header.magic_number) == 0x00000803);
    // where each image is 28 x 28
    enforce(image_header.numRows() == 28);
    enforce(image_header.numCols() == 28);

    auto data = decompressed_file_data[ImageFileHeader.sizeof .. $];
    assert(image_header.numImages() * image_header.numRows() * image_header.numCols() == data.length);

    return data;
}


/*******************************************************************************

    Header present at the beginning of each decompressed image file

    The opening bytes of the decompressed image file can be copied to this data
    structure in order to read the file's header data.

*******************************************************************************/

private struct ImageFileHeader
{
    /***************************************************************************

        32-bit integer magic number encoded in big endian

        The first two bytes are always 0.
        The third byte encodes the type of the data:

            - 0x08 unsigned byte
            - 0x09 signed byte
            - 0x0B short (2 bytes)
            - 0x0C int (4 bytes)
            - 0x0D float (4 bytes)
            - 0x0E double (8 bytes)

        The fourth bytes encodes the number of dimensions:

            - 0x01 vector
            - 0x02 matrix
            - 0x03 3rd order tensor
            - ...

    ***************************************************************************/

    private int magic_number;


    /***************************************************************************

        32-bit integer number of images encoded in big endian

    ***************************************************************************/

    private int num_images;


    /***************************************************************************

        32-bit integer number of rows of an image encoded in big endian

    ***************************************************************************/

    private int num_rows;


    /***************************************************************************

        32-bit integer number of cols of an image encoded in big endian

    ***************************************************************************/

    private int num_cols;


    /***************************************************************************

        Returns:
            the number of images according to this image file header

    ***************************************************************************/

    public int numImages ()
    {
        return ntohl((&this).num_images);
    }


    /***************************************************************************

        Returns:
            the number of rows of an image according to this image file
            header

    ***************************************************************************/

    public int numRows ()
    {
        return ntohl((&this).num_rows);
    }


    /***************************************************************************

        Returns:
            the number of cols of an image according to this image file
            header

    ***************************************************************************/

    public int numCols ()
    {
        return ntohl((&this).num_cols);
    }
}


/*******************************************************************************

    Loads MNIST labels from a compressed data file, validating and stripping
    the header data

    The labels are returned in an array. Each element represent a label between
    `0` and `9`, indicating the digit of the corresponding hand written digit
    in the image data.

    Params:
        filename = name of the file providing gzipped label data

    Throws:
        if filename cannot be opened or the decompressed file's header is
        ill-formed

    Returns:
        array of MNIST label data where each label is represented by `ubyte`
        value between 0 and 9 (the i-th label corresponds to the i-th element
        of this array)

*******************************************************************************/

private ubyte[] mnistLabels (cstring filename)
{
    auto decompressed_file_data = decompress(filename);

    LabelFileHeader label_header;
    memcpy(&label_header, decompressed_file_data.ptr, label_header.sizeof);
    // must encode a 1-dimensional array of ubyte
    enforce(ntohl(label_header.magic_number) == 0x00000801);

    auto data = decompressed_file_data[LabelFileHeader.sizeof .. $];
    enforce(label_header.numLabels() == data.length);

    return data;
}


/*******************************************************************************

    Header present at the beginning of each decompressed label file

    The opening bytes of the decompressed label file can be copied to this data
    structure in order to read the file's header data.

*******************************************************************************/

private struct LabelFileHeader
{
    /***************************************************************************

        32-bit integer magic number encoded in big endian

    ***************************************************************************/

    private int magic_number;


    /***************************************************************************

        32-bit integer number of labels encoded in big endian

    ***************************************************************************/

    private int num_labels;


    /***************************************************************************

        Returns:
            the number of labels according to this label file header

    ***************************************************************************/

    public int numLabels ()
    {
        return ntohl((&this).num_labels);
    }
}


/*******************************************************************************

    Loads a gzipped file from disk and decompresses its contents into an array

    Params:
        filename = name of the gzipped file

    Returns:
        the decompressed contents of file with given filename

*******************************************************************************/

private ubyte[] decompress (cstring filename)
{
    void[] buffer;
    File.get(filename, buffer);
    ubyte[] data = cast(ubyte[]) buffer;

    auto decompress = new ZlibStreamDecompressor();
    ubyte[] decompressed_data;
    decompress.start(ZlibStreamDecompressor.Encoding.Gzip);
    while (data.length)
    {
        auto offset = min(1024, data.length);
        decompress.decodeChunk(data[0 .. offset],
                               (ubyte[] decompressed_chunk)
                               {
                                   decompressed_data ~= decompressed_chunk;
                               });
        data = data[offset .. $];
    }
    decompress.end();

    return decompressed_data;
}
