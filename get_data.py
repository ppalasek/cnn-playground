from __future__ import print_function
import os
import sys
import shutil
import requests
import hashlib
import gzip
import tarfile


def check_mnist():
    """
    Check ...
    """
    output_dir = './data/mnist/'
    output_files = {'train-images-idx3-ubyte': False,
                    'train-labels-idx1-ubyte': False,
                    't10k-images-idx3-ubyte': False,
                    't10k-labels-idx1-ubyte': False}
    # Check if files exist
    for file in output_files.iteritems():
        output_file = "%s%s" % (output_dir, file[0])
        if os.path.isfile(output_file):
            output_files[file[0]] = True
    # Return True if all files exist
    return all(item[1] == True for item in output_files.iteritems())


def get_mnist():
    """

    """
    remove_gz_file = True
    output_dir = './data/mnist/'
    mnist_files_md5sums = {'train-images-idx3-ubyte.gz': 'f68b3c2dcbeaaa9fbdd348bbdeb94873',
                           'train-labels-idx1-ubyte.gz': 'd53e105ee54ea40749a09fcbcd1e9432',
                           't10k-images-idx3-ubyte.gz': '9fb629c4189551a2d022fa330f9573f3',
                           't10k-labels-idx1-ubyte.gz': 'ec29112dd5afa0611ce80d1b7f02629c'}

    for file, file_md5sum in mnist_files_md5sums.iteritems():

        # Download file
        print(" -- Download '%s'..." % file, end="")
        sys.stdout.flush()
        url = 'http://yann.lecun.com/exdb/mnist/%s' % file
        response = requests.get(url, stream=True)
        output_file_gz = "%s%s" % (output_dir, file)
        output_file = output_file_gz.replace(".gz", "")
        with open(output_file_gz, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response
        print("Done!")
        # Check md5sum
        if hashlib.md5(open(output_file_gz, 'rb').read()).hexdigest() == file_md5sum:
            print("    -- Check md5sum: PASS")
        else:
            print("    -- Check md5sum: FAIL")
            md5sum_error = "Check of md5sum failed for file '%s'" % file
            raise ValueError(md5sum_error)
        # Extract gunzip file
        print("    -- Extract .gz file...", end="")
        sys.stdout.flush()
        inF = gzip.open(output_file_gz, 'rb')
        outF = open(output_file, 'wb')
        outF.write(inF.read())
        inF.close()
        outF.close()
        print("Done!")
        if remove_gz_file:
            print("    -- Delete .gz file...", end="")
            os.remove(output_file_gz)
            print("Done!")


def check_cifar10():
    """

    """
    output_dir = './data/cifar10/'
    output_files = {'batches.meta': False,
                    'data_batch_1': False,
                    'data_batch_2': False,
                    'data_batch_3': False,
                    'data_batch_4': False,
                    'data_batch_5': False,
                    'readme.html': False,
                    'test_batch': False}

    # Check if files exist
    for file in output_files.iteritems():
        output_file = "%s%s" % (output_dir, file[0])
        if os.path.isfile(output_file):
            output_files[file[0]] = True
    # Return True if all files exist
    return all(item[1] == True for item in output_files.iteritems())


def get_cifar10():
    """

    """
    remove_targz_file = False
    output_dir = './data/cifar10/'
    file = 'cifar-10-python.tar.gz'
    # Download file
    print(" -- Download '%s'..." % file, end="")
    sys.stdout.flush()
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    response = requests.get(url, stream=True)
    output_file_gz = "%s%s" % (output_dir, file)
    with open(output_file_gz, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response
    print("Done!")
    # Check md5sum
    if hashlib.md5(open(output_file_gz, 'rb').read()).hexdigest() == "c58f30108f718f92721af3b95e74349a":
        print("    -- Check md5sum: PASS")
    else:
        print("    -- Check md5sum: FAIL")
        md5sum_error = "Check of md5sum failed for file '%s'" % file
        raise ValueError(md5sum_error)
    # Extract gunzip file
    print("    -- Extract .tar.gz file...", end="")
    sys.stdout.flush()
    tar = tarfile.open(output_file_gz, "r:gz")
    tar.extractall(path=output_dir)
    tar.close()
    print("Done!")
    if remove_targz_file:
        print("    -- Delete .tar.gz file...", end="")
        os.remove(output_file_gz)
        print("Done!")
    # Move files in `output_dir` and delete `output_dir`/cifar-10-batches-py/
    print("    -- Move dataset files and delete temp ones...", end="")
    sys.stdout.flush()
    src = "%scifar-10-batches-py/" % output_dir
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, output_dir)
    shutil.rmtree(src)
    print("Done!")


def check_cifar100():
    """

    """
    output_dir = './data/cifar100/'
    output_files = {'meta': False,
                    'train': False,
                    'test': False}

    # Check if files exist
    for file in output_files.iteritems():
        output_file = "%s%s" % (output_dir, file[0])
        if os.path.isfile(output_file):
            output_files[file[0]] = True
    # Return True if all files exist
    return all(item[1] == True for item in output_files.iteritems())


def get_cifar100():
    """

    """
    remove_targz_file = False
    output_dir = './data/cifar100/'
    file = 'cifar-100-python.tar.gz'
    # Download file
    print(" -- Download '%s'..." % file, end="")
    sys.stdout.flush()
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    response = requests.get(url, stream=True)
    output_file_gz = "%s%s" % (output_dir, file)
    with open(output_file_gz, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response
    print("Done!")
    # Check md5sum
    if hashlib.md5(open(output_file_gz, 'rb').read()).hexdigest() == "eb9058c3a382ffc7106e4002c42a8d85":
        print("    -- Check md5sum: PASS")
    else:
        print("    -- Check md5sum: FAIL")
        md5sum_error = "Check of md5sum failed for file '%s'" % file
        raise ValueError(md5sum_error)
    # Extract gunzip file
    print("    -- Extract .tar.gz file...", end="")
    sys.stdout.flush()
    tar = tarfile.open(output_file_gz, "r:gz")
    tar.extractall(path=output_dir)
    tar.close()
    print("Done!")
    if remove_targz_file:
        print("    -- Delete .tar.gz file...", end="")
        os.remove(output_file_gz)
        print("Done!")
    # Move files in `output_dir` and delete `output_dir`/cifar-100-python/
    print("    -- Move dataset files and delete temp ones...", end="")
    sys.stdout.flush()
    src = "%scifar-100-python/" % output_dir
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, output_dir)
    shutil.rmtree(src)
    print("Done!")


def check_svhn():
    """

    """
    output_dir = './data/svhn/'
    output_files = {'train_32x32.mat': False,
                    'test_32x32.mat': False}

    # Check if files exist
    for file in output_files.iteritems():
        output_file = "%s%s" % (output_dir, file[0])
        if os.path.isfile(output_file):
            output_files[file[0]] = True
    # Return True if all files exist
    return all(item[1] == True for item in output_files.iteritems())


def get_svhn():
    """

    """
    output_dir = './data/svhn/'
    svhn_files_md5sums = {'train_32x32.mat': 'e26dedcc434d2e4c54c9b2d4a06d8373',
                          'test_32x32.mat': 'eb5a983be6a315427106f1b164d9cef3'}

    for file, file_md5sum in svhn_files_md5sums.iteritems():
        # Download file
        print(" -- Download '%s'..." % file, end="")
        sys.stdout.flush()
        url = 'http://ufldl.stanford.edu/housenumbers/%s' % file
        response = requests.get(url, stream=True)
        output_file = "%s%s" % (output_dir, file)
        with open(output_file, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response
        print("Done!")
        # Check md5sum
        if hashlib.md5(open(output_file, 'rb').read()).hexdigest() == file_md5sum:
            print("    -- Check md5sum: PASS")
        else:
            print("    -- Check md5sum: FAIL")
            md5sum_error = "Check of md5sum failed for file '%s'" % file
            raise ValueError(md5sum_error)


def main():
    """

    """
    # Create './data' dir, if does not exist
    if not os.path.exists("./data"):
        os.makedirs("./data")

    datasets = {1: 'MNIST',
                2: 'CIFAR-10',
                3: 'CIFAR-100',
                4: 'SVHN'}

    print("Choose one or more dataset(s) to be downloaded:")
    print("(Separate your choices with spaces; e.g., 1 3 4)")
    print("")
    print(" -- MNIST     : 1")
    print(" -- CIFAR-10  : 2")
    print(" -- CIFAR-100 : 3")
    print(" -- SVHN      : 4")

    # Get user's choices
    while True:
        users_choices = [int(x) for x in raw_input().split()]
        if not(set(users_choices) <= set([1, 2, 3, 4])) or \
           len(users_choices) == 0 or \
           len(users_choices) > 4:
            print("Please choose one or more from {1,2,3,4}:")
            continue
        else:
            break

    # Remove the last printed line in stdout
    print('\x1b[1A' + '\x1b[2K')

    print("You are going to download the following datasets:")
    for dataset in datasets.iteritems():
        if dataset[0] in users_choices:
            print(" - %s" % dataset[1])
    print("")

    for choice in users_choices:
        if choice == 1:
            # Get the MNIST dataset
            if not(check_mnist()):
                print("Downloading MNIST...")
                get_mnist()
            else:
                print(" # MNIST dataset is already stored in ./data/mnist/")
        if choice == 2:
            # Get the CIFAR-10 dataset
            if not(check_cifar10()):
                get_cifar10()
            else:
                print(" # CIFAR-10 dataset is already stored in ./data/cifar10/")
        if choice == 3:
            # Get the CIFAR-100 dataset
            if not(check_cifar100()):
                get_cifar100()
            else:
                print(" # CIFAR-100 dataset is already stored in ./data/cifar100/")
        if choice == 4:
            # Get the SVHN dataset
            if not(check_svhn()):
                get_svhn()
            else:
                print(" # SVHN dataset is already downloaded and ./data/cifar100/")


if __name__ == "__main__":
    main()
