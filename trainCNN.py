import argparse


def main():
    """

    """
    # Set up a parser for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    parser.add_argument("-a", "--architecture", type=str, choices=['CCFF'], default="ccff", help="choose CNN architecture")
    parser.add_argument("-e", "--num_epochs", type=int, default=10, help="set number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=256, help="set batch size")
    parser.add_argument("-l", '--loss', type=str, choices=['cross_entropy', 'hinge'], default="cross_entropy", help="choose loss (objective) function")
    args = parser.parse_args()

    # Access command line arguments as follows:
    print(args.verbose)
    print(args.architecture)
    print(args.num_epochs)
    print(args.batch_size)
    print(args.loss)


if __name__ == "__main__":
    main()