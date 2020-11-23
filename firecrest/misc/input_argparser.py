from argparse import ArgumentParser
import os.path


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, "r")  # return an open file handle


parser = ArgumentParser(description="JSON configuration file")
parser.add_argument(
    "-i",
    "--input",
    dest="filename",
    required=True,
    help="input JSON file with printhead waveform run configuration",
    metavar="FILE",
    type=lambda x: is_valid_file(parser, x),
)
