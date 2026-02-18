import sys
import argparse
from .cmd import sndls
from ..utils.fmt import (
    printc_exit as print_exit,
    exit_warning
)
from sndls import (
    __description__,
    __version_repr__
)


def get_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__description__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False
    )
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default=".",
        help="input audio file, .csv file or folder containing audio files"
    )
    parser.add_argument(
    "-e", "--extension",
        type=str,
        nargs="+",
        default=[".wav"],
        help="audio file extension(s)"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="search input folder recursively"
    )
    parser.add_argument(
        "-d", "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="data type used to read audio files and calculate stats"
    )
    parser.add_argument(
        "-m", "--meta",
        action="store_true",
        help="show metadata information only"
    )
    parser_filter_select = parser.add_mutually_exclusive_group()
    parser_filter_select.add_argument(
        "-f", "--filter",
        type=str,
        help="filter files meeting a certain condition"
    )
    parser_filter_select.add_argument(
        "-s", "--select",
        type=str,
        help="select files meeting a certain condition"
    )
    parser_hash = parser.add_mutually_exclusive_group()
    parser_hash.add_argument(
        "--sha256",
        action="store_true",
        help="compute sha256 hash"
    )
    parser_hash.add_argument(
        "--sha256-short",
        action="store_true",
        help="compute sha256 hash and print only last 8 characters"
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="save output to a .csv file"
    )
    parser.add_argument(
        "--sample",
        type=float,
        help=(
            "number of files to randomly sample from all input files. If the "
             "value is between 0.0 and 1.0, it will be interpreted as a "
             "percentage"
        )
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print summary only"
    )
    parser.add_argument(
        "--silent-thresh",
        type=float,
        default=-80.0,
        help="root mean square (RMS) threshold in decibels below which a file "
             "is considered silent"
    )
    parser.add_argument(
        "--silent-frame-size-ms",
        type=float,
        help=(
            "if set, the root mean square (RMS) level for determining file "
            "silence is computed per frame, with frame size given in "
            "milliseconds"
        )
    )
    parser.add_argument(
        "--silent-frame-mode",
        choices=["all", "any", "mean", "median", "max"],
        default="any",
        help=(
            "method to flag silence from frame root mean square (RMS) level: "
            "'any', 'all', 'mean', 'median', or 'max'"
        )
    )
    parser.add_argument(
        "--silent-hop-size",
        type=float,
        default=0.5,
        help=(
            "percentage of --silent-frame-size-ms used as hop size for "
            "framewise calculations (only used if --silent-frame-size-ms is "
            "enabled)"
        )
    )
    parser.add_argument(
        "--fft-size",
        type=int,
        choices=[2048, 1024, 512],
        default=2048,
        help="fft size for spectral analysis"
    )
    parser.add_argument(
        "--hop-size",
        type=int,
        choices=[512, 256, 128],
        default=512,
        help="hop size used for spectral analysis"
    )
    parser.add_argument(
        "--spectral-rolloff",
        type=float,
        help=(
            "computes the mean spectral rolloff in hertz for a given "
            "cumulative energy percentage between 0.0 and 1.0"
        )
    )
    parser.add_argument(
        "--spectral-rolloff-detail",
        action="store_true",
        help="shows spectral rollof in min ≤ mean ≤ max format"
    )
    parser.add_argument(
        "-p", "--post-action",
        choices=["cp", "mv", "rm", "mv+sp", "cp+sp", "dump", "dump+sp"],
        help=(
            "action to execute after listing all files (cp=copy, mv=move, "
            "rm=remove, sp=split, dump=dump to file)"
        )
    )
    parser.add_argument(
        "--post-action-output",
        type=str,
        help=(
            "post action output (required for --post-action "
            "{cp,mv,cp+sp,mv+sp, dump, dump+sp})"
        )
    )
    parser.add_argument(
        "--post-action-preserve-subfolders",
        action="store_true",
        help=(
            "preserve subfolder structure with respect to --input when copying"
            " or moving files using --post-action"
        )
    )
    parser.add_argument(
        "--post-action-num-splits",
        type=int,
        help="number of partitions (required for --post-action {cp+sp,mv+sp}, "
             "optional for --post-action {dump})"
    )
    parser.add_argument(
        "--post-action-split-dirname",
        type=str,
        default="split_",
        help="split folder name (only valid if --post-action {cp+sp,mv+sp})"
    )
    parser.add_argument(
        "--max-fname-chars",
        type=int,
        default=40,
        help="maximum filename characters to display"
    )
    parser.add_argument(
        "--csv-input-file-col",
        type=str,
        default="file",
        help="name of the column containing files if the input is a .csv file"
    )
    parser.add_argument(
        "--csv-ignore-errors",
        action="store_true",
        help="skip rows with errors when reading .csv files"
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=60 * 60 * 3,
        help="skip reading audio files larger than this duration in seconds"
    )
    parser.add_argument(
        "--skip-invalid-files",
        action="store_true",
        help="skip and report files that cannot be parsed"
    )
    parser.add_argument(
        "--preload",
        type=str,
        help="preload .csv, .tsv or .txt file to use as input for "
             "--filter/--select"
    )
    parser.add_argument(
        "--preload-truncate-ragged-lines",
        action="store_true",
        help="truncate ragged lines when found in preloaded files"
    )
    parser.add_argument(
        "--preload-has-header",
        action="store_true",
        help="if enabled, the first row of --preload will be ignored"
    )
    parser.add_argument(
        "--csv-overwrite",
        action="store_true",
        help="overwrites .csv file if --csv is enabled and file already exists"
    )
    parser.add_argument(
        "-u", "--unattended",
        action="store_true",
        help="perform --post-action without user confirmation"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1234,
        help="random seed used if --sample is enabled"
    )

    return parser


def main() -> None:
    # Print tool version
    if (len(sys.argv) == 2 and sys.argv[1] in ("-v", "--version")):
        print_exit(__version_repr__, code=0)
    
    # NOTE: If no input is given, then current directory is assumed
    parser = get_parser()
    args = parser.parse_args()

    try:
        sndls(args)
    
    except KeyboardInterrupt:
        exit_warning("Process terminated by the user")
