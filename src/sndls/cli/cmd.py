import os
import csv
import random
import shutil
import numpy as np
import polars as pl
from copy import deepcopy
from time import perf_counter
from decimal import Decimal
from numbers import Number
from argparse import Namespace
from tqdm import tqdm
from typing import List
from ..utils.config import (
    get_allowed_audio_file_extensions,
    get_sppbar_color
)
from ..utils.io import (
    ask_confirmation,
    get_dir_files,
    read_audio,
    read_audio_metadata
)
from ..utils.collections import flatten_nested_list
from ..utils.fmt import (
    bytes_to_str,
    exit_error,
    exit_warning,
    printc as print,
    print_error,
    print_warning,
    time_to_str
)
from ..utils.guards import is_file_with_ext
from ..utils.hash import generate_sha256_from_file
from ..utils.audio import (
    ms_to_samples,
    is_anomalous,
    is_clipped,
    is_silent,
    peak_db,
    rms_db,
    spectral_rolloff
)


def _matches_filter(
        data: dict,
        preload: pl.DataFrame,
        expr: str
) -> bool:
    """Matches a filter expression against a set of file specifications.
    
    Args:
        data (dict): Audio file specifications.
        preload (pl.DataFrame): Preloaded data.
        expr (str): Filter expression.
    
    Returns:
        bool: `True` of the filter matches the contents of `data`, `False`
            otherwise.
    """
    try:
        # Set constrained globals and locals
        if preload is not None:
            data["preload"] = preload

        result = eval(expr, {}, deepcopy(data))

        if not isinstance(result, bool):
            raise ValueError("Invalid return type")
    
    except Exception as e:
        fields_repr = ", ".join(k for k in data)

        exit_error(
            f"Invalid --filter/--select expression '{expr}': {e}.\n"
            "Only python expressions returning a bool value are valid. To "
            "create a filter expression you can access any of the following "
            f"fields of each file: {fields_repr}. "
            "Please look at the repository README.md for further details",
            writer=tqdm
        )
    
    return result


def _preload_file(
        file: str,
        has_header: bool = False,
        truncate_ragged_lines: bool = False,
        ignore_errors: bool = False
) -> pl.DataFrame:
    """Preloads a file in memory to be used with --filter/--select option.
    
    Args:
        file (str): File to be loaded in memory as a `pl.DataFrame`.
        has_header (bool): If `True`, the first column of the preloaded file
            is assumed to be a header and will be ignored.
    """
    if not os.path.isfile(file):
        exit_error(f"--preload file '{file}' not found")
    
    elif not is_file_with_ext(file, ext=[".csv", ".tsv", ".txt"]):
        exit_error("--preload option only supports .csv, .tsv or .txt files")
    
    elif is_file_with_ext(file, ext=".csv"):
        separator = ","
    
    elif is_file_with_ext(file, ext=".txt"):
        separator = " "
    
    elif is_file_with_ext(file, ext=".tsv"):
        separator = "\t"
    
    else:
        raise AssertionError
    
    try:
        preload = pl.read_csv(
            file,
            separator=separator,
            has_header=has_header,
            truncate_ragged_lines=truncate_ragged_lines,
            ignore_errors=ignore_errors
        )
        
    except pl.exceptions.ComputeError as e:
        exit_error(
            f"The following error occurred while opening '{file}': {e}\n\n"
            "If the issue was caused by 'truncate_ragged_lines', please "
            "consider using --preload-truncate-ragged-lines"
        )
    
    return preload


def _audio_file_meta_repr_from_dict(data: dict, max_fname_chars: int) -> str:
    """Creates a printable string representation of a set of audio file
    specifications.

    !!! note
        Differently from `_audio_file_repr_from_dict`, the specifications
        in `data` of this method contain exclusively metadata information.
    
    Args:
        data (dict): Audio data.
        max_fname_chars (int): Maximum name of characters from the file path
            to be printed to the terminal.
        
    Returns:
        str: `str` representation of the audio file specifications.
    """
    # Get filename repr
    filename_repr = (
        f"...{data['file'][-max_fname_chars:]}"
        if len(data["file"]) > max_fname_chars + 3 else data["file"]
    ).ljust(max_fname_chars + 3)

    # Get length repr
    num_samples_repr = str(data["num_samples_per_channel"])

    if len(num_samples_repr) > 10:
        f"{Decimal(num_samples_repr):.5e}"
    
    if data["is_invalid"]:
        len_repr = "-".rjust(20)

    else:
        len_repr = (
            num_samples_repr
            + "x" + str(data["num_channels"])
            + "@" + str(data["fs"]) + "hz"
        ).rjust(20)

    # Get mem repr
    mem_repr = bytes_to_str(data["size_bytes"]).rjust(7)

    # Get format repr
    data_fmt = "-" if data["fmt"] is None else data["fmt"]
    data_subtype = "-" if data["subtype"] is None else data["subtype"]

    fmt_repr = data_fmt.ljust(6) + " " + data_subtype.ljust(14)

    # Assemble representation
    repr = f"{filename_repr} {mem_repr} {fmt_repr} {len_repr}"

    if data["is_invalid"]:
        repr = f"<error>{repr}</error>"

    return repr


def _audio_file_repr_from_dict(
        data: dict,
        max_fname_chars: int,
        abbrev_hash: bool
) -> str:
    """Creates a printable string representation of a set of audio file
    specifications.
    
    !!! note
        Differently from `_audio_file_meta_repr_from_dict`, the specifications
        in `data` contain metadata information and audio based statistics.
    
    Args:
        data (dict): Audio data.
        max_fname_chars (int): Maximum name of characters from the file path
            to be printed to the terminal.
        
    Returns:
        str: `str` representation of the audio file specifications.
    """
    # Get filename repr
    filename_repr = (
        f"...{data['file'][-max_fname_chars:]}"
        if len(data["file"
                    ]) > max_fname_chars + 3 else data["file"]
    ).ljust(max_fname_chars + 3)

    # Get length repr
    num_samples_repr = str(data["num_samples_per_channel"])

    if len(num_samples_repr) > 10:
        f"{Decimal(num_samples_repr):.5e}"
    
    if data["is_invalid"]:
        len_repr = "-".rjust(20)
    
    else:
        len_repr = (
            num_samples_repr
            + "x" + str(data["num_channels"])
            + "@" + str(data["fs"]) + "hz"
        ).rjust(20)

    # Get mem repr
    mem_repr = bytes_to_str(data["size_bytes"]).rjust(7)

    # Get format repr
    data_fmt = "-" if data["fmt"] is None else data["fmt"]
    data_subtype = "-" if data["subtype"] is None else data["subtype"]

    fmt_repr = data_fmt.ljust(6) + " " + data_subtype.ljust(14)

    # Audio stats repr
    if data["is_invalid"]:
        db_repr = "- dBrms".rjust(16) + " " + "- dBpeak".rjust(16)
    
    else:
        if all(
            s in data
            for s in (
                "spectral_rolloff_min",
                "spectral_rolloff",
                "spectral_rolloff_max")
        ):
            db_repr = " ".join(
                [
                    f"{r:.1f}dBrms:{idx}".rjust(16)
                    + f"{p:.1f}dBpeak:{idx}".rjust(16)
                    + "  "
                    + f"{s_min / 1000.0:.1f}".rjust(4) 
                    + " ≤"
                    + f"{s_avg / 1000.0:.1f}".rjust(4)
                    + " ≤"
                    + f"{s_max / 1000.0:.1f}".rjust(4)
                    + f"kHz:{idx}".ljust(6)
                    for idx, (r, p, s_min, s_avg, s_max)
                    in enumerate(
                        zip(
                            data["rms_db"],
                            data["peak_db"],
                            data["spectral_rolloff_min"],
                            data["spectral_rolloff"],
                            data["spectral_rolloff_max"]
                        )
                    )
                ]
            )

        elif "spectral_rolloff" in data:
            db_repr = " ".join(
                [
                    f"{r:.1f}dBrms:{idx}".rjust(16)
                    + f"{p:.1f}dBpeak:{idx}".rjust(16)
                    + f"{s / 1000.0:.1f}kHz:{idx}".rjust(12)
                    for idx, (r, p, s)
                    in enumerate(
                        zip(
                            data["rms_db"],
                            data["peak_db"],
                            data["spectral_rolloff"]
                        )
                    )
                ]
            )
        
        else:
            db_repr = " ".join(
                [
                    f"{r:.1f}dBrms:{idx}".rjust(16)
                    + f"{p:.1f}dBpeak:{idx}".rjust(16)
                    for idx, (r, p)
                    in enumerate(zip(data["rms_db"], data["peak_db"]))
                ]
            )

    # Assemble representation
    repr = f"{filename_repr} {mem_repr} {fmt_repr} {len_repr} {db_repr}"

    if "sha256" in data:
        sha256 = data["sha256"][-8:] if abbrev_hash else data["sha256"]
        repr = (
            f"{filename_repr}  {sha256} {mem_repr} {fmt_repr} {len_repr} "
            f"{db_repr}"
        )

    if (
        data["is_clipped"]
        or data["is_anomalous"]
        or data["is_silent"]
        or data["is_invalid"]
    ):
        repr = f"<error>{repr}</error>"

    return repr


def _create_dir_if_missing(dir: str) -> None:
    """Creates a folder if it does not exist.
    
    Args:
        dir (str): Folder to be created.
    """
    if not os.path.isdir(dir):
        print(f"Creating post action output folder '{dir}'")

        try:
            os.makedirs(dir)
        
        except Exception as e:
            exit_error(f"An unexpected error ocurred: {e}")


def _perform_post_action(files: List[str], args: Namespace) -> None:
    """Perform post actions such as copying, moving or deleting files that
    match a certain filter.
    
    Args:
        files (List[str]): List of targeted files.
        args (Namespace): Main namespace containing user provided input.
    """
    if args.post_action == "cp":
        output = args.post_action_output
        print(f"\n{len(files)} file(s) will be copied to '{output}'")

        if not args.unattended:
            ask_confirmation()
        
        _create_dir_if_missing(args.post_action_output)

        copied_files = 0

        # Copy files and warn user if a file already exists
        for f in tqdm(
            files,
            desc="Copying files",
            leave=False,
            unit="file",
            colour=get_sppbar_color()
        ):
            try:
                if args.recursive and args.post_action_preserve_subfolders:
                    # Get relative path
                    dst = f.replace(args.input, "", 1)
                    dst = dst.split(os.sep)  # Avoid double-slashes xplatform
                    dst = os.path.join(args.post_action_output, *dst)

                    # Create subfolders if structure does not exist yet
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                
                else:
                    dst = os.path.join(output, os.path.basename(f))

                if os.path.exists(dst):
                    print_warning(
                        f"File '{dst}' already exists",
                        writer=tqdm
                    )
                    continue

                else:
                    shutil.copy(f, dst)
                    copied_files += 1
            
            except Exception as e:
                print_warning(
                    f"An error ocurred while copying '{f}' to '{dst}': {e}",
                    writer=tqdm
                )
            
        print_fn = (print_warning if copied_files != len(files) else print)
        print_fn(f"{copied_files}/{len(files)} file(s) copied to '{output}'")
    
    elif args.post_action == "mv":
        output = args.post_action_output
        print(f"\n{len(files)} file(s) will be moved to '{output}'")

        if not args.unattended:
            ask_confirmation()
        
        _create_dir_if_missing(args.post_action_output)
    
        # Move files and warn user if a file already exists
        moved_files = 0

        # Move audio files and warn user if a file already exists
        for f in tqdm(
            files,
            desc="Moving audio files",
            leave=False,
            unit="file",
            colour=get_sppbar_color()
        ):
            try:
                if args.recursive and args.post_action_preserve_subfolders:
                    # Get relative path
                    dst = f.replace(args.input, "", 1)
                    dst = dst.split(os.sep)  # Avoid double-slashes xplatform
                    dst = os.path.join(args.post_action_output, *dst)

                    # Create subfolders if structure does not exist yet
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                
                else:
                    dst = os.path.join(output, os.path.basename(f))

                if os.path.exists(dst):
                    print_warning(f"File '{dst}' already exists", writer=tqdm)
                    continue
                    
                else:
                    shutil.move(f, dst)
                    moved_files += 1
            
            except Exception as e:
                print_warning(
                    f"An error ocurred while moving '{f}' to '{dst}': {e}",
                    writer=tqdm
                )
            
        print_fn = (print_warning if moved_files != len(files) else print)
        print_fn(f"{moved_files}/{len(files)} file(s) moved to '{output}'")
    
    elif args.post_action == "rm":
        print(
            f"\n{len(files)} file(s) will be deleted. This action cannot be "
            "undone"
        )

        if not args.unattended:
            ask_confirmation()
        
        # Delete files
        for f in tqdm(
            files,
            desc="Deleting files",
            leave=False,
            unit="file",
            colour=get_sppbar_color()
        ):
            try:
                os.remove(f)
            
            except Exception as e:
                print_warning(
                    f"An error ocurred while deleting '{f}': {e}",
                    writer=tqdm
                )
    
    elif args.post_action == "cp+sp":
        # Check splits are possible without 0 files in any of them
        if len(files) < args.post_action_num_splits:
            exit_error(
                "The number of requested splits "
                f"({args.post_action_num_splits}) is bigger than the number of"
                f" file(s) ({len(files)})"
            )

        print(
            f"\n{len(files)} file(s) will be copied and split into "
            f"{args.post_action_num_splits} partition(s)"
        )

        if not args.unattended:
            ask_confirmation()
        
        _create_dir_if_missing(args.post_action_output)

        # Shuffle files and create splits specs
        random.seed(args.random_seed)
        random.shuffle(files)
        splits = np.array_split(files, args.post_action_num_splits)
        split_dirname_zfill = len(str(args.post_action_num_splits - 1))
        copied_files = 0

        for split_idx, split_data in enumerate(splits):
            split_files = split_data.tolist()
            split_dir = os.path.join(
                args.post_action_output,
                f"{args.post_action_split_dirname}"
                f"{str(split_idx).zfill(split_dirname_zfill)}"
            )
            _create_dir_if_missing(split_dir)

            # Copy files and warn user if a file already exists
            for f in tqdm(
                split_files,
                desc=(
                    f"Copying files of partition {split_idx + 1}/"
                    f"{args.post_action_num_splits}"
                ),
                leave=False,
                unit="file",
                colour=get_sppbar_color()
            ):
                try:
                    if args.recursive and args.post_action_preserve_subfolders:
                        # Get relative path
                        dst = f.replace(args.input, "", 1)
                        dst = dst.split(os.sep)  # Avoid double-slashes
                        dst = os.path.join(split_dir, *dst)

                        # Create subfolders if structure does not exist yet
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                    
                    else:
                        dst = os.path.join(split_dir, os.path.basename(f))
                    
                    if os.path.exists(dst):
                        print_warning(
                            f"File '{dst}' already exists",
                            writer=tqdm
                        )
                        continue

                    else:
                        shutil.copy(f, dst)
                        copied_files += 1
                
                except Exception as e:
                    print_warning(
                        f"An error ocurred while copying '{f}' to '{dst}': {e}",
                        writer=tqdm
                    )

        print_fn = (print_warning if copied_files != len(files) else print)
        print_fn(
           f"{copied_files}/{len(files)} file(s) copied to "
           f"'{args.post_action_output}'"
        )
    
    elif args.post_action == "mv+sp":
        # Check splits are possible without 0 files in any of them
        if len(files) < args.post_action_num_splits:
            exit_error(
                "The number of requested splits "
                f"({args.post_action_num_splits}) is bigger than the number of"
                f" file(s) ({len(files)})"
            )

        print(
            f"\n{len(files)} file(s) will be moved and split into "
            f"{args.post_action_num_splits} partition(s)"
        )

        if not args.unattended:
            ask_confirmation()
        
        _create_dir_if_missing(args.post_action_output)

        # Create splits specs
        random.seed(args.random_seed)
        random.shuffle(files)
        splits = np.array_split(files, args.post_action_num_splits)
        split_dirname_zfill = len(str(args.post_action_num_splits - 1))
        moved_files = 0

        for split_idx, split_data in enumerate(splits):
            split_files = split_data.tolist()
            split_dir = os.path.join(
                args.post_action_output,
                f"{args.post_action_split_dirname}"
                f"{str(split_idx).zfill(split_dirname_zfill)}"
            )
            _create_dir_if_missing(split_dir)

            # Copy files and warn user if a file already exists
            for f in tqdm(
                split_files,
                desc=(
                    f"Moving files of partition {split_idx + 1}/"
                    f"{args.post_action_num_splits}"
                ),
                leave=False,
                unit="file",
                colour=get_sppbar_color()
            ):
                try:
                    if args.recursive and args.post_action_preserve_subfolders:
                        # Get relative path
                        dst = f.replace(args.input, "", 1)
                        dst = dst.split(os.sep)  # Avoid double-slashes
                        dst = os.path.join(split_dir, *dst)

                        # Create subfolders if structure does not exist yet
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                    
                    else:
                        dst = os.path.join(split_dir, os.path.basename(f))
                    
                    if os.path.exists(dst):
                        print_warning(
                            f"File '{dst}' already exists",
                            writer=tqdm
                        )
                        continue

                    else:
                        shutil.move(f, dst)
                        moved_files += 1
                
                except Exception as e:
                    print_warning(
                        f"An error ocurred while moving '{f}' to '{dst}': {e}",
                        writer=tqdm
                    )

        print_fn = (print_warning if moved_files != len(files) else print)
        print_fn(
           f"{moved_files}/{len(files)} file(s) moved to "
           f"'{args.post_action_output}'"
        )

    elif args.post_action == "dump":
        print(
            f"\n{len(files)} file path(s) will be dumped to "
            f"'{args.post_action_output}'"
        )

        if not args.unattended:
            ask_confirmation()

        with open(args.post_action_output, "w") as f:
            f.write("\n".join(files))
        
        print(
            f"\n{len(files)} file path(s) dumped to "
            f"'{args.post_action_output}'"
        )

    elif args.post_action == "dump+sp":
        if args.post_action_num_splits > len(files):
            exit_error(
                "--post-action-num-splits should be equal or smaller than "
                "the number of files"
            )
        
        split_file_repr, ext = os.path.splitext(args.post_action_output)
        split_file_repr += f"_*{ext}"
        
        print(
            f"\n{len(files)} file path(s) will be dumped to "
            f"{args.post_action_num_splits} file(s) '{split_file_repr}'"
        )

        if not args.unattended:
            ask_confirmation()
        
        splits = np.array_split(files, args.post_action_num_splits)
        zfill = len(str(len(splits)))

        for split_idx, split in enumerate(splits):
            filename, ext = os.path.splitext(args.post_action_output)
            filename += f"_{str(split_idx).zfill(zfill)}{ext}"

            with open(filename, "w") as f:
                f.write("\n".join(split))
         
        print(
            f"\n{len(files)} file(s) dumped to "
            f"{args.post_action_num_splits} file(s) '{split_file_repr}'"
        )

    else:
        raise AssertionError


def sndls(args: Namespace) -> None:
    """Main routine triggered by the `sndls` command.
    
    Args:
        args (Namespace): Main namespace containing user provided input.
    """
    # Check file extensions
    for ext in args.extension:
        if ext not in get_allowed_audio_file_extensions():
            exit_error(
                "Currently only the following extensions are supported: "
                f"{', '.join(get_allowed_audio_file_extensions())}"
            )
    
    # Check incompatible args that are not handled by mutually exclusive groups
    if args.meta and (
        args.sha256
        or args.sha256_short
        or args.csv
        or args.filter
        or args.select 
        or args.spectral_rolloff
    ):
        exit_error(
            "--meta not allowed with: --sha256, --sha256-short, --csv, "
            "--filter, --select, --spectral-rolloff"
        )
    
    # Check spectral-rolloff if enabled
    if (
        args.spectral_rolloff is not None
        and (args.spectral_rolloff > 1.0 or args.spectral_rolloff < 0.0)
    ):
        exit_error("--spectral-rolloff should be a value between 0.0 and 1.0")

    # Check csv does not exist already if it should be written
    if args.csv and os.path.isfile(args.csv) and not args.csv_overwrite:
        exit_error(
            f"'{args.csv}' already exists. Please choose a different filename "
            "or use --csv-overwrite to allow overwriting existing files"
        )
    
    if args.sample is not None and args.sample <= 0.0:
        exit_error(
            "--sample should be 1 or greater to sample a concrete number of "
            "samples or between 0.0 and 1.0 to sample a percentage of all "
            "samples"
        )
    
    # Preload file if requested
    if args.preload is not None:
        preload = _preload_file(
            file=args.preload,
            has_header=args.preload_has_header,
            truncate_ragged_lines=args.preload_truncate_ragged_lines,
            ignore_errors=args.csv_ignore_errors
        )
    
    else:
        preload = None
    
    # Get file(s)
    if is_file_with_ext(file=args.input, ext=args.extension):
        files = [args.input]
    
    elif is_file_with_ext(file=args.input, ext=".csv"):
        # Read file and check if the col column exists
        df = pl.read_csv(args.input, ignore_errors=args.csv_ignore_errors)

        if args.csv_input_file_col not in df.columns:
            exit_error(
                f"No '{args.csv_input_file_col}' column found in the provided "
                f".csv file '{args.input}'"
            )

        # Read files and check all exist since col can contain garbage values
        files = df[args.csv_input_file_col].to_list()

        for row_idx, file in enumerate(tqdm(
            files,
            desc=f"Verifying '{args.csv_input_file_col}' column data",
            colour=get_sppbar_color(),
            leave=False,
            unit="row"
        )):
            if not is_file_with_ext(file=file, ext=args.extension):
                # NOTE: +2 because the count starts from 1 and the header
                # row is skipped in the count
                exit_error(
                    f"Row #{row_idx + 2} of '{args.csv_input_file_col}' column"
                    f" contains an invalid filename '{file}'. Please check "
                    "that such a file exists, has a valid --extension option, "
                    "and can be reached"
                )
        
    elif os.path.isdir(args.input):
        # Show progress bar in case folder is too big
        with tqdm(
            total=1,
            desc="Fetching files... This may take some time for large folders",
            bar_format="{desc}",
            leave=False
        ) as pbar:
            pbar.update(1)
            files = get_dir_files(
                dir=args.input,
                ext=args.extension,
                recursive=args.recursive
            )
        
    else:
        exit_error(f"Invalid input file or folder '{args.input}'")

    # Check folder is not empty
    if len(files) == 0:
        if not args.recursive:
            exit_warning(
                f"0 audio files found in '{args.input}'. Use --recursive or -r"
                " if you intended to perform a recursive search"
            )
        else:
            exit_warning(f"0 audio files found in '{args.input}'")

    # Check splits are provided
    if (
        args.post_action in ("mv+sp", "cp+sp", "dump+sp")
        and args.post_action_num_splits is None
    ):
        exit_error(
            "--post-action-num-splits must be defined if --post-action is "
            "mv+sp, cp+sp or dump+sp"
        )
    
    # Sample files if --sample is enabled
    if args.sample:
        if args.sample >= 1.0 and len(files) < int(args.sample):
            exit_error(
                "Not enough files to sample from. The current input has only "
                f"{len(files)} file(s)"
            )
        
        # Sample percentage
        elif args.sample > 0.0 and args.sample < 1.0:
            rng = random.Random(args.random_seed)
            files = rng.sample(files, k=int(args.sample * len(files)))

        # Sample concrete number
        else:
            rng = random.Random(args.random_seed)
            files = rng.sample(files, k=int(args.sample))

    # Collect target files if --post-action
    if args.post_action:
        if (
            args.post_action in (
                "cp",
                "mv",
                "cp+sp",
                "mv+sp",
                "dump",
                "dump+sp"
            )
            and args.post_action_output is None
        ):
            exit_error(
                "--post-action-output must be defined if --post-action is one "
                "of cp, mv, cp+sp, mv+sp, dump or dump+sp"
            )
        
        post_action_files = []
    
    # Check --post-action-preserve-subfolders is enabled with --recursive
    if args.post_action_preserve_subfolders and not args.recursive:
        exit_error(
            "--post-action-preserve-subfolders can only be used if --recursive"
            " is enabled"
        )
    
    # Check
    if args.silent_hop_size <= 0.0 or args.silent_hop_size > 1.0:
        exit_error(
            "--silent-hop-size must be greater than 0.0 and less than or equal"
            " to 1.0"
        )

    # Global stats to collect
    glob_stats = {
        "fs": [],
        "mono_files": 0,
        "stereo_files": 0,
        "multichannel_files": 0,
        "skipped_files": 0,
        "invalid_files": 0,
        "anomalous_files": 0,
        "clipped_files": 0,
        "silent_files": 0,
        "min_duration": None,
        "max_duration": None,
        "total_duration": 0,
        "total_size_bytes": 0
    }

    # Create .csv file if requested
    if args.csv is not None:
        # Header cols (mandatory fields)
        cols = [
            "file",
            "size_bytes",
            "subtype",
            "fmt",
            "fs",
            "num_channels",
            "num_samples_per_channel",
            "duration_seconds",
            "peak_db",
            "rms_db",
            "is_clipped",
            "is_anomalous",
            "is_silent",
            "is_invalid"
        ]

        if args.spectral_rolloff:
            if args.spectral_rolloff_detail:
                for c in (
                    "spectral_rolloff_min",
                    "spectral_rolloff",
                    "spectral_rolloff_max"
                ):
                    cols.insert(-4, c)
            
            else:
                cols.insert(-4, "spectral_rolloff")

        # Optional fields
        if args.sha256 or args.sha256_short:
            cols.insert(1, "sha256")

        with open(args.csv, "w") as f:
            writer = csv.writer(f)
            writer.writerow(cols)
    
    # Mark start
    start_time = perf_counter()

    for file in tqdm(
        files,
        desc="Analyzing audio files",
        colour=get_sppbar_color(),
        leave=False,
        unit="file"
    ):
        # Get metadata
        if args.skip_invalid_files:
            try:
                audio_meta = read_audio_metadata(file)
                audio_meta["file"] = file
                audio_meta["filename"] = os.path.basename(file)
                audio_meta["size_bytes"] = os.path.getsize(file)
                audio_meta["is_invalid"] = False
            
            except Exception as _:
                audio_meta = {}
                audio_meta["file"] = file
                audio_meta["filename"] = os.path.basename(file)
                audio_meta["size_bytes"] = os.path.getsize(file)
                audio_meta["fs"] = None
                audio_meta["num_channels"] = 0
                audio_meta["num_samples_per_channel"] = 0
                audio_meta["duration_seconds"] = 0
                audio_meta["fmt"] = None
                audio_meta["subtype"] = None
                audio_meta["is_invalid"] = True

        else:
            try:
                audio_meta = read_audio_metadata(file)
                audio_meta["file"] = file
                audio_meta["filename"] = os.path.basename(file)
                audio_meta["size_bytes"] = os.path.getsize(file)
                audio_meta["is_invalid"] = False

            except Exception as e:
                exit_error(
                    f"File '{file}' could not be parsed due to the following "
                    f"error: {e}. Use --skip-invalid-files to ignore "
                    "unparseable files and continue analysis"
                )
        
        # Ingest data if requested
        if not args.meta:
            # Skip long files
            if audio_meta["duration_seconds"] > args.max_duration:
                glob_stats["skipped_files"] += 1
                continue
            
            # Update audio stats
            if args.skip_invalid_files:
                try:
                    audio, fs = read_audio(file, dtype=args.dtype)
                    audio_peak_db = flatten_nested_list(
                        peak_db(audio, axis=-1).tolist()
                    )
                    audio_rms_db = flatten_nested_list(
                        rms_db(audio, axis=-1).tolist()
                    )
                    audio_is_clipped = is_clipped(audio)
                    audio_is_anomalous = is_anomalous(audio)
                    silent_frame_size_samples = (
                        ms_to_samples(
                            args.silent_frame_size_ms,
                            fs=fs,
                            truncate=True
                        )
                        if args.silent_frame_size_ms is not None else None
                    )
                    audio_is_silent = is_silent(
                        x=audio,
                        thresh_db=args.silent_thresh,
                        frame_size=silent_frame_size_samples,
                        hop_size=args.silent_hop_size,
                        axis=-1,
                        mode=args.silent_frame_mode
                    )
                    audio_meta["peak_db"] = audio_peak_db
                    audio_meta["rms_db"] = audio_rms_db
                    audio_meta["is_clipped"] = audio_is_clipped
                    audio_meta["is_anomalous"] = audio_is_anomalous
                    audio_meta["is_silent"] = audio_is_silent
                    audio_meta["is_invalid"] = False

                    if args.sha256 or args.sha256_short:
                        audio_meta["sha256"] = generate_sha256_from_file(file)
                
                except Exception as _:
                    audio_is_clipped = False
                    audio_is_anomalous = False
                    audio_is_silent = False
                    audio_meta["peak_db"] = None
                    audio_meta["rms_db"] = None
                    audio_meta["is_clipped"] = False
                    audio_meta["is_anomalous"] = False
                    audio_meta["is_silent"] = False
                    audio_meta["is_invalid"] = True

                    if args.spectral_rolloff is not None:
                        if args.spectral_rolloff_detail:
                            _spectral_rolloff = spectral_rolloff(
                                audio,
                                fs,
                                args.fft_size,
                                args.hop_size,
                                rolloff=args.spectral_rolloff
                            )
                            _min_spectral_rolloff = flatten_nested_list(
                                np.min(
                                    _spectral_rolloff,
                                    axis=-1,
                                    keepdims=True
                                ).tolist()
                            )
                            _max_spectral_rolloff = flatten_nested_list(
                                    np.max(
                                    _spectral_rolloff,
                                    axis=-1,
                                    keepdims=True
                                ).tolist()
                            )
                            _average_spectral_rolloff = flatten_nested_list(
                                    np.mean(
                                    spectral_rolloff(
                                        audio,
                                        fs,
                                        args.fft_size,
                                        args.hop_size,
                                        rolloff=args.spectral_rolloff
                                    ),
                                    axis=-1,
                                    keepdims=True
                                ).tolist()
                            )
                            audio_meta["spectral_rolloff_min"] = (
                                _min_spectral_rolloff
                            )
                            audio_meta["spectral_rolloff"] = (
                                _average_spectral_rolloff
                            )
                            audio_meta["spectral_rolloff_max"] = (
                                _max_spectral_rolloff
                            )
                        
                        else:
                            audio_meta["spectral_rolloff"] = (
                                    flatten_nested_list(
                                        np.mean(
                                        spectral_rolloff(
                                            audio,
                                            fs,
                                            args.fft_size,
                                            args.hop_size,
                                            rolloff=args.spectral_rolloff
                                        ),
                                        axis=-1,
                                        keepdims=True
                                    ).tolist()
                                )
                            )

                    if args.sha256 or args.sha256_short:
                        audio_meta["sha256"] = generate_sha256_from_file(file)
            
            else:
                try:
                    audio, fs = read_audio(file, dtype=args.dtype)
                    audio_peak_db = flatten_nested_list(
                        peak_db(audio, axis=-1).tolist()
                    )
                    audio_rms_db = flatten_nested_list(
                        rms_db(audio, axis=-1).tolist()
                    )
                    audio_is_clipped = is_clipped(audio)
                    audio_is_anomalous = is_anomalous(audio)
                    silent_frame_size_samples = (
                        ms_to_samples(
                            args.silent_frame_size_ms,
                            fs=fs,
                            truncate=True
                        )
                        if args.silent_frame_size_ms is not None else None
                    )
                    audio_is_silent = is_silent(
                        x=audio,
                        thresh_db=args.silent_thresh,
                        frame_size=silent_frame_size_samples,
                        hop_size=args.silent_hop_size,
                        axis=-1,
                        mode=args.silent_frame_mode
                    )
                    audio_meta["peak_db"] = audio_peak_db
                    audio_meta["rms_db"] = audio_rms_db
                    audio_meta["is_clipped"] = audio_is_clipped
                    audio_meta["is_anomalous"] = audio_is_anomalous
                    audio_meta["is_silent"] = audio_is_silent

                    if args.spectral_rolloff is not None:
                        if args.spectral_rolloff_detail:
                            _spectral_rolloff = spectral_rolloff(
                                audio,
                                fs,
                                args.fft_size,
                                args.hop_size,
                                rolloff=args.spectral_rolloff
                            )
                            _min_spectral_rolloff = flatten_nested_list(
                                np.min(
                                    _spectral_rolloff,
                                    axis=-1,
                                    keepdims=True
                                ).tolist()
                            )
                            _max_spectral_rolloff = flatten_nested_list(
                                    np.max(
                                    _spectral_rolloff,
                                    axis=-1,
                                    keepdims=True
                                ).tolist()
                            )
                            _average_spectral_rolloff = flatten_nested_list(
                                    np.mean(
                                    spectral_rolloff(
                                        audio,
                                        fs,
                                        args.fft_size,
                                        args.hop_size,
                                        rolloff=args.spectral_rolloff
                                    ),
                                    axis=-1,
                                    keepdims=True
                                ).tolist()
                            )
                            audio_meta["spectral_rolloff_min"] = (
                                _min_spectral_rolloff
                            )
                            audio_meta["spectral_rolloff"] = (
                                _average_spectral_rolloff
                            )
                            audio_meta["spectral_rolloff_max"] = (
                                _max_spectral_rolloff
                            )
                        
                        else:
                            audio_meta["spectral_rolloff"] = (
                                    flatten_nested_list(
                                        np.mean(
                                        spectral_rolloff(
                                            audio,
                                            fs,
                                            args.fft_size,
                                            args.hop_size,
                                            rolloff=args.spectral_rolloff
                                        ),
                                        axis=-1,
                                        keepdims=True
                                    ).tolist()
                                )
                            )

                    if args.sha256 or args.sha256_short:
                        audio_meta["sha256"] = generate_sha256_from_file(file)
                
                except Exception as e:
                    exit_error(
                        f"File '{file}' could not be parsed due to the "
                        f"following error: {e}. Use --skip-invalid-files to "
                        "ignore unparseable files and continue analysis"
                    )
            
            # Apply filters
            if (
                (
                    args.filter is not None
                    and _matches_filter(
                        data=audio_meta,
                        preload=preload,
                        expr=args.filter
                    )
                ) or (
                    args.select is not None
                    and not _matches_filter(
                        data=audio_meta,
                        preload=preload,
                        expr=args.select,
                    )
                )
            ):
                continue
  
            if not args.summary:
                file_repr = _audio_file_repr_from_dict(
                    audio_meta,
                    args.max_fname_chars,
                    abbrev_hash=bool(args.sha256_short)
                )
                print(file_repr, writer=tqdm)

        else:
            # Format current file representation
            if not args.summary:
                file_repr = _audio_file_meta_repr_from_dict(
                    audio_meta,
                    args.max_fname_chars
                )
                print(file_repr, writer=tqdm)
            
        # Collect files for --post-action if any 
        if args.post_action:
            post_action_files.append(file)
        
        # Write data to csv
        if args.csv:
            with open(args.csv, "a") as f:
                writer = csv.DictWriter(f, fieldnames=cols)

                # Remove fields not written to .csv
                del audio_meta["filename"]

                writer.writerow(audio_meta)
        
        # Update global stats based on metadata
        if isinstance(audio_meta["duration_seconds"], Number):
            # NOTE: It may not be a number in invalid files
            glob_stats["total_duration"] += audio_meta["duration_seconds"]
        
        # Update size
        glob_stats["total_size_bytes"] += audio_meta["size_bytes"]

        # Update duration stats
        if (
            (glob_stats["min_duration"] is None)
            or (audio_meta["duration_seconds"] < glob_stats["min_duration"])
        ):
            glob_stats["min_duration"] = audio_meta["duration_seconds"]
        
        if (
            (glob_stats["max_duration"] is None)
            or (audio_meta["duration_seconds"] > glob_stats["max_duration"])
        ):
            glob_stats["max_duration"] = audio_meta["duration_seconds"]
        
        # Update channel stats
        if audio_meta["num_channels"] == 1:
            glob_stats["mono_files"] += 1
            
        elif audio_meta["num_channels"] == 2:
            glob_stats["stereo_files"] += 1
        
        elif audio_meta["num_channels"] > 2:
            glob_stats["multichannel_files"] += 1
        
        # Update sample rates
        if audio_meta["fs"] not in glob_stats["fs"]:
            glob_stats["fs"].append(audio_meta["fs"])
        
        # Update global stats based on audio data
        if not args.meta:
            if audio_is_silent:
                glob_stats["silent_files"] += 1
            
            if audio_is_anomalous:
                glob_stats["anomalous_files"] += 1
            
            if audio_is_clipped:
                glob_stats["clipped_files"] += 1
            
            if audio_meta["is_invalid"]:
                glob_stats["invalid_files"] += 1
            
    # Get elapsed time
    elapsed_time = perf_counter() - start_time

    # Print global stats
    if not args.summary:
        print("")
    
    print(
        "Total file(s):".ljust(22) + str(
            glob_stats["mono_files"]
            + glob_stats["stereo_files"]
            + glob_stats["multichannel_files"]
            + glob_stats["invalid_files"]
        )
    )

    if glob_stats["invalid_files"] > 0:
        print_error(
            "Invalid file(s):".ljust(22) + f"{glob_stats['invalid_files']}"
        )

    print("Mono file(s):".ljust(22) + f"{glob_stats['mono_files']}")
    print("Stereo file(s):".ljust(22) + f"{glob_stats['stereo_files']}")
    print(
        "Multichannel file(s):".ljust(22)
        + f"{glob_stats['multichannel_files']}",
    )

    if len(glob_stats["fs"]) == 0:
        fs_repr = "-"

    else:
        fs_repr = ", ".join(
            f"{fs}hz" if fs is not None
            else "unknown" for fs in glob_stats["fs"]
        )

    if "unknown" in fs_repr:
        print_error("Sample rate(s):".ljust(22) + f"{fs_repr}")
    
    else:
        print("Sample rate(s):".ljust(22) +  f"{fs_repr}")
    
    # Data dependant summary lines
    if not args.meta:
        skipped_files_repr = (
            "Skipped files:".ljust(22) + str(glob_stats["skipped_files"])
        )

        if glob_stats["skipped_files"] > 0:
            print_error(skipped_files_repr)
        
        else:
            print(skipped_files_repr)

        clipped_files_repr = (
            "Clipped files:".ljust(22) + str(glob_stats["clipped_files"])
        )

        if glob_stats["clipped_files"] > 0:
            print_error(clipped_files_repr)

        else:
            print(clipped_files_repr)

        anomalous_files_repr = (
            "Anomalous files:".ljust(22) + str(glob_stats["anomalous_files"])
        )

        if glob_stats["anomalous_files"] > 0:
            print_error(anomalous_files_repr)
        
        else:
            print(anomalous_files_repr)

        silent_files_repr = (
            "Silent files:".ljust(22) + str(glob_stats["silent_files"])
        )

        if glob_stats["silent_files"] > 0:
            print_error(silent_files_repr)
        
        else:
            print(silent_files_repr)

    print(
        "Total duration:".ljust(22)
        + time_to_str(glob_stats['total_duration']),
    )

    # NOTE: Total files are recalculated because some files may have been
    # filtered from len(files)
    total_files = (
        glob_stats["mono_files"]
        + glob_stats["stereo_files"]
        + glob_stats["multichannel_files"]
    )

    if total_files > 1:
        print(
            "Minimum duration:".ljust(22)
            + time_to_str(glob_stats['min_duration'])
        )
        print(
            "Maximum duration:".ljust(22)
            + time_to_str(glob_stats['max_duration'])
        )
        print(
            "Average duration:".ljust(22)
            + time_to_str(glob_stats['total_duration'] / total_files)
        )

    print(
        "Total size:".ljust(22) + bytes_to_str(glob_stats['total_size_bytes']),
    )
    print("")
    print(f"Elapsed time: {time_to_str(elapsed_time, abbrev=False)}")

    # Perform --post-action if any
    if args.post_action:
        _perform_post_action(post_action_files, args)
