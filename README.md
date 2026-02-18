# `sndls`: An audio-friendly `ls`, with a little something extra

`sndls` (sound `ls`) is a command-line tool designed for quick and efficient inspection of audio data. It provides functionalities such as:

- Saving search results to a `.csv` file for later analysis.
- Detecting clipped, silent, or anomalous files that may impact machine learning pipelines.
- Computing and verifying SHA-256 hashes to detect file modifications or corruption.
- Filtering files using `python` expressions to identify those matching specific criteria.
- Performing fast, metadata-based file inspection.
- Executing post-processing actions, such as removing clipped files, copying files that meet certain conditions, and more.

`sndls` currently supports the following extensions:
`.aif`, `.aiff`, `.mp3`, `.flac`, `.ogg`, `.wav`, `.wave`.

# Table of contents
- [Installation](#installation)
    - [Install through pip](#install-through-pip)
    - [Install in developer mode](#install-in-developer-mode)
    - [Install through uv](#install-through-uv)
- [Tutorial](#tutorial)
    - [Quickstart](#quickstart)
    - [Help](#help)
    - [Recursive search](#recursive-search)
    - [Generating SHA-256 hash](#generating-sha-256-hash)
    - [Fast metadata search](#fast-metadata-search)
    - [Saving output to csv file](#saving-output-to-csv-file)
    - [Filtering by extension](#filtering-by-extension)
    - [Filtering by python expressions](#filtering-by-python-expressions)
    - [Filtering by using preloaded files](#filtering-by-using-preloaded-files)
    - [Post-actions](#post-actions)
    - [Random data sampling and splitting](#random-data-sampling-and-splitting)
- [Cite](#cite)
- [License](#license)

# Installation
## Install through pip
To install `sndls`, run:
```bash
pip install sndls
```
Verify the installation with:
```bash
sndls --version
```
This should output:
```
sndls version x.y.z yyyy-zzzz developed by Esteban Gómez
```
Where:
- `x.y.z` represents the major, minor, and patch version
- `yyyy-zzzz` indicates the development start year and the current 

## Install in developer mode
Developer mode installation is intended for those developing new features for the tool. To set it up:
1. Clone the repository to your desired folder using:
```bash
git clone <repository_url>
```
2. Navigate to the root directory (where `pyproject.toml` is located):
```bash
cd <repository_folder>
```
3. Install in developer mode with:
```bash
python -m flit install -s
```
This will allow immediate reflection of any code modifications when the tool is executed in the terminal.

Before proceeding, ensure that Flit is installed. If not, install it with:
```bash
python -m pip install flit
```
For more information on `flit`, refer to the [Flit Command Line Interface documentation](https://flit.pypa.io/en/stable/).

## Install through `uv`
Alternatively, you can install the tool using `uv`. This is adequate for when you can to keep it isolated from your `python`
environment setup and just run it to analyze a certain data collection.

1. Install `uv` and `uvx` following the instructions for your operating system in [`uv` website](https://docs.astral.sh/uv/getting-started/installation/).
2. Run:
```bash
uv tool install sndls
```
3. Verify the installation with
```bash
uv tool run sndls --version
```
or you can use the shortcut version `uvx`:
```bash
uvx sndls --version
```
This should output:
```
sndls version x.y.z yyyy-zzzz developed by Esteban Gómez
```
Where:
- `x.y.z` represents the major, minor, and patch version
- `yyyy-zzzz` indicates the development start year and the current

# Tutorial
This quick tutorial is structured into multiple sections, each focusing on a
fundamental aspect of `sndls` and its core functionalities.

## Quickstart
To inspect the audio data in a certain folder, run:
```bash
sndls /path/to/folder
```
If no path is provided, the current directory will be used as the default input.
If your folder contains audio files, you should see output similar to the
following in your terminal (the information will vary based on your folder's contents):

```bash
/path/to/audio/dir/000_audio.wav    120.0K WAV  PCM_16        50000x1@16000hz     -18.5dBrms:0    -5.0dBpeak:0
/path/to/audio/dir/001_audio.wav    115.0K WAV  PCM_16        52000x1@16000hz     -19.0dBrms:0    -5.5dBpeak:0
/path/to/audio/dir/002_audio.wav     95.0K WAV  PCM_16        48000x1@16000hz     -17.0dBrms:0    -4.5dBpeak:0
/path/to/audio/dir/003_audio.wav    130.0K WAV  PCM_16        65000x1@16000hz     -18.0dBrms:0    -3.0dBpeak:0

Total file(s):        4
Mono file(s):         4
Stereo file(s):       0
Multichannel file(s): 0
Sample rate(s):       16000hz
Skipped files:        0
Clipped files:        0
Anomalous files:      0
Silent files:         0
Total duration:       14.5 second(s)
Minimum duration:     3.0 second(s)
Maximum duration:     4.0 second(s)
Average duration:     3.6 second(s)
Total size:           460.0K

Elapsed time: 5.0 ms
```

## Help
For a detailed description of all available options, run:
```bash
sndls --help
```
This will display all parameters along with their descriptions.

## Recursive search
By default, `sndls` searches for audio files only within the specified input folder.
To include audio files from nested directories, enable recursive search using `--recursive` or `-r`:

```bash
sndls /path/to/root/dir --recursive
```

## Generating SHA-256 hash
In addition to retrieving audio metadata and data for each file, you can generate the corresponding SHA-256 hash. To visualize the full SHA-256, use the `--sha256` option. If you'd prefer to see only the last 8 characters of the SHA-256, use the `--sha256-short` option instead:
```bash
sndls /path/to/audio/dir --sha256
```
This will make your output appear as follows:
```bash
/path/to/audio/dir/000_audio.wav    d4f72a9b8cfd7e33ab32e4f24cfdb7f8a28f85a4b7f29de96b0b2b74369b48e5  106.3K WAV  PCM_16        52782x1@16000hz     -18.3dBrms:0    -2.5dBpeak:0
/path/to/audio/dir/001_audio.wav    a6d1a0c02a5e55d531b29c6cf97c09cb68fe9b0f758bdf45c1ec8f7d915e9b63  111.7K WAV  PCM_16        61425x1@16000hz     -21.0dBrms:0    -4.2dBpeak:0
/path/to/audio/dir/002_audio.wav    0f2a4d6b19b6f9cf5d8f7d47d088dc9be7b964f017028d7389f1acb46a18c8b9   90.6K WAV  PCM_16        49200x1@16000hz     -16.8dBrms:0    -3.2dBpeak:0
/path/to/audio/dir/004_audio.wav    6a55cfef36e1a8937d66b9082f74c19bc82cdbf4db7a1c98a3f1b0883c1a7456  127.9K WAV  PCM_16        68042x1@16000hz     -19.1dBrms:0    -1.9dBpeak:0

...
```

If `--sha256-short` is used instead, you should see:

```bash
/path/to/audio/dir/000_audio.wav    369b48e5  106.3K WAV  PCM_16        52782x1@16000hz     -18.3dBrms:0    -2.5dBpeak:0
/path/to/audio/dir/001_audio.wav    915e9b63  111.7K WAV  PCM_16        61425x1@16000hz     -21.0dBrms:0    -4.2dBpeak:0
/path/to/audio/dir/002_audio.wav    6a18c8b9   90.6K WAV  PCM_16        49200x1@16000hz     -16.8dBrms:0    -3.2dBpeak:0
/path/to/audio/dir/004_audio.wav    3c1a7456  127.9K WAV  PCM_16        68042x1@16000hz     -19.1dBrms:0    -1.9dBpeak:0

...
```

## Fast metadata search
Inspecting large folders or those containing long audio files can take considerable time.
In some cases, it's preferable to extract only metadata without reading the actual audio samples.
For such cases, the `--meta`  or `-m` option is available. In this case, only metadata
based information will be printed to the terminal. Information such as `peak_db`, `rms_db` will
not be calculated.
```bash
sndls /path/to/audio/dir --meta
```
For small folders, the difference in runtime may be negligible, but for larger datasets, it can be
substantial.

## Saving output to `.csv` file
The results of a given search can also be saved to a `.csv` file as tabular data for later inspection.
To do this, simply provide the `--csv` argument followed by the name of your desired output file:
```bash
sndls /path/to/audio/dir --csv output.csv
```

Please note that the `.csv` file will include the full file path and full SHA-256 (if `--sha256`
or `--sha256-short` is enabled). The results included in the `.csv` will be the exact results that match your search.

## Filtering by extension
Listed files can be filtered by many ways, including their extension. Only certain audio file extensions
that can be parsed by `soundfile` are currently supported. Use the `--extension` or `-e` option if you want
to restrict your results to a certain extension or extensions:
```bash
sndls /path/to/audio/dir --extension .wav .flac
```
In this case, the search will include only `.wav` and `.flac` files, ignoring all other extensions.

## Filtering by `python` expressions
In addition to filtering by extension using the `--extension` or `-e` option, you can create custom
filters to find files with specific traits. This can be useful for tasks like:

- Finding clipped, silent, or anomalous files
- Finding files within a specific duration range
- Finding files with a particular sample rate

For these cases, the `--select` or `-s`) option allows you to select files that meet certain criteria, while
the `--filter` or `-f` option lets you select all files except those that match the filter. Both options
accept `python` expressions for greater flexibility in your search. 

Note that these options are mutually exclusive, meaning only one can be used at a time.

For example, to search for only clipped mono files, run:
```bash
sndls /path/to/audio/dir --select "is_clipped and num_channels == 1"
```

To filter out files shorter than 3.0 seconds, run:
```bash
sndls /path/to/audio/dir --filter "duration_seconds < 3.0"
```

Please note that some fields contain lists of values, where the length depends on the
number of channels in the file, such as `peak_db` or `rms_db`. In such cases, methods
like `any()` or `all()` can be useful.

For example, to find all files where all channels have peak values in decibels (`peak_db`)
greater than -3.0 dB, you can do the following:
```bash
sndls /path/to/audio/dir --select "all(db > -3.0 for db in peak_db)"
```

Here is a list of all fields that can be used to refine your search:
| Field                      | Description                                                                                                  | Data type     |
|----------------------------|--------------------------------------------------------------------------------------------------------------|---------------|
| `file`                     | Audio file path                                                                                              | `str`         |
| `filename`                 | Audio filename                                                                                               | `str`         |
| `fs`                       | Audio sample rate in hertz (e.g. 16000, 48000)                                                               | `int`         |
| `num_channels`             | Number of channels in the file                                                                               | `int`         |
| `num_samples_per_channels` | Number of samples per channels                                                                               | `int`         |
| `duration_seconds`         | Duration of the file in seconds                                                                              | `float`       |
| `size_bytes`               | Size of the file in bytes                                                                                    | `int`         |
| `fmt`                      | File format (`WAV`, `RF64`, etc)                                                                             | `str`         |
| `subtype`                  | File subtype (`PCM_16`, `PCM_24`, `FLOAT`, etc)                                                              | `str`         |
| `peak_db`                  | Per-channel peak value in decibels                                                                           | `List[float]` |
| `rms_db`                   | Per-channel root mean square value in decibels                                                               | `List[float]` |
| `spectral_rolloff`         | Average spectral-rolloff in hertz (only available with `--spectral-rolloff`)                                 | `List[float]` |
| `spectral_rolloff_min`     | Minimum spectral-rolloff in hertz (only available with `--spectral-rolloff` and `--spectral-rolloff-detail`) | `List[float]` |
| `spectral_rolloff_max`     | Maximum spectral-rolloff in hertz (only available with `--spectral-rolloff` and `--spectral-rolloff-detail`) | `List[float]` |
| `is_silent`                | `True` if all channels have less than `--silent-thresh` dB RMS                                               | `bool`        |
| `is_clipped`               | `True` if any channel contains values outside the `-1.0` to `1.0` range                                      | `bool`        |
| `is_anomalous`             | `True` if any sample is `NaN`, `inf` or `-inf`                                                               | `bool`        |
| `is_invalid`               | `True` if the file could not be read. Only valid with `--skip-invalid-files`                                 | `bool`        |
| `sha256`                   | SHA-256 hash (only available if `--sha256` or `--sha256-short` is enabled                                    | `str`         |
| `preload`                  | Preloaded `DataFrame` (only available with `--preload`)                                                      | `DataFrame`   |

## Filtering by using preloaded files
`sndls` provides a `--preload` option to load a `.csv`, `.tsv`, or `.txt` file that can be used with the `--filter` and `--select` options. This feature allows you to expand your search and filtering capabilities, such as matching files from a specific file or finding a particular set of SHA-256 hashes, etc. To preload a file, you can do the following:
```bash
sndls /path/to/audio/dir --preload /path/to/preload/file
```

In all cases, your preloaded file will be interpreted as tabular data. To exclude the first row when it contains header information, use the `--preload-has-header` option. Otherwise, every row will be treated as data. All data from your preloaded file will be availabl
 under the preload variable when writing `--filter` or `--select` expressions. You can use it as a regular `DataFrame`. If there is no header
 information, the columns will be automatically numbered as `column_1`, `column_2`, etc.

 ```bash
 sndls /path/to/audio/dir --preload /path/to/preload/file --select "((preload['column_1'].str.contains(filename)) & (preload['column_2'] == 'TARGET')).any()"
 ```

 This expression will match all files whose filename is in `column_1` and `column_2` contains the value of `TARGET`. Please keep in mind that every file must be matched against your entire preload file, so using the `--preload` option for selection or filtering is expected to take longer than regular search expressions. However, it can be much more powerful in certain cases.

## Post-actions
In some cases, we want not just to see files matching a certain criteria, but also perform actions on them (e.g., remove clipped files or silent files from a dataset). For such cases, the `--post-action` option exists. It has five available values: `cp`, `mv`, `rm`, `cp+sp`, and `mv+sp`, where:  
- `cp` will copy the files to `--post-action-output`.  
- `mv` will move the files to `--post-action-output`.  
- `rm` will delete the files (this action cannot be undone).  
- `cp+sp` will first copy the files to `--post-action-output` and then create `--post-action-num-splits` splits of the data.  
- `mv+sp` will first move the files to `--post-action-output` and then create `--post-action-num-splits` splits of the data.
- `dump` will create a file with all the file paths. This can be useful for using `rsync` with `--files-from` option.
- `dump+sp` will create `--post-action-num-splits` files, each one containing a subset of all the file paths.

In all cases, you will be asked to confirm the action through the command line. Here is an example:

```bash
sndls /path/to/audio/dir --post-action cp --post-action-output /post/action/output
...
N file(s) will be copied to '/post/action/output'  
Do you want to continue? [y/n]:
```
Write `y` or `n` and then press enter. The action will then be executed.
If you are using this tool as part of an automated pipeline, you may want to skip user input. In such cases, there is the `--unattended` or `-u` option. When used, it will skip the confirmation prompt, but ensure that your action is correctly set up beforehand:
```bash
sndls /path/to/audio/dir --post-action cp --post-action-output /post/action/output --unattended
...
N file(s) will be copied to '/post/action/output'  
Creating post action output folder '/post/action/output'  
N/N file(s) copied to '/post/action/output'
```
The additional output lines show if all your files were correctly copied, moved, or deleted. Please note that moving or copying files will not overwrite already existing files.

## Random data sampling and splitting
`sndls` can be useful for sampling files that meet certain conditions from a large dataset, especially when copying everything or manually filtering the files might be time-consuming. The `--sample` option allows you to achieve this. In summary, this option can randomly sample a given number of files from your search results as follows:
```bash
sndls /path/to/audio/dir --sample 20
```
This command randomly samples 20 audio files from `/path/to/audio/dir`. These files can be used with the `--post-action` option to copy them to another folder for later inspection:
```bash
sndls /path/to/audio/dir --sample 20 --post-action cp --post-action-output /path/to/output/dir
```
This allows you to randomly sample data based on specific conditions, as it can be combined with the `--filter`, `--select`, or any other available options. To change the random seed used for selecting the files, you can do so as follows:
```bash
sndls /path/to/audio/dir --sample 20 --post-action cp --post-action-output /path/to/output/dir --random-seed 3673
```
Where 3673 can be any integer number that will be used as a random seed.

Additionally, if a `float` between `0.0` and `1.0` is provided with the `--sample` option, it will be interpreted as a percentage of the total number of files.

# Cite
If this tool contributed to your work, please consider citing it:

```
@misc{sndls,
  author = {Esteban Gómez},
  title  = {sndls},
  year   = 2024,
  url    = {https://github.com/eagomez2/sndls}
}
```

This tool was developed by <a href="https://estebangomez.me/" target="_blank">Esteban Gómez</a>, member of the <a href="https://www.aalto.fi/en/department-of-information-and-communications-engineering/speech-interaction-technology" target="_blank">Speech Interaction Technology group from Aalto University</a>.

# License
For further details about the license of this tool, please see [LICENSE](LICENSE).