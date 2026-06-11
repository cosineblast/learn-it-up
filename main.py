# A ssc file is a sequence of a bunch of key value pairs
# of the form #KEY:VALUE;
#
# The first key of a file is VERSION (we expect 0.81)
# then, the file will contain several attributes about all the charts contained in the file
# such as song name, song author, genre, expected bpm, audio file, background video file.
#
# After the overall information about the file, comes the information about each chart.
# Each chart starts with the key NOTEDATA and an empty value, followed by the chart data.
# The most important ones are OFFSET, BPMS (Beats Per Minute) and NOTES.
#
# - NOTES is a newline separated
# list of numbers representing the steps at each position. The notes value
# also has lines containing a single comma separating measures.
#
# - OFFSET is a number that tells how much time (in seconds) should the music audio be displaced
# to align with the chart.
#
# - BPMS is a list of pairs informing the moments of BPM changes in the songs. The list
# items are separated by commas, and the pair are of the form TIME=BPM.
#
# For visual gimmicks, the main keys are STOPS, DELAY, WARP, TIMESIGNATURES,
# TICKCOUNTS, COMBOS, SPEEDS, SCROLLS and FAKES.
# Out of these, SPEEDS and SCROLLS do not affect note timing.
#
# LASTSECONDHINT is used by the game to figure out the song length in seconds.

# There is a project that implements parsing ssc files more conveniently, but it uses a
# broken python dependency fs, so we use this stream-based parser instead.

import ssc_util
import audio_util
import misc_commands

import argparse
import glob
import re
import os
import json
import random
import itertools
from pathlib import Path
import pickle


def main():
    args = cli_parser().parse_args()

    match args.command:
        case "inspect":
            misc_commands.inspect_chart(args.filename)

        case "simulate":
            misc_commands.simulate_chart(args.filename, args.chart)

        case "extract_single":
            misc_commands.extract_single(args.input_file, args.output_file)

        case "parse_single":
            misc_commands.parse_single(args.input_file, args.output_file)

        case "resample_single":
            misc_commands.resample_single(args.features, args.sscbin, args.chart, args.destination)

        case "parse_all":
            parse_all()

        case "extract_all":
            extract_all()

        case "partition":
            partition(args.seed)
        case command:
            print("Invalid command '{}'".format(command))


def cli_parser():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    inspect = subparsers.add_parser(
        "inspect", help="Inspect relevant attributes of a PIU SSC file"
    )
    inspect.add_argument("filename")

    simulate = subparsers.add_parser(
        "simulate", help="Simulate timing of a PIU SSC chart"
    )
    simulate.add_argument("filename")
    simulate.add_argument("--chart", default=None)

    extract = subparsers.add_parser(
        "extract_single", help="Extract audio information from file"
    )
    extract.add_argument("input_file")
    extract.add_argument("output_file")

    parse_single = subparsers.add_parser(
        "parse_single", help="Parse SSC file into a refined file with absoute time info"
    )
    parse_single.add_argument("input_file")
    parse_single.add_argument("output_file")

    resample_single = subparsers.add_parser(
        "resample_single", help="Resample an attribute file to fit BPMs of a chart (debug)"
    )
    resample_single.add_argument("features")
    resample_single.add_argument("--sscbin")
    resample_single.add_argument("--chart")
    resample_single.add_argument("destination")

    parse_all = subparsers.add_parser(
        "parse_all", help="Parse all charts in the data/songs into data/parsed"
    )

    extract_all = subparsers.add_parser(
        "extract_all", help="Extract audio information from all files in data/parsed"
    )

    partition = subparsers.add_parser(
        "partition", help="Compute training test and validation partitions"
    )
    partition.add_argument("--seed", default=42, type=int)

    return parser


# This command parses all the SSC files for the model's dataset.
# it:
# - parses all SSC files in data/songs
# - filters out charts that are considered invalid (e.g containing stops or warps)
# - finds absolute BPM information for all these charts
# - generates the three mirrored versions of the charts (horizontal, vertical, horizontal+vertical)
# - saves the result to data/parsed/${pack_name}_${song_title}.ssc.bin
def parse_all():
    print("LET THE BASS KICK")

    sscs = sorted(glob.glob("data/songs/**/*.ssc", recursive=True))

    print("About to parse {} .ssc files".format(len(sscs)))

    os.makedirs("data/parsed", exist_ok=True)

    for file in sscs:
        print("<<< Parsing SSC file", file)

        stepfile = ssc_util.load_ssc(file)

        charts = [
            chart for chart in stepfile.charts if ssc_util.is_applicable_chart(chart)
        ]

        charts += [
            new_chart for chart in charts for new_chart in _generate_permutations(chart)
        ]

        destination_path = _get_destination_path_for_ssc(file)

        refined_stepfile = ssc_util.refine_stepfile(stepfile._replace(charts=charts), file)

        print(">>> Saving parsed result to", destination_path)
        print()


        with open(destination_path, "wb") as f:
            ssc_util.dump_refined_stepfile(refined_stepfile, f)


# This command extracts features for all song files
def extract_all():
    print("JOOOOOOOOOOO")

    refined_sscs = sorted(glob.glob("data/parsed/*.ssc.bin"))

    print("About to load {} audio files".format(len(refined_sscs)))

    os.makedirs("data/features", exist_ok=True)

    for file in refined_sscs:
        print("<<< Reading refined stepfile", file)

        with open(file, "rb") as f:
            refined_stepfile = ssc_util.load_refined_stepfile(f)

        source = refined_stepfile.info["MUSIC"]
        destination = _get_destination_path_for_features(file)

        print(">>> Extracting features for {} into {}...".format(source, destination))

        loader = audio_util.AudioFeatureLoader(use_tqdm=False)

        features = loader.load(source)

        with open(destination, "wb") as f:
            pickle.dump(features, f)

        print(">>> OK!")

# This command creates a file data/partitions.json containing information about which files are in which
# dataset partitions: training, validation and test
def partition(seed):

    random.seed(seed)

    ratios = [8, 1, 1]
    partitions = ["training", "validation", "testing"]

    prefix_sum_ratios = list(itertools.accumulate(ratios))

    # This makes so that 8/10 of values get training, 1/10 get validation and 1/10 get testing
    def get_partition_for_index(index):
        remainder = index % prefix_sum_ratios[-1]

        for partition, ratio in zip(partitions, prefix_sum_ratios):
            if remainder < ratio:
                return partition

    files_per_partition = {partition: [] for partition in partitions}

    all_refined_files = glob.glob("data/parsed/*.bin")

    random.shuffle(all_refined_files)

    for i, file in enumerate(all_refined_files):
        files_per_partition[get_partition_for_index(i)].append(file)

    with open("data/partitions.json", "w") as f:
        json.dump(files_per_partition, f, indent=2)


def _get_destination_path_for_ssc(filepath):
    def sanitize_name(name):
        without_prefix = None

        # Remove '123 - ' prefix
        match re.findall("^([0-9]+ - )(.+)$", name):
            case [(_, suffix)]:
                without_prefix = suffix
            case _:
                without_prefix = name

        return str(without_prefix).replace(" ", "_")

    path = Path(filepath)
    pack_name = sanitize_name(path.parents[1].name)
    filename = sanitize_name(path.name)

    result = Path("data/parsed/") / Path(pack_name + "___" + filename + ".bin")

    return result


def _get_destination_path_for_features(refined_filepath):
    assert str(refined_filepath).endswith(".ssc.bin")
    return Path("data/features") / (Path(Path(refined_filepath).stem).stem + ".feat.bin")


def _generate_permutations(chart):
    # down left, up left, middle, up right, down right
    dl, ul, m, ur, dr = 0, 1, 2, 3, 4

    vertical = {dl: ul, ul: dl, m: m, ur: dr, dr: ur}
    horizontal = {dl: dr, ul: ur, m: m, ur: ul, dr: dl}

    permute = lambda step, permutation: ''.join([step[permutation[i]] for i in range(5)])

    apply = lambda permutation: (
        [[permute(step, permutation) for step in measure] for measure in chart.NOTES]
    )

    return [
        chart._replace(DESCRIPTION=chart.DESCRIPTION + name, NOTES=apply(permutation))
        for permutation, name in [
            # TODO: apply vertical flip on low level charts
            # (vertical, "_V"),
            (horizontal, '_H'),
        ]
    ]


if __name__ == "__main__":
    main()
