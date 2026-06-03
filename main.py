
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
from pathlib import Path


def main():
    args = cli_parser().parse_args()

    match args.command:
        case 'inspect':
            misc_commands.inspect_chart(args.filename)

        case 'simulate':
            misc_commands.simulate_chart(args.filename, args.chart)

        case 'extract_single':
            misc_commands.extract_single(args.input_file, args.output_file)

        case 'parse_single':
            misc_commands.parse_single(args.input_file, args.output_file)

        case 'parse_all':
            parse_all()

        case command:
            misc_commands.print("Invalid command '{}'".format(command))

def cli_parser():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    inspect = subparsers.add_parser('inspect', help='Inspect relevant attributes of a PIU SSC file')
    inspect.add_argument('filename')

    simulate = subparsers.add_parser('simulate', help='Simulate timing of a PIU SSC chart')
    simulate.add_argument('filename')
    simulate.add_argument('--chart', default=None)

    extract = subparsers.add_parser('extract_single', help='Extract audio information from file')
    extract.add_argument('input_file')
    extract.add_argument('output_file')

    parse_single = subparsers.add_parser('parse_single', help='Parse SSC file into a json file with absoute time info')
    parse_single.add_argument('input_file')
    parse_single.add_argument('output_file')

    parse_all = subparsers.add_parser('parse_all', help='Parse all charts in the data/songs into data/ssc_json')

    return parser

# This command parses all the SSC files for the training dataset, it:
# - parses all SSC files in data/songs
# - filters out charts that are considered invalid (e.g containing stops or warps)
# - finds absolute BPM information for all these charts
# - generates the three mirrored versions of the charts (horizontal, vertical, horizontal+vertical)
# - saves the result to data/json_ssc/${pack_name}_${song_title}.ssc.json
def parse_all():
    print('LET THE BASS KICK')

    sscs = sorted(glob.glob('data/songs/**/*.ssc', recursive=True))

    print('About to parse {} .ssc files'.format(len(sscs)))

    os.makedirs('data/parsed', exist_ok=True)

    for file in sscs:
        print('<<< Parsing SSC file', file)

        stepfile = ssc_util.load_ssc(file)

        charts = [chart for chart in stepfile.charts if ssc_util.is_applicable_chart(chart)]
        
        charts += [new_chart for chart in charts for new_chart in generate_permutations(chart)]

        destination_path = get_destination_path_for_ssc(file)

        print('>>> Saving parsed result to', destination_path)
        print()

        content = misc_commands.stepfile_to_dicts(stepfile._replace(charts=charts))

        with open(destination_path, 'w') as f:
            json.dump(content, f)


def get_destination_path_for_ssc(filepath):
    def sanitize_name(name):
        without_prefix = None

        # Remove '123 - ' prefix
        match re.findall('^([0-9]+ - )(.+)$', name):
            case [(_, suffix)]: without_prefix = suffix
            case _: without_prefix = name

        return str(without_prefix).replace(' ', '_')
        

    path = Path(filepath)
    pack_name = sanitize_name(path.parents[1].name)
    filename = sanitize_name(path.name)

    result = Path('data/parsed/') / Path(pack_name + '___' + filename + '.json')

    return result

def generate_permutations(chart):
    # down left, up left, middle, up right, down right
    dl, ul, m, ur, dr = 0, 1, 2, 3, 4

    vertical    = { dl: ul, ul: dl, m: m, ur: dr, dr: ur }
    horizontal  = { dl: dr, ul: ur, m: m, ur: ul, dr: dl }

    permute = lambda step, permutation: [step[permutation[i]] for i in range(5)]

    apply = lambda permutation: (
        [[permute(step, permutation) for step in measure] for measure in chart.NOTES]
    )

    return [
        chart._replace(DESCRIPTION=chart.DESCRIPTION+ name, NOTES=apply(permutation))
        for permutation, name in [
            (vertical, '_V'),
            # TODO: apply horizontal flip on low level charts
            # (horizontal, '_H'),
        ]
    ]

    

if __name__ == '__main__':
    main()

