
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

import argparse
import pickle


def main():
    args = cli_parser().parse_args()

    match args.command:
        case 'inspect':
            inspect_chart(args.filename)

        case 'simulate':
            simulate_chart(args.filename, args.chart)

        case 'extract_single':
            extract_single(args.input_file, args.output_file)

        case command:
            print("Invalid command '{}'".format(command))


def inspect_chart(filename: str):
    print('Reading file {}'.format(filename))

    stepfile = ssc_util.parse_ssc(filename)


    for key in ['TITLE', 'ARTIST']:
        print('{}: {}'.format(key, stepfile.info[key]))

    charts = [chart for chart in stepfile.charts if ssc_util.is_applicable_chart(chart)]

    print('available charts:', [chart.DESCRIPTION for chart in charts])


def simulate_chart(filename: str, chart_name):
    print('Reading file {}'.format(filename))

    stepfile = ssc_util.parse_ssc(filename)

    charts = [chart for chart in stepfile.charts if ssc_util.is_applicable_chart(chart)]

    if chart_name is None or chart_name == '':
        print('Available charts:', [chart.DESCRIPTION for chart in charts])
        chart_name = input('Chart to play:')

    for chart in charts:
        if chart.DESCRIPTION == chart_name:
            info = ssc_util.compute_steps_absolute_times(chart.OFFSET, chart.BPMS, chart.NOTES)
            ssc_util.run_chart(info)

def extract_single(source, destination):
    print('Extracting features for {} into {}'.format(source, destination))

    loader = audio_util.AudioFeatureLoader(use_tqdm=True)

    features = loader.load(source)

    print('feature shape:', features.shape)

    with open(destination, 'wb') as f:
        pickle.dump(features, f)
    

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

    return parser

if __name__ == '__main__':
    main()

