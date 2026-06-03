
import ssc_util
import audio_util

import argparse
import pickle

import json
import os

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

def parse_single(source, destination):
    print('Parsing SSC file {} into {}'.format(source, destination))

    stepfile = ssc_util.parse_ssc(source)

    charts = [chart for chart in stepfile.charts if ssc_util.is_applicable_chart(chart)]

    content_to_write = stepfile_to_dicts(stepfile._replace(charts=charts))

    with open(destination, 'w') as f:
        json.dump(content_to_write, f, indent=2)


def stepfile_to_dicts(stepfile):

    chart_to_dict = lambda chart: ({
        'description': chart.DESCRIPTION,
        'notes': chart.NOTES,
        'offset': chart.OFFSET,
        'bpms': chart.BPMS,
    })

    return {
        'title': stepfile.info['TITLE'],
        'artist': stepfile.info['ARTIST'],
        'music': os.path.abspath(stepfile.info['MUSIC']),
        'offset': stepfile.info['OFFSET'],
        'charts': [chart_to_dict(chart) for chart in stepfile.charts]
    }

    
