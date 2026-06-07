import ssc_util
import audio_util

import argparse
import pickle

import json
import os


def inspect_chart(filename: str):
    print("Reading file {}".format(filename))

    stepfile = ssc_util.load_ssc(filename)

    for key in ["TITLE", "ARTIST"]:
        print("{}: {}".format(key, stepfile.info[key]))

    charts = [
        chart
        for chart in stepfile.charts
        if ssc_util.is_applicable_chart(chart, stepfile_name=stepfile.info["TITLE"])
    ]

    print("available charts:", [chart.DESCRIPTION for chart in charts])


def simulate_chart(filename: str, chart_name):
    print("Reading file {}".format(filename))

    stepfile = ssc_util.load_ssc(filename)

    charts = [chart for chart in stepfile.charts if ssc_util.is_applicable_chart(chart)]

    if chart_name is None or chart_name == "":
        print("Available charts:", [chart.DESCRIPTION for chart in charts])
        chart_name = input("Chart to play:")

    for chart in charts:
        if chart.DESCRIPTION == chart_name:
            refined = ssc_util.refine_chart(chart)
            ssc_util.run_chart(refined)


def extract_single(source, destination):
    print("Extracting features for {} into {}".format(source, destination))

    loader = audio_util.AudioFeatureLoader(use_tqdm=True)

    features = loader.load(source)

    print("feature shape:", features.shape)

    with open(destination, "wb") as f:
        pickle.dump(features, f)

def resample_single(features_file, refined_file, chartname, destination):
    print("Resampling features {} of for chart {} of stepfile into {}".format(features_file, chartname, refined_file, destination))

    with open(refined_file, 'rb') as f:
        stepfile = ssc_util.load_refined_stepfile(f)

    with open(features_file, 'rb') as f:
        features = pickle.load(f)

    match next(iter(chart for chart in stepfile.charts if chart.description == chartname), None):
        case None:
            raise Exception(f'Chart {chartname} not found in stepfile {refined_file}')
        case value:
            chart = value

    resampled = audio_util.resample_features(features, chart.beat_start_end_times)

    print(resampled.shape)

    with open(destination, 'wb') as f:
        pickle.dump(resampled, f)



def parse_single(source, destination):
    print("Parsing SSC file {} into {}".format(source, destination))

    stepfile = ssc_util.load_ssc(source)

    charts = [chart for chart in stepfile.charts if ssc_util.is_applicable_chart(chart)]

    refined = ssc_util.refine_stepfile(stepfile._replace(charts=charts), source)

    with open(destination, "wb") as f:
        ssc_util.dump_refined_stepfile(refined, f)
