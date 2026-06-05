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


def parse_single(source, destination):
    print("Parsing SSC file {} into {}".format(source, destination))

    stepfile = ssc_util.load_ssc(source)

    charts = [chart for chart in stepfile.charts if ssc_util.is_applicable_chart(chart)]

    refined_charts = [ssc_util.refine_chart(chart) for chart in charts]

    content_to_write = ssc_util.refined_stepfile_to_dicts(
        stepfile.info,
        refined_charts,
        source
    )

    with open(destination, "w") as f:
        json.dump(content_to_write, f, indent=2)
