import ssc_util
import audio_util

import argparse
import pickle

import json
import os
import time

from playsound3 import playsound, AVAILABLE_BACKENDS

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


def simulate_chart(filename: str, chart_name, audio):
    print("Reading file {}".format(filename))

    stepfile = ssc_util.load_ssc(filename)

    charts = [chart for chart in stepfile.charts if ssc_util.is_applicable_chart(chart)]

    if chart_name is None or chart_name == "":
        print("Available charts:", [chart.DESCRIPTION for chart in charts])
        chart_name = input("Chart to play:")

    for chart in charts:
        if chart.DESCRIPTION == chart_name:
            refined = ssc_util.refine_chart(chart)
            _run_chart(refined, audio)


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


def _run_chart(chart: ssc_util.RefinedChart, audio_path=None):
    """Simulates in real time, the notes of a refined chart."""
        
    steps = [step for step in chart.steps if step.stepcode != "00000"]

    if audio_path is None:
        input("PRESS ENTER FOR FIRST STEP...")
    else:
        playsound(audio_path, block=False, backend='ffplay')
        time.sleep(steps[0].time_in_seconds)

    def show_step(step):
        code = "".join(["-" if x == "0" else x for x in step.stepcode])
        print(
            "[{:10.4f}]({:10.4f}) {}".format(
                float(step.time_in_beats), step.time_in_seconds, code
            )
        )
        
    show_step(steps[0])

    base_music_time = steps[0].time_in_seconds
    base_real_time = time.time()

    for step in steps[1:]:
        to_sleep = (step.time_in_seconds - base_music_time) - (
            time.time() - base_real_time
        )

        time.sleep(max(0, to_sleep))
        show_step(step)

    time.sleep(2)


def parse_single(source, destination):
    print("Parsing SSC file {} into {}".format(source, destination))

    stepfile = ssc_util.load_ssc(source)

    charts = [chart for chart in stepfile.charts if ssc_util.is_applicable_chart(chart)]

    refined = ssc_util.refine_stepfile(stepfile._replace(charts=charts), source)

    with open(destination, "wb") as f:
        ssc_util.dump_refined_stepfile(refined, f)


def add_subparsers(subparsers):
    inspect = subparsers.add_parser(
        "inspect", help="Inspect relevant attributes of a PIU SSC file"
    )
    inspect.add_argument("filename")

    simulate = subparsers.add_parser(
        "simulate", help="Simulate timing of a PIU SSC chart"
    )
    simulate.add_argument("filename")
    simulate.add_argument("--chart", default=None)
    simulate.add_argument("--audio", default=None)

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
