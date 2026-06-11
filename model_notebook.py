import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import glob
    import pickle
    import numpy as np

    import loading
    import models

    return F, glob, loading, models, np, pickle, torch


@app.cell
def _(torch):
    print('HasCuda:',torch.cuda.is_available() )
    return


@app.cell(hide_code=True)
def _(glob, pickle):
    all_stepfiles = []
    stepfile_paths = glob.glob('data/parsed/*.ssc.bin')
    for _path in stepfile_paths:
        with open(_path, 'rb') as _f:
            all_stepfiles.append(pickle.load(_f))
    return all_stepfiles, stepfile_paths


@app.cell(hide_code=True)
def _():
    from pathlib import Path

    def get_feature_path_for(refined_stepfile_path):
        assert str(refined_stepfile_path).endswith(".ssc.bin")
        return Path("data/features") / (Path(Path(refined_stepfile_path).stem).stem + ".feat.bin")


    return


@app.cell
def _(loading):
    loader = loading.LoadFeaturesCached()
    return (loader,)


@app.cell
def _(all_stepfiles, loader, loading, stepfile_paths):
    dataset = loading.PumpItUpConvolutionCNNOnsetDataset(all_stepfiles, stepfile_paths, loader)
    return (dataset,)


@app.cell
def _(dataset):
    len(dataset)
    return


@app.cell
def _(models):
    cnn_model = models.PumpItUpConvolutionCNNOnset()
    return (cnn_model,)


@app.cell
def _(cnn_difficulty_data, cnn_model, cnn_sample_data):
    cnn_model(cnn_sample_data, cnn_difficulty_data)
    return


@app.cell
def _(np, torch):
    cnn_sample_data = torch.tensor(np.random.randn(3, 15, 80)[None, :]).type(torch.float32)
    cnn_sample_data.shape
    return (cnn_sample_data,)


@app.cell
def _(F, torch):
    cnn_difficulty_data = F.one_hot(torch.tensor([15]), num_classes=25).type(torch.float32)
    cnn_difficulty_data.shape
    return (cnn_difficulty_data,)


@app.cell
def _(full_difficulty_data, full_model, full_model_data):
    full_model(full_model_data, full_difficulty_data)
    return


@app.cell
def _(models):
    full_model = models.PumpItUpConvolutionOnset()
    return (full_model,)


@app.cell
def _(np, torch):
    full_model_data = torch.tensor(np.random.randn(1, 100, 3, 15, 80)).type(torch.float32)
    full_model_data.shape
    return (full_model_data,)


@app.cell
def _(F, torch):
    full_difficulty_data = F.one_hot(torch.tensor([15]), num_classes=25).type(torch.float32)
    full_difficulty_data.shape
    return (full_difficulty_data,)


@app.cell
def _(full_difficulty_data, torch):
    torch.reshape(full_difficulty_data, (1, 1, 25)).repeat((1, 100, 1)).shape
    return


if __name__ == "__main__":
    app.run()
