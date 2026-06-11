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


    return F, loading, mo, models, nn, np, pickle, torch


@app.cell
def _(torch):
    print('HasCuda:', torch.cuda.is_available())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return (device,)


@app.cell(hide_code=True)
def _():
    import json

    with open('data/partitions.json') as _f:
        partitions = json.load(_f)

    training_paths   = partitions['training']
    test_paths       = partitions['testing']
    validation_paths = partitions['validation']
    return test_paths, training_paths, validation_paths


@app.cell(hide_code=True)
def _(pickle, test_paths, training_paths, validation_paths):

    def _get_stepfiles(paths):
        files = []
        for _path in paths:
            with open(_path, 'rb') as _f:
                files.append(pickle.load(_f))
        return files


    training_stepfiles   = _get_stepfiles(training_paths)
    test_stepfiles       = _get_stepfiles(test_paths)
    validation_stepfiles = _get_stepfiles(validation_paths)
    return test_stepfiles, training_stepfiles, validation_stepfiles


@app.cell(hide_code=True)
def _():
    from pathlib import Path

    def get_feature_path_for(refined_stepfile_path):
        assert str(refined_stepfile_path).endswith(".ssc.bin")
        return Path("data/features") / (Path(Path(refined_stepfile_path).stem).stem + ".feat.bin")


    return


@app.cell
def _(loading):
    audio_loader = loading.LoadFeaturesCached()
    return (audio_loader,)


@app.cell
def _(audio_loader, loading, training_paths, training_stepfiles):
    training_dataset = loading.PumpItUpConvolutionCNNOnsetDataset(training_stepfiles, training_paths, audio_loader)
    len(training_dataset)
    return (training_dataset,)


@app.cell
def _(audio_loader, loading, test_paths, test_stepfiles):
    testing_dataset = loading.PumpItUpConvolutionCNNOnsetDataset(test_stepfiles, test_paths, audio_loader)
    len(testing_dataset)
    return


@app.cell
def _(audio_loader, loading, validation_paths, validation_stepfiles):
    validation_dataset = loading.PumpItUpConvolutionCNNOnsetDataset(validation_stepfiles, validation_paths, audio_loader)
    len(validation_dataset)
    return


@app.cell
def _(device, models):
    cnn_model = models.PumpItUpConvolutionCNNOnset().float().to(device)
    cnn_model
    return (cnn_model,)


@app.cell
def _():
    from torch.utils.data import DataLoader

    return (DataLoader,)


@app.cell
def _(DataLoader, training_dataset):
    training_loader =  DataLoader(training_dataset, batch_size=256, shuffle=True)
    return (training_loader,)


@app.cell
def _(cnn_model, nn, torch):
    loss_fn = nn.BCEWithLogitsLoss()
    # TODO: pick better optimizer
    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=1e-1)
    return loss_fn, optimizer


@app.cell
def _(training_loader):
    from itertools import islice
    from tqdm import tqdm
    thing = iter(training_loader)
    return islice, thing, tqdm


@app.cell
def _(cnn_model, device, islice, loss_fn, optimizer, thing, torch, tqdm):


    def train_epoch():
        size = 100

        bar = tqdm(enumerate(islice(thing, size)), total=size)
        for batch, ((frames, difficulties), y) in bar:
            # (Batch, 15, 80, 3) -> (Batch, 3, 15, 80)
            frames = frames.transpose(1, 3).transpose(2, 3).float().to(device)
            difficulties = difficulties.float().to(device)
            y = y.float().to(device)

            pred = cnn_model(frames, difficulties)

            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()

            torch.nn.utils.clip_grad_norm_(cnn_model.parameters(), max_norm=5.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad()

            if batch % 10 == 0:
                bar.set_description(f'batch: {batch} loss: {loss.item()} Progress')


    return (train_epoch,)


@app.cell
def _(train_epoch):

    train_epoch()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Testing DDC models with random data
    """)
    return


@app.cell
def _(np, torch):
    cnn_sample_data = torch.tensor(np.random.randn(10, 3, 15, 80)).type(torch.float32)
    cnn_sample_data.shape
    return (cnn_sample_data,)


@app.cell
def _(F, torch):
    cnn_difficulty_data = F.one_hot(torch.tensor([15]).repeat(10), num_classes=25).type(torch.float32)
    cnn_difficulty_data.shape
    return (cnn_difficulty_data,)


@app.cell
def _(cnn_difficulty_data, cnn_model, cnn_sample_data):
    cnn_model(cnn_sample_data, cnn_difficulty_data)
    return


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
