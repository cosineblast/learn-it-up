import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""
    # Training DDC models
    """)
    return


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
    return (validation_dataset,)


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
def _(DataLoader, training_dataset, validation_dataset):
    BATCH_SIZE = 256

    training_loader =  DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)
    return BATCH_SIZE, training_loader, validation_loader


@app.cell
def _(cnn_model, nn, torch):
    loss_fn = nn.BCEWithLogitsLoss()
    # TODO: pick better optimizer
    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=0.1, weight_decay=1.0)
    return loss_fn, optimizer


@app.cell
def _(training_loader):
    from itertools import islice
    from tqdm import tqdm
    thing = iter(training_loader)
    return


@app.cell
def _(
    BATCH_SIZE,
    cnn_model,
    device,
    loss_fn,
    mo,
    optimizer,
    torch,
    training_dataset,
    training_loader,
):
    from math import ceil 

    def train_epoch():
        cnn_model.train()

        size = ceil(len(training_dataset) / BATCH_SIZE)

        with mo.status.progress_bar(total=size, title='Training...', remove_on_exit=True) as bar:

            for batch, ((frames, difficulties), y) in enumerate(training_loader):
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
    
                if batch % 100 == 0:
                    bar.update(increment=100, subtitle=f'Batch {batch}/{size} Loss: {loss.item()}')


    return ceil, train_epoch


@app.cell
def _(train_epoch):
    train_epoch()
    return


@app.cell
def _(
    BATCH_SIZE,
    ceil,
    cnn_model,
    device,
    loss_fn,
    mo,
    torch,
    validation_dataset,
    validation_loader,
):
    from collections import namedtuple 

    def evaluate_validation():
        cnn_model.eval()

        size = ceil(len(validation_dataset) / BATCH_SIZE)

        loss_mean = torch.tensor(0.0).to(device)
        batch_count = 0

        total_positives = 0.0
        total_label_positives = 0.0
        total_true_positives = 0.0

        with mo.status.progress_bar(total=size, title='Validating...', remove_on_exit=True) as bar:
            with torch.no_grad():
                for batch, ((frames, difficulties), y) in enumerate(validation_loader):
                    # (Batch, 15, 80, 3) -> (Batch, 3, 15, 80)
                    frames = frames.transpose(1, 3).transpose(2, 3).float().to(device)
                    difficulties = difficulties.float().to(device)
                    y = y.float().to(device)
    
                    pred = cnn_model(frames, difficulties)
    
                    loss = loss_fn(pred, y)
    
                    loss_mean += torch.mean(loss)
    
                    positives = y == 1.0
                    label_positives = pred > 0.5
                    true_positives = positives & label_positives
    
                    total_positives += torch.sum(positives)
                    total_label_positives += torch.sum(label_positives)
                    total_true_positives = torch.sum(true_positives)
    
                    if batch % 100 == 0:
                        bar.update(increment=100, subtitle=f'Batch {batch}/{size} Loss: {loss.item()}')
    
                    batch_count += 1 

        precision = 0.0 if total_label_positives == 0 else total_true_positives / total_label_positives
        recall = total_true_positives / total_positives

        return (loss_mean / batch_count, precision, recall)




    return (evaluate_validation,)


@app.cell
def _(evaluate_validation):
    evaluate_validation()
    return


@app.cell
def _(cnn_model, evaluate_validation, torch, train_epoch):
    def train_epochs(epochs):
        losses = []

        for epoch in range(epochs):
            train_epoch()

            loss = evaluate_validation()

            losses.append(loss)

            print()
            print(f'epoch {epoch+1}/{epochs}. eval={loss}')

            torch.save(cnn_model.state_dict(), f'cnn_model_{epoch}.pth')
        return losses

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Testing DDC models with random data
    """)
    return


@app.cell
def _(device, np, torch):
    cnn_sample_data = torch.tensor(np.random.randn(10, 3, 15, 80)).type(torch.float32).to(device)
    cnn_sample_data.shape
    return (cnn_sample_data,)


@app.cell
def _(F, device, torch):
    cnn_difficulty_data = F.one_hot(torch.tensor([15]).repeat(10), num_classes=25).type(torch.float32).to(device)
    cnn_difficulty_data.shape
    return (cnn_difficulty_data,)


@app.cell
def _(cnn_difficulty_data, cnn_model, cnn_sample_data):
    cnn_model(cnn_sample_data, cnn_difficulty_data)
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
def _(full_difficulty_data, full_model, full_model_data):
    full_model(full_model_data, full_difficulty_data)
    return


if __name__ == "__main__":
    app.run()
