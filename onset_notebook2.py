import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Training DDC onset models
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Loading dataset
    """)
    return


@app.cell(hide_code=True)
def _(json):
    # Querying dataset partition splits

    with open('data/partitions.json') as _f:
        partitions = json.load(_f)

    training_paths   = partitions['training']
    test_paths       = partitions['testing']
    validation_paths = partitions['validation']
    return test_paths, training_paths, validation_paths


@app.cell(hide_code=True)
def _(pickle, test_paths, training_paths, validation_paths):
    # Loading refined stepfiles

    def _get_stepfiles(paths):
        files = []
        for _path in paths:
            with open(_path, 'rb') as _f:
                files.append(pickle.load(_f))
        return files


    training_stepfiles   = _get_stepfiles(training_paths)
    test_stepfiles       = _get_stepfiles(test_paths)
    validation_stepfiles = _get_stepfiles(validation_paths)
    return training_stepfiles, validation_stepfiles


@app.cell(hide_code=True)
def _(loading, test_paths, training_paths, validation_paths):
    # Loading audio features

    from pathlib import Path

    def _get_feature_path_for(refined_stepfile_path):
        assert str(refined_stepfile_path).endswith(".ssc.bin")
        return Path("data/features") / (Path(Path(refined_stepfile_path).stem).stem + ".feat.bin")

    audio_loader = loading.LoadFeaturesCached()

    training_features = [audio_loader(_get_feature_path_for(path)) for path in training_paths]
    validation_features = [audio_loader(_get_feature_path_for(path)) for path in validation_paths]
    test_features = [audio_loader(_get_feature_path_for(path)) for path in test_paths]
    return training_features, validation_features


@app.cell
def _(loading, training_features, training_stepfiles):
    UNROLL_SIZE = 16

    training_dataset = loading.ppc.PPC_AlignedOnsetDataset(
        training_stepfiles, training_features, 
        UNROLL_SIZE, 
        transform=loading.MaskAndPaddingTransform(UNROLL_SIZE, skip=1)
    )
    len(training_dataset)
    return (training_dataset,)


@app.cell(hide_code=True)
def _(mo):
    measure_time_button = mo.ui.run_button(label='Measure dataset iteration time')

    mo.center(measure_time_button)
    return (measure_time_button,)


@app.cell(hide_code=True)
def _(measure_time_button, mo, training_dataset):
    if measure_time_button.value:
        with mo.status.progress_bar(total=len(training_dataset)) as bar:
            for i in range(len(training_dataset)):
                _ = training_dataset[i]

                if i % 10 == 0:
                    bar.update(increment=10)
    return


@app.cell
def _(DataLoader, training_dataset):
    BATCH_SIZE = 32

    training_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return BATCH_SIZE, training_loader


@app.cell
def _(device, models):
    model = models.ppc.PumpPumpConvolutionAlignedOnset().float().to(device)
    model
    return (model,)


@app.cell
def _(model, nn, torch):
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    return loss_fn, optimizer


@app.cell
def _(
    BATCH_SIZE,
    device,
    loss_fn,
    mo,
    model,
    optimizer,
    torch,
    training_dataset,
    training_loader,
):
    from math import ceil 

    def train_epoch():
        model.train()

        size = ceil(len(training_dataset) / BATCH_SIZE)

        with mo.status.progress_bar(total=size, title='Training...', remove_on_exit=True) as bar:

            for batch, (x, nps, bpms, y, mask) in enumerate(training_loader):
                x = x.to(device) # (batch x unroll x 5 32 x 80 x 3)
                nps = nps.float().to(device) # (batch)
                bpms = bpms.float().to(device) # (batch x unroll)
                y = y.float().to(device) # (batch x unroll x 48)
                mask = mask.float().to(device) # (batch x unroll)

                pred = model(x, nps, bpms)

                y = torch.flatten(y, start_dim=0, end_dim=1)
                pred = torch.flatten(pred, start_dim=0, end_dim=1)
                mask = torch.flatten(mask, start_dim=0, end_dim=1)  # (batch * unroll)

                loss_per_batch = loss_fn(pred, y) # (batch * unroll x 48)

                loss = torch.mean(loss_per_batch.transpose(0, 1) * mask)

                # Backpropagation
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad()

                bar.update(increment=1, subtitle=f'Batch {batch}/{size} Loss: {loss.item()}')


    return (train_epoch,)


@app.cell
def _(
    evaluate_validation_per_chart,
    model,
    np,
    should_save_models,
    should_validate_training,
    torch,
    train_epoch,
    training_features,
    training_stepfiles,
):
    def train_epochs(epochs):
        losses = []
        training_losses = []
        best_score = -np.inf

        for epoch in range(epochs):
            train_epoch()

            result = evaluate_validation_per_chart()

            losses.append(result)

            print()
            print(f'epoch {epoch+1}/{epochs}. evaluation={result}')

            if result.avg_aligned_auc_score > best_score and should_save_models.value:
                torch.save(model.state_dict(), f'best_onset_model.pth')
                best_score = result.avg_aligned_auc_score

            if epoch % 10 == 0 and should_validate_training.value:
                result = evaluate_validation_per_chart(training_stepfiles, training_features)
                training_losses.append(result)
                print(f'epoch {epoch+1}/{epochs}. training evaluation={result}')

        return losses, training_losses

    return (train_epochs,)


@app.cell(hide_code=True)
def _(mo):
    epoch_count_input = mo.ui.number()
    should_save_models = mo.ui.checkbox(label='Save best models')
    should_validate_training = mo.ui.checkbox(label='Validate models in training set sometimes')
    train_button = mo.ui.run_button(label='Train')


    mo.center(
        mo.vstack([
            mo.md('Train epochs'),
            epoch_count_input,
            should_save_models,
            should_validate_training,
            train_button
        ])
    )
    return (
        epoch_count_input,
        should_save_models,
        should_validate_training,
        train_button,
    )


@app.cell
def _(epoch_count_input, train_button, train_epochs):
    if train_button.value:
        metrics = train_epochs(epoch_count_input.value)
    return


@app.cell
def _(
    defaultdict,
    device,
    evaluation,
    loss_fn,
    mo,
    model,
    namedtuple,
    np,
    validation_features,
    validation_stepfiles,
):
    FullEvaluation = namedtuple('FullEvaluation', 
                                ['avg_precision',  'avg_recall', 'avg_fscore', 
                                 'avg_loss', 'avg_raw_auc_score', 'avg_aligned_auc_score', 'avg_accuracy', 'avg_best_thresholds'])

    def evaluate_validation_per_chart(stepfiles=validation_stepfiles, file_features=validation_features):
        chart_count = sum(len(stepfile.charts) for stepfile in stepfiles)

        sum_precision = 0
        sum_recall = 0
        sum_fscore = 0

        sum_mean_loss = 0

        sum_raw_auc = 0
        sum_aligned_auc = 0
        sum_accuracy = 0

        best_thresholds = defaultdict(list)

        model.eval()

        with mo.status.progress_bar(total=chart_count, title='Validating', remove_on_exit=True) as bar:
            for stepfile, features in zip(stepfiles, file_features):

                for chart in stepfile.charts:
                    result = evaluation.measure_aligned_onset_performance(model, chart, features, loss_fn, device)

                    sum_precision += result.precision 
                    sum_recall    += result.recall 
                    sum_fscore    += result.fscore

                    sum_mean_loss += result.mean_loss

                    sum_raw_auc += result.raw_auc_score
                    sum_aligned_auc += result.aligned_auc_score
                    sum_accuracy += result.accuracy

                    best_thresholds[chart.difficulty].append(result.threshold_ideal)

                    bar.update()

        return FullEvaluation(sum_precision / chart_count,  sum_recall / chart_count,  sum_fscore / chart_count,
                              sum_mean_loss / chart_count,
                              sum_raw_auc / chart_count, sum_aligned_auc / chart_count,
                              sum_accuracy / chart_count,
                              {difficulty:np.mean(best_thresholds[difficulty]) for difficulty in best_thresholds})

    return (evaluate_validation_per_chart,)


@app.cell(hide_code=True)
def _(mo):
    evaluate_button = mo.ui.run_button(label='Evaluate')
    save_button = mo.ui.run_button(label='Save')

    mo.center(
        mo.vstack([
            evaluate_button, 
            save_button
        ])
    )
    return evaluate_button, save_button


@app.cell
def _(
    evaluate_button,
    evaluate_validation_per_chart,
    mo,
    model,
    save_button,
    torch,
):
    if evaluate_button.value:
        mo.output.append(evaluate_validation_per_chart()._asdict())

    if save_button.value:
        _path = 'onset_model.pth'
        torch.save(model.state_dict(), _path)
        print('Saved model to ', _path)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Testing DDC models with random data
    """)
    return


@app.cell
def _(device, models):
    sample_aligned_model = models.ppc.PumpPumpConvolutionAlignedOnset().float().to(device)
    return (sample_aligned_model,)


@app.cell
def _(device, np, torch):
    sample_aligned_x = torch.tensor(np.random.standard_normal((10, 8, 5, 32, 80, 3))).float().to(device)
    sample_aligned_x.shape
    return (sample_aligned_x,)


@app.cell
def _(device, np, torch):
    sample_nps = torch.tensor(np.random.standard_normal((10,))).float().to(device)
    sample_nps.shape
    return (sample_nps,)


@app.cell
def _(device, np, torch):
    sample_bpms = torch.tensor(np.random.standard_normal((10, 8))).float().to(device)
    sample_bpms.shape
    return (sample_bpms,)


@app.cell
def _(sample_aligned_model, sample_aligned_x, sample_bpms, sample_nps):
    sample_aligned_model(sample_aligned_x, sample_nps, sample_bpms)
    return


@app.cell
def _(device, torch, training_dataset):
    thing_x, thing_nps, thing_bpms, thign_y, _mask = training_dataset[0]

    thing_x = torch.tensor(thing_x[None, :]).float().to(device)
    thing_nps = torch.tensor([thing_nps]).float().to(device)
    thing_bpms = torch.tensor(thing_bpms[None, :]).float().to(device)

    print(thing_x.shape)
    print(thing_nps)
    print(thing_bpms.shape)
    return thing_bpms, thing_nps, thing_x


@app.cell
def _(model, thing_bpms, thing_nps, thing_x):
    model(thing_x, thing_nps, thing_bpms)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Appendix
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
    import loading.ppc
    import models
    import models.ppc
    import json
    import evaluation
    from collections import namedtuple, defaultdict
    from torch.utils.data import DataLoader

    return (
        DataLoader,
        defaultdict,
        evaluation,
        json,
        loading,
        mo,
        models,
        namedtuple,
        nn,
        np,
        pickle,
        torch,
    )


@app.cell(hide_code=True)
def _(torch):
    print('HasCuda:', torch.cuda.is_available())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return (device,)


if __name__ == "__main__":
    app.run()
