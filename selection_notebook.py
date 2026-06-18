import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Training Learn it Up DDC selection models
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
    return test_stepfiles, training_stepfiles, validation_stepfiles


@app.cell
def _():
    UNROLL = 100
    BATCH_SIZE = 64
    return BATCH_SIZE, UNROLL


@app.cell
def _(UNROLL, loading, training_stepfiles):
    training_dataset = loading.PumpItUpConvolutionSelectionLSTMDataset(training_stepfiles, unroll_length=UNROLL,
                                                                       transform=loading.MaskAndPaddingTransform(UNROLL))
    len(training_dataset)
    return (training_dataset,)


@app.cell
def _(UNROLL, loading, test_stepfiles):
    testing_dataset = loading.PumpItUpConvolutionSelectionLSTMDataset(test_stepfiles, unroll_length=UNROLL)
    len(testing_dataset)
    return


@app.cell
def _(BATCH_SIZE, DataLoader, training_dataset):
    training_loader =  DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return (training_loader,)


@app.cell
def _(device, models):
    Model = models.PumpItUpConvolutionSelectionLSTM 
    selection_model = Model().float().to(device)
    selection_model
    return (selection_model,)


@app.cell
def _(nn, selection_model, torch):
    # reduction=none is because we want the loss funciton to spit the loss for each element of the input sequence, 
    # so we can apply the mask due to shorter sequences
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(selection_model.parameters(), lr=0.001, weight_decay=1e-4)
    return loss_fn, optimizer


@app.cell
def _(
    BATCH_SIZE,
    device,
    loss_fn,
    mo,
    optimizer,
    selection_model,
    torch,
    training_dataset,
    training_loader,
):
    from math import ceil 

    def train_epoch():
        selection_model.train()

        size = ceil(len(training_dataset) / BATCH_SIZE)

        with mo.status.progress_bar(total=size, title='Training...', remove_on_exit=True) as bar:

            for batch, (x, deltas, y, mask) in enumerate(training_loader):
                batch_size = x.shape[0]

                x = x.float().to(device)
                deltas = deltas.float().to(device)
                mask = mask.float().to(device)
                y = y.to(device)

                pred = selection_model(x, deltas)

                y = torch.flatten(y, start_dim=0, end_dim=1)
                pred = torch.flatten(pred, start_dim=0, end_dim=1)
                mask = torch.flatten(mask, start_dim=0, end_dim=1)

                loss_per_batch = loss_fn(pred, y)
                loss = torch.mean(loss_per_batch * mask)

                loss.backward()

                #torch.nn.utils.clip_grad_norm_(cnn_model.parameters(), max_norm=5.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad()

                bar.update(subtitle=f'Batch {batch}/{size} Loss: {loss.item()}')

    return (train_epoch,)


@app.cell
def _(
    evaluate_validation_per_chart,
    np,
    selection_model,
    should_save_models,
    should_validate_training,
    torch,
    train_epoch,
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

            if result.avg_accuracy > best_score and should_save_models.value:
                torch.save(selection_model.state_dict(), f'selection_model_{epoch+1}.pth')
                best_score = result.avg_accuracy
            
            if epoch % 10 == 0 and should_validate_training.value:
                result = evaluate_validation_per_chart(training_stepfiles)
                training_losses.append(result)
                print(f'epoch {epoch+1}/{epochs}. training evaluation={result}')

        return losses, training_losses

    return (train_epochs,)


@app.cell(hide_code=True)
def _(mo):
    should_train = mo.ui.run_button(label='Run')
    should_save_models = mo.ui.checkbox(label='Save best models')
    should_validate_training = mo.ui.checkbox(label='Validate models in training set sometimes')

    epoch_count = mo.ui.number()

    mo.center(
        mo.vstack([
            mo.md('Train epochs'),
            epoch_count,
            should_save_models,
            should_validate_training,
            should_train    
        ])
    )

    return (
        epoch_count,
        should_save_models,
        should_train,
        should_validate_training,
    )


@app.cell(hide_code=True)
def _(epoch_count, should_train, train_epochs):
    if should_train.value:
        train_epochs(epoch_count.value)
    return


@app.cell
def _(
    device,
    evaluation,
    loss_fn,
    mo,
    namedtuple,
    selection_model,
    validation_stepfiles,
):
    FullStepEvaluation = namedtuple('FullStepEvaluation', 
                                ['avg_loss', 'avg_accuracy'])

    def evaluate_validation_per_chart(stepfiles=validation_stepfiles):
        chart_count = sum(len(stepfile.charts) for stepfile in stepfiles)

        sum_mean_loss = 0
        sum_accuracy = 0

        selection_model.eval()

        with mo.status.progress_bar(total=chart_count, title='Validating', remove_on_exit=True) as bar:
            for stepfile in stepfiles:
                for chart in stepfile.charts:
                    result = evaluation.measure_selection_performance(selection_model, chart, loss_fn, device)

                    sum_mean_loss += result.mean_loss
                    sum_accuracy += result.accuracy

                    bar.update()

        return FullStepEvaluation(sum_mean_loss / chart_count, sum_accuracy / chart_count)

    return (evaluate_validation_per_chart,)


@app.cell(hide_code=True)
def _(mo):
    should_evaluate = mo.ui.run_button(label='Evaluate model')
    should_evaluate

    should_save = mo.ui.run_button(label='Save model')

    mo.center(
    mo.vstack([should_evaluate, should_save])
    )
    return should_evaluate, should_save


@app.cell(hide_code=True)
def _(
    evaluate_validation_per_chart,
    selection_model,
    should_evaluate,
    should_save,
    torch,
):
    if should_evaluate.value:
        evaluate_validation_per_chart()

    if should_save.value:
        _path = 'selection_model.pth' 
        torch.save(selection_model, _path)
        print('Saved model to', _path)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Testing DDC models with random data
    """)
    return


@app.cell
def _(loading, training_stepfiles):
    selection_dataset = loading.PumpItUpConvolutionSelectionLSTMDataset(training_stepfiles, 100)
    return


@app.cell
def _(device, np, torch):
    step_data = torch.tensor(np.random.rand(20, 10, 5, 4) > 0.5).float().to(device)
    step_data.shape
    return (step_data,)


@app.cell
def _(device, np, torch):
    delta_time_data = torch.tensor(np.random.rand(20, 10, 3)).float().to(device)
    delta_time_data.shape
    return (delta_time_data,)


@app.cell
def _(delta_time_data, selection_model, step_data):
    selection_model(step_data, delta_time_data)
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
    import models
    import json
    import evaluation
    from collections import namedtuple
    from torch.utils.data import DataLoader

    return (
        DataLoader,
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
