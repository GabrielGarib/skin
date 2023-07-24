import numpy as np
import os
import time
import warnings
import copy

import torch
import torch.nn as nn

def accuracy(predictions, labels):
    #round predictions to the closest integer
    probs    = torch.softmax(predictions, dim=1)
    winners  = probs.argmax(dim=1)
    corrects = (winners == labels).float()
    acc      = corrects.sum() / len(corrects)
    return acc

def train_epoch(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc  = 0
    model.train()
    for batch in iterator:

        optimizer.zero_grad()

        img, label, metadata = batch

        img      = img.to(device)
        label    = label.to(device)
        metadata = metadata.to(device)

        predictions = model(img, metadata)

        loss = criterion(predictions, label)
        acc  = accuracy(predictions, label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc  += acc.item()

    return epoch_loss/len(iterator), epoch_acc/len(iterator)

def eval_epoch(model, iterator, criterion, device):

    #initialize every epoch
    epoch_loss = 0
    epoch_acc  = 0

    #deactivating dropout layers
    model.eval()

    #deactivates autograd
    with torch.no_grad():
        for batch in iterator:

            img, label, metadata = batch
            img      = img.to(device)
            label    = label.to(device)
            metadata = metadata.to(device)

            predictions = model(img, metadata)

            #compute loss and accuracy
            loss = criterion(predictions, label)
            acc  = accuracy(predictions, label)

            #keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc  += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def train(model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, device, n_epochs,
          saved_models_dir, saved_scores_dir, save_path, early_stop=15, printfreq=10, temp_save=False):

    train_loss_hist = []
    train_acc_hist  = []
    val_loss_hist   = []
    val_acc_hist    = []

    best_val_loss  = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    early_stop_count = 0

    t0 = time.time()
    print(' Epoch    Train Loss    Val Loss    Train Acc    Val Acc    Best      lr      Time [min]')
    print('-'*89)

    for epoch in range(n_epochs):
        t1 = time.time()
        st = '        '

        # Training metrics
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, criterion, device)
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)

        # Validation metrics
        val_loss, val_acc     = eval_epoch(model, val_dataloader, criterion, device)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

        # Update best model
        if val_loss < best_val_loss:
            early_stop_count = 0 # reset count
            st = '     ***'
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(saved_models_dir, 'best_' + save_path))
        else:
            early_stop_count += 1
            
        epoch_lr = optimizer.param_groups[0]['lr']
        if (epoch + 1) % printfreq == 0:
            t2 = (time.time() - t1)/60
            s = f'{epoch+1:6}{train_loss:12.4f}{val_loss:13.4f}{train_acc:13.4f}{val_acc:12.4f}{st}{epoch_lr:11.1e}{t2:10.1f}'
            print(s)
            
        if temp_save:
            # Save current model and training information
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(saved_models_dir, save_path))

        # Update learning rate
        #scheduler.step()  #original: torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        scheduler.step(val_loss) # optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        # Save the loss and accuracy values
        np.save(os.path.join(saved_scores_dir, f'{save_path}_train_loss'), train_loss_hist)
        np.save(os.path.join(saved_scores_dir, f'{save_path}_train_acc'), train_acc_hist)
        np.save(os.path.join(saved_scores_dir, f'{save_path}_val_loss'), val_loss_hist)
        np.save(os.path.join(saved_scores_dir, f'{save_path}_val_acc'), val_acc_hist)

        if (early_stop is not None) and (early_stop_count >= early_stop):
            print('Training stopped early')
            break

    tfinal = (time.time() - t0)/60
    print('-'*89)
    print(f'Total time [min] for {epoch + 1} Epochs: {tfinal:.1f}')
    return
