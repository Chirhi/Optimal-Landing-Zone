import os
import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, val_loss_min=np.Inf):
        """
        Initializes an instance of the EarlyStopping class with the following parameters:
        
        Args:
            patience (int, optional): The number of epochs to wait for improvement in validation loss before stopping training. Defaults to 5.
            verbose (bool, optional): Whether to print messages during training. Defaults to False.
            delta (float, optional): The threshold for improvement in validation loss. Defaults to 0.
            val_loss_min (float, optional): The minimum validation loss. Defaults to positive infinity.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = -val_loss_min
        self.early_stop = False
        self.val_loss_min = val_loss_min
        self.delta = delta

    def __call__(self, val_loss, model, epoch, optimizer, scheduler, scaler, train_losses, val_losses):
        """
        Calls the EarlyStopping instance to check if the model's validation loss has improved.

        Args:
            val_loss (float): The current validation loss of the model.
            model (torch.nn.Module): The model being trained.
            epoch (int): The current epoch number.
            optimizer (torch.optim.Optimizer): The optimizer being used to train the model.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler being used.
            scaler (torch.cuda.amp.GradScaler): The gradient scaler being used.
            train_losses (list): A list of training losses.
            val_losses (list): A list of validation losses.
        """
        score = -val_loss

        if score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}. Validation loss: {val_loss:.6f}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer, scheduler, scaler, train_losses, val_losses)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, optimizer, scheduler, scaler, train_losses, val_losses):
        """
        Saves a checkpoint of the model during training.

        Args:
            val_loss (float): The current validation loss of the model.
            model (torch.nn.Module): The model being trained.
            epoch (int): The current epoch number.
            optimizer (torch.optim.Optimizer): The optimizer being used to train the model.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler being used.
            scaler (torch.cuda.amp.GradScaler): The gradient scaler being used.
            train_losses (list): A list of training losses.
            val_losses (list): A list of validation losses.
            full_save (bool, optional): Whether to save the full model with metrics. Defaults to False.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')

        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
        }

        os.makedirs('models', exist_ok=True)

        torch.save(state, 'models/checkpoint.pt')
        torch.save(model.state_dict(), 'models/checkpoint_weights.pth')

        self.val_loss_min = val_loss
