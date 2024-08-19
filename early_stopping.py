import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, val_loss_min=np.Inf):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = -val_loss_min
        self.early_stop = False
        self.val_loss_min = val_loss_min
        self.delta = delta

    def __call__(self, val_loss, model, epoch, optimizer, scheduler, scaler, train_losses, val_losses):
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

        torch.save(state, 'checkpoint.pt')
        torch.save(model.state_dict(), 'checkpoint_weights.pth')
        self.val_loss_min = val_loss

