import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix
import numpy as np
import os


class OnlineOCCELoss(nn.Module):
    """
    Online anti-class cross-entropy with confusion matrix-based prior:
      L = - (1/(C-1)) * sum_{j != y} \tilde{y}_{y,j} * log softmax(-logits)_j
    where \tilde{y} = (1-a) * U + a * Prior_tau,
    U = uniform over anti-classes, Prior_tau = temperature-smoothed confusion-based prior.
    
    The prior is based on the confusion matrix from predictions.
    """
    def __init__(self, incr: int, device: torch.device, a: float = 0.5, 
                 eps: float = 1e-8, save_distribution: bool=False):
        """
        save_distribution: If True, saves the average prior distribution of every epoch to a file.
        """
        super().__init__()
        assert incr >= 2
        
        self.incr = incr
        self.a = float(a)              # trade-off prior vs uniform (anti-class)
        self.tau = float(1.0)          # temperature applied ONLY to prior
        self.eps = float(eps)
        
        self.current_classes = 0 
        self.task_id = -1
        self._device = device
        self.save_distribution = save_distribution
        self.soft_targets_over_epoch = []
        
        # Accumulators for predictions and targets to build confusion matrix
        self.epoch_predictions = []
        self.epoch_targets = []


    def compute_confusion_matrix_prior(self, y_true, y_pred):
        """
        Compute confusion matrix and convert it to a prior distribution.
        
        The confusion matrix shows P(predicted=j | true=i).
        For OCCE, we want P(anti-class=j | true=i, j!=i), normalized over wrong predictions.
        
        Args:
            y_true: numpy array of true labels
            y_pred: numpy array of predicted labels
        
        Returns:
            prior: torch tensor of shape [C, C] where prior[i,j] is the probability
                   of predicting class j when true class is i (with diagonal set to 0)
        """
        # Compute confusion matrix: cm[i,j] = count of (true=i, pred=j)
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(self.current_classes))
        
        # Convert to percentages (row-wise normalization)
        # cm_percentage[i,j] = P(pred=j | true=i)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)  # avoid division by zero
        cm_percentage = cm.astype('float') / row_sums

        # Set diagonal to 0 (we don't want correct predictions in anti-class distribution)
        np.fill_diagonal(cm_percentage, 0.0)
        
        # Re-normalize rows to sum to 1 (only over wrong predictions)
        row_sums = cm_percentage.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        cm_percentage = cm_percentage / row_sums
        
        # Convert to torch tensor
        prior = torch.from_numpy(cm_percentage).to(
            dtype=torch.float32, device=self._device
        )
        
        # Ensure numerical stability
        prior = prior.clamp_min(self.eps)
        prior.fill_diagonal_(0.0)
        row_sum = prior.sum(dim=1, keepdim=True).clamp_min(self.eps)
        prior = prior / row_sum
        
        return prior
    
    def update_prior_from_confusion_matrix(self):
        """
        Update the prior matrix based on accumulated predictions and targets.
        """
        if len(self.epoch_predictions) == 0 or len(self.epoch_targets) == 0:
            # No predictions yet, keep uniform prior
            return
        
        # Concatenate all predictions and targets from this epoch
        y_pred = np.concatenate(self.epoch_predictions)
        y_true = np.concatenate(self.epoch_targets)


        # Compute confusion matrix-based prior
        confusion_prior = self.compute_confusion_matrix_prior(y_true, y_pred)

        # Update the prior
        self.prior.copy_(confusion_prior)
        self.soft_targets_over_epoch.append(confusion_prior.cpu())
 
    @torch.no_grad()
    def step(self, logits: torch.Tensor, target: torch.Tensor): 
        """
        Accumulate predictions and targets to compute confusion matrix.
        
        Args:
            logits: Model logits [B, C]
            target: Target labels [B]
        """
        # Get predicted classes
        pred_classes = logits.argmax(dim=1)  # [B]
        
        # Store predictions and targets (move to CPU for efficiency)
        self.epoch_predictions.append(pred_classes.cpu().numpy())
        self.epoch_targets.append(target.cpu().numpy())
 
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute OCCE with current prior.
        Temperature is applied only to the prior branch.
        """
        B, C = logits.shape
        assert C == self.current_classes
 
        # anti-class log-probs
        log_ycomp = F.log_softmax(-logits, dim=1)  # [B, C]
 
        # Uniform anti-class distribution U_y (zero on diag, 1/(C-1) elsewhere)
        U = torch.ones((B, C), device=logits.device, dtype=logits.dtype)
        U.scatter_(1, target.unsqueeze(1), 0.0)
        U = U / (C - 1)
 
        # Temperature-smoothed prior rows for each target class
        P = self.prior.index_select(0, target)  # [B, C]
        if self.tau != 1.0:
            # Smooth via log/tau then softmax
            P = P.clamp_min(self.eps)
            P = (P.log() / self.tau).softmax(dim=1)
        
        # Force exact zeros on diagonal
        P.scatter_(1, target.unsqueeze(1), 0.0)
        P = P / P.sum(dim=1, keepdim=True).clamp_min(self.eps)
 
        # Mix prior with uniform over anti-classes
        Tmix = (1.0 - self.a) * U + self.a * P  # [B, C]
 
        # One-cold CE
        loss = - (Tmix * log_ycomp).sum(dim=1)
        return loss.mean()
 
    @torch.no_grad()
    def next_epoch(self):
        """
        Compute confusion matrix from accumulated predictions and update prior.
        Then reset accumulators.
        """
        # Update prior based on confusion matrix
        self.update_prior_from_confusion_matrix()
        
        # Reset accumulators
        self.epoch_predictions = []
        self.epoch_targets = []
    
    def before_task(self, current_classes):
        self.current_classes = current_classes
        self.task_id += 1
        self.soft_targets_over_epoch = []
        self.epoch_predictions = []
        self.epoch_targets = []

        if self.task_id == 0:
            # Prior: row-stochastic, zero diagonal; init uniform off-diagonal
            prior = torch.full((self.current_classes, self.current_classes), 
                             1.0 / (self.current_classes - 1))
            prior.fill_diagonal_(0.0)
            self.register_buffer("prior", prior.to(self._device))  # [C, C]
        else:
            # Delete old buffer
            delattr(self, 'prior')

            # Create new prior with different size
            new_prior = torch.full((self.current_classes, self.current_classes), 
                                  1.0 / (self.current_classes - 1))
            new_prior.fill_diagonal_(0.0)

            # Re-register
            self.register_buffer("prior", new_prior.to(self._device))

    def visualize_soft_targets_heatmap(self, filename, task, title="Soft Targets Heatmap"):
        """
        Visualizes the soft targets as a heatmap with integer labels for the axes.
        """
        import seaborn as sns
        import matplotlib.pyplot as plt

        if not os.path.exists(filename + 'soft_targets_heatmap/'):
            os.makedirs(filename + 'soft_targets_heatmap/')

        soft_targets_array = np.stack(self.soft_targets_over_epoch, axis=0)
        soft_targets_ema = np.mean(soft_targets_array, axis=0)

        # SAVE TARGETS
        np.savez(filename + 'soft_targets_heatmap/task_{}.npz'.format(task), 
                soft_targets_ema=soft_targets_ema)
        print(f"Average soft targets saved.")

        soft_targets = soft_targets_ema
        N = soft_targets.shape[0]
        
        # Create integer labels for the axes
        tick_labels = np.arange(N)
        
        # Add grid lines every 10 classes
        fig, ax = plt.subplots(figsize=(50, 50))
        sns.heatmap(
            soft_targets,
            xticklabels=tick_labels,
            yticklabels=tick_labels,
            cmap="viridis",
            square=True,
            cbar=True,
            annot=True,
            fmt=".3f",
            ax=ax
        )
        
        # Add grid lines every 10 classes
        grid_positions = [10 * x for x in range(1, (N // 10) + 1)]
        for pos in grid_positions:
            ax.axhline(y=pos, color='red', linewidth=2, linestyle='-')
            ax.axvline(x=pos, color='red', linewidth=2, linestyle='-')
        
        plt.xlabel("Predicted / Soft Class")
        plt.ylabel("True Class")
        plt.title(title)
        plt.savefig(filename + 'soft_targets_heatmap/task_{}.png'.format(task), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Soft targets heatmap saved for task {task}.")