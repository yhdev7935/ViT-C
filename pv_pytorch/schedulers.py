
import numpy as np

class WarmupScheduler():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, warmup_period = 4000, initial_steps = 0):

        self._optimizer    = optimizer
        self.warmup_period = warmup_period
        # self.lr_max        = lr_max
        self.n_steps       = initial_steps

        self.d_model       = d_model

        lr = self.get_lr()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


    def get_lr(self):
        #np.power(self.d_model, -0.5)
        return np.power(self.d_model, -0.5)*np.min([np.power(self.n_steps, -0.5) if self.n_steps > 0 else 1, self.n_steps*np.power(self.warmup_period, -1.5)])#(self.n_steps / self.warmup_period) * self.lr_max


    def step(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.get_lr()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
