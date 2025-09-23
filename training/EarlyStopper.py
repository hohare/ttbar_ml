

class EarlyStopper:
    def __init__(self, tolerance, delta):
        self.tolerance = tolerance
        self.min_delta = delta

        self.best_loss = None
        self.stop_early = False
        self.counter = 0

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.stop_early = True
        else:
            self.best_loss = loss
            self.counter = 0