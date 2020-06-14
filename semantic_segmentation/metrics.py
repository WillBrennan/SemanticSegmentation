import torch
from ignite import metrics


def thresholded_transform(threshold: float):
    def _fn(items):
        y_pred, y = items
        y = torch.round(y).int()
        y_pred = (y_pred > threshold).int()

        return y_pred, y

    return _fn


class IoUMetric(metrics.Metric):
    def __init__(self, output_transform, device=None):
        super().__init__(output_transform=output_transform, device=device)

    def reset(self):
        self.num_intersection = 0
        self.num_total = 0

    def update(self, output):
        y_pred, y = output

        num_total = (y_pred == 1).sum() + (y == 1).sum()
        num_intersection = ((y_pred == 1) & (y == 1)).sum()

        self.num_total += num_total.item()
        self.num_intersection += num_intersection.item()

    def compute(self):
        return self.num_intersection / max(1, self.num_total - self.num_intersection)
