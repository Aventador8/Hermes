"""
preds.shape: (N,1,H,W) || (N,1,H,W,D) || (N,H,W) || (N,H,W,D)
labels.shape: (N,1,H,W) || (N,1,H,W,D) || (N,H,W) || (N,H,W,D)
"""

import torch

class Dice:

    def __init__(self, name='Dice', class_indexs=[1], class_names=['xx']) -> None:
        super().__init__()
        self.name = name
        self.class_indexs = class_indexs
        self.class_names = class_names
    def __call__(self, preds, labels):
        res = {}
        preds = torch.argmax(torch.softmax(preds, dim=1),dim=1)
        for class_index, class_name in zip(self.class_indexs, self.class_names):
            preds_ = (preds == class_index).to(torch.int)
            labels_ = (labels == class_index).to(torch.int)
            # print('preds_shape:', preds_.shape)
            # print('labels_shape:', labels_.shape)
            intersection = (preds_ * labels_).sum()
            try:
                res[class_name] = (2 * intersection) / (preds_.sum() + labels_.sum() + 1e-5).item()
            except ZeroDivisionError:
                res[class_name] = 1.0
            # res[class_name] = metric.dc(preds_.cpu().numpy(), labels_.cpu().numpy())
        return res

class Jaccard:
    def __init__(self, name='Jaccard', class_indexs=[1], class_names=['xx']) -> None:
        super().__init__()
        self.name = name
        self.class_indexs = class_indexs
        self.class_names = class_names

    def __call__(self, preds, labels):
        res = {}
        preds = torch.argmax(torch.softmax(preds, dim=1), dim=1)
        for class_index, class_name in zip(self.class_indexs, self.class_names):
            preds_ = (preds == class_index).to(torch.int)
            labels_ = (labels == class_index).to(torch.int)
            intersection = (preds_ * labels_).sum()
            union = ((preds_ + labels_) != 0).sum()
            # res[class_name] = intersection / ( union + 1e-5 )
            try:
                res[class_name] = intersection / ( union + 1e-5)
            except ZeroDivisionError:
                res[class_name] = 1.0
            # res[class_name] = metric.jc(preds_.cpu().numpy(), labels_.cpu().numpy())
        return res



