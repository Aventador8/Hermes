from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F



class PixelContrastLoss(nn.Module):
    def __init__(self,
                 temperature=0.07,
                 base_temperature=0.07,
                 max_samples=1024,
                 max_views=100,
                 ignore_index=-1,
                 device='cuda:2'):
        super(PixelContrastLoss, self).__init__()

        self.temperature = temperature
        self.base_temperature = base_temperature

        self.ignore_label = ignore_index

        self.max_samples = max_samples
        self.max_views = max_views

        self.device = device

    def _hard_anchor_sampling(self, X, y_hat, y):  # X: B * HW * C  y_hat: B * D1  y: B * D2
        batch_size, feat_dim = X.shape[0], X.shape[-1]  # feat_dim: channel
        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            # print('this_classes(unique): ', this_classes)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]
            # print('after filter: ', this_classes)
            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        # print('total_classes:',total_classes)
        n_view = self.max_samples // total_classes  # self.max_samples = 1024
        # print('n_view:',n_view)
        n_view = min(n_view, self.max_views)  # self.max_views = 100

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).to(self.device) # BD2 * 100 * C  24 * 42 *16
        y_ = torch.zeros(total_classes, dtype=torch.float).to(self.device)  # Batch * n_class = 24

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]  #  一个batch所有像素标签 D1
            this_y = y[ii]  #  一个batch所有像素预测的标签 D1
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    print('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id  # y_: class
                X_ptr += 1

        return X_, y_  # X_: BD2 * 100 * C

    def _contrastive(self, feats_, labels_):  # feats_: 12 * 85 * 16       labels_: 12
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)  # BD2 * 1
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().to(self.device)  # BD2 * BD2

        contrast_count = n_view
        # contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0) # 1008 * 256

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).to(self.device) ,
                                                     0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


    def forward(self, feats, labels=None, predict=None): # feats: N * 16 * 512 * 512  labels: N * 512 * 512  predict: N * 512 * 512
        #feats                                      segstrong_feature: 8 * 16 * 512 * 512  psuedoLabels: 8 * 512 * 512
        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        predict = predict.unsqueeze(1).float().clone()
        predict = torch.nn.functional.interpolate(predict,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        predict = predict.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)  # N * 262144
        predict = predict.contiguous().view(batch_size, -1) # N * 262144
        feats = feats.permute(0, 2, 3, 1)  # B * H * W * C   //    N * 512 * 512 * 16
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])  # B * HW * C    //   N * 262144 * 16

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)
        if feats_ is None :
            loss = torch.tensor(0.0).to(self.device)
            return loss

        loss = self._contrastive(feats_, labels_)
        return loss


# if __name__ == "__main__":
#     feats = torch.rand((4,16,150,100))
#     labels = torch.ones((4,150,100))
#     pred = torch.ones((4,150,100))
#     loss_func1 = PixelContrastLoss()
#     loss1 = loss_func1(feats,labels,pred)
#     loss_func2 = PixelContrastLoss2()
#     loss2 = loss_func2(feats,labels,pred)
#
#     print(loss1)
#     print(loss2)
#










