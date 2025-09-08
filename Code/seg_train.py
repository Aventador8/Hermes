import argparse
import random
from itertools import cycle
import os
import gc
from tqdm import tqdm

from Code.dataset.BUSI import BUSI_Dataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import make_grid
from Code.loss.soft_supconloss_no_reweight import SoftSupConLoss
from Code.model.ResNet18 import pretrained_resnet18
from Code.loss.loss_manager import *
from Code.metric.dice import calc_dice_score
from tensorboardX import SummaryWriter
from Code.metric.acc import accuracy
from Code.model.pretrained_unet import preUnet
from Code.utils import ramps
from Code.utils.helpers import evaluate_iou, AverageMeter, tensorboard_write_images, tensorboard_write_scalars, \
    get_classwise_thresholds, filter_pseudo_labels_by_threshold
from Code.utils.ramps import get_current_consistency_weight

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser()
parser.add_argument('--cls_weight', type=float, default=1.1, help='hyper parameter')
parser.add_argument('--seg_weight', type=float, default=0.5, help='hyper parameter')
parser.add_argument('--cls_contrast_weight', type=float, default=0.2, help='hyper parameter')
parser.add_argument('--seg_contrast_weight', type=float, default=0.8, help='hyper parameter')
parser.add_argument('--maxiter', type=int, default=3000, help='epoch number')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=(224,224), help='training dataset size')
parser.add_argument('--labeled_data_path', type=str, default='/data/zhangpeng/UDIAT2/train/10/labeled')
parser.add_argument('--unlabeled_data_path', type=str, default='/data/zhangpeng/UDIAT2/train/10/unlabeled')
parser.add_argument('--eval_data_path', type=str, default='/data/zhangpeng/UDIAT2/test')
parser.add_argument('--save_path', type=str, default='/data/zhangpeng/rebuttal/visualization_selection_process_seg/UDIAT2/10/dual_threshold')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--resume', type=bool, default=False, help='whether to resume')

def adjust_lr(optimizer, init_lr, epoch, max_epoch):
    lr_ = init_lr * (1.0 - epoch / max_epoch) ** 0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_

class_dict = ['background', 'tumor']
args = parser.parse_args()
device = args.device
lr = args.lr
# make result file
os.makedirs(args.save_path, exist_ok=True)
tensorboard_path = os.path.join(args.save_path, 'log')
os.makedirs(tensorboard_path, exist_ok=True)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


writer = SummaryWriter(tensorboard_path)

label_train_data = BUSI_Dataset(args.labeled_data_path, image_size=224, mode='train_l')
label_trainloader = DataLoader(label_train_data,batch_size=args.batchsize,
                               shuffle=True, drop_last=True, num_workers=4)

unlabeled_train_data = BUSI_Dataset(args.unlabeled_data_path, image_size=224, mode='train_u')
unlabeled_trainloader = DataLoader(unlabeled_train_data, batch_size=args.batchsize,
                                   shuffle=True, num_workers=4, drop_last=True)

eval_data = BUSI_Dataset(args.eval_data_path, image_size=224, mode='val')
eval_loader = DataLoader(eval_data,batch_size=args.batchsize, num_workers=4)

print("unlabeled_data_loader: {}, labeled_data_loader: {}, val_loader: {}".format(len(unlabeled_trainloader),
                                                                                        len(label_trainloader),
                                                                                        len(eval_loader)))

# model
seg_model = preUnet(num_classes=2, device=device,
                    seg_pretrained=True,consistency_button=True,
                    seg_contrast_button=True, contrast_feature='3').to(device)
cls_model = pretrained_resnet18(num_classes=2,pretrained=True,
                        device=device, cls_contrast_button=True,
                        consistency_button=True).to(device)

seg_optimizer = torch.optim.Adam(seg_model.parameters(), lr=lr)
# cls_optimizer = torch.optim.Adam(cls_model.parameters(), lr=lr)
cls_optimizer = torch.optim.SGD(cls_model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)

step = 1
eval_step = 1
start_epoch = 1
previous_best_dice = 0.0
previous_best_iou = 0.0
previous_best_acc = 0.0
save_image_interval = 20
pretrain = False
max_iter = args.maxiter
max_epoch = args.maxiter // len(unlabeled_trainloader)

best_tumor = 0.0

# if pretrain:
#     checkpoint = torch.load('without_result2/BUSI/train/109/latest_resume.pth', map_location=device)
#     model.load_state_dict(checkpoint['model'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     start_epoch = checkpoint['epoch'] + 1  # 设置开始的epoch
#     previous_best_dice = checkpoint['previous_best_iou']
#     # step = checkpoint['step']
#     step = checkpoint['step']

# fp 16
scaler = GradScaler()
criterion_dice = DiceLoss_(weight=1.0, n_classes=2, class_weight=[0.0, 1.0])
# loss functions
cls_contrast_loss = SoftSupConLoss(device=device).to(device)
pixel_contrast_loss = PixelContrastLoss(device=device).to(device)

with (tqdm(total=max_iter - step, bar_format='[{elapsed}<{remaining}] ') as pbar):
    for epoch in range(max_epoch):
        seg_model.train()
        cls_model.train()
        train_total_loss = 0
        # gc.collect()
        # torch.cuda.empty_cache()
        for batch_idx, data in enumerate(zip(unlabeled_trainloader,cycle(label_trainloader))):
        # for batch, data in enumerate(zip(unlabeled_trainloader, label_trainloader)):
            gc.collect()
            torch.cuda.empty_cache()
            images = {}
            scalars = {}
            save_image = True if (step) % save_image_interval == 0 else False

            img_w, img_s = data[0][0], data[0][1]
            img, mask, label = data[1][0], data[1][1], data[1][2]

            img_s, img_w =  img_s.to(device), img_w.to(device)
            img, mask, label = img.to(device), mask.to(device), label.to(device)

            with autocast():
                seg_output_s = seg_model(img_s, cls_model, cls_optimizer)
                cls_output_s = cls_model(img_s)
                seg_contrast_feature_s = seg_output_s['contrast_embed']
                cls_contrast_feature_s = cls_output_s['contrast_embed']
                cls_consistency = cls_output_s['consistency']
                seg_consistency = seg_output_s['consistency']
                outSeg_s = seg_output_s['output']
                outCls_s = cls_output_s['output']

            # del out_s
            torch.cuda.empty_cache()

            with autocast():
                seg_output_w = seg_model(img_w, cls_model, cls_optimizer)
                cls_output_w = cls_model(img_w)

                seg_contrast_feature_w = seg_output_w['contrast_embed']
                cls_contrast_feature_w = cls_output_w['contrast_embed']
                outSeg_w = seg_output_w['output']
                outCls_w = cls_output_w['output']
                # saliency_w = out_w['saliency']

            # del out_w
            torch.cuda.empty_cache()

            with autocast():
                seg_output = seg_model(img, cls_model, cls_optimizer)
                cls_output = cls_model(img)
                # intra_cls_consistency_lb = out_label['intra_cls_consistency_embed']
                # intra_seg_consistency_lb = out_label['intra_seg_consistency_embed']
                outCls_lb = cls_output['output']
                outSeg_lb = seg_output['output']
                # saliency_lb = out_label['saliency']
            # del out_label
            torch.cuda.empty_cache()
            # labeled data for classification and segmentation
            with autocast():
                # supervised losses
                loss_lb = calc_supervised_loss(outSeg_lb, mask, outCls_lb, label)

            # classification
            with autocast():
                # unlabeled cls loss
                probs_u_w = torch.softmax(outCls_w.detach(), dim=-1)  # N*D
                max_probs, p_targets_u = torch.max(probs_u_w, dim=-1)  # N * 1
                classwise_thresholds = get_classwise_thresholds(p_targets_u, 2)  # num_classes 2
                high_confidence_mask = filter_pseudo_labels_by_threshold(probs_u_w, p_targets_u,
                                                                         classwise_thresholds)
                cls_uncertainty = -1.0 * torch.sum(probs_u_w * torch.log(probs_u_w + 1e-6), dim=1)
                cls_threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(step, max_iter)) * np.log(2)
                cls_certainty_mask = (cls_uncertainty < cls_threshold)


                loss_u_cls = (F.cross_entropy(outCls_s, p_targets_u, reduction='none') * high_confidence_mask * cls_certainty_mask).mean()
                if epoch > 10:
                    # for supervised classification contrastive learning
                    cls_label = p_targets_u.clone()
                    cls_features = torch.cat([cls_contrast_feature_s.unsqueeze(1), cls_contrast_feature_w.unsqueeze(1)], dim=1)
                    # cls_mean = torch.mean(torch.stack([F.softmax(outCls_s, dim=-1), F.softmax(outCls_w, dim=-1)]), dim=0)

                    cls_contrast = cls_contrast_loss(cls_features, cls_label, reduction=None)
                    cls_contrast = (cls_contrast * high_confidence_mask * cls_certainty_mask).mean()
                    scalars['cls_certainty_rate'] = torch.sum(cls_certainty_mask == True) / \
                                                    (torch.sum(cls_certainty_mask == True) + torch.sum(
                                                        cls_certainty_mask == False))
                    scalars['loss/cls_contrast'] = cls_contrast
                else:
                    # CLScontrast = sum(cls_features.view(-1, 1)) * 0.0
                    cls_contrast = torch.tensor(0.0).to(device)
                # del cls_features
                torch.cuda.empty_cache()
                scalars['train_loss/cls_contrast'] = cls_contrast
                scalars['train_loss/cls_label_ce'] = loss_lb['cls_loss']
                scalars['train_loss/cls_unlabel_ce'] = loss_u_cls

            Loss_cls = args.cls_weight * (loss_lb['cls_loss'] + 0.1 * loss_u_cls + args.cls_contrast_weight * cls_contrast)
            cls_optimizer.zero_grad()
            scaler.scale(Loss_cls).backward(retain_graph=True)
            scaler.step(cls_optimizer)
            scaler.update()

            # segmentation
            with autocast():
                # unlabeled seg loss
                softmax_seg_weak = torch.softmax(outSeg_w.detach(), dim=1)
                seg_conf_w, pseudo_label_seg_w = torch.max(softmax_seg_weak, dim=1)
                uncertainty = -1.0 * torch.sum(softmax_seg_weak * torch.log(softmax_seg_weak + 1e-6), dim=1)
                uncertainty_threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(step, max_iter)) * np.log(2)
                uncertainty_mask = (uncertainty > uncertainty_threshold)

                segwise_thresholds = get_classwise_thresholds(pseudo_label_seg_w, 2)  # num_classes 2
                seg_high_confidence_mask = filter_pseudo_labels_by_threshold(softmax_seg_weak, pseudo_label_seg_w,
                                                                             segwise_thresholds)
                ignore_mask = 1 - seg_high_confidence_mask.int()

                loss_u_seg = criterion_dice(outSeg_s, pseudo_label_seg_w, ignore=(ignore_mask * uncertainty_mask).float())
                if epoch > 10:
                    # segmentation
                    # calculate uncertainty label (考虑强弱一致性的平均)
                    # strong view
                    preds_s = torch.argmax(torch.softmax(outSeg_s, dim=1), dim=1).to(torch.float)

                    certainty_pseudo = pseudo_label_seg_w.clone()
                    certainty_pseudo[uncertainty_mask] = -1  # 忽略的像素值为-1
                    certainty_pseudo[ignore_mask] = -1
                    # unlabeled + labeled
                    embed_mean = torch.mean(torch.stack([seg_contrast_feature_s, seg_contrast_feature_w]), dim=0)
                    pixel_conLoss = pixel_contrast_loss(embed_mean, certainty_pseudo, preds_s)
                    scalars['seg_uncertainty_rate'] = torch.sum(uncertainty_mask == True) / \
                                                      (torch.sum(uncertainty_mask == True) + torch.sum(
                                                          uncertainty_mask == False))
                    if save_image:
                        grid_image = make_grid(img_w, 4, normalize=False, padding=6, pad_value=255)
                        images['train/seg/original_image'] = grid_image
                        grid_image = make_grid(pseudo_label_seg_w.unsqueeze(dim=1) * 50., 4, normalize=False, padding=6,
                                               pad_value=255)
                        images['train/seg/mean_preds'] = grid_image
                        grid_image = make_grid(certainty_pseudo.unsqueeze(dim=1) * 50., 4, normalize=False, padding=6,
                                               pad_value=255)
                        images['train/seg/certainty_pseudo'] = grid_image
                        grid_image = make_grid(uncertainty.unsqueeze(dim=1), 4, normalize=False, padding=6,
                                               pad_value=255)
                        images['train/seg/uncertainty'] = grid_image
                        grid_image = make_grid(uncertainty_mask.unsqueeze(dim=1).float(), 4, normalize=False, padding=6,
                                               pad_value=255)
                        images['train/seg/uncertainty_mask'] = grid_image
                        # grid_image = make_grid(saliency_lb.unsqueeze(dim=1), 4, normalize=False, padding=6,
                        #                        pad_value=255)
                        # images['train/seg/saliency'] = grid_image
                        # grid_image = make_grid(certainty_mask.unsqueeze(dim=1), 4, normalize=False, padding=20, pad_value=127)
                        # images['train/seg/certainty'] = grid_image

                        # add_figures(writer, saliency_w, step, padding=4)
                        tensorboard_write_images(writer, images, step)
                else:
                    pixel_conLoss = torch.tensor(0.0).to(device)

                scalars['train_loss/seg_unlabeled_dice'] = loss_u_seg
                scalars['train_loss/seg_dice'] = loss_lb['seg_dice_loss']
                scalars['train_loss/seg_ce'] = loss_lb['seg_ce_loss']
                scalars['train_loss/seg_total'] = loss_lb['seg_total_loss']

                # task consistency loss
                consistency_weight = get_current_consistency_weight(step // 100)
                task_consistency = segAndclass_consistency(cls_consistency, seg_consistency)

                # del preds_w, preds_lb
                torch.cuda.empty_cache()
                Loss_seg = args.seg_weight * (loss_lb['seg_total_loss'] + 0.1 * loss_u_seg + args.seg_contrast_weight * pixel_conLoss)\
                            + consistency_weight * task_consistency

                scalars['train_loss/seg_contrast'] = pixel_conLoss
                # scalars['train_loss/task_consistency'] = task_consistency
                scalars['loss/total'] = Loss_cls + Loss_seg

                if step % 10 == 0:
                    scalars.update({'seg_lr': seg_optimizer.param_groups[0]['lr']})
                    tensorboard_write_scalars(writer, scalars, step)
                    # for key, value in scalars.items():
                    #     if isinstance(value, Tensor):
                    #         scalars[key] = value.item()
                    #     writer.add_scalar(key, value, step)
                if step == 100 :
                    checkpoint = {
                        'cls_model': cls_model.state_dict(),
                        'seg_model': seg_model.state_dict(),
                        'cls_optimizer': cls_optimizer.state_dict(),
                        'seg_optimizer': seg_optimizer.state_dict(),
                        'epoch': epoch,
                        'step': step
                    }
                    torch.save(checkpoint, os.path.join(args.save_path, '100_iterations.pth'))

                if step == 1000 :
                    checkpoint = {
                        'cls_model': cls_model.state_dict(),
                        'seg_model': seg_model.state_dict(),
                        'cls_optimizer': cls_optimizer.state_dict(),
                        'seg_optimizer': seg_optimizer.state_dict(),
                        'epoch': epoch,
                        'step': step
                    }
                    torch.save(checkpoint, os.path.join(args.save_path, '1000_iterations.pth'))

                if step == 3000 :
                    checkpoint = {
                        'cls_model': cls_model.state_dict(),
                        'seg_model': seg_model.state_dict(),
                        'cls_optimizer': cls_optimizer.state_dict(),
                        'seg_optimizer': seg_optimizer.state_dict(),
                        'epoch': epoch,
                        'step': step
                    }
                    torch.save(checkpoint, os.path.join(args.save_path, '3000_iterations.pth'))

            train_total_loss += Loss_cls.item() + Loss_seg.item()

            seg_optimizer.zero_grad()
            scaler.scale(Loss_seg).backward()
            scaler.step(seg_optimizer)
            scaler.update()

            del cls_contrast_feature_s, seg_contrast_feature_s
            torch.cuda.empty_cache()

            adjust_lr(seg_optimizer, lr, step, max_iter)
            adjust_lr(cls_optimizer, lr, step, max_iter)
            # adjust_learning_rate_poly(optimizer, epoch, max_epoch, lr, 0.9)



            with open(os.path.join(args.save_path, 'train_detail_loss.txt'), 'a') as f:
                f.write(f'Epoch[{epoch}/{max_epoch}]' + ' ' +'step: ' +str(step)
                    + ' ' + 'loss_seg_dice: ' + str(format(loss_lb['seg_dice_loss'], '.5f'))
                    + ' ' + 'loss_seg_ce: ' + str(format(loss_lb['seg_ce_loss'], '.5f'))
                    + ' ' + 'loss_cls_lb_ce: ' + str(format(loss_lb['cls_loss'], '.5f'))
                    + ' ' + 'loss_cls_ulb_ce: ' + str(format(loss_u_cls, '.5f'))
                    + ' ' + 'loss_seg_ulb_dice: ' + str(format(loss_u_seg, '.5f'))
                    + ' ' + 'cls_contrast:' + str(format(cls_contrast.item(), '.5f'))
                    + ' ' + 'seg_contrast:' + str(format(pixel_conLoss.item(), '.5f'))  + '\n')



            step += 1
            pbar.update(1)
        # train_epoch_loss = train_total_loss / (len(unlabeled_trainloader) + len(label_trainloader))
        f.close()

        with open(os.path.join(args.save_path,'train_epoch_loss.txt'),'a') as g:
            g.write(f'Epoch[{epoch}/{max_epoch}]'+ ' ' + 'Loss=' + str(format(train_total_loss / (len(unlabeled_trainloader) + len(label_trainloader)), '.5f'))+'\n')

        g.close()
        gc.collect()
        torch.cuda.empty_cache()
        writer.add_scalar('train/epoch_loss', train_total_loss / (len(unlabeled_trainloader) + len(label_trainloader)), epoch)

        seg_model.eval()
        cls_model.eval()
        eval_intersection_meter = AverageMeter()
        eval_union_meter = AverageMeter()
        eval_total_loss = 0
        eval_dice_class = [0.0]*2
        eval_acc = AverageMeter()
        with open(os.path.join(args.save_path, 'eval_metric.txt'), 'a') as h:
            h.write('===========> Epoch: {:},  Previous best Dice: {:.2f}, IoU: {:.2f}, Acc: {:.2f}\n'.format(
                    epoch,  previous_best_dice, previous_best_iou, previous_best_acc))

        for batch, (img, mask, label) in enumerate(eval_loader):
            gc.collect()
            torch.cuda.empty_cache()

            img, mask, label = img.to(device), mask.to(device), label.to(device)
            with autocast():

                out_seg = seg_model(img)['output']
                out_cls = cls_model(img)['output']
                # del outs
                seg_soft = torch.softmax(out_seg, dim=1)
                preds = torch.argmax(seg_soft, dim=1).to(torch.float)
                # iou
                intersection, union, _ = evaluate_iou(out_seg, mask, nclass=2)
                eval_intersection_meter.update(intersection)
                eval_union_meter.update(union)
                # dice
                calc_dice_score(eval_dice_class, out_seg, mask, nclass=2)
                # Acc
                prec1 = accuracy(out_cls, label)
                eval_acc.update(prec1[0].item())
                # loss
                # outputs = calc_supervised_loss(out_seg, mask, out_cls, label)

            if eval_step % 20 == 0:
                images = {}
                # print('img shape: ', img.shape)
                # print('pred shape: ', preds.unsqueeze(dim=1).shape)
                # print('mask shape: ', mask.shape)
                grid_image = make_grid(img, 4, normalize=False, padding=6, pad_value=255)
                images['val/images'] = grid_image
                grid_image = make_grid(preds.unsqueeze(dim=1) * 50., 4, normalize=False, padding=6,
                                       pad_value=255)
                images['val/preds'] = grid_image
                grid_image = make_grid(mask.unsqueeze(dim=1) * 50., 4, normalize=False)
                images['val/mask'] = grid_image

                tensorboard_write_images(writer, images, eval_step)
            # with open(os.path.join(save_path,'eval_detail_loss.txt'),'a') as f:
            #     f.write(f'Epoch[{epoch}/{max_epoch}]'+ ' '+ 'batch: '+ str(batch) +' '+'step: '+ str(eval_step) + ' '
            #             +'seg_total_loss:'+str(format(outputs['seg_total_loss'].item(), '.5f'))+' '
            #             +'seg_dice_loss:'+str(format(outputs['seg_dice_loss'].item(), '.5f'))+' '
            #             +'seg_ce_loss:'+str(format(outputs['seg_ce_loss'].item(), '.5f'))+' '
            #             +'cls_loss:'+str(format(outputs['cls_loss'].item(), '.5f'))+'\n')
            # eval_total_loss += outputs['total_loss'].item()
            eval_step += 1

        f.close()
        # del outputs
        torch.cuda.empty_cache()
        # eval_epoch_loss = eval_total_loss / len(eval_loader)
        eval_iou_class = eval_intersection_meter.sum / (eval_union_meter.sum + 1e-10) * 100.0
        eval_mIOU = np.mean(eval_iou_class)

        eval_dice_class = [dice * 100.0 / len(eval_loader) for dice in eval_dice_class]
        eval_mean_dice = sum(eval_dice_class) / len(eval_dice_class)

        # print(f'Epoch[{epoch}/{max_epoch}]'+' '+"Eval_loss="+str(format(eval_epoch_loss, '.3f'))+' '
        #       +'mDice:'+str(format(eval_mean_dice, '.2f'))+' '+'mIoU:'+str(format(eval_mIOU.item(), '.2f'))+' '
        #       +'Acc:'+str(format(eval_acc.avg, '.2f'))+ ' '+ 'tumor:'+ str(format(eval_dice_class[1], '.2f'))+  '\n')

        print(f'Epoch[{epoch}/{max_epoch}]' + ' ' + 'mDice:' + str(format(eval_mean_dice, '.2f')) + ' ' +
              'mIoU:' + str(format(eval_mIOU.item(), '.2f')) + ' '
              + 'Acc:' + str(format(eval_acc.avg, '.2f')) + ' ' + 'tumor:' + str(
            format(eval_dice_class[1], '.2f')) + ' ' + 'fore_iou:' + str(format(eval_iou_class[1], '.2f')) + '\n')

        with open(os.path.join(args.save_path, 'short_results.txt'), 'a') as t:
            t.write(f'Epoch[{epoch}/{max_epoch}]' + ' ' + 'mDice:' + str(format(eval_mean_dice, '.2f')) + ' ' +
              'mIoU:' + str(format(eval_mIOU.item(), '.2f')) + ' '
              + 'Acc:' + str(format(eval_acc.avg, '.2f')) + ' ' + 'tumor:' + str(
            format(eval_dice_class[1], '.2f')) + ' ' + 'fore_iou:' + str(format(eval_iou_class[1], '.2f')) + '\n')

        # with open(os.path.join(save_path,'eval_metric.txt'),'a') as g:
        #     g.write(f'Epoch[{epoch}/{max_epoch}]'+' '+"Eval_loss="+str(format(eval_epoch_loss, '.5f'))+
        #             ', '+'Acc:'+str(format(eval_acc.avg, '.2f'))+'\n')
        with open(os.path.join(args.save_path, 'eval_metric.txt'), 'a') as g:
            g.write(f'Epoch[{epoch}/{max_epoch}]' + ' '  + 'Acc:' + str(format(eval_acc.avg, '.2f')) + '\n')

            g.write('===========> IOU <===========\n')
            for (cls_idx, iou) in enumerate(eval_iou_class):
                g.write('***** Evaluation ***** >>>> Class [{:} {:}] '
                                'IoU: {:.2f}\n'.format(cls_idx, class_dict[cls_idx], iou))

            g.write('===========> DICE <===========\n')
            for (cls_idx, dice) in enumerate(eval_dice_class):
                g.write('***** Evaluation ***** >>>> Class [{:} {:}] Dice: '
                            '{:.2f}\n'.format(cls_idx, class_dict[cls_idx], dice))

            g.write('***** Evaluation ***** >>>> MeanIoU: {:.2f}, MeanDice: {:.2f}\n\n'.format(eval_mIOU, eval_mean_dice))

        # writer.add_scalar('eval/epoch_loss',eval_epoch_loss, epoch)
        writer.add_scalar('eval/mDice',eval_mean_dice, epoch)
        writer.add_scalar('eval/mIoU',eval_mIOU.item(), epoch)
        writer.add_scalar('eval/Acc',eval_acc.avg, epoch)
        # writer.add_scalar('eval/Dice', eval_dice_class[1], epoch)
        # writer.add_scalar('eval/IoU', eval_iou_class[1], epoch)

        for i, iou in enumerate(eval_iou_class):
            writer.add_scalar('eval/%s_IoU' % (class_dict[i]), iou, epoch)

        for i, dice in enumerate(eval_dice_class):
            writer.add_scalar('eval/%s_dice' % (class_dict[i]), dice, epoch)

        is_best_dice = eval_mean_dice > previous_best_dice
        previous_best_dice = max(eval_dice_class[1], previous_best_dice)
        previous_best_iou = max(eval_iou_class[1], previous_best_iou)
        previous_best_acc = max(eval_acc.avg, previous_best_acc)
        checkpoint = {
            'cls_model': cls_model.state_dict(),
            'seg_model': seg_model.state_dict(),
            'cls_optimizer': cls_optimizer.state_dict(),
            'seg_optimizer': seg_optimizer.state_dict(),
            'epoch': epoch,
            'previous_best_iou': previous_best_dice,
            'step': step
        }
        torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
        if is_best_dice:
            torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))

        g.close()
        gc.collect()
        torch.cuda.empty_cache()

writer.close()





