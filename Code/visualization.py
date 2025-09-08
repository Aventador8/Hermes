import argparse
import os

import imageio
import numpy as np
import torch
import yaml
from torchvision import transforms
from torch.utils.data import DataLoader

from Code.dataset.BUSI import BUSI_Dataset
from Code.model.pretrained_unet_copy_discard import preUnet
from Code.utils.logger import get_logger, create_dir

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
parser.add_argument('--trainsize', type=int, default=(224,224), help='training dataset size')
parser.add_argument('--dataset', type=str, default='SZ-TUS', help='dataset name')
parser.add_argument('--threshold', type=float, default=0.5, help='threshold')
'----------------------------------------------------------------------------------------'
parser.add_argument('--eval_data_path', type=str, default='/data/zhangpeng/SZ-TUS/test')
'----------------------------------------------------------------------------------------------'

opt = parser.parse_args()

class Test(object):
    def __init__(self):
        # self._init_configure()
        self._init_logger()
        self.model_2 = preUnet(num_classes=2, device=opt.device,
                            seg_pretrained=False, consistency_button=False,
                            seg_contrast_button=False)


    def _init_configure(self):
        with open('configs/config.yml') as fp:
            self.cfg = yaml.safe_load(fp)

    def _init_logger(self):

        log_dir = '/data/zhangpeng/logs/' + opt.dataset + '/Janus'

        self.logger = get_logger(log_dir)
        print('RUNDIR: {}'.format(log_dir))

        self.save_path = log_dir
        self.image_save_path_2 = log_dir
        create_dir(self.image_save_path_2)
        create_dir(self.image_save_path_2+'/image')
        create_dir(self.image_save_path_2+'/predicted_mask')
        create_dir(self.image_save_path_2 + '/ground_truth')
        create_dir(self.image_save_path_2 + '/merge')

        self.model_2_load_path = '/data/zhangpeng/Janus/SZ-TUS/60/best.pth'


    def visualize_val_input(self, var_map, i):
        count = i
        im = transforms.ToPILImage()(var_map.squeeze_(0).detach().cpu()).convert("RGB")
        name = '{:02d}_input.png'.format(count)
        imageio.imwrite(self.image_save_path_2+'/image' + "/val_" + name, im)

    def visualize_gt(self, var_map, i):
        count = i
        for kk in range(var_map.shape[0]):
            pred_edge_kk = var_map[kk, :, :, :]
            pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
            color_mask = np.zeros((pred_edge_kk.shape[0], pred_edge_kk.shape[1], 3), dtype=np.uint8)
            color_mask[pred_edge_kk==1] = [255, 0, 0]
            name = '{:02d}_gt.png'.format(count)
            imageio.imwrite(self.image_save_path_2+'/ground_truth' + "/val_" + name, color_mask)


    def visualize_prediction2(self, var_map, i):
        count = i
        for kk in range(var_map.shape[0]):
            pred_edge_kk = var_map[kk, :, :, :]
            pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze().astype(np.float32)
            pred_edge_kk *= 255.0
            pred_edge_kk = pred_edge_kk.astype(np.uint8)
            name = '{:02d}_pred_2.png'.format(count)
            imageio.imwrite(self.image_save_path_2+ '/predicted_mask'+ "/val_" + name, pred_edge_kk)

    def visualize_uncertainity(self, var_map, i):
        count = i
        for kk in range(var_map.shape[0]):
            pred_edge_kk = var_map[kk, :, :, :]
            pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
            pred_edge_kk *= 255.0
            pred_edge_kk = pred_edge_kk.astype(np.uint8)
            name = '{:02d}_pred.png'.format(count)
            imageio.imwrite(self.image_save_path_1 + "/uncertainity_" + name, pred_edge_kk)

    def merge_mask_and_prediction(self, mask, prediction, i):
        count = i
        for kk in range(prediction.shape[0]):
            # mask
            gt = mask[kk, :, :, :]
            gt = gt.detach().cpu().numpy().squeeze()

            # prediction
            predict = prediction[kk, :, :, :]
            predict = predict.detach().cpu().numpy().squeeze()

            color_mask = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
            color_mask[gt==1] = [255, 0, 0]  # mask is red
            color_mask[predict == 1] = [0, 255, 0]  # pred is green

            # overlapping
            color_mask[(gt==1)&(predict == 1)] = [255, 255, 0]

            name = '{:02d}_pred_2.png'.format(count)
            imageio.imwrite(self.image_save_path_2+ '/merge'+ "/val_" + name, color_mask)


    def run(self):
        # build models
        checkpoint = torch.load(self.model_2_load_path)
        self.model_2.load_state_dict(checkpoint['seg_model'])
        self.model_2.cuda()

        eval_data = BUSI_Dataset(opt.eval_data_path, image_size=224, mode='val')
        eval_loader = DataLoader(eval_data, batch_size=1, num_workers=4)

        for i, pack in enumerate(eval_loader, start=1):
            with torch.no_grad():
                images, gts, _ = pack

                images = images.cuda()
                gts = gts.cuda()

                feat_map_2 = self.model_2(images)['output']
                prediction2 = torch.argmax(feat_map_2, dim=1, keepdim=True)

            self.visualize_val_input(images, i)
            self.visualize_gt(gts.unsqueeze(1), i)

            self.visualize_prediction2(prediction2, i)
            self.merge_mask_and_prediction(gts.unsqueeze(1), prediction2, i)

        # self.evaluate_model_2('logs/kvasir/test/saved_images_2/')


if __name__ == '__main__':
    Test_network = Test()
    Test_network.run()
