from configparser import Interpolation
import os
import random
import argparse
import time
import torch
import numpy as np
import scipy
from torch.optim import optimizer
import torch.nn.functional as F
from tqdm import tqdm
from datasets.mvtec import FSAD_Dataset_train, FSAD_Dataset_test, CLASS_NAMES
from utils.utils import time_file_str, time_string, convert_secs2time, AverageMeter, print_log, visualize_results, denormalization, visualize_augment_image
from models.siamese import Encoder, Predictor
from models.stn import stn_net
from losses.norm_loss import CosLoss
from utils.funcs import embedding_concat, mahalanobis_torch, rot_img, translation_img, hflip_img, rot90_img, grey_img
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import average_precision_score, auc
from utils.eval_metric import iou_curve, get_aupr_curve

from scipy.ndimage import gaussian_filter
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

'''
1. 将提取的features进行平均，得到不同layer下的平均feature map
'''
def get_layers_feature_map(test_outputs):

    # 得到layer1, layer2, layer3的输出
    layer_feature1 = test_outputs['layer1']
    layer_feature2 = test_outputs['layer2']
    layer_feature3 = test_outputs['layer3']

    output_layers1 = []
    output_layers2 = []
    output_layers3 = []

    for index in range(len(layer_feature1)):
        output_layers = torch.mean(layer_feature1[index], dim=1, keepdim=False)
        output_layers = output_layers.squeeze().cpu().numpy()
        output_layers1.append(output_layers)

    for index in range(len(layer_feature2)):
        output_layers = torch.mean(layer_feature2[index], dim=1, keepdim=False)
        output_layers = output_layers.squeeze().cpu().numpy()
        output_layers2.append(output_layers)
    
    for index in range(len(layer_feature3)):
        output_layers = torch.mean(layer_feature3[index], dim=1, keepdim=False)
        output_layers = output_layers.squeeze().cpu().numpy()
        output_layers3.append(output_layers)
    
    return output_layers1, output_layers2, output_layers3


def usr_parser():
    parser = argparse.ArgumentParser(description='RegAD on MVtec')
    parser.add_argument('--obj',        type=str, default='hazelnut')
    parser.add_argument('--gpu',        type=int, default=0)
    parser.add_argument('--vis',        type=int, default=1)
    parser.add_argument('--data_type',  type=str, default='mvtec')
    parser.add_argument('--data_path',  type=str, default='./MVTec/MVTec_AD')
    parser.add_argument('--epochs',     type=int, default=50, help='maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size',   type=int, default=224)
    parser.add_argument('--lr',         type=float, default=0.01, help='learning rate in SGD')
    parser.add_argument('--momentum',   type=float, default=0.9,  help='momentum of SGD')
    parser.add_argument('--seed',       type=int,   default=668,  help='manual seed')
    parser.add_argument('--shot',       type=int,   default=2,    help='shot count')
    parser.add_argument('--inferences', type=int,   default=10,   help='number of rounds per inference')
    parser.add_argument('--stn_mode',   type=str,   default='rotation_scale', help='[affine, translation, rotation, scale, shear, rotation_scale, translation_scale, rotation_translation, rotation_translation_scale]')
    args = parser.parse_args()
    return args

def get_save_path(args, inference_round, add_str=''):
    # save image path
    image_dir = f'./vis_result/{args.obj}/{args.shot}/{inference_round}/'

    if add_str != '':
        image_dir = image_dir + add_str + '/'

    if not os.path.exists(image_dir):
        os.makedirs(image_dir, exist_ok=True)

    return image_dir

def main(args):

    torch.cuda.set_device(args.gpu)

    args.input_channel = 3
    if args.seed is None:
        args.seed = random.randint(1, 10000)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    args.prefix = time_file_str()

    args.save_dir = './logs_mvtec_test/'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    log = open(os.path.join(args.save_dir, 'log_{}_{}.txt'.format(str(args.shot),args.obj)), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)

    STN = stn_net(args).to(device)
    ENC = Encoder().to(device)
    PRED = Predictor().to(device)

    print_log('-------start test on {} with {} shot----------------'.format(args.obj, args.shot), log)

    # load models
    CKPT_name  = f'./save_checkpoints/{args.shot}/{args.obj}/{args.obj}_{args.shot}_rotation_scale_model.pt'
    model_CKPT = torch.load(CKPT_name)
    STN.load_state_dict(model_CKPT['STN'])
    ENC.load_state_dict(model_CKPT['ENC'])
    PRED.load_state_dict(model_CKPT['PRED'])
    models = [STN, ENC, PRED]

    print('Loading Datasets')
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = FSAD_Dataset_test(args.data_path, class_name=args.obj, is_train=False, resize=args.img_size, shot=args.shot)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    print('Loading Fixed Support Set')
    fixed_fewshot_list = torch.load(f'./support_set/{args.obj}/{args.shot}_{args.inferences}.pt')

    print('Start Testing:')
    image_auc_list  = []
    pixel_auc_list  = []
    image_aupr_list = []
    pixel_aupr_list = []
    iou_score_list  = []
    for inference_round in range(args.inferences):
        print('Round {}:'.format(inference_round))
        cur_few_list = fixed_fewshot_list[inference_round]
        scores_list, test_imgs, gt_list, gt_mask_list, query_features = test(args, models, inference_round, fixed_fewshot_list, test_loader, **kwargs)
        scores = np.asarray(scores_list)
        
        # Normalization
        max_anomaly_score = scores.max()
        min_anomaly_score = scores.min()
        scores            = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

        # ground truth
        gt_list    = np.asarray(gt_list)
        gt_mask   = np.asarray(gt_mask_list)
        gt_mask   = (gt_mask > 0.5).astype(np.int_)

        # AUC--------------------------------------------------------------------
        # calculate image-level AUC 
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)        
        img_auc    = roc_auc_score(gt_list, img_scores)
        image_auc_list.append(img_auc)

        # calculate perpixel level AUC        
        pixel_auc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        pixel_auc_list.append(pixel_auc)

        # AUPR--------------------------------------------------------------------
        # calculate image-level AUPR
        img_aupr, cls_th, _, _     = get_aupr_curve(gt_list.flatten(), img_scores.flatten())
        image_aupr_list.append(img_aupr)

        # calculate perpixel level AUPR
        pixel_aupr, seg_th, _, _ = get_aupr_curve(gt_mask.flatten(), scores.flatten())
        pixel_aupr_list.append(pixel_aupr)

        # iou
        fpr, iou, thresh_ = iou_curve(gt_mask.flatten(), scores.flatten())
        iou_score         = iou.max()
        seg_thresh        = float(scipy.interpolate.interp1d(iou, thresh_)(iou_score))
        iou_score_list.append(iou_score)

        # PR95--------------------------------------------------------------------
        #img_fpr, img_tpr, img_thresh         = roc_curve(gt_list ,img_scores)
        #pixel_fpr,  pixel_tpr,  pixel_thresh = roc_curve(gt_mask.flatten() ,scores.flatten())

        # print results
        print_log('cur Image-level AUC/AUPR: {:.4f} {:.4f}, cur Pixel-level AUC/AUPR/IOU: {:.4f} {:.4f} {:.4f}'.format(img_auc, img_aupr, pixel_auc, pixel_aupr, iou_score), log)

        if (args.vis and inference_round == 0):
            # save image path
            image_dir = get_save_path(args, inference_round)
            visualize_results(test_imgs, scores, img_scores, gt_mask_list, query_features, seg_th, cls_th, cur_few_list, image_dir, args.obj)

    image_auc_list  = np.array(image_auc_list)
    pixel_auc_list  = np.array(pixel_auc_list)
    image_aupr_list = np.array(image_aupr_list)
    pixel_aupr_list = np.array(pixel_aupr_list)
    iou_score_list  = np.array(iou_score_list)
    mean_img_auc    = np.mean(image_auc_list, axis = 0)
    mean_pixel_auc  = np.mean(pixel_auc_list, axis = 0)
    mean_img_aupr   = np.mean(image_aupr_list, axis = 0)
    mean_pixel_aupr = np.mean(pixel_aupr_list, axis = 0)
    mean_iou_score  = np.mean(iou_score_list, axis = 0)
    print_log('Image-level AUC/AUPR: {:.4f} {:.4f}, Pixel-level AUC/AUPR: {:.4f} {:.4f} {:.4f}'.format(mean_img_auc, mean_img_aupr, mean_pixel_auc, mean_pixel_aupr, mean_iou_score), log)


def get_support_augment_images(args, fixed_fewshot_list, cur_epoch):
    
    support_img         = fixed_fewshot_list[cur_epoch]
    augment_support_img = support_img
    # rotate img with small angle
    for angle in [-np.pi/4, -3 * np.pi/16, -np.pi/8, -np.pi/16, np.pi/16, np.pi/8, 3 * np.pi/16, np.pi/4]:
        rotate_img          = rot_img(support_img, angle)
        augment_support_img = torch.cat([augment_support_img, rotate_img], dim=0)
    
    # translate img
    for a,b in [(0.2,0.2), (-0.2,0.2), (-0.2,-0.2), (0.2,-0.2), (0.1,0.1), (-0.1,0.1), (-0.1,-0.1), (0.1,-0.1)]:
        trans_img           = translation_img(support_img, a, b)
        augment_support_img = torch.cat([augment_support_img, trans_img], dim=0)
    
    # hflip img
    flipped_img         = hflip_img(support_img)
    augment_support_img = torch.cat([augment_support_img, flipped_img], dim=0)
    
    # rgb to grey img
    greyed_img          = grey_img(support_img)
    augment_support_img = torch.cat([augment_support_img, greyed_img], dim=0)
    
    # rotate img in 90 degree
    for angle in [1,2,3]:
        rotate90_img        = rot90_img(support_img, angle)
        augment_support_img = torch.cat([augment_support_img, rotate90_img], dim=0)
    
    image_dir = get_save_path(args, cur_epoch, 'support')
    visualize_augment_image(augment_support_img, image_dir, args.obj)
    
    # rand shuffle support image
    augment_support_img = augment_support_img[torch.randperm(augment_support_img.size(0))]

    # visualize and save
    #visualize_results(support_img, None, None, None, None, None, None, None, './vis_reuslt/rotate/', 'rotate')

    return augment_support_img



def test(args, models, cur_epoch, fixed_fewshot_list, test_loader, **kwargs):
    STN = models[0]
    ENC = models[1]
    PRED = models[2]

    STN.eval()
    ENC.eval()
    PRED.eval()

    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    test_outputs  = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    augment_support_img = get_support_augment_images(args, fixed_fewshot_list, cur_epoch)
    
    # torch version
    with torch.no_grad():
        support_feat = STN(augment_support_img.to(device))
    support_feat = torch.mean(support_feat, dim=0, keepdim=True)
    train_outputs['layer1'].append(STN.stn1_output)
    train_outputs['layer2'].append(STN.stn2_output)
    train_outputs['layer3'].append(STN.stn3_output)

    for k, v in train_outputs.items():
        train_outputs[k] = torch.cat(v, 0)

    # Embedding concat
    embedding_vectors = train_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name], True)

    # calculate multivariate Gaussian distribution
    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W)
    mean = torch.mean(embedding_vectors, dim=0)
    cov  = torch.zeros(C, C, H * W).to(device)
    I    = torch.eye(C).to(device)
    for i in range(H * W):
        cov[:, :, i] = torch.cov(embedding_vectors[:, :, i].T) + 0.01 * I
    train_outputs = [mean, cov]

    # torch version
    query_imgs = []
    gt_list = []
    mask_list = []
    score_map_list = []
    query_features = []

    for (query_img, _, mask, y) in tqdm(test_loader):
        query_imgs.extend(query_img.cpu().detach().numpy())
        gt_list.extend(y.cpu().detach().numpy())
        mask_list.extend(mask.cpu().detach().numpy())
        
        # model prediction
        query_feat = STN(query_img.to(device))
        z1 = ENC(query_feat)
        z2 = ENC(support_feat)
        p1 = PRED(z1)
        p2 = PRED(z2)

        loss = CosLoss(p1,z2, Mean=False)/2 + CosLoss(p2,z1, Mean=False)/2
        loss_reshape = F.interpolate(loss.unsqueeze(1), size=query_img.size(2), mode='bilinear',align_corners=False).squeeze(0)
        score_map_list.append(loss_reshape.cpu().detach().numpy())

        test_outputs['layer1'].append(STN.stn1_output)
        test_outputs['layer2'].append(STN.stn2_output)
        test_outputs['layer3'].append(STN.stn3_output)

    query_features = get_layers_feature_map(test_outputs)
    
    for k, v in test_outputs.items():
        test_outputs[k] = torch.cat(v, 0)

    # Embedding concat
    embedding_vectors = test_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name], True)

    # calculate distance matrix
    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W)
    dist_list = []

    for i in tqdm(range(H * W)):
        mean = train_outputs[0][:, i]
        conv_inv = torch.linalg.inv(train_outputs[1][:, :, i])
        dist = [mahalanobis_torch(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
        dist_list.append(dist)

    dist_list = torch.tensor(dist_list).transpose(1, 0).reshape(B, H, W)

    # upsample
    score_map = F.interpolate(dist_list.unsqueeze(1), size=query_img.size(2), mode='bilinear',
                              align_corners=False).squeeze().numpy()

    # apply gaussian smoothing on the score map
    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i], sigma=4)

    return score_map, query_imgs, gt_list, mask_list, query_features

if __name__ == '__main__':
    args = usr_parser()
    for obj in CLASS_NAMES:
        args.obj = obj
        main(args)
