import time
import random
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime()))
    return string


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def time_file_str():
    ISOTIMEFORMAT = '%Y-%m-%d'
    string = '{}'.format(time.strftime(ISOTIMEFORMAT, time.localtime()))
    return string + '-{}'.format(random.randint(1, 10000))


def print_log(print_string, log):
    print("{:}".format(print_string))
    log.write('{:}\n'.format(print_string))
    log.flush()


'''
def get_custom_support_set(args):
    kwargs              = {'num_workers': 4, 'pin_memory': True} 
    support_dataset     = customdataset(args.data_path, class_name=args.obj, is_train=True, resize=args.img_size)
    support_data_loader = torch.utils.data.DataLoader(support_dataset, batch_size=args.shot, shuffle=True, **kwargs)
    save_img_list = []
    for i in range(args.inferences):
        support_img = iter(support_data_loader).next()
        save_img_list.append(support_img)
    torch.save(save_img_list, f'./support_set/{args.obj}/{args.shot}_{args.inferences}.pt')
'''


def visualize_results(test_img, scores, img_scores, gts, query_features, threshold, cls_threshold, cur_few_list, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    vmax = vmax * 0.5 + vmin * 0.5
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        #kernel = morphology.disk(4)
        #mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 7, figsize=(15, 3), gridspec_kw={'width_ratios': [4, 4, 4, 4, 4, 4, 3]})

        fig_img.subplots_adjust(wspace=0.05, hspace=0)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Input image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax_img[2].imshow(heat_map, cmap='jet', norm=norm, interpolation='none')
        ax_img[2].imshow(vis_img, cmap='gray', alpha=0.7, interpolation='none')
        ax_img[2].title.set_text('Segmentation')
        
        for j in range(3, 6):
            featuremap = query_features[j-3][i]
            ax_img[j].imshow(featuremap, cmap='jet')
            ax_img[j].title.set_text('featuremap {}'.format(j-3))
        

        black_mask = np.zeros((int(mask.shape[0]), int(3 * mask.shape[1] / 4)))
        ax_img[6].imshow(black_mask, cmap='gray')
        ax = plt.gca()
        if img_scores[i] > cls_threshold:
            cls_result = 'nok'
        else:
            cls_result = 'ok'

        ax.text(0.05,
                0.89,
                'Detected anomalies',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.79,
                '------------------------',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.72,
                'Results',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.67,
                '------------------------',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.59,
                '\'{}\''.format(cls_result),
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='r',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.47,
                'Anomaly scores: {:.2f}'.format(img_scores[i]),
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.37,
                '------------------------',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.30,
                'Thresholds',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.25,
                '------------------------',
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.17,
                'Classification: {:.2f}'.format(cls_threshold),
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax.text(0.05,
                0.07,
                'Segementation: {:.2f}'.format(threshold),
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=ax.transAxes,
                fontdict=dict(
                    fontsize=8,
                    color='w',
                    family='sans-serif',
                ))
        ax_img[6].title.set_text('Classification')
        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=300, bbox_inches='tight')
        plt.close()

    for i in range(cur_few_list.shape[0]):
        img = cur_few_list[i]
        img = img.numpy()
        img = denormalization(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imencode('.png', img)[1].tofile(os.path.join(save_dir, 'support', class_name + '_fewshot_support_{}.png'.format(i)))

def visualize_augment_image(augment_image, save_dir, class_name):
    num            = len(augment_image)
    imagew, imageh = np.int(augment_image[0].shape[1]), np.int(augment_image[0].shape[2])
    augment_image1 = []
    augment_image2 = []

    for i in range(num):
        if i % 2 == 0:
            augment_image1.append(augment_image[i])
        else:
            augment_image2.append(augment_image[i])
    
    show_image1 = np.zeros((imagew * 5, imageh * 5, 3), dtype=np.uint8)
    show_image2 = np.zeros((imagew * 5, imageh * 5, 3), dtype=np.uint8)

    # 一张图像 扩充到22张图像, 5*5张进行排列
    for i in range(len(augment_image1)):
        img     = augment_image1[i]
        img     = img.numpy()
        img     = denormalization(img)
        x       = i // 5
        y       = i % 5
        start_x = int(x * imagew)
        start_y = int(y * imageh)
        show_image1[start_x:(start_x+imagew), start_y:(start_y+imageh), :] = img
    
    for i in range(len(augment_image2)):
        img     = augment_image[i]
        img     = img.numpy()
        img     = denormalization(img)
        x       = i // 5
        y       = i % 5
        start_x = int(x * imagew)
        start_y = int(y * imageh)
        show_image2[start_x:(start_x+imagew), start_y:(start_y+imageh), :] = img

    show_image1 = cv2.cvtColor(show_image1, cv2.COLOR_RGB2BGR)
    show_image2 = cv2.cvtColor(show_image2, cv2.COLOR_RGB2BGR)
    cv2.imencode('.png', show_image1)[1].tofile(os.path.join(save_dir, class_name + '_augment_1.png'))
    cv2.imencode('.png', show_image2)[1].tofile(os.path.join(save_dir, class_name + '_augment_2.png'))

def denormalization(x):
    #mean = np.array([0.485, 0.456, 0.406])
    #std  = np.array([0.229, 0.224, 0.225])
    #x    = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

    x    = (x.transpose(1, 2, 0) * 255.).astype(np.uint8)
    
    return x