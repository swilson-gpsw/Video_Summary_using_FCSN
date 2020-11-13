from torchvision import transforms, models
import torch
from torch import nn
from PIL import Image
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--video_dir', type=str, help='directory containing mp4 file of specified dataset.', default='/mnt/hd02/CVPR/test/')
parser.add_argument('--data_folder', type=str, help='save folder for data ', default='/mnt/hd02/CVPR/results_timed/')
# parser.add_argument('--vsumm_data', type=str, help='preprocessed dataset path from this repo: https://github.com/KaiyangZhou/pytorch-vsumm-reinforce, which should be a hdf5 file. We copy cps and some other info from it.', default='../data/eccv_datasets/eccv16_dataset_tvsum_google_pool5.h5')

args = parser.parse_args()
video_dir = args.video_dir
data_folder = args.data_folder
# vsumm_data = h5py.File(args.vsumm_data)


class Rescale(object):
    """Rescale a image to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is matched to output_size. If int, smaller of image edges is matched to output_size keeping aspect ratio the same.
    """

    def __init__(self, *output_size):
        self.output_size = output_size

    def __call__(self, image):
        """
        Args:
            image (PIL.Image) : PIL.Image object to rescale
        """
        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = image.resize((new_w, new_h), resample=Image.BILINEAR)
        return img


transform = transforms.Compose([
    Rescale(224, 224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


net = models.googlenet(pretrained=True).float().cuda()
net.eval()
fea_net = nn.Sequential(*list(net.children())[:-2])


def sum_fscore(overlap_arr, true_sum_arr, oracle_sum):
    fscores = []
    for overlap, true_sum in zip(overlap_arr, true_sum_arr):
        precision = overlap / (oracle_sum + 1e-8);
        recall = overlap / (true_sum + 1e-8);
        if precision == 0 and recall == 0:
            fscore = 0
        else:
            fscore = 2 * precision * recall / (precision + recall)
        fscores.append(fscore)
    return sum(fscores) / len(fscores)


def get_oracle_summary(user_summary):
    n_user, n_frame = user_summary.shape
    oracle_summary = np.zeros(n_frame)
    overlap_arr = np.zeros(n_user)
    oracle_sum = 0
    true_sum_arr = user_summary.sum(axis=1)
    priority_idx = np.argsort(-user_summary.sum(axis=0))
    best_fscore = 0
    for idx in priority_idx:
        oracle_sum += 1
        for usr_i in range(n_user):
            overlap_arr[usr_i] += user_summary[usr_i][idx]
        cur_fscore = sum_fscore(overlap_arr, true_sum_arr, oracle_sum)
        if cur_fscore > best_fscore:
            best_fscore = cur_fscore
            oracle_summary[idx] = 1
        else:
            break
    tqdm.write('Overlap: '+str(overlap_arr))
    tqdm.write('True summary n_key: '+str(true_sum_arr))
    tqdm.write('Oracle smmary n_key: '+str(oracle_sum))
    tqdm.write('Final F-score: '+str(best_fscore))
    return oracle_summary


def video2fea(video_path, data_folder):
    video = cv2.VideoCapture(video_path.as_uri())
    idx = video_path.as_uri().split('.')[0].split('/')[-1]
    tqdm.write('Processing video '+idx)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    ratio = length//320
    # ratio = max(fps//30, 1)
    fea = []
    i = 0
    success, frame = video.read()
    time = -1./fps
    times = []
    while success:
        time += 1./fps
        if ratio>0 and (i+1) % ratio == 0:
            times.append(time)
            fea.append(fea_net(transform(Image.fromarray(frame)).cuda().unsqueeze(0)).squeeze().detach().cpu())
        i += 1
        success, frame = video.read()

    fea = fea[:320]
    times = times[:320]
    dname = data_folder + idx + '.csv'
    header = """frame_count, fps
    %i, %i
    length of time: %i
    """%(length, fps, %len(times))  + ''.join(['%f, ' for i in range(len(times))]) %(tuple(times)) + """

    features:
    """
    try:
        fea = torch.stack(fea)
        np.savetxt(dname, fea, delimiter = ",", header = header)
    except:
        fea = [-1]
        np.savetxt(dname, fea, delimiter = ",", header = header)


def make_dataset(video_dir, data_folder):
    video_dir = Path(video_dir).resolve()
    video_list = list(video_dir.glob('*.MP4'))
    video_list.sort()
    file_list = set(os.listdir(data_folder))
    for video_path in tqdm(video_list, desc='Video', ncols=80, leave=False):
        idx = video_path.as_uri().split('.')[0].split('/')[-1]
        file_name = idx + '.csv'
        if file_name not in file_list:
            video2fea(video_path, data_folder)



if __name__ == '__main__':
    make_dataset(video_dir, data_folder)


# vsumm_data.close()
