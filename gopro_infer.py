import numpy as np
import os
import torch
from fcsn import FCSN
import json

from config import Config


"""
Load trained model

Infer model

Access

"""

def load_data(file):

    with open(file, 'r') as fid:
        fid.readline()
        meta = fid.readline()[1:]
        frame_count, fps = np.array(meta.split(','), dtype = 'float')
        ratio = frame_count//320
        frame_no = 1
        time = []
        data = None
        for line in fid.readlines()[2:]:
            frame_no += ratio
            time.append(frame_no/fps)

            d_line = np.array(line.split(','), dtype = 'float')
            data = d_line if data is None else np.vstack((data, d_line))
    return time, data

def load_model(model_path = 'save_dir/epoch-49.pkl'):
    model = FCSN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def infer_model(model, data, config):
    feature = torch.tensor(data.T, dtype = torch.float32)
    # print(feature.shape)
    if config.gpu:
        feature = feature.cuda()
        model = model.cuda()
    dim1 = min(feature.shape[1], 320)
    feature = feature[:,:dim1]
    feature = feature.unsqueeze(0)
    pred_score = model(feature).squeeze(0)

    score = torch.softmax(pred_score, dim=0)

    score1 = score[1].cpu()
    score1 = score1.detach().numpy()
    score1 = list(score1.astype(float))

    score0 = score[0].cpu()
    score0 = score0.detach().numpy()
    score0 = list(score0.astype(float))



    return score0, score1

if __name__ == "__main__":
    data_fol = "/mnt/hd02/CVPR/h5/"
    files = [data_fol + fi for fi in os.listdir(data_fol)]
    # files = ['/mnt/hd02/CVPR/h5/594457e9546fd50001e4f637.csv']

    results_fol = '/mnt/hd02/CVPR/results/'
    model = load_model()
    config = Config(mode='test')
    for fi in files:
        if os.path.getsize(fi) < 8000:
            continue
        print(fi.split('/')[-1], end = '')
        time, data = load_data(fi)
        data = torch.tensor(data)
        score0, score1 = infer_model(model, data, config)
        datum_id = fi.split('/')[-1].split('.')[0]
        output = {
            'time': time,
            'score0': score0,
            'score1': score1,
            'datum_id': datum_id
        }
        line = json.dumps(output) + '/n'

        with open(results_fol + datum_id + '_results.json','w') as fod:
            fod.write(line)
        print('...Success!')
        # except:
        #     print('...Failuer!')
