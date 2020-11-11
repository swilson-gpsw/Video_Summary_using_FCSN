import numpy as np
import os
import torch
from fcsn import FCSN


"""
Load trained model

Infer model

Access

"""

def load_data(file):

    with open(data_file, 'r') as fid:
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


if __name__ == "__main__":
    # data_fol = "/Users/swilson/Projects/CVPR_20/data/"
    # for fi in os.listdir(data_fol)[]:
    #     data_file = data_fol + fi
    #
    #     time, data = load_data(data_file)

    model = load_model()
