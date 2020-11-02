from typing import Dict
from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18,resnet50,resnet34
from tqdm import tqdm
from prettytable import PrettyTable
from pathlib import Path
import os
from torch import Tensor
from pathlib import Path
from argparse import ArgumentParser
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST
from tqdm import tqdm
import torch.distributed
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel
import ignite.distributed as idist
try:
    from tensorboardX import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        raise RuntimeError(
            "This module requires either tensorboardX or torch >= 1.2.0. "
            "You may install tensorboardX with command: \n pip install tensorboardX \n"
            "or upgrade PyTorch using your package manager of choice (pip or conda)."
        )

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator, Engine
# from ignite.metrics import Accuracy, Loss
from ignite.handlers import Checkpoint, DiskSaver
from ignite.utils import manual_seed
from datetime import datetime
import pandas as pd
"""lstm_train_test.py runs the LSTM baselines training/inference on forecasting dataset.

Note: The training code for these baselines is covered under the patent <PATENT_LINK>.

Example usage:
python lstm_train_test.py 
    --model_path saved_models/lstm.pth.tar 
    --test_features ../data/forecasting_data_test.pkl 
    --train_features ../data/forecasting_data_train.pkl 
    --val_features ../data/forecasting_data_val.pkl 
    --use_delta --normalize
"""
import matplotlib.pyplot as plt

import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Tuple, Union
import argparse
import joblib
from joblib import Parallel, delayed
import numpy as np
import pickle as pkl
from termcolor import cprint
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
# from logger import Logger
import utils.baseline_config as config
import utils.baseline_utils as baseline_utils
from utils.map_features_utils import MapFeaturesUtils

from utils.raster_utils import RasterDataset

global_step = 0
best_loss = float("inf")
np.random.seed(100)

ROLLOUT_LENS = [1, 10, 30]
args = argparse.Namespace()
args.end_epoch=5000
args.joblib_batch_size=100
args.lr=0.001
args.model_path=None
args.normalize=True
args.obs_len=20
args.pred_len=30
args.test=False
args.test_batch_size=512
args.test_features='forecasting_features_val.pkl'
args.train_batch_size=512
args.train_features='forecasting_features_val.pkl'
args.traj_save_path=None
args.use_delta=False
args.use_map=False
args.use_social=False
args.val_batch_size=512
args.val_features='forecasting_features_val.pkl'
# key for getting feature set
    # Get features
if args.use_map and args.use_social:
    baseline_key = "map_social"
elif args.use_map:
    baseline_key = "map"
elif args.use_social:
    baseline_key = "social"
else:
    baseline_key = "none"



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


weight_path="model_multi_update_lyft_public.pth"


class LyftMultiModel(nn.Module):

    def __init__(self, num_modes=3):
        super().__init__()

        # TODO: support other than resnet18?
        backbone = resnet34(pretrained=True, progress=True)
        self.backbone = backbone

        num_in_channels = 3 

        self.backbone.conv1 = nn.Conv2d(
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False,
        )

        # This is 512 for resnet18 and resnet34;
        # And it is 2048 for the other resnets 
        # low layer: 100352 - high layer but no pool :25088
        backbone_out_features = 512
#         self.cov_insted_max=nn.Conv2d(
#             self.backbone.conv1.out_channels,
#             self.backbone.conv1.out_channels,
#             kernel_size=self.backbone.maxpool.kernel_size,
#             stride=self.backbone.maxpool.stride,
#             padding=self.backbone.maxpool.padding,
#         )
#         self.cov_insted_avg=nn.Conv2d(
#             kernel_size=self.backbone.avgpool.kernel_size,
#         )
        # X, Y coords for the future positions (output shape: Bx50x2)

        # You can add more layers here.
        self.head = nn.Sequential(
            nn.Linear(in_features=backbone_out_features, out_features=4096),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(8192, out_features=4096),
#             nn.Dropout(0.1),
#             nn.ReLU(),
#             nn.Linear(4096, out_features=2048)
        )

        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes
        
        
        self.rlu = nn.ReLU()
        self.logit = nn.Linear(4096, out_features=60)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.head(x)
        x = self.logit(x)

        return x



def pytorch_neg_multi_log_likelihood_batch(
    gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor
) -> Tensor:
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (batch_size, future_len, num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    assert torch.allclose(torch.sum(confidences, dim=1), confidences.new_ones((batch_size,))), "confidences should sum to 1"
    assert avails.shape == (batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(gt).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
    # print("error", error)
    return torch.mean(error)






def log_model_weights(engine, model=None, fp=None, **kwargs):
    """Helper method to log norms of model weights: print and dump into a file
    """
    assert model and fp
    output = {"total": 0.0}
    max_counter = 5
    for name, p in model.named_parameters():
        name = name.replace(".", "/")
        n = torch.norm(p)
        if max_counter > 0:
            output[name] = n
        output["total"] += n
        max_counter -= 1

    msg = "{} | {}: {}".format(
        engine.state.epoch, engine.state.iteration, " - ".join(["{}:{:.4f}".format(m, v) for m, v in output.items()])
    )

    with open(fp, "a") as h:
        h.write(msg)
        h.write("\n")


def log_model_grads(engine, model=None, fp=None, **kwargs):
    """Helper method to log norms of model gradients: print and dump into a file
    """
    assert model and fp
    output = {"grads/total": 0.0}
    max_counter = 5
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        name = name.replace(".", "/")
        n = torch.norm(p.grad)
        if max_counter > 0:
            output["grads/{}".format(name)] = n
        output["grads/total"] += n
        max_counter -= 1

    msg = "{} | {}: {}".format(
        engine.state.epoch, engine.state.iteration, " - ".join(["{}:{:.4f}".format(m, v) for m, v in output.items()])
    )

    with open(fp, "a") as h:
        h.write(msg)
        h.write("\n")


def log_data_stats(engine, fp=None, **kwargs):
    """Helper method to log mean/std of input batch of images and median of batch of targets.
    """
    assert fp
    x, y = engine.state.batch
    output = {
        "batch xmean": x.mean().item(),
        "batch xstd": x.std().item(),
        "batch ymedian": y.median().item(),
    }

    msg = "{} | {}: {}".format(
        engine.state.epoch, engine.state.iteration, " - ".join(["{}:{:.7f}".format(m, v) for m, v in output.items()])
    )

    with open(fp, "a") as h:
        h.write(msg)
        h.write("\n")


        
def get_data_loaders(args, baseline_key):
    
    data_dict = baseline_utils.get_data(args, baseline_key)
    # # Get PyTorch Dataset
    train_dataset = RasterDataset(data_dict, args, "train")
    print(train_dataset)
    print(len(train_dataset))
    val_dataset = RasterDataset(data_dict, args, "val")
    print(val_dataset)
    print(len(val_dataset))
    return train_dataset,val_dataset,

def forward(data, model, device,criterion):
    
    img = data[2].to(device)
    targets = data[1].to(device)
    
    # Forward pass
    outputs = model(img)
    loss = criterion(targets, outputs, )
    return  loss,outputs


