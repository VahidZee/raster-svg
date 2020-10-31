from argparse import ArgumentParser
import typing as th
from torch.utils.data import DataLoader, Subset, random_split, Dataset
import pandas as pd
import numpy as np
from deepsvg.config import _Config
import torch.nn as nn
from src.lyft.data import build_rasterizer, AgentDataset
from l5kit.data import LocalDataManager, ChunkedDataset
import argparse
import importlib
from l5kit.configs import load_config_data
from src.lyft.data import AgentDataset

from src.lyft.models.model_trajectory import ModelTrajectory
from src.lyft.train.utils import neg_multi_log_likelihood

from deepsvg.utils import Stats, TrainVars, Timer
import torch
from deepsvg import utils
from datetime import datetime
from tensorboardX import SummaryWriter
from deepsvg.utils.stats import SmoothedValue
import os

from torch.utils.data.dataloader import default_collate
from collections import defaultdict


def my_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    # print(batch)
    # batch = filter(lambda x:x is not None, batch)
    batch = list(filter(None, batch))
    # print(batch)
    if len(batch) > 0:
        return default_collate(batch)
    else:
      return


def train(model_cfg:_Config,data_cfg, data_path, model_name, experiment_name="", log_dir="./logs", debug=False, resume=" "):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # set env variable for data
    model_cfg.print_params()
    dm = LocalDataManager(data_path)
    # get config
    rasterizer = build_rasterizer(data_cfg, dm)

    train_zarr = ChunkedDataset(dm.require(data_cfg["train_dataloader"]["split"])).open()
    val_zarr = ChunkedDataset(dm.require(data_cfg["val_dataloader"]["split"])).open()

    train_dataset = AgentDataset(data_cfg=data_cfg, zarr_dataset = train_zarr, rasterizer = rasterizer,
                                 model_args=model_cfg.model_args, max_num_groups=model_cfg.max_num_groups,
                                 max_seq_len=model_cfg.max_seq_len,min_frame_future=50)
    val_dataset = AgentDataset(data_cfg=data_cfg, zarr_dataset = val_zarr, rasterizer = rasterizer,
                                 model_args=model_cfg.model_args, max_num_groups=model_cfg.max_num_groups,
                                 max_seq_len=model_cfg.max_seq_len,min_frame_future=50)

    if model_cfg.train_idxs is not None:
        train_dataset = Subset(train_dataset, model_cfg.train_idxs)
    if model_cfg.val_idxs is not None:
        val_dataset = Subset(val_dataset, model_cfg.val_idxs)

    train_dataloader = DataLoader(train_dataset, batch_size=model_cfg.train_batch_size, shuffle=True,
                            num_workers=model_cfg.loader_num_workers,collate_fn=my_collate)
    validat_dataloader = DataLoader(val_dataset, batch_size=model_cfg.val_batch_size, shuffle=False,
                                  num_workers=model_cfg.loader_num_workers,collate_fn=my_collate)

    model = ModelTrajectory(model_cfg=model_cfg, data_config= data_cfg, modes=3).to(device)
    stats = Stats(num_steps=model_cfg.num_steps, num_epochs=model_cfg.num_epochs, steps_per_epoch=len(train_dataloader),
                  stats_to_print=model_cfg.stats_to_print)
    stats.stats['val'] = defaultdict(SmoothedValue)
    print(stats.stats.keys())
    train_vars = TrainVars()
    timer = Timer()

    stats.num_parameters = utils.count_parameters(model)
    print(f"#Parameters: {stats.num_parameters:,}")

    # Summary Writer
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    experiment_identifier = f"{model_name}_{experiment_name}_{current_time}"

    summary_writer = SummaryWriter(os.path.join(log_dir, "tensorboard", "debug" if debug else "full", experiment_identifier))
    checkpoint_dir = os.path.join(log_dir, "models", model_name, experiment_name)
    print(checkpoint_dir)

    # model_cfg.set_train_vars(train_vars, train_dataloader)

    # Optimizer, lr & warmup schedulers
    optimizers = model_cfg.make_optimizers(model)
    scheduler_lrs = model_cfg.make_schedulers(optimizers, epoch_size=len(train_dataloader))
    scheduler_warmups = model_cfg.make_warmup_schedulers(optimizers, scheduler_lrs)

    loss_fns = [l.to(device) for l in model_cfg.make_losses()]

    if not resume == " ":
        ckpt_exists = utils.load_ckpt_list(checkpoint_dir, model, None, optimizers, scheduler_lrs, scheduler_warmups, stats, train_vars)

    if not resume == " " and ckpt_exists:
        print(f"Resuming model at epoch {stats.epoch+1}")
        stats.num_steps = model_cfg.num_epochs * len(train_dataloader)
    if True:
        # Run a single forward pass on the single-device model for initialization of some modules
        single_foward_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True,
                                              num_workers=0 , collate_fn=my_collate)
        data = next(iter(single_foward_dataloader))
        if data is not None:
            model_args, params_dict = [data['image'][arg].to(device) for arg in model_cfg.model_args], model_cfg.get_params(0, 0)
            entery = [*model_args,{},True]
            out = model(entery)


    model = nn.DataParallel(model)

    epoch_range = utils.infinite_range(stats.epoch) if model_cfg.num_epochs is None else range(stats.epoch, cfg.num_epochs)
    print(epoch_range)
    timer.reset()
    print(timer.get_elapsed_time())
    for epoch in epoch_range:
        print(f"Epoch {epoch+1}")
        print(timer.get_elapsed_time())
        for n_iter, data in enumerate(train_dataloader):
            if data is None:
                continue
            print(timer.get_elapsed_time())
            step = n_iter + epoch * len(train_dataloader)

            if model_cfg.num_steps is not None and step > model_cfg.num_steps:
                return

            model.train()
            model_args = [data['image'][arg].to(device) for arg in model_cfg.model_args]
            params_dict, weights_dict = model_cfg.get_params(step, epoch), model_cfg.get_weights(step, epoch)

            for i, (loss_fn, optimizer, scheduler_lr, scheduler_warmup, optimizer_start) in enumerate(zip(loss_fns, optimizers, scheduler_lrs, scheduler_warmups, model_cfg.optimizer_starts), 1):
                optimizer.zero_grad()
                entery = [*model_args, params_dict, True]
                output,conf = model(entery)
                loss_dict = {}
                loss_dict['loss'] = neg_multi_log_likelihood(data['target_positions'].to(device), output, conf, data.get('target_availabilities', None).to(device)).mean()
                if step >= optimizer_start:
                    loss_dict['loss'].backward()
                    if model_cfg.grad_clip is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), model_cfg.grad_clip)

                    optimizer.step()
                    if scheduler_lr is not None:
                        scheduler_lr.step()
                    if scheduler_warmup is not None:
                        scheduler_warmup.step()

                stats.update_stats_to_print("train", loss_dict)
                stats.update("train", step, epoch, {
                    ("lr" if i == 1 else f"lr_{i}"): optimizer.param_groups[0]['lr'],
                    **loss_dict
                })

            stats.update("train", step, epoch, {
                **weights_dict,
                "time": timer.get_elapsed_time()
            })

            if step % model_cfg.log_every == 0:
                print(stats.get_summary("train"))
                stats.write_tensorboard(summary_writer, "train")
                summary_writer.flush()
                timer.reset()

            if step % model_cfg.val_every == 0:
                timer.reset()
                # validation(validat_dataloader, model, model_cfg, device, epoch, stats, summary_writer, timer,optimizer.param_groups[0]['lr'])
                timer.reset()

            if not debug and step % model_cfg.ckpt_every == 0:
                utils.save_ckpt_list(checkpoint_dir, model, model_cfg, optimizers, scheduler_lrs, scheduler_warmups, stats, train_vars)
                print("save checkpoint")




def validation(val_dataloader,model,model_cfg,device,epoch,stats,summary_writer,timer,lr):
    model.eval()
    for n_iter, data in enumerate(val_dataloader):
        if data is None:
            continue
        step = n_iter

        if model_cfg.val_num_steps is not None and step > model_cfg.val_num_steps:
            return

        model_args = [data['image'][arg].to(device) for arg in model_cfg.model_args]
        params_dict, weights_dict = model_cfg.get_params(step, epoch), model_cfg.get_weights(step, epoch)

        entery = [*model_args, params_dict, True]
        output,conf = model(entery)
        loss_dict = {}
        loss_dict['loss'] = neg_multi_log_likelihood(data['target_positions'].to(device), output, conf, data.get('target_availabilities', None).to(device)).mean()

        stats.update_stats_to_print("val", loss_dict)

        stats.update("val", step, epoch, {
            **loss_dict
        })

        stats.update("val", step, epoch, {
            **weights_dict,
            "time": timer.get_elapsed_time()
        })

        if step % model_cfg.log_every == 0:
            print(stats.get_summary("val"))
            stats.write_tensorboard(summary_writer, "val")
            summary_writer.flush()
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepSVG Trainer')
    parser.add_argument("--config-module", type=str, required=True)
    parser.add_argument("--config-data", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default=" ")
    parser.add_argument("--val_idxs", type=str, default=None)
    parser.add_argument("--train_idxs", type=str, default=None)

    args = parser.parse_args()

    cfg = importlib.import_module(args.config_module).Config()
    model_name, experiment_name = args.config_module.split(".")[-2:]
    print(model_name,experiment_name)
    if args.val_idxs is not None:
        cfg.val_idxs = args.val_idxs
    if args.train_idxs is not None:
        cfg.train_idxs = args.train_idxs
    config_data = load_config_data(args.config_data)
    train(cfg, config_data, args.data_path, model_name, experiment_name, log_dir=args.log_dir, debug=args.debug, resume=args.resume)
