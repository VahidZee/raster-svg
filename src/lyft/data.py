import torch
from l5kit.dataset import AgentDataset as _AgentDataset
from l5kit.rasterization import build_rasterizer as _build_rasterizer

from l5kit.rasterization import SemanticRasterizer, SemBoxRasterizer
from src.lyft.rasterizer import (
    render_semantic_map, rasterize_semantic, rasterize_sem_box, rasterize_box, get_frame)

import functools
import copy
import types
from deepsvg.svglib.svg import SVG
from src.lyft.utils import apply_colors


def build_rasterizer(config, data_manager):
    map_type = config['raster_params']['map_type']
    config_prime = copy.deepcopy(config)
    base_map_type = config_prime['raster_params']['map_type'] = map_type.replace('svg_', '').replace('tensor_', '')

    rasterizer = _build_rasterizer(config_prime, data_manager)
    tl_face_color = not config['raster_params']['disable_traffic_light_faces']
    svg = map_type.startswith('svg_')
    tensor = map_type.startswith('tensor_')
    if svg or tensor:
        svg_args = config['raster_params'].get('svg_args', dict())
        render_semantics = functools.partial(render_semantic_map, tl_face_color=tl_face_color)
        if isinstance(rasterizer, SemanticRasterizer):
            rasterize_sem = functools.partial(
                rasterize_semantic, svg=svg, svg_args=svg_args)
            rasterizer.render_semantic_map = types.MethodType(render_semantics, rasterizer)
            rasterizer.rasterize = types.MethodType(rasterize_sem, rasterizer)

        if isinstance(rasterizer, SemBoxRasterizer):
            rasterize_sem = functools.partial(rasterize_semantic, svg=False, svg_args=None)
            rasterize_sembox = functools.partial(rasterize_sem_box, svg=svg, svg_args=svg_args)
            rasterizer.sat_rast.render_semantic_map = types.MethodType(render_semantics, rasterizer.sat_rast)
            rasterizer.sat_rast.rasterize = types.MethodType(rasterize_sem, rasterizer.sat_rast)
            rasterizer.rasterize = types.MethodType(rasterize_sembox, rasterizer)
            rasterizer.box_rast.rasterize = types.MethodType(rasterize_box, rasterizer.box_rast)

    return rasterizer


def agent_dataset(cfg: dict, zarr_dataset, rasterizer, perturbation=None, agents_mask=None, min_frame_history=10,
                  min_frame_future=1):
    data = _AgentDataset(cfg, zarr_dataset, rasterizer, perturbation, agents_mask, min_frame_history, min_frame_future)
    map_type = cfg['raster_params']['map_type']
    svg = map_type.startswith('svg_')
    tensor = map_type.startswith('tensor_')
    if svg or tensor:
        data.get_frame = types.MethodType(get_frame, data)
    return data


class AgentDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: dict, zarr_dataset, rasterizer, perturbation=None, agents_mask=None, min_frame_history=10,
                 min_frame_future=1):
        super().__init__()
        map_type = cfg['raster_params']['map_type']
        self.svg_args = cfg['raster_params'].get('svg_args', dict())
        self.svg = map_type.startswith('svg_')
        self.svg_cmds = self.svg_args.get('return_cmds', True)
        self.tensor = map_type.startswith('tensor_')
        self.data = agent_dataset(
            cfg, zarr_dataset, rasterizer, perturbation, agents_mask, min_frame_history, min_frame_future)

    def __getitem__(self, i):
        item = self.data[i]
        if self.svg and self.svg_cmds:
            tens = SVG.from_tensor(item['path']).simplify().split_paths().to_tensor(concat_groups=False)
            svg = apply_colors(tens, item['path_type'])
            del item['path']
            del item['path_type']
            item['svg'] = svg
        return item

    def __len__(self):
        return len(self.data)
