import torch
import functools
import copy
import types
from deepsvg.svglib.svg import SVG, Bbox
from deepsvg.config import _Config
from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svglib.svg import SVG
from deepsvg.svglib.geom import Point

import math
import torch
import torch.utils.data
import random
from typing import List, Union, Dict, Any
import pandas as pd
import os
import pickle

import csv

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict: Dict[str, Any], args: Any, mode: str):
        """Initialize the Dataset.

        Args:
            data_dict: Dict containing all the data
            args: Arguments passed to the baseline code
            mode: train/val/test mode

        """
        self.data_dict = data_dict
        self.args = args
        self.mode = mode

        # Get input
        self.input_data = data_dict["{}_input".format(mode)]
        if mode != "test":
            self.output_data = data_dict["{}_output".format(mode)]
        self.data_size = self.input_data.shape[0]

        # Get helpers
        self.helpers = self.get_helpers()
        self.helpers = list(zip(*self.helpers))
        
        
        
        from argoverse.map_representation.map_api import ArgoverseMap

        self.avm = ArgoverseMap()
        self.mf=MapFeaturesUtils()
        

    def __len__(self):
        """Get length of dataset.

        Returns:
            Length of dataset

        """
        return self.data_size

    def __getitem__(self, idx: int
                    ) -> Tuple[torch.FloatTensor, Any, Dict[str, np.ndarray]]:
        """Get the element at the given index.

        Args:
            idx: Query index

        Returns:
            A list containing input Tensor, Output Tensor (Empty if test) and viz helpers. 

        """
        helper=self.helpers[idx]
        cnt_lines,img,cnt_lines_norm=self.mf.get_candidate_centerlines_for_trajectory(
                        helper[0] if self.mode != "test"  else  helper[0][:20],
                        yaw_deg=helper[5],centroid=helper[0][0],
                        city_name=helper[1][0],avm=self.avm,
            viz=True,
            seq_len = 80,
            max_candidates=10,
            )
        
        
        res = torch.cat([linear_path_to_tensor(path, -1) for path in cnt_lines_norm], 0)

        return {"history_positions": torch.FloatTensor(self.input_data[idx]),
                "target_positions": torch.empty(1) if self.mode == "test" else torch.FloatTensor(self.output_data[idx]),
                "path":res,
               }
    
    def get_helpers(self) -> Tuple[Any]:
        """Get helpers for running baselines.

        Returns:
            helpers: Tuple in the format specified by LSTM_HELPER_DICT_IDX

        Note: We need a tuple because DataLoader needs to index across all these helpers simultaneously.

        """
        helper_df = self.data_dict[f"{self.mode}_helpers"]
        candidate_centerlines = helper_df["CANDIDATE_CENTERLINES"].values
#         print("ss",candidate_centerlines)
        candidate_nt_distances = helper_df["CANDIDATE_NT_DISTANCES"].values
        xcoord = np.stack(helper_df["FEATURES"].values
                          )[:, :, config.FEATURE_FORMAT["X"]].astype("float")
        ycoord = np.stack(helper_df["FEATURES"].values
                          )[:, :, config.FEATURE_FORMAT["Y"]].astype("float")
        centroids = np.stack((xcoord, ycoord), axis=2)
        _DEFAULT_HELPER_VALUE = np.full((centroids.shape[0]), None)
        city_names = np.stack(helper_df["FEATURES"].values
                              )[:, :, config.FEATURE_FORMAT["CITY_NAME"]]
        seq_paths = helper_df["SEQUENCE"].values
        translation = (helper_df["TRANSLATION"].values
                       if self.args.normalize else _DEFAULT_HELPER_VALUE)
        rotation = (helper_df["ROTATION"].values
                    if self.args.normalize else _DEFAULT_HELPER_VALUE)

        use_candidates = self.args.use_map and self.mode == "test"

        candidate_delta_references = (
            helper_df["CANDIDATE_DELTA_REFERENCES"].values
            if self.args.use_map and use_candidates else _DEFAULT_HELPER_VALUE)
        delta_reference = (helper_df["DELTA_REFERENCE"].values
                           if self.args.use_delta and not use_candidates else
                           _DEFAULT_HELPER_VALUE)

        helpers = [None for i in range(len(config.LSTM_HELPER_DICT_IDX))]

        # Name of the variables should be the same as keys in LSTM_HELPER_DICT_IDX
        for k, v in config.LSTM_HELPER_DICT_IDX.items():
            helpers[v] = locals()[k.lower()]

        return tuple(helpers)

