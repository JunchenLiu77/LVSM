# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import random
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from einops import rearrange
import imageio



def create_video_from_frames(frames, output_video_file, framerate=30):
    """
    Creates a video from a sequence of frames.

    Parameters:
        frames (numpy.ndarray): Array of image frames (shape: N x H x W x C).
        output_video_file (str): Path to save the output video file.
        framerate (int, optional): Frames per second for the video. Default is 30.
    """
    frames = np.asarray(frames)

    # Normalize frames if values are in [0,1] range
    if frames.max() <= 1:
        frames = (frames * 255).astype(np.uint8)

    imageio.mimsave(output_video_file, frames, fps=framerate, quality=8)



class ProcessData(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @torch.no_grad()
    def compute_rays(self, c2w, fxfycxcy, h=None, w=None, device="cuda"):
        """
        Args:
            c2w (torch.tensor): [b, v, 4, 4]
            fxfycxcy (torch.tensor): [b, v, 4]
            h (int): height of the image
            w (int): width of the image
        Returns:
            ray_o (torch.tensor): [b, v, 3, h, w]
            ray_d (torch.tensor): [b, v, 3, h, w]
        """

        b, v = c2w.size()[:2]
        c2w = c2w.reshape(b * v, 4, 4)

        fx, fy, cx, cy = fxfycxcy[:,:, 0], fxfycxcy[:,:,  1], fxfycxcy[:,:,  2], fxfycxcy[:,:,  3]
        h_orig = int(2 * cy.max().item())  # Original height (estimated from the intrinsic matrix)
        w_orig = int(2 * cx.max().item())  # Original width (estimated from the intrinsic matrix)
        if h is None or w is None:
            h, w = h_orig, w_orig

        # in case the ray/image map has different resolution than the original image
        if h_orig != h or w_orig != w:
            fx = fx * w / w_orig
            fy = fy * h / h_orig
            cx = cx * w / w_orig
            cy = cy * h / h_orig

        fxfycxcy = fxfycxcy.reshape(b * v, 4)
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        y, x = y.to(device), x.to(device)
        x = x[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        y = y[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
        y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
        z = torch.ones_like(x)
        ray_d = torch.stack([x, y, z], dim=2)  # [b*v, h*w, 3]
        ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [b*v, h*w, 3]
        ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # [b*v, h*w, 3]
        ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [b*v, h*w, 3]

        ray_o = rearrange(ray_o, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)
        ray_d = rearrange(ray_d, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)

        return ray_o, ray_d
    
    def fetch_views(self, data_batch, num_input_views, num_target_views, num_ss_views, num_ood_target_views, has_target_image=True, training=True):
        """
        Splits the input data batch into ss, input and target sets.
        
        Args:
            data_batch (dict): Contains input tensors with the following keys:
                - 'image' (torch.Tensor): Shape [b, v, c, h, w], optional for some target views
                - 'fxfycxcy' (torch.Tensor): Shape [b, v, 4]
                - 'c2w' (torch.Tensor): Shape [b, v, 4, 4]
            num_input_views (int): Number of encoder input views.
            num_target_views (int): Number of target views.
            num_ss_views (int): Number of views for self-supervision.
            num_ood_target_views (int): Number of ood target views.
            has_target_image (bool): If True, target views have image supervision.
            training (bool): If True, use training mode.

        Returns:
            tuple: (ss_dict, input_dict, target_dict), all as EasyDict objects.

        """
        # # randomize input views if dynamic_input_view_num is True and not in inference mode
        # if (self.config.training.get("dynamic_input_view_num", False) 
        #     and (not self.config.inference.get("if_inference", False))):
        #     num_input_views = np.random.randint(2, 5)

        input_dict, target_dict, ss_dict, ood_target_dict = {}, {}, {}, {}
        # index = [] save for future use if we want to select specific views

        num_views, bs = data_batch["c2w"].size(1), data_batch["image"].size(0)
        assert num_input_views + num_target_views + num_ss_views + num_ood_target_views == num_views, f"We have {num_views} views, but we want to select {num_input_views} input views, {num_target_views} target views, {num_ss_views} self-supervision views, and {num_ood_target_views} ood target views. This is more than the total number of views we have."
        assert num_ss_views >= 2, f"We need at least 2 self-supervision views, but we want to select {num_ss_views} self-supervision views."

        if training:
            # During training, target views can be the same as the input views and ood_target views can be the same as the ss views
            input_indices = torch.tensor(
                [[j for j in range(num_input_views)] for _ in range(bs)], dtype=torch.long, device=data_batch["image"].device)
            target_indices = torch.tensor(
                [random.sample(range(num_input_views + num_target_views), num_target_views) for _ in range(bs)], dtype=torch.long, device=data_batch["image"].device)
            ss_indices = torch.tensor(
                [[j + num_input_views + num_target_views for j in range(num_ss_views)] for _ in range(bs)], dtype=torch.long, device=data_batch["image"].device)
            ood_target_indices = torch.tensor(
                [random.sample(range(num_input_views + num_target_views + num_ss_views, num_views), num_ood_target_views) for _ in range(bs)], dtype=torch.long, device=data_batch["image"].device)
        else:
            # Otherwise, the indices has been specified in the index file. We follow the indices specified.
            input_indices = torch.tensor(
                [[j for j in range(num_input_views)] for _ in range(bs)], dtype=torch.long, device=data_batch["image"].device)
            target_indices = torch.tensor(
                [[j + num_input_views for j in range( num_target_views)] for _ in range(bs)], dtype=torch.long, device=data_batch["image"].device)
            ss_indices = torch.tensor(
                [[j + num_input_views + num_target_views for j in range(num_ss_views)] for _ in range(bs)], dtype=torch.long, device=data_batch["image"].device)
            ood_target_indices = torch.tensor(
                [[j + num_input_views + num_target_views + num_ss_views for j in range(num_ood_target_views)] for _ in range(bs)], dtype=torch.long, device=data_batch["image"].device)
        
        input_indices = torch.sort(input_indices, dim=1).values # [b, num_input_views]
        target_indices = torch.sort(target_indices, dim=1).values # [b, num_target_views]
        ss_indices = torch.sort(ss_indices, dim=1).values # [b, num_ss_views]
        ood_target_indices = torch.sort(ood_target_indices, dim=1).values # [b, num_ood_target_views]

        for key, value in data_batch.items():
            if key == "scene_name":
                input_dict[key] = value
                target_dict[key] = value
                ss_dict[key] = value
                ood_target_dict[key] = value
                continue

            to_expand_dim = value.shape[2:] # [b, v, (value dim)] -> [value dim], e.g. [c, h, w] or [4] or [4, 4]
            
            expanded_input_index = input_indices.view(input_indices.shape[0], input_indices.shape[1], *(1,) * len(to_expand_dim)).expand(-1, -1, *to_expand_dim)
            expanded_target_index = target_indices.view(target_indices.shape[0], target_indices.shape[1], *(1,) * len(to_expand_dim)).expand(-1, -1, *to_expand_dim)
            expanded_ss_index = ss_indices.view(ss_indices.shape[0], ss_indices.shape[1], *(1,) * len(to_expand_dim)).expand(-1, -1, *to_expand_dim)
            expanded_ood_target_index = ood_target_indices.view(ood_target_indices.shape[0], ood_target_indices.shape[1], *(1,) * len(to_expand_dim)).expand(-1, -1, *to_expand_dim)
            
            input_dict[key] = torch.gather(value, dim=1, index=expanded_input_index)
            target_dict[key] = torch.gather(value, dim=1, index=expanded_target_index)
            ss_dict[key] = torch.gather(value, dim=1, index=expanded_ss_index)
            ood_target_dict[key] = torch.gather(value, dim=1, index=expanded_ood_target_index)
        
        height, width = data_batch["image"].shape[3], data_batch["image"].shape[4]
        input_dict["image_h_w"] = (height, width)
        target_dict["image_h_w"] = (height, width)
        ss_dict["image_h_w"] = (height, width)
        ood_target_dict["image_h_w"] = (height, width)
        input_dict, target_dict, ss_dict, ood_target_dict = edict(input_dict), edict(target_dict), edict(ss_dict), edict(ood_target_dict)
        return input_dict, target_dict, ss_dict, ood_target_dict


    
    @torch.no_grad()
    def forward(self, data_batch, num_input_views, num_target_views, num_ss_views, num_ood_target_views, has_target_image=True, training=True, compute_rays=True):
        """
        Preprocesses the input data batch and (optionally) computes ray_o and ray_d.

        Args:
            data_batch (dict): Contains input tensors with the following keys:
                - 'image' (torch.Tensor): Shape [b, v, c, h, w]
                - 'fxfycxcy' (torch.Tensor): Shape [b, v, 4]
                - 'c2w' (torch.Tensor): Shape [b, v, 4, 4]
            num_input_views (int): Number of encoder input views.
            num_target_views (int): Number of target views.
            num_ss_views (int): Number of views for self-supervision.
            num_ood_target_views (int): Number of ood target views.
            has_target_image (bool): If True, target views have image supervision.
            training (bool): If True, use training mode.
            compute_rays (bool): If True, compute ray_o and ray_d.
                
        Returns:
            Input and Target data_batch (dict): Contains processed tensors with the following keys:
                - 'image' (torch.Tensor): Shape [b, v, c, h, w]
                - 'fxfycxcy' (torch.Tensor): Shape [b, v, 4]
                - 'c2w' (torch.Tensor): Shape [b, v, 4, 4]
                - 'ray_o' (torch.Tensor): Shape [b, v, 3, h, w]
                - 'ray_d' (torch.Tensor): Shape [b, v, 3, h, w]
                - 'image_h_w' (tuple): (height, width)
        """
        input_dict, target_dict, ss_dict, ood_target_dict = self.fetch_views(data_batch, num_input_views=num_input_views, num_target_views=num_target_views, num_ss_views=num_ss_views, num_ood_target_views=num_ood_target_views, has_target_image=has_target_image, training=training)

        if compute_rays:
            for dict in [input_dict, target_dict, ss_dict, ood_target_dict]: 
                c2w = dict["c2w"]
                fxfycxcy = dict["fxfycxcy"]
                image_height, image_width = dict["image_h_w"]    
                ray_o, ray_d = self.compute_rays(c2w, fxfycxcy, image_height, image_width, device=data_batch["image"].device)
                dict["ray_o"], dict["ray_d"] = ray_o, ray_d

        return input_dict, target_dict, ss_dict, ood_target_dict