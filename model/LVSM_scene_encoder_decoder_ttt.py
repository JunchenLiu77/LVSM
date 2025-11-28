# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from easydict import EasyDict as edict
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import traceback
from utils import camera_utils, data_utils
from .transformer import QK_Norm_TransformerBlock, init_weights
from .loss import LossComputer
import math

amp_dtype_mapping = {
    "fp16": torch.float16, 
    "bf16": torch.bfloat16, 
    "fp32": torch.float32, 
    'tf32': torch.float32
}

class Images2LatentScene(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.process_data = data_utils.ProcessData(config)

        # Initialize both input tokenizers, and output de-tokenizer
        self._init_tokenizers()
        
        # Initialize transformer blocks
        self._init_transformer()
        
        # Initialize TTT blocks (or say learnable optimizers)
        self._init_ttt()
        if self.config.model.ttt.distill_factor > 0.0:
            print(f"Enable encoder-optimizer distillation with factor={self.config.model.ttt.distill_factor}")
        else:
            print("No encoder-optimizer distillation is enabled")
        
        # Initialize loss computer
        self.loss_computer = LossComputer(config)
        
        # Count TTT parameters for logging
        self.ttt_param_counts = {
            'total_ttt_params': sum(p.numel() for block in self.ttt_blocks for p in block.parameters()) if self.ttt_blocks is not None else 0,
            'trainable_ttt_params': sum(p.numel() for block in self.ttt_blocks for p in block.parameters() if p.requires_grad) if self.ttt_blocks is not None else 0,
            'blocks': []
        }
        
        # Add learnable state_lr parameters to the count if they exist
        if self.config.model.ttt.state_lr_mode == "learnable":
            for lr_param in self.ttt_lrnet:
                self.ttt_param_counts['total_ttt_params'] += lr_param.numel()
                self.ttt_param_counts['trainable_ttt_params'] += lr_param.numel() if lr_param.requires_grad else 0
        elif "adaptive" in self.config.model.ttt.state_lr_mode:
            for lrnet in self.ttt_lrnet:
                self.ttt_param_counts['total_ttt_params'] += sum(p.numel() for p in lrnet.parameters())
                self.ttt_param_counts['trainable_ttt_params'] += sum(p.numel() for p in lrnet.parameters() if p.requires_grad)
        
        for i, block in enumerate(self.ttt_blocks) if self.ttt_blocks is not None else []:
            block_params = sum(p.numel() for p in block.parameters())
            block_trainable = sum(p.numel() for p in block.parameters() if p.requires_grad)
            self.ttt_param_counts['blocks'].append({
                'total': block_params,
                'trainable': block_trainable
            })


    def _create_tokenizer(self, in_channels, patch_size, d_model):
        """Helper function to create a tokenizer with given config"""
        tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=patch_size,
                pw=patch_size,
            ),
            nn.Linear(
                in_channels * (patch_size**2),
                d_model,
                bias=False,
            ),
        )
        tokenizer.apply(init_weights)
        return tokenizer


    def _init_tokenizers(self):
        """Initialize the image and target pose tokenizers, and image token decoder"""
        # Image tokenizer
        self.image_tokenizer = self._create_tokenizer(
            in_channels = self.config.model.image_tokenizer.in_channels,
            patch_size = self.config.model.image_tokenizer.patch_size,
            d_model = self.config.model.transformer.d
        )
        
        # Target pose tokenizer
        self.target_pose_tokenizer = self._create_tokenizer(
            in_channels = self.config.model.target_pose_tokenizer.in_channels,
            patch_size = self.config.model.target_pose_tokenizer.patch_size,
            d_model = self.config.model.transformer.d
        )
        
        # Image token decoder (decode image tokens into pixels)
        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(self.config.model.transformer.d, bias=False),
            nn.Linear(
                self.config.model.transformer.d,
                (self.config.model.target_pose_tokenizer.patch_size**2) * 3,
                bias=False,
            ),
            nn.Sigmoid()
        )
        self.image_token_decoder.apply(init_weights)


    def _init_transformer(self):
        """Initialize transformer blocks"""
        config = self.config.model.transformer
        use_qk_norm = config.use_qk_norm

        # latent vectors for LVSM encoder-decoder
        self.n_light_field_latent = nn.Parameter(
            torch.randn(
                config.n_latent_vectors,
                config.d,
            )
        )
        nn.init.trunc_normal_(self.n_light_field_latent, std=0.02)

        # Create transformer blocks
        self.transformer_encoder = [
            QK_Norm_TransformerBlock(
                config.d, config.d_head, use_qk_norm=use_qk_norm
            ) for _ in range(config.encoder_n_layer)
        ]

        self.transformer_decoder = [
            QK_Norm_TransformerBlock(
                config.d, config.d_head, use_qk_norm=use_qk_norm
            ) for _ in range(config.decoder_n_layer)
        ]
        
        # Apply special initialization if configured
        if config.special_init:
            # Encoder
            for idx, block in enumerate(self.transformer_encoder):
                if config.depth_init:
                    weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                else:
                    weight_init_std = 0.02 / (2 * config.encoder_n_layer) ** 0.5
                block.apply(lambda module: init_weights(module, weight_init_std))

            # Decoder
            for idx, block in enumerate(self.transformer_decoder):
                if config.depth_init:
                    weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                else:
                    weight_init_std = 0.02 / (2 * config.decoder_n_layer) ** 0.5
                block.apply(lambda module: init_weights(module, weight_init_std))  
        else:
            # Encoder
            for block in self.transformer_encoder:
                block.apply(init_weights)

            # Decoder
            for block in self.transformer_decoder:
                block.apply(init_weights)

                
        self.transformer_encoder = nn.ModuleList(self.transformer_encoder)
        self.transformer_decoder = nn.ModuleList(self.transformer_decoder)
        self.transformer_input_layernorm_decoder = nn.LayerNorm(config.d, bias=False)


    def _init_ttt(self):
        if self.config.model.ttt.opt_model == "dit":
            # Using DiT architecture to update the state, we dont explicitly update the state with some certain learning rate, 
            # instead, we use the DiT architecture to update the state and modulate the update with current time step.
            from .dit import DiT

            # Initialization will be done in the DiT class
            self.ttt_blocks = nn.ModuleList([DiT(
                hidden_size=self.config.model.transformer.d * 2,
                depth=self.config.model.ttt.n_blocks_per_layer,
                num_heads=self.config.model.transformer.d * 2 // self.config.model.transformer.d_head,
                mlp_ratio=4.0
            ) for _ in range(self.config.model.ttt.n_layer)])
            self.ttt_grad_normalizers = nn.ModuleList([nn.LayerNorm(self.config.model.transformer.d, bias=False) for _ in range(self.config.model.ttt.n_layer)])
            # initialize the state normalizer to be identity
            self.ttt_state_normalizers = nn.ModuleList([nn.Identity() for _ in range(self.config.model.ttt.n_layer)])
            return
        
        # Initialize state learning rate based on configuration
        state_lr_mode = self.config.model.ttt.state_lr_mode
        if state_lr_mode == 'learnable':
            # Initialize learnable state_lr parameters for each TTT layer
            self.ttt_lrnet = nn.ParameterList()
            init_value = self.config.model.ttt.state_lr_init
            
            for _ in range(self.config.model.ttt.n_layer):
                # Create a learnable gating vector with shape [D]
                lr_param = nn.Parameter(torch.full((self.config.model.transformer.d,), init_value))
                self.ttt_lrnet.append(lr_param)
            
            print(f"Initialized learnable state_lr with init_value={init_value}")
        elif state_lr_mode == "fixed":
            # Use fixed state_lr from config
            print(f"Using fixed state_lr={self.config.model.ttt.state_lr}")
        elif state_lr_mode == "adaptive":
            # Use adaptive state_lr which take in the magnitude of gradient [B, L, D] and produce the state_lr [B, L, D]
            self.ttt_lrnet = nn.ModuleList()
            for _ in range(self.config.model.ttt.n_layer):
                # Build the transformer block(s) and sigmoid
                lrnet = nn.Sequential(
                    # Normalization helps with performance a little bit
                    nn.LayerNorm(self.config.model.transformer.d, bias=False) if self.config.model.ttt.normalizer_type == "layer_norm" else nn.RMSNorm(self.config.model.transformer.d),
                    *[QK_Norm_TransformerBlock(
                        self.config.model.transformer.d,
                        self.config.model.transformer.d_head,
                        use_qk_norm=True,
                        use_positional_encoding=self.config.model.ttt.use_positional_encoding
                    ) for _ in range(self.config.model.ttt.n_blocks_per_layer_lrnet)],
                    nn.Sigmoid()
                )
                lrnet.apply(init_weights)
                self.ttt_lrnet.append(lrnet)
            print(f"Using adaptive state_lr with {self.config.model.ttt.n_blocks_per_layer_lrnet} blocks per layer")
        elif state_lr_mode == "adaptive_mlp":
            # Similar to adaptive, but use a MLP to compute the state_lr
            self.ttt_lrnet = nn.ModuleList()
            for _ in range(self.config.model.ttt.n_layer):
                lrnet = nn.Sequential(
                    # Normalization helps with performance a little bit
                    nn.LayerNorm(self.config.model.transformer.d, bias=False) if self.config.model.ttt.normalizer_type == "layer_norm" else nn.RMSNorm(self.config.model.transformer.d),
                    nn.Linear(self.config.model.transformer.d, self.config.model.transformer.d * 4, bias=False),
                    nn.GELU(),
                    nn.Linear(self.config.model.transformer.d * 4, self.config.model.transformer.d, bias=False),
                )
                lrnet.apply(init_weights)
                self.ttt_lrnet.append(lrnet)
            print(f"Using adaptive_mlp state_lr with 2 layers per layer")
        elif state_lr_mode == "adaptive_scale_shift":
            # Source: https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L113-L116
            # Use adaLN to modulate the output scale
            self.ttt_lrnet = nn.ModuleList()
            for _ in range(self.config.model.ttt.n_layer):
                lrnet = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(self.config.model.transformer.d, 2 * self.config.model.transformer.d, bias=True),
                )
                # zero out the modulation layers
                # nn.init.constant_(lrnet[-1].weight, 0.0)
                # nn.init.constant_(lrnet[-1].bias, 0.0)
                lrnet.apply(init_weights)
                self.ttt_lrnet.append(lrnet)
            print(f"Using adaptive_layernorm state_lr with adaLN to modulate the output scale")
        elif state_lr_mode == "adaptive_mlp_with_time":
            raise NotImplementedError("adaptive_mlp_with_time is not implemented")
        
        # Initialize LayerNorm modules for gradient and state normalization.
        if self.config.model.ttt.normalizer_type == "layer_norm":
            normalizer_template = nn.LayerNorm(
                self.config.model.transformer.d, 
                bias=False, 
                elementwise_affine=self.config.model.ttt.normalizer_affine, 
                eps=self.config.model.ttt.normalizer_eps
            )
        elif self.config.model.ttt.normalizer_type == "rms_norm":
            normalizer_template = nn.RMSNorm(
                self.config.model.transformer.d, 
                elementwise_affine=self.config.model.ttt.normalizer_affine,
                eps=self.config.model.ttt.normalizer_eps
            )
        else:
            raise ValueError(f"Invalid normalizer type: {self.config.model.ttt.normalizer_type}")
        
        # Deep copy the templates to create independent instances
        self.ttt_state_normalizers = nn.ModuleList()
        self.ttt_grad_normalizers = nn.ModuleList()
        for _ in range(self.config.model.ttt.n_layer):
            self.ttt_state_normalizers.append(copy.deepcopy(normalizer_template))
            self.ttt_grad_normalizers.append(copy.deepcopy(normalizer_template))
        
        if self.config.model.ttt.opt_model == "adam":
            # Adam option does not instantiate learnable optimizer blocks.
            # Updates are computed using torch.optim.Adam during ttt_forward.
            self.ttt_blocks = None
            print("Use Adam optimizers for TTT blocks, which will be created during ttt_forward")
            return
        
        self.ttt_blocks = nn.ModuleList()
        for _ in range(self.config.model.ttt.n_layer):
            if self.config.model.ttt.opt_model == "mlp":
                # Instantiate TTT blocks as a simple MLP
                self.ttt_blocks.append(
                    nn.Sequential(
                        nn.Linear(self.config.model.transformer.d * 2, self.config.model.transformer.d * 4, bias=False),
                        nn.GELU(),
                        nn.Linear(self.config.model.transformer.d * 4, self.config.model.transformer.d, bias=False),
                    )
                )
                print("Initialized TTT blocks as a simple MLP")
            elif self.config.model.ttt.opt_model == "flatten_mlp":
                # Instantiate TTT blocks as a simple MLP, but flatten the input to perform global fusion.
                self.ttt_blocks.append(
                    nn.Sequential(
                        # flatten the input [b, n_latent_vectors, 2*d] to [b, 2*d*n_latent_vectors]
                        Rearrange(
                            "b n d -> b (n d)",
                            n=self.config.model.transformer.n_latent_vectors,
                            d=self.config.model.transformer.d * 2
                        ),
                        nn.Linear(self.config.model.transformer.d * 2 * self.config.model.transformer.n_latent_vectors, self.config.model.ttt.mlp_dim, bias=False),
                        nn.GELU(),
                        nn.Linear(self.config.model.ttt.mlp_dim, self.config.model.transformer.d * self.config.model.transformer.n_latent_vectors, bias=False),
                        # unflatten the output [b, d*n_latent_vectors] to [b, n_latent_vectors, d]
                        Rearrange(
                            "b (n d) -> b n d",
                            n=self.config.model.transformer.n_latent_vectors,
                            d=self.config.model.transformer.d
                        ),
                    )
                )
                print("Initialized TTT blocks as a simple MLP, but flatten the input to perform global fusion")
            elif self.config.model.ttt.opt_model == "transformer":
                # Instantiate TTT blocks as a simple MLP and a transformer block.
                # TTT block take in concatenated state tokens and their gradients [b, n_latent_vectors, 2*d]
                # and output the updated state tokens [b, n_latent_vectors, d]
                self.ttt_blocks.append(
                    nn.Sequential(
                        nn.Linear(self.config.model.transformer.d * 2, self.config.model.transformer.d * 4, bias=False),
                        nn.GELU(),
                        nn.Linear(self.config.model.transformer.d * 4, self.config.model.transformer.d, bias=False),
                        nn.LayerNorm(self.config.model.transformer.d, bias=False) if self.config.model.ttt.normalizer_type == "layer_norm" else nn.RMSNorm(self.config.model.transformer.d),
                        *[QK_Norm_TransformerBlock(
                            self.config.model.transformer.d,
                            self.config.model.transformer.d_head,
                            use_qk_norm=True,
                            use_positional_encoding=self.config.model.ttt.use_positional_encoding
                        ) for _ in range(self.config.model.ttt.n_blocks_per_layer)],
                        # nn.Linear(self.config.model.transformer.d, self.config.model.transformer.d, bias=False),
                    )
                )
                print(f"Initialized TTT blocks as a simple MLP and {self.config.model.ttt.n_blocks_per_layer} transformer blocks")
            elif self.config.model.ttt.opt_model == "transformer2":
                # more transformer blocks, use qk norm, and put linear layers after each transformer block
                self.ttt_blocks.append(
                    nn.Sequential(
                        *[QK_Norm_TransformerBlock(
                            self.config.model.transformer.d * 2, 
                            self.config.model.transformer.d_head, 
                            use_qk_norm=True, 
                            use_positional_encoding=self.config.model.ttt.use_positional_encoding
                        ) for _ in range(self.config.model.ttt.n_blocks_per_layer)],
                        nn.Linear(self.config.model.transformer.d * 2, self.config.model.transformer.d, bias=False),
                    )
                )
                print(f"Initialized TTT blocks as {self.config.model.ttt.n_blocks_per_layer} transformer blocks and put linear layers after each transformer block")
            elif self.config.model.ttt.opt_model == "transformer3":
                # just use transformer blocks and no linear layers, the model only take in the grad_s
                self.ttt_blocks.append(
                    nn.Sequential(
                        *[QK_Norm_TransformerBlock(
                            self.config.model.transformer.d, 
                            self.config.model.transformer.d_head, 
                            use_qk_norm=True, 
                            use_positional_encoding=self.config.model.ttt.use_positional_encoding
                        ) for _ in range(self.config.model.ttt.n_blocks_per_layer)],
                    )
                )
                print(f"Initialized TTT blocks as {self.config.model.ttt.n_blocks_per_layer} transformer blocks and no linear layers")
        
        # initialize ttt blocks weights
        for block in self.ttt_blocks:
            block.apply(init_weights)
            # nn.init.zeros_(block[-1].weight)
            # init the last layer of the ttt blocks to be all zeros, so that the model is a residual connection
            # for i in range(1, self.config.model.ttt.n_blocks_per_layer + 1):
            #     nn.init.zeros_(block[-i].attn.fc.weight)
            #     nn.init.zeros_(block[-i].mlp.mlp[-2].weight)


    def train(self, mode=True):
        """Override the train method to keep the loss computer in eval mode"""
        super().train(mode)
        self.loss_computer.eval()


    def pass_layers(self, transformer_blocks, input_tokens, gradient_checkpoint=False, checkpoint_every=1):
        """
        Helper function to pass input tokens through all transformer blocks with optional gradient checkpointing.
        
        Args:
            input_tokens: Tensor of shape [batch_size, num_views * num_patches, hidden_dim]
                The input tokens to process through the transformer blocks.
            gradient_checkpoint: bool, default False
                Whether to use gradient checkpointing to save memory during training.
            checkpoint_every: int, default 1 
                Number of transformer layers to group together for gradient checkpointing.
                Only used when gradient_checkpoint=True.
                
        Returns:
            Tensor of shape [batch_size, num_views * num_patches, hidden_dim]
                The processed tokens after passing through all transformer blocks.
        """
        num_layers = len(transformer_blocks)
        
        if not gradient_checkpoint:
            # Standard forward pass through all layers
            for layer in transformer_blocks:
                input_tokens = layer(input_tokens)
            return input_tokens
            
        # Gradient checkpointing enabled - process layers in groups
        def _process_layer_group(tokens, start_idx, end_idx):
            """Helper to process a group of consecutive layers."""
            for idx in range(start_idx, end_idx):
                tokens = transformer_blocks[idx](tokens)
            return tokens
            
        # Process layer groups with gradient checkpointing
        for start_idx in range(0, num_layers, checkpoint_every):
            end_idx = min(start_idx + checkpoint_every, num_layers)
            input_tokens = torch.utils.checkpoint.checkpoint(
                _process_layer_group,
                input_tokens,
                start_idx,
                end_idx,
                use_reentrant=False
            )
            
        return input_tokens


    def get_posed_input(self, images=None, ray_o=None, ray_d=None, method="default_plucker"):
        '''
        Args:
            images: [b, v, c, h, w]
            ray_o: [b, v, 3, h, w]
            ray_d: [b, v, 3, h, w]
            method: Method for creating pose conditioning
        Returns:
            posed_images: [b, v, c+6, h, w] or [b, v, 6, h, w] if images is None
        '''

        if method == "custom_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            pose_cond = torch.cat([ray_d, nearest_pts], dim=2)
            
        elif method == "aug_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d, nearest_pts], dim=2)
            
        else:  # default_plucker
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d], dim=2)

        if images is None:
            return pose_cond
        else:
            return torch.cat([images * 2.0 - 1.0, pose_cond], dim=2)


    def _maybe_corrupt_images_for_ss(self, images, training=True):
        """
        Optionally corrupt images via diffusion-style interpolation with Gaussian noise for SS loss augmentation.
        
        Controlled by config keys in ttt.yaml:
          - model.ttt.corrupt_training_images: bool (default False)
        
        Formula (per (b, v) sample):
          x_t = sqrt(1 - t) * image + sqrt(t) * eps,  t ~ Uniform(0, 1),  eps ~ N(0, I)
        """
        b, v = images.shape[:2]
        device = images.device
        # Sample a scalar t per (b, v) and broadcast to pixels/channels
        t = torch.rand((b, v, 1, 1, 1), device=device) * self.config.model.ttt.corrupt_max_t  # Uniform(0, corrupt_max_t)
        # Normalize to [-1, 1], apply diffusion-style noise, then map back to [0, 1]
        x = images * 2.0 - 1.0
        eps = torch.randn_like(x)
        x_t = torch.sqrt(1.0 - t) * x + torch.sqrt(t) * eps
        images_noisy = (x_t + 1.0) * 0.5
        return images_noisy.clamp(0.0, 1.0)


    def _maybe_corrupt_state(self, s):
        """
        Optionally corrupt the latent state 's' during training.
        We normalize 's' to unit variance, apply diffusion-style mixing with Gaussian noise,
        then map back using the original statistics.
        
        Controlled by:
          - model.ttt.corrupt_training_state: bool (default False)
        """
        # Compute per-sample statistics over the last dimension
        s_mean = s.mean(dim=(-1), keepdim=True)
        s_std = s.std(dim=(-1), keepdim=True) + 1e-10
        s_norm = (s - s_mean) / s_std
        
        # Sample a scalar t per batch item to control the corruption strength
        b = s.shape[0]
        device = s.device
        t = torch.rand((b, 1, 1), device=device) * self.config.model.ttt.corrupt_max_t  # Uniform(0, corrupt_max_t)
        eps = torch.randn_like(s_norm)
        s_t = torch.sqrt(1.0 - t) * s_norm + torch.sqrt(t) * eps
        
        # Map back to the original domain
        return s_t * s_std + s_mean


    def encode(self, input, ss=None, training=True):
        """
        Encode the light_field_latent into latent_tokens with input posed images.
        """
        checkpoint_every = self.config.training.grad_checkpoint_every
        n_latent_vectors = self.config.model.transformer.n_latent_vectors
        
        images = input.image
        ray_o = input.ray_o
        ray_d = input.ray_d
        if ss is not None:
            images = torch.cat([images, ss.image], dim=1)
            ray_o = torch.cat([ray_o, ss.ray_o], dim=1)
            ray_d = torch.cat([ray_d, ss.ray_d], dim=1)
        
        # Process input images
        posed_input_images = self.get_posed_input(
            images=images, ray_o=ray_o, ray_d=ray_d
        )
        b, v_input, c, h, w = posed_input_images.size()

        input_img_tokens = self.image_tokenizer(posed_input_images)  # [b*v, n_patches, d]
        _, n_patches, d = input_img_tokens.size()  # [b*v, n_patches, d]
        input_img_tokens = input_img_tokens.reshape(b, v_input * n_patches, d)  # [b, v*n_patches, d]
        latent_vector_tokens = self.n_light_field_latent.expand(b, -1, -1) # [b, n_latent_vectors, d]
        encoder_input_tokens = torch.cat((latent_vector_tokens, input_img_tokens), dim=1) # [b, n_latent_vectors + v*n_patches, d]
        intermediate_tokens = self.pass_layers(self.transformer_encoder, encoder_input_tokens, gradient_checkpoint=self.config.training.grad_checkpoint and training, checkpoint_every=checkpoint_every)
        encoded_latents, input_img_tokens = intermediate_tokens.split([n_latent_vectors, v_input * n_patches], dim=1) # [b, n_latent_vectors, d], [b, v*n_patches, d]
        return encoded_latents


    def decode(self, target, latent_tokens, target_pose_tokens=None, training=True):
        """
        Decode the target view images with the latent tokens and target poses.
        """
        checkpoint_every = self.config.training.grad_checkpoint_every
        n_latent_vectors = self.config.model.transformer.n_latent_vectors
        b, v_target = target.image.size()[:2]
        if target_pose_tokens is None:
            target_pose_cond = self.get_posed_input(ray_o=target.ray_o, ray_d=target.ray_d)  # [b, v_target, c, h, w]
            target_pose_tokens = self.target_pose_tokenizer(target_pose_cond)  # [b*v_target, n_patches, d]
        
        _, n_patches, d = target_pose_tokens.size()
        height, width = target.image_h_w
        patch_size = self.config.model.target_pose_tokenizer.patch_size

        # Chunked VRAM-saving decoding
        chunk_size = getattr(self.config.model.ttt, 'decode_chunk_size', None)
        if chunk_size is None or chunk_size > v_target or chunk_size <= 0:
            chunk_size = v_target

        rendered_chunks = []
        for start in range(0, v_target, chunk_size):
            # print(f"Decoding chunk {start} to {start + chunk_size} of {v_target}")
            end = min(start + chunk_size, v_target)
            this_v = end - start

            # Per-chunk pose tokens
            this_target_pose_tokens = target_pose_tokens[b*start:b*end, :, :].contiguous() # [b*this_v, n_patches, d]
            this_latent_tokens = repeat(latent_tokens, 'b nl d -> (b v) nl d', v=this_v)

            decoder_input_tokens = torch.cat((this_target_pose_tokens, this_latent_tokens), dim=1)  # [b*this_v, n_latent_vectors + n_patches, d]
            decoder_input_tokens = self.transformer_input_layernorm_decoder(decoder_input_tokens)
            # print(f"[decode {start} to {end}, before decoder]: alloced {torch.cuda.memory_allocated() / 1024**3:.2f}GB, cached {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
            transformer_output_tokens = self.pass_layers(
                self.transformer_decoder,
                decoder_input_tokens,
                gradient_checkpoint=self.config.training.grad_checkpoint and training,
                checkpoint_every=checkpoint_every
            )
            # print(f"[decode {start} to {end}, after decoder]: alloced {torch.cuda.memory_allocated() / 1024**3:.2f}GB, cached {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
            
            # Discard the latent tokens
            target_image_tokens, _ = transformer_output_tokens.split([n_patches, n_latent_vectors], dim=1)  # [b*this_v, n_patches, d]

            rendered_images = self.image_token_decoder(target_image_tokens) # [b*this_v, n_patches, p*p*3]
            rendered_images = rearrange(
                rendered_images,
                "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
                v=this_v,
                h=height // patch_size,
                w=width // patch_size,
                p1=patch_size,
                p2=patch_size,
                c=3
            )
            rendered_chunks.append(rendered_images)
            # torch.cuda.empty_cache()

        rendered_images = torch.cat(rendered_chunks, dim=1)  # [b, v_target, c, H, W]
        return rendered_images, target_pose_tokens


    def _update_state_with_loss(
        self, 
        s, 
        decoder_input, 
        grad_norm, 
        state_norm, 
        opt, 
        input_loss, 
        lrnet=None,
        need_grad=False,
        t=None,
        grad_s=None,
    ):
        """
        Update state using the computed loss.
        
        Args:
            s: Current state tensor [b, n_latent_vectors, d]
            decoder_input: Decoder input (may be detached from s)
            grad_norm: Gradient normalizer
            state_norm: State normalizer
            opt: Optimizer
            input_loss: Computed input loss
            lrnet: (Optional) Learnable state learning rate
            need_grad: Whether to keep tracking gradients
        Returns:
            Tuple of (updated_s, layer_metrics)
        """
        layer_metrics = {}
        
        if grad_s is None:
            if self.config.model.ttt.grad_mode == "normal":
                # retrain_graph is needed for traversing the same computational graph the second time
                grad_s = torch.autograd.grad(input_loss, decoder_input, create_graph=False, retain_graph=True)[0]
            elif self.config.model.ttt.grad_mode == "zero":
                grad_s = torch.zeros_like(s)
            elif self.config.model.ttt.grad_mode == "random":
                grad_s = torch.randn_like(s)
        else:
            print(f"use saved grad_s, grad_s.mean: {grad_s.mean().item()}, grad_s.std: {grad_s.std().item()}")

        if self.config.model.ttt.detach_grad:
            # If detach grad, the gradient will not flow into the decoder 
            grad_s = grad_s.detach()

        # log gradient statistics before normalizer
        layer_metrics["orig_grad_max"] = torch.max(torch.abs(grad_s)).item()
        layer_metrics["orig_grad_mean"] = torch.mean(torch.abs(grad_s)).item()
        layer_metrics["orig_grad_std"] = torch.std(grad_s).item()

        # calculate the domain of s
        s_mean = s.mean(dim=(-1), keepdim=True)
        s_std = s.std(dim=(-1), keepdim=True)
        # print(f"s mean: {s.mean().item()}, s std: {s.std().item()}")

        # normalize gradient after detach. otherwise the normalizer will get no gradients.
        grad_s_normed = grad_s / (grad_s.std(dim=(-1), keepdim=True) + 1e-10) # [b, n_latent_vectors, d]
        grad_s_normed = grad_norm(grad_s_normed) # [b, n_latent_vectors, d]

        # log the scale factor of the normalizer
        if (isinstance(grad_norm, nn.RMSNorm) or isinstance(grad_norm, nn.LayerNorm)) and grad_norm.elementwise_affine:
            layer_metrics["grad_norm_scaler"] = grad_norm.weight.mean().item()

        # log gradient statistics
        layer_metrics["grad_max"] = torch.max(torch.abs(grad_s_normed)).item()
        layer_metrics["grad_mean"] = torch.mean(torch.abs(grad_s_normed)).item()
        layer_metrics["grad_std"] = torch.std(grad_s_normed).item()
        
        # update state with loss
        if self.config.model.ttt.opt_model == "adam":
            # If ttt block is None, we initialize it as a adam optimizer with the current state
            if self.ttt_blocks is None:
                self.ttt_blocks = torch.optim.Adam(
                    [s], 
                    lr=self.config.model.ttt.adam.lr, 
                    betas=(self.config.model.ttt.adam.beta1, self.config.model.ttt.adam.beta2), 
                    eps=self.config.model.ttt.adam.eps, 
                    weight_decay=self.config.model.ttt.adam.weight_decay
                )
            # update the state
            self.ttt_blocks.zero_grad()
            s.grad = grad_s_normed
            self.ttt_blocks.step()

            return s, grad_s, layer_metrics
            
            # # Create Adam optimizer with the current state as parameter
            # state_param = nn.Parameter(s.clone().detach().requires_grad_(True))
            # state_param.grad = grad_s_normed
            
            # # Get Adam hyperparameters
            # adam_lr = self.config.model.ttt.adam.lr
            # adam_beta1 = self.config.model.ttt.adam.beta1
            # adam_beta2 = self.config.model.ttt.adam.bet
            # adam_eps = self.config.model.ttt.adam.eps
            # adam_weight_decay = self.config.model.ttt.adam.weight_decay
            
            # # Create Adam optimizer
            # optimizer = torch.optim.Adam(
            #     [state_param], 
            #     lr=adam_lr, 
            #     betas=(adam_beta1, adam_beta2), 
            #     eps=adam_eps, 
            #     weight_decay=adam_weight_decay
            # )
            
            # # update the state
            # optimizer.step()
            # delta_s = state_param.data - s
        else:
            if self.config.model.ttt.opt_model == "transformer3":
                opt_input = grad_s_normed # [b, n_latent_vectors, d]
            else:
                if self.config.model.ttt.detach_opt_input:
                    # If detach opt input, the gradient will not flow into the opt input state.
                    opt_input_s = s.detach()
                else:
                    opt_input_s = s

                # normalize the opt input state after detach as well.
                opt_input_s = state_norm(opt_input_s)

                # log the scale factor of the normalizer
                # if (isinstance(state_norm, nn.RMSNorm) or isinstance(state_norm, nn.LayerNorm)) and state_norm.elementwise_affine:
                #     layer_metrics["state_norm_scaler"] = state_norm.weight.mean().item()

                # log the opt input state -- state after normalizer
                layer_metrics["opt_state_max"] = torch.max(opt_input_s).item()
                layer_metrics["opt_state_mean"] = torch.mean(opt_input_s).item()
                layer_metrics["opt_state_std"] = torch.std(opt_input_s).item()

                opt_input = torch.cat((opt_input_s, grad_s_normed), dim=-1) # [b, n_latent_vectors, 2*d]

            if self.config.model.ttt.opt_model == "dit":
                # inject the time step also
                t_vec = torch.full((opt_input.shape[0],), t, device=opt_input.device)
                delta_s = opt(opt_input, t_vec) # [b, n_latent_vectors, 2*d]
            else:    
                delta_s = opt(opt_input) # [b, n_latent_vectors, d]

            # pull delta_s to the same domain as s
            # print(f"delta_s mean: {delta_s.mean().item()}, delta_s std: {delta_s.std().item()}")
            # delta_s = (delta_s - delta_s.mean(dim=(-1), keepdim=True)) / (delta_s.std(dim=(-1), keepdim=True) + 1e-10)
            # delta_s = delta_s * (s_std + 1e-10) + s_mean
            if not self.config.model.ttt.opt_model == "dit":
                delta_s = delta_s / (delta_s.std(dim=(-1), keepdim=True) + 1e-10) * s_std
            # delta_s = delta_s * s_std
        
        # get the effective state_lr for this layer
        if self.config.model.ttt.opt_model == "dit":
            assert self.config.model.ttt.state_lr_mode == "fixed", f"expect state_lr_mode to be fixed for DiT, but got {self.config.model.ttt.state_lr_mode}"
            assert self.config.model.ttt.state_lr == 1.0, f"expect state_lr to be 1.0 for DiT, but got {self.config.model.ttt.state_lr}"
        if self.config.model.ttt.state_lr_mode == "learnable":
            assert lrnet is not None, "lrnet is required for learnable state_lr"
            # Use learnable state_lr with sigmoid activation
            state_lr = torch.sigmoid(lrnet)  # [D]
            # Expand to match delta_s shape for element-wise multiplication
            state_lr = state_lr.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
        elif self.config.model.ttt.state_lr_mode in ["adaptive", "adaptive_mlp"]:
            assert lrnet is not None, "lrnet is required for adaptive state_lr"
            # Pass the "magnitude" of gradient to the lrnet. Since the gradient can be small, we use the log scale as the input.
            log_abs_grad_s = torch.log(torch.abs(grad_s) + 1e-10)
            # adding a bias term to make the output around -2 before sigmoid, the learning rate will be around 0.1-0.2 at the beginning.
            # This makes the residual update smaller at the beginning while maintaining relatively big gradient for the lrnet.
            state_lr = torch.sigmoid(lrnet(log_abs_grad_s) + self.config.model.ttt.state_lr_init) # [b, n_latent_vectors, d]
        elif self.config.model.ttt.state_lr_mode == "adaptive_scale_shift":
            assert lrnet is not None, "lrnet is required for adaptive_scale_shift state_lr"
            # Pass the "magnitude" of gradient to the lrnet. Since the gradient can be small, we use the log scale as the input.
            log_abs_grad_s = torch.log(torch.abs(grad_s) + 1e-10) # [b, n_latent_vectors, d]
            shift, scale = lrnet(log_abs_grad_s).chunk(2, dim=-1) # [b, n_latent_vectors, d]
            # modulate the delta_s
            def modulate(x, shift, scale):
                return x * (1 + scale) + shift
            delta_s = modulate(delta_s, shift, scale)
            state_lr = torch.ones_like(shift) * self.config.model.ttt.state_lr
        else:
            # Use fixed state_lr from config
            state_lr = self.config.model.ttt.state_lr

        # apply the cosine decay to the state_lr
        # def cosine_scheduler(t, s=0.008):
        #     return math.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        # state_lr = state_lr * cosine_scheduler(t)
        
        # Apply update with effective learning rate
        s_update = delta_s * state_lr
        
        # Apply update
        if self.config.model.ttt.is_residual and self.config.model.ttt.opt_model != "dit":
            new_s = s_update + (s.detach() if self.config.model.ttt.detach_residual else s)
            if not self.config.model.ttt.opt_model == "adam":
                # pull the new s to the same domain as previous s
                new_s = new_s / (new_s.std(dim=(-1), keepdim=True) + 1e-10) * s_std
        else:
            new_s = s_update[..., :self.config.model.transformer.d]

        # log state statistics
        layer_metrics["state_max"] = torch.max(new_s).item()
        layer_metrics["state_mean"] = torch.mean(new_s).item()
        layer_metrics["state_std"] = torch.std(new_s).item()
        
        # log learnable lr statistics if applicable
        if self.config.model.ttt.state_lr_mode == "learnable" or "adaptive" in self.config.model.ttt.state_lr_mode:
            layer_metrics["state_lr_mean"] = torch.mean(state_lr).item()
            layer_metrics['state_lr_max'] = torch.max(state_lr).item()
            layer_metrics['state_lr_std'] = torch.std(state_lr).item()
        
        return new_s, grad_s, layer_metrics


    def ttt_forward(self, input, target, ss, ood_target, n_iters=None):
        raise NotImplementedError("TTT forward is not updated yet")
        return None


    def ttt_forward_g3r(
        self, 
        input, 
        target, 
        ss, 
        ood_target, 
        layer_idx, 
        iter_idx, 
        t, 
        s=None, 
        ss_pose_tokens=None, 
        target_pose_tokens=None, 
        ood_target_pose_tokens=None, 
        is_first=False,
        is_last=False,
        training=True,
        input_views_ss=False,
        ood_target_views_ss=False,
    ):
        """
        Forward the latent tokens with the TTT blocks for G3R supervision. Returns the updated state and TTT metrics for logging.
        Args:
            input: Input data batch
            target: Target data batch
            ss: Self-supervision data batch
            ood_target: OOD target data batch
            input: Input data batch
            target: Target data batch
            layer_idx: Layer index
            iter_idx: Iteration index
            t: Time step
            s: (Optional) Current state tensor [b, n_latent_vectors, d]
            input_pose_tokens: (Optional) Cached pose tokens for input views
            target_pose_tokens: (Optional) Cached pose tokens for target views
            is_first: Whether this is the first iteration
            is_last: Whether to update the state
            training: gradient checkpointing will always be disabled during inference
            input_views_ss: Whether to use input views to calculate ss loss
            ood_target_views_ss: Whether to use ood target views to calculate ss loss
        Returns:
            input_loss_metrics: Input loss metrics, only calculated on the last iteration
            target_loss_metrics: Target loss metrics, calculated on the updated state
            ss_loss_metrics: Self-supervision loss metrics, calculated on the state before update
            ood_target_loss_metrics: OOD target loss metrics, calculated on the updated state
            rendered_input: Rendered input
            rendered_target: Rendered target
            rendered_ss: Rendered self-supervision
            rendered_ood_target: Rendered OOD target
            loss: Total loss
            s: Updated state
            ss_pose_tokens: Cached pose tokens for self-supervision views
            target_pose_tokens: Cached pose tokens for target for target views
            ood_target_pose_tokens: Cached pose tokens for OOD target for OOD target views
            layer_metrics: Layer metrics
        """
        assert self.config.model.ttt.supervise_mode == "g3r", "supervise_mode must be g3r for G3R supervision"
        assert layer_idx is not None and iter_idx is not None, "layer_idx and iter_idx must be provided for G3R supervision"
        assert self.config.model.ttt.distill_factor == 0.0, "distill_factor must be 0.0 for G3R supervision"

        if s is None:
            assert is_first, "s must be provided for G3R supervision when is_first is False"
            # use encoder to absorb input views
            if training and self.config.model.ttt.corrupt_training_images:
                enc_input = copy.deepcopy(input)
                enc_input.image = self._maybe_corrupt_images_for_ss(input.image)
            else:
                enc_input = input
            if self.config.model.input_4views:
                s = self.encode(enc_input, ss, training=training)
            else:
                s = self.encode(enc_input, training=training)

            if self.config.model.ttt.opt_model == "adam":
                # Reset the optimizer for the new sample
                self.ttt_blocks = None
                # detach the state to make a leaf node
                s = s.detach()
        
        if training and self.config.model.ttt.corrupt_training_states:
            s = self._maybe_corrupt_state(s)
        s = s.requires_grad_(True)

        if not (self.config.model.ttt.supervise_s0 and is_first) or is_last or not training:
            # if supervise s0 and this is the first iteration, we calculate target loss on the s0 state and dont do state update
            # Compute self-supervision losses which is calculated on the ss views
            ss_loss = 0.0
            with torch.enable_grad():
                # calculate ss loss
                rendered_ss, ss_pose_tokens = self.decode(ss, s, target_pose_tokens=ss_pose_tokens, training=training)
                ss_loss_metrics = self.loss_computer(rendered_ss, ss.image)
                ss_loss += ss_loss_metrics["loss"]

                if input_views_ss:
                    # calculate input views ss loss
                    rendered_input, _ = self.decode(input, s, training=training)
                    input_views_ss_loss_metrics = self.loss_computer(rendered_input, input.image)
                    ss_loss += input_views_ss_loss_metrics["loss"]
                
                if ood_target_views_ss:
                    # calculate ood target views ss loss
                    rendered_ood_target, ood_target_pose_tokens = self.decode(ood_target, s, target_pose_tokens=ood_target_pose_tokens, training=training)
                    ood_target_views_ss_loss_metrics = self.loss_computer(rendered_ood_target, ood_target.image)
                    ss_loss += ood_target_views_ss_loss_metrics["loss"]
        else:
            ss_loss_metrics = None
            rendered_ss = None

        if not (self.config.model.ttt.supervise_s0 and is_first) or not training:
            # Update state with self-supervision losses except for the last layer
            grad_norm = self.ttt_grad_normalizers[layer_idx]
            state_norm = self.ttt_state_normalizers[layer_idx]
            opt = None
            if self.config.model.ttt.opt_model != "adam":
                opt = self.ttt_blocks[layer_idx]
            lrnet = None
            if self.config.model.ttt.state_lr_mode in ["learnable"] or "adaptive" in self.config.model.ttt.state_lr_mode:
                lrnet = self.ttt_lrnet[layer_idx]

            new_s, grad_s, layer_metrics = self._update_state_with_loss(
                s, s, grad_norm, state_norm, opt, ss_loss, lrnet,
                need_grad=True, t=t
            )

            if self.config.model.ttt.enable_unroll:
                if input_views_ss or ood_target_views_ss:
                    raise NotImplementedError("Unroll with input views ss or ood target views ss is not supported yet")
                # compute the ss loss again with the new state, we dont need gradient this time
                with torch.no_grad():
                    rendered_ss, _ = self.decode(
                        ss, new_s,
                        target_pose_tokens=ss_pose_tokens, 
                        training=training
                    )
                    new_ss_loss_metrics = self.loss_computer(rendered_ss, ss.image)
                
                # only update state if the new ss loss is smaller
                if new_ss_loss_metrics["loss"] < ss_loss_metrics["loss"]:
                    print(f"{iter_idx}th iter new ss loss is smaller than cur ss loss: {new_ss_loss_metrics['loss']:.4f} < {ss_loss_metrics['loss']:.4f}, update the state")
                    s = new_s
                    ss_loss_metrics = new_ss_loss_metrics
                else:
                    print(f"{iter_idx}th iter new ss loss is larger than cur ss loss: {new_ss_loss_metrics['loss']:.4f} > {ss_loss_metrics['loss']:.4f}, keep the current state")
            else:
                # if unroll is disabled, we use the new state directly
                s = new_s
        else:
            layer_metrics = {}

        if is_last:
            # render input views, only for visualization
            # we only calculate the input loss once per data sample so we dont cache it during training.
            rendered_input, _ = self.decode(input, s, training=training)
            input_loss_metrics = self.loss_computer(rendered_input, input.image)
            layer_metrics["input_loss"] = input_loss_metrics["loss"].item()
        else:
            rendered_input = None
            input_loss_metrics = None
            layer_metrics["input_loss"] = 0.0
        
        # calculate loss on target and ood target views
        rendered_target, target_pose_tokens = self.decode(target, s, target_pose_tokens=target_pose_tokens, training=training)
        target_loss_metrics = self.loss_computer(rendered_target, target.image)
        layer_metrics["target_loss"] = target_loss_metrics["loss"].item()

        rendered_ood_target, ood_target_pose_tokens = self.decode(ood_target, s, target_pose_tokens=ood_target_pose_tokens, training=training)
        ood_target_loss_metrics = self.loss_computer(rendered_ood_target, ood_target.image)
        layer_metrics["ood_target_loss"] = ood_target_loss_metrics["loss"].item()

        assert self.config.training.supervision == "target", "supervision must be target for G3R supervision"
        loss = target_loss_metrics["loss"] + ood_target_loss_metrics["loss"]

        # return loss metrics, rendered images, loss, updated state, pose tokens, and layer metrics
        return input_loss_metrics, target_loss_metrics, ss_loss_metrics, ood_target_loss_metrics, rendered_input, rendered_target, rendered_ss, rendered_ood_target, loss, s, ss_pose_tokens, target_pose_tokens, ood_target_pose_tokens, layer_metrics


    def forward(
        self, 
        data_batch, 
        num_input_views, 
        num_target_views, 
        num_ss_views, 
        num_ood_target_views,
        is_g3r, 
        has_target_image=True, 
        training=True, 
        layer_idx=None, 
        iter_idx=None, 
        t=None, 
        n_iters=None, 
        **kwargs
    ):
        assert has_target_image, "JC: we might need to support this?"
        if kwargs.get("input") is None or kwargs.get("target") is None or kwargs.get("ss") is None or kwargs.get("ood_target") is None:
            if "input" in kwargs:
                kwargs.pop("input")
            if "target" in kwargs:
                kwargs.pop("target")
            if "ss" in kwargs:
                kwargs.pop("ss")
            if "ood_target" in kwargs:
                kwargs.pop("ood_target")
            input, target, ss, ood_target = self.process_data(
                data_batch, 
                num_input_views=num_input_views, 
                num_target_views=num_target_views, 
                num_ss_views=num_ss_views, 
                num_ood_target_views=num_ood_target_views, 
                has_target_image=has_target_image, 
                training=training, 
                compute_rays=True
            )
        else:
            input, target, ss, ood_target = kwargs.pop("input"), kwargs.pop("target"), kwargs.pop("ss"), kwargs.pop("ood_target")
        
        if is_g3r:
            assert layer_idx is not None and iter_idx is not None, "layer_idx and iter_idx must be provided for G3R supervision"
            assert layer_idx == 0, "G3R always use the same optimizer network"
            forward_res = self.ttt_forward_g3r(input, target, ss, ood_target, layer_idx, iter_idx, t, training=training, **kwargs)
            return input, target, ss, ood_target, *forward_res
        else:    
            forward_res = self.ttt_forward(input, target, ss, ood_target, n_iters)
            return input, target, ss, ood_target, *forward_res


    @torch.no_grad()
    def render_video(self, data_batch, traj_type="interpolate", num_frames=60, loop_video=False, order_poses=False):
        """
        Render a video from the model.
        
        Args:
            result: Edict from forward pass or just data
            traj_type: Type of trajectory
            num_frames: Number of frames to render
            loop_video: Whether to loop the video
            order_poses: Whether to order poses
            
        Returns:
            result: Updated with video rendering
        """
    
        raise NotImplementedError("Need some closer look here.")
        if data_batch.input is None:
            ss, input, target = self.process_data(data_batch, num_ss_views=num_ss_views, num_input_views=num_input_views, num_target_views=num_target_views, has_target_image=False, training=False, compute_rays=True)
            data_batch = edict(input=input, target=target)
        else:
            input, target = data_batch.input, data_batch.target
        
        # Prepare input tokens; [b, v, 3+6, h, w]
        posed_images = self.get_posed_input(
            images=input.image, ray_o=input.ray_o, ray_d=input.ray_d
        )
        bs, v_input, c, h, w = posed_images.size()

        input_img_tokens = self.image_tokenizer(posed_images)  # [b*v_input, n_patches, d]
        _, n_patches, d = input_img_tokens.size()  # [b*v_input, n_patches, d]
        input_img_tokens = input_img_tokens.reshape(bs, v_input * n_patches, d)  # [b, v_input*n_patches, d]

        latent_vector_tokens = self.n_light_field_latent.expand(bs, -1, -1) # [b, n_latent_vectors, d]
        encoder_input_tokens = torch.cat((latent_vector_tokens, input_img_tokens), dim=1) # [b, n_latent_vectors + v*n_patches, d]

        # Process through encoder
        intermediate_tokens = self.pass_layers(self.transformer_encoder, encoder_input_tokens, gradient_checkpoint=False)
        latent_tokens, _ = intermediate_tokens.split(
            [self.config.model.transformer.n_latent_vectors, v_input * n_patches], dim=1
        ) # [b, n_latent_vectors, d]

        if traj_type == "interpolate":
            c2ws = input.c2w # [b, v, 4, 4]
            fxfycxcy = input.fxfycxcy #  [b, v, 4]
            device = input.c2w.device

            # Create intrinsics from fxfycxcy
            intrinsics = torch.zeros((c2ws.shape[0], c2ws.shape[1], 3, 3), device=device) # [b, v, 3, 3]
            intrinsics[:, :,  0, 0] = fxfycxcy[:, :, 0]
            intrinsics[:, :,  1, 1] = fxfycxcy[:, :, 1]
            intrinsics[:, :,  0, 2] = fxfycxcy[:, :, 2]
            intrinsics[:, :,  1, 2] = fxfycxcy[:, :, 3]

            # Loop video if requested
            if loop_video:
                c2ws = torch.cat([c2ws, c2ws[:, [0], :]], dim=1)
                intrinsics = torch.cat([intrinsics, intrinsics[:, [0], :]], dim=1)

            # Interpolate camera poses
            all_c2ws, all_intrinsics = [], []
            for b in range(input.image.size(0)):
                cur_c2ws, cur_intrinsics = camera_utils.get_interpolated_poses_many(
                    c2ws[b, :, :3, :4], intrinsics[b], num_frames, order_poses=order_poses
                )
                all_c2ws.append(cur_c2ws.to(device))
                all_intrinsics.append(cur_intrinsics.to(device))

            all_c2ws = torch.stack(all_c2ws, dim=0) # [b, num_frames, 3, 4]
            all_intrinsics = torch.stack(all_intrinsics, dim=0) # [b, num_frames, 3, 3]

            # Add homogeneous row to c2ws
            homogeneous_row = torch.tensor([[[0, 0, 0, 1]]], device=device).expand(all_c2ws.shape[0], all_c2ws.shape[1], -1, -1)
            all_c2ws = torch.cat([all_c2ws, homogeneous_row], dim=2)

            # Convert intrinsics to fxfycxcy format
            all_fxfycxcy = torch.zeros((all_intrinsics.shape[0], all_intrinsics.shape[1], 4), device=device)
            all_fxfycxcy[:, :, 0] = all_intrinsics[:, :, 0, 0]  # fx
            all_fxfycxcy[:, :, 1] = all_intrinsics[:, :, 1, 1]  # fy
            all_fxfycxcy[:, :, 2] = all_intrinsics[:, :, 0, 2]  # cx
            all_fxfycxcy[:, :, 3] = all_intrinsics[:, :, 1, 2]  # cy

        # Compute rays for rendering
        rendering_ray_o, rendering_ray_d = self.process_data.compute_rays(
            fxfycxcy=all_fxfycxcy, c2w=all_c2ws, h=h, w=w, device=device
        )

        # Get pose conditioning for target views
        target_pose_cond = self.get_posed_input(
            ray_o=rendering_ray_o.to(input.image.device), 
            ray_d=rendering_ray_d.to(input.image.device)
        )
                
        _, num_views, c, h, w = target_pose_cond.size()
    
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond) # [bs*v_target, n_patches, d]
        _, n_patches, d = target_pose_tokens.size()  # [b*v_target, n_patches, d]
        target_pose_tokens = target_pose_tokens.reshape(bs, num_views * n_patches, d)  # [b, v_target*n_patches, d]

        view_chunk_size = 4
        video_rendering_list = []
        
        for cur_chunk in range(0, num_views, view_chunk_size):
            cur_view_chunk_size = min(view_chunk_size, num_views - cur_chunk)
            
            # Get current chunk of target pose tokens
            start_idx, end_idx = cur_chunk * n_patches, (cur_chunk + cur_view_chunk_size) * n_patches
            cur_target_pose_tokens = rearrange(target_pose_tokens[:, start_idx:end_idx,: ], 
                                               "b (v_chunk p) d -> (b v_chunk) p d", 
                                               v_chunk=cur_view_chunk_size, p=n_patches)

            cur_repeated_latent_tokens = repeat(
                latent_tokens,
                'b nl d -> (b v_chunk) nl d', 
                v_chunk=cur_view_chunk_size
                )

            decoder_input_tokens = torch.cat((cur_target_pose_tokens, cur_repeated_latent_tokens), dim=1)
            decoder_input_tokens = self.transformer_input_layernorm_decoder(decoder_input_tokens)

            transformer_output_tokens = self.pass_layers(
                self.transformer_decoder, 
                decoder_input_tokens, 
                gradient_checkpoint=False
            )

            target_image_tokens, _ = transformer_output_tokens.split(
                [n_patches, self.config.model.transformer.n_latent_vectors], dim=1
            )

            # Decode to images
            height, width = target.image_h_w
            patch_size = self.config.model.target_pose_tokenizer.patch_size
            
            video_rendering = self.image_token_decoder(target_image_tokens)
            video_rendering = rearrange(
                video_rendering, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
                v=cur_view_chunk_size,
                h=height // patch_size, 
                w=width // patch_size, 
                p1=patch_size, 
                p2=patch_size, 
                c=3
            ).cpu()

            video_rendering_list.append(video_rendering)

        # Combine all chunks
        video_rendering = torch.cat(video_rendering_list, dim=1)
        data_batch.video_rendering = video_rendering

        return data_batch


    @torch.no_grad()
    def load_ckpt(self, load_path):
        if os.path.isdir(load_path):
            ckpt_names = [file_name for file_name in os.listdir(load_path) if file_name.endswith(".pt")]
            ckpt_names = sorted(ckpt_names, key=lambda x: x)
            ckpt_paths = [os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names]
        else:
            ckpt_paths = [load_path]
        try:
            checkpoint = torch.load(ckpt_paths[-1], map_location="cpu", weights_only=True)
        except:
            traceback.print_exc()
            print(f"Failed to load {ckpt_paths[-1]}")
            return None
        
        # This function is called during inference, so we need to load the model strictly
        status = self.load_state_dict(checkpoint["model"], strict=True)
        print(f"Loaded model from {ckpt_paths[-1]}, the status is {status}")
        return 0
