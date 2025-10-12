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

        # Flag to track if we are in TTT mode to disable gradient checkpointing
        self._in_ttt_mode = False
        
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
        elif self.config.model.ttt.state_lr_mode == "adaptive" or self.config.model.ttt.state_lr_mode == "adaptive_mlp":
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
                    nn.Sigmoid()
                )
                lrnet.apply(init_weights)
                self.ttt_lrnet.append(lrnet)
            print(f"Using adaptive_mlp state_lr with 2 layers per layer")
        
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
    
    def encode(self, input):
        """
        Encode the light_field_latent into latent_tokens with input posed images.
        """
        checkpoint_every = self.config.training.grad_checkpoint_every
        n_latent_vectors = self.config.model.transformer.n_latent_vectors
        
        # Process input images
        posed_input_images = self.get_posed_input(
            images=input.image, ray_o=input.ray_o, ray_d=input.ray_d
        )
        b, _, c, h, w = posed_input_images.size()

        # latent token with only using the first n_encoder_inputs input views
        v_input = self.config.model.ttt.n_encoder_inputs
        partial_posed_input_images = posed_input_images[:, :v_input, ...]
        input_img_tokens = self.image_tokenizer(partial_posed_input_images)  # [b*v, n_patches, d]
        _, n_patches, d = input_img_tokens.size()  # [b*v, n_patches, d]
        input_img_tokens = input_img_tokens.reshape(b, v_input * n_patches, d)  # [b, v*n_patches, d]
        latent_vector_tokens = self.n_light_field_latent.expand(b, -1, -1) # [b, n_latent_vectors, d]
        encoder_input_tokens = torch.cat((latent_vector_tokens, input_img_tokens), dim=1) # [b, n_latent_vectors + v*n_patches, d]
        intermediate_tokens = self.pass_layers(self.transformer_encoder, encoder_input_tokens, gradient_checkpoint=self.config.training.grad_checkpoint and not self._in_ttt_mode, checkpoint_every=checkpoint_every)
        partial_encoded_latents, input_img_tokens = intermediate_tokens.split([n_latent_vectors, v_input * n_patches], dim=1) # [b, n_latent_vectors, d], [b, v*n_patches, d]
        
        full_encoded_latents = None
        if self.config.model.ttt.distill_factor > 0.0:
            # latent token with using all input views, which provide teacher signal for distillation
            v_input = posed_input_images.size(1)
            input_img_tokens = self.image_tokenizer(posed_input_images)  # [b*v, n_patches, d]
            _, n_patches, d = input_img_tokens.size()  # [b*v, n_patches, d]
            input_img_tokens = input_img_tokens.reshape(b, v_input * n_patches, d)  # [b, v*n_patches, d]
            latent_vector_tokens = self.n_light_field_latent.expand(b, -1, -1) # [b, n_latent_vectors, d]
            encoder_input_tokens = torch.cat((latent_vector_tokens, input_img_tokens), dim=1) # [b, n_latent_vectors + v*n_patches, d]
            intermediate_tokens = self.pass_layers(self.transformer_encoder, encoder_input_tokens, gradient_checkpoint=self.config.training.grad_checkpoint and not self._in_ttt_mode, checkpoint_every=checkpoint_every)
            full_encoded_latents, input_img_tokens = intermediate_tokens.split([n_latent_vectors, v_input * n_patches], dim=1) # [b, n_latent_vectors, d], [b, v*n_patches, d]
        
        return partial_encoded_latents, full_encoded_latents

    def decode(self, target, latent_tokens, target_pose_tokens=None, last_n=None):
        """
        Decode the target view images with the latent tokens and target poses.
        """
        checkpoint_every = self.config.training.grad_checkpoint_every
        n_latent_vectors = self.config.model.transformer.n_latent_vectors
        b, v_target = target.image.size()[:2]
        if last_n is not None:
            v_target = last_n
        
        if target_pose_tokens is None:
            target_pose_cond = self.get_posed_input(ray_o=target.ray_o, ray_d=target.ray_d)  # [b, v_target, c, h, w]
            target_pose_cond = target_pose_cond[:, -v_target:, ...]
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
            transformer_output_tokens = self.pass_layers(
                self.transformer_decoder,
                decoder_input_tokens,
                gradient_checkpoint=self.config.training.grad_checkpoint and not self._in_ttt_mode,
                checkpoint_every=checkpoint_every
            )

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

        rendered_images = torch.cat(rendered_chunks, dim=1)  # [b, v_target, c, H, W]
        return rendered_images, target_pose_tokens
    
    def _compute_ttt_loss(
        self, 
        s, 
        input, 
        target, 
        full_encoded_latents=None, 
        input_pose_tokens=None, 
        target_pose_tokens=None, 
        compute_target_loss=False,
        input_need_grad=False,
        target_need_grad=False,
    ):
        """
        Compute losses for TTT layer.
        
        Args:
            s: Current state tensor [b, n_latent_vectors, d]
            input: Input batch data
            target: Target batch data
            full_encoded_latents: (Optional) Full encoded latents for distillation
            input_pose_tokens: (Optional) Cached pose tokens for input views
            target_pose_tokens: (Optional) Cached pose tokens for target views
            compute_target_loss: (Optional) Whether to compute target loss
            input_need_grad: Whether to keep tracking gradients for input loss
            target_need_grad: Whether to keep tracking gradients for target loss
        
        Returns:
            Tuple of (decoder_input, input_loss, target_loss, distillation_loss, layer_metrics, input_pose_tokens, target_pose_tokens)
        """
        if self.config.model.ttt.detach_decoder_input:
            # If detach decoder input, the gradient will not flow into the decoder input.
            decoder_input = s.detach().requires_grad_(True)
        else:
            decoder_input = s

        layer_metrics = {}
        
        # render input views and compute input loss
        with torch.no_grad() if not input_need_grad else torch.enable_grad():
            rendered_input, input_pose_tokens = self.decode(
                input, 
                decoder_input, 
                target_pose_tokens=input_pose_tokens, 
                last_n=self.config.training.num_input_views - self.config.model.ttt.n_encoder_inputs
            )
            input_loss_metrics = self.loss_computer(rendered_input, input.image[:, self.config.model.ttt.n_encoder_inputs:, ...])
            layer_metrics["input_loss"] = input_loss_metrics["loss"].item()

        # render target views and compute target loss
        rendered_target = None
        target_loss_metrics = None
        if self.config.model.ttt.supervise_mode == "average" or self.config.model.ttt.supervise_mode == "g3r" or compute_target_loss:
            with torch.no_grad() if not target_need_grad else torch.enable_grad():
                rendered_target, target_pose_tokens = self.decode(target, decoder_input, target_pose_tokens=target_pose_tokens)
                target_loss_metrics = self.loss_computer(rendered_target, target.image)
                layer_metrics["target_loss"] = target_loss_metrics["loss"].item()
        
        # compute the distillation loss
        distillation_loss = None
        if self.config.model.ttt.distill_factor > 0.0:
            raise NotImplementedError("check whether no_grad is needed here")
            assert full_encoded_latents is not None, "full_encoded_latents is required for distillation"
            distillation_loss = F.mse_loss(s, full_encoded_latents)
            layer_metrics["distillation_loss"] = distillation_loss.item()
        
        return decoder_input, input_loss_metrics, target_loss_metrics, distillation_loss, layer_metrics, input_pose_tokens, target_pose_tokens, rendered_input, rendered_target
    

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
        
        if self.config.model.ttt.grad_mode == "normal":
            # retrain_graph is needed for traversing the same computational graph the second time
            grad_s = torch.autograd.grad(input_loss, decoder_input, create_graph=False, retain_graph=True)[0]
        elif self.config.model.ttt.grad_mode == "zero":
            grad_s = torch.zeros_like(s)
        elif self.config.model.ttt.grad_mode == "random":
            grad_s = torch.randn_like(s)

        if self.config.model.ttt.detach_grad:
            # If detach grad, the gradient will not flow into the decoder 
            grad_s = grad_s.detach()

        # log gradient statistics before normalizer
        layer_metrics["orig_grad_max"] = torch.max(torch.abs(grad_s)).item()
        layer_metrics["orig_grad_mean"] = torch.mean(torch.abs(grad_s)).item()
        layer_metrics["orig_grad_std"] = torch.std(grad_s).item()

        with torch.no_grad() if not need_grad else torch.enable_grad():
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
                # Create Adam optimizer with the current state as parameter
                state_param = nn.Parameter(s.clone().detach().requires_grad_(True))
                state_param.grad = grad_s_normed
                
                # Get Adam hyperparameters
                adam_lr = self.config.model.ttt.adam.lr
                adam_beta1 = self.config.model.ttt.adam.beta1
                adam_beta2 = self.config.model.ttt.adam.beta2
                adam_eps = self.config.model.ttt.adam.eps
                adam_weight_decay = self.config.model.ttt.adam.weight_decay
                
                # Create Adam optimizer
                optimizer = torch.optim.Adam(
                    [state_param], 
                    lr=adam_lr, 
                    betas=(adam_beta1, adam_beta2), 
                    eps=adam_eps, 
                    weight_decay=adam_weight_decay
                )
                
                # update the state
                optimizer.step()
                delta_s = state_param.data - s
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
                    if (isinstance(state_norm, nn.RMSNorm) or isinstance(state_norm, nn.LayerNorm)) and state_norm.elementwise_affine:
                        layer_metrics["state_norm_scaler"] = state_norm.weight.mean().item()

                    # log the opt input state -- state after normalizer
                    layer_metrics["opt_state_max"] = torch.max(opt_input_s).item()
                    layer_metrics["opt_state_mean"] = torch.mean(opt_input_s).item()
                    layer_metrics["opt_state_std"] = torch.std(opt_input_s).item()

                    opt_input = torch.cat((opt_input_s, grad_s_normed), dim=-1) # [b, n_latent_vectors, 2*d]

                delta_s = opt(opt_input) # [b, n_latent_vectors, d]
            
            # get the effective state_lr for this layer
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
                state_lr = lrnet(log_abs_grad_s) # [b, n_latent_vectors, d]
            else:
                # Use fixed state_lr from config
                state_lr = self.config.model.ttt.state_lr
            
            # Apply update with effective learning rate
            s_update = delta_s * state_lr
            
            # Apply update
            if self.config.model.ttt.is_residual:
                s = s_update + (s.detach() if self.config.model.ttt.detach_residual else s)
            else:   
                s = s_update

            # log state statistics
            layer_metrics["state_max"] = torch.max(s).item()
            layer_metrics["state_mean"] = torch.mean(s).item()
            layer_metrics["state_std"] = torch.std(s).item()
            
            # log learnable lr statistics if applicable
            if self.config.model.ttt.state_lr_mode == "learnable" or self.config.model.ttt.state_lr_mode == "adaptive" or self.config.model.ttt.state_lr_mode == "adaptive_mlp":
                layer_metrics["state_lr_mean"] = torch.mean(state_lr).item()
                layer_metrics['state_lr_max'] = torch.max(state_lr).item()
                layer_metrics['state_lr_std'] = torch.std(state_lr).item()
        
        return s, layer_metrics
    
    
    def ttt_forward(self, input, target):
        """
        Update the latent tokens with the TTT blocks. Returns the updated state and TTT metrics for logging.
        Args:
            input: Input data batch
            target: Target data batch
            partial_encoded_latents: Latent tokens with only using the first n_encoder_inputs input views [b, n_latent_vectors, d]
            full_encoded_latents: (Optional) Latent tokens with using all input views [b, n_latent_vectors, d]
        Returns:
            s: Updated latent tokens [b, n_latent_vectors, d]
            ttt_metrics: TTT metrics
        """
        # Disable gradient checkpointing during TTT to avoid double differentiation
        self._in_ttt_mode = True

        partial_encoded_latents, full_encoded_latents = self.encode(input)
        s = partial_encoded_latents
        if self.config.model.ttt.detach_s0:
            # If detach s0, the gradient will not flow into encoder, tokenizer and the register_token.
            # We need to detach s but then make it require grad again for autograd.grad to work
            s = s.detach()

        # iterate over layers and iterations
        input_pose_tokens = None
        target_pose_tokens = None
        ttt_metrics = {'layers': []}
        losses = []
        for layer_idx in range(self.config.model.ttt.n_layer):
            if self.config.model.ttt.supervise_mode == "random_last":
                # if random_last, we randomly choose the number of iterations between [min_layer, max_layer]
                max_iter = random.randint(self.config.model.ttt.min_layer, self.config.model.ttt.max_layer)
            else:
                max_iter = self.config.model.ttt.n_iters_per_layer
            
            for iter_idx in range(max_iter):
                # input: always need gradient -- We need to calculate gradient later.
                # target:
                # - training time: only when supervise average need gradient.
                # - inference time: no need gradient.
                s = s.requires_grad_(True)
                decoder_input, input_loss_metrics, target_loss_metrics, distillation_loss, loss_metrics, input_pose_tokens, target_pose_tokens, _, _ = self._compute_ttt_loss(
                    s, input, target, full_encoded_latents, input_pose_tokens, target_pose_tokens,
                    input_need_grad=True,
                    target_need_grad=self.config.model.ttt.supervise_mode == "average" and self.config.training.supervision == "target" and not self.config.inference.if_inference,
                )

                if self.config.model.ttt.supervise_mode == "average":
                    if self.config.training.supervision == "input":
                        loss = input_loss_metrics["loss"]
                    elif self.config.training.supervision == "target":
                        loss = target_loss_metrics["loss"]
                    if self.config.model.ttt.distill_factor > 0.0:
                        loss += distillation_loss * self.config.model.ttt.distill_factor
                    losses.append(loss)
                
                # Update state with self-supervision losses
                grad_norm = self.ttt_grad_normalizers[layer_idx]
                state_norm = self.ttt_state_normalizers[layer_idx]
                opt = self.ttt_blocks[layer_idx]
                lrnet = None
                if self.config.model.ttt.state_lr_mode in ["learnable", "adaptive", "adaptive_mlp"]:
                    lrnet = self.ttt_lrnet[layer_idx]
                
                # training time: if detach middle, we dont keep tracking gradients before the last layer
                # inference time: always dont keep tracking gradients
                s, update_metrics = self._update_state_with_loss(
                    s, decoder_input, grad_norm, state_norm, opt, input_loss_metrics["loss"], lrnet,
                    need_grad=(not self.config.model.ttt.detach_middle or (iter_idx == max_iter - 1)) and not self.config.inference.if_inference,
                )
                
                # Merge metrics
                layer_metrics = {**loss_metrics, **update_metrics}
                ttt_metrics['layers'].append(layer_metrics)

                if self.config.model.ttt.detach_middle and (iter_idx != max_iter - 1):
                    s = s.detach()

                # debug: print the memory usage after each update
                print(f"[{layer_idx}, {iter_idx}]: allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB, cached memory: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")

        # Compute the last layer losses
        # input: dont need gradient unless supervise input at training time
        # target: always need gradient if supervise target at training time.
        decoder_input, last_input_loss_metrics, last_target_loss_metrics, last_distillation_loss, last_layer_metrics, _, _, _, rendered_target = self._compute_ttt_loss(
            s, input, target, full_encoded_latents, 
            input_pose_tokens=None, 
            target_pose_tokens=target_pose_tokens, 
            compute_target_loss=True,
            input_need_grad=self.config.training.supervision == "input" and not self.config.inference.if_inference,
            target_need_grad=self.config.training.supervision == "target" and not self.config.inference.if_inference,
        )

        if self.config.training.supervision == "input":
            loss = last_input_loss_metrics["loss"]
        elif self.config.training.supervision == "target":
            loss = last_target_loss_metrics["loss"]
        if self.config.model.ttt.distill_factor > 0.0:
            loss += last_distillation_loss * self.config.model.ttt.distill_factor
        losses.append(loss)

        # TODO: hacky replacing
        # Render full input images, only for visualization
        rendered_input, _ = self.decode(input, decoder_input)

        ttt_metrics["last_input_loss"] = last_layer_metrics["input_loss"]
        ttt_metrics["last_target_loss"] = last_layer_metrics["target_loss"]
        if self.config.model.ttt.distill_factor > 0.0:
            ttt_metrics["last_distillation_loss"] = last_layer_metrics["distillation_loss"]

        loss = sum(losses) / len(losses)
        ttt_metrics["loss"] = loss.item()

        # Re-enable gradient checkpointing after TTT
        self._in_ttt_mode = False

        print(f"[last layer]: allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB, cached memory: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")

        return input, target, last_input_loss_metrics, last_target_loss_metrics, last_distillation_loss, rendered_input, rendered_target, loss, ttt_metrics

    
    def ttt_forward_g3r(self, input, target, layer_idx, iter_idx, s=None, full_encoded_latents=None, input_pose_tokens=None, target_pose_tokens=None, update=True):
        """
        Forward the latent tokens with the TTT blocks for G3R supervision. Returns the updated state and TTT metrics for logging.
        Args:
            input: Input data batch
            target: Target data batch
            layer_idx: Layer index
            iter_idx: Iteration index
            s: (Optional) Current state tensor [b, n_latent_vectors, d]
            full_encoded_latents: (Optional) Latent tokens with using all input views [b, n_latent_vectors, d]
            input_pose_tokens: (Optional) Cached pose tokens for input views
            target_pose_tokens: (Optional) Cached pose tokens for target views
            update: Whether to update the state
        Returns:
            s: Updated latent tokens [b, n_latent_vectors, d]
        """
        assert layer_idx is not None and iter_idx is not None, "layer_idx and iter_idx must be provided for G3R supervision"
        raise NotImplementedError("check whether gradient in this case is needed")

        # Disable gradient checkpointing during TTT to avoid double differentiation
        self._in_ttt_mode = True

        if s is None:
            assert layer_idx == 0 and iter_idx == 0, "layer_idx and iter_idx must be 0 for G3R supervision when s is not provided"
            partial_encoded_latents, full_encoded_latents = self.encode(input)
            s = partial_encoded_latents
        
        if self.config.model.ttt.detach_s0:
            # If detach s0, the gradient will not flow into encoder, tokenizer and the register_token.
            # We need to detach s but then make it require grad again for autograd.grad to work
            s = s.detach().requires_grad_(True)

        # Compute self-supervision losses
        decoder_input, input_loss_metrics, target_loss_metrics, distillation_loss, loss_metrics, input_pose_tokens, target_pose_tokens, rendered_input, rendered_target = self._compute_ttt_loss(
            s, input, target, full_encoded_latents, input_pose_tokens, target_pose_tokens
        )
        
        # Update state with self-supervision losses if update is True
        layer_metrics = loss_metrics
        if update:
            grad_norm = self.ttt_grad_normalizers[layer_idx]
            state_norm = self.ttt_state_normalizers[layer_idx]
            opt = self.ttt_blocks[layer_idx]
            lrnet = None
            if self.config.model.ttt.state_lr_mode == "learnable":
                lrnet = self.ttt_lrnet[layer_idx]

            s, update_metrics = self._update_state_with_loss(s, decoder_input, grad_norm, state_norm, opt, input_loss_metrics["loss"], lrnet)
            layer_metrics = {**loss_metrics, **update_metrics}
        else:
            # TODO: hacky replacing
            rendered_input, _ = self.decode(input, decoder_input)

        if self.config.training.supervision == "input":
            loss = input_loss_metrics["loss"]
        elif self.config.training.supervision == "target":
            loss = target_loss_metrics["loss"]
        if self.config.model.ttt.distill_factor > 0.0:
            loss += distillation_loss * self.config.model.ttt.distill_factor

        # Re-enable gradient checkpointing after TTT
        self._in_ttt_mode = False

        return input, target, input_loss_metrics, target_loss_metrics, distillation_loss, rendered_input, rendered_target, loss, s, full_encoded_latents, input_pose_tokens, target_pose_tokens, layer_metrics
    
    
    def forward(self, data_batch, has_target_image=True, layer_idx=None, iter_idx=None, **kwargs):
        assert has_target_image, "JC: we might need to support this?"
        if kwargs.get("input") is None or kwargs.get("target") is None:
            if "input" in kwargs:
                kwargs.pop("input")
            if "target" in kwargs:
                kwargs.pop("target")
            input, target = self.process_data(data_batch, has_target_image=has_target_image, target_has_input = self.config.training.target_has_input, compute_rays=True)
        else:
            input, target = kwargs.pop("input"), kwargs.pop("target")
        
        if self.config.model.ttt.supervise_mode == "g3r":
            assert layer_idx is not None and iter_idx is not None, "layer_idx and iter_idx must be provided for G3R supervision"
            return self.ttt_forward_g3r(input, target, layer_idx, iter_idx, **kwargs)
        else:    
            return self.ttt_forward(input, target)


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
            input, target = self.process_data(data_batch, has_target_image=False, target_has_input=self.config.training.target_has_input, compute_rays=True)
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
