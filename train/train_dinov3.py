import os
import random

from peft import LoraConfig, get_peft_model
import yaml
import torch
from argparse import ArgumentParser, Namespace
from torch.utils.tensorboard import SummaryWriter
from dino3d.losses.corr_loss import ConsistencyLoss
from dino3d.losses.corr_map_model import Correlation2Displacement
from dino3d.models.dino3d import DINO3D
from dino3d.models.base_dino import DINO
from dino3d.checkpointing import CheckPoint
from tqdm import tqdm

from dino3d.train.train import fix_random_seeds, get_dataloaders, get_train_dataloader, run_iter, log_pca, get_scheduler, load_config, SafeNamespace
from dinov3.models.vision_transformer import DinoVisionTransformer
import torch.multiprocessing as mp




def get_dinoV33d_model(args, spatial_reduction=True, use_sa_ffn=False, plucker_emb_dim=128, device=None):
    """
    Create the DinoV3 3D model with custom layers for training.
    
    Args:
        args: Arguments containing model configuration
        spatial_reduction: Whether to use spatial reduction in adapters
        use_sa_ffn: Whether to use FFN in spatial adapters
        plucker_emb_dim: Dimension of plucker embedding (0 to disable)
        device: Device to load the model on
    
    Returns:
        DinoVisionTransformer model with loaded checkpoint and frozen base parameters
    """
    # Create the base DinoV3 ViT small model with custom extensions
    # Parameters based on dinov3_vits16plus from backbones.py
    model = DinoVisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=6,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=True,
        spatial_reduction=spatial_reduction,
        use_sa_ffn=use_sa_ffn,
        plucker_emb_dim=plucker_emb_dim,
        # Other parameters with defaults
        device=device,
    )
    
        # Load the pretrained checkpoint
    checkpoint_path = "/home/admina/Dino3D/dinov3_checkpoint/dinov3_vits16plus.pth"
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"Loaded checkpoint from {checkpoint_path}")
    except Exception as e:
        print(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        print("Initializing model weights from scratch")
        model.init_weights()
        checkpoint = model.state_dict()  # Use current state as checkpoint
    
    # Load the state dict, filtering out mismatched keys due to custom layers
    model_state_dict = model.state_dict()
    pretrained_state_dict = {}
    
    for k, v in checkpoint.items():
        if k in model_state_dict:
            if model_state_dict[k].shape == v.shape:
                pretrained_state_dict[k] = v
            else:
                print(f"Skipping {k} due to shape mismatch: {v.shape} vs {model_state_dict[k].shape}")
        else:
            print(f"Skipping {k} as it's not in the model")
    
    # Load the filtered state dict
    missing_keys, unexpected_keys = model.load_state_dict(pretrained_state_dict, strict=False)
    print(f"Loaded {len(pretrained_state_dict)} matched parameters")
    if missing_keys:
        print(f"Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
    
    # Freeze all parameters except the custom layers
    for name, param in model.named_parameters():
        if 'spatial_adapters' in name or 'plucker_embed' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    if getattr(args, "use_lora", False):
        # Auto-detect actual submodule attribute names that are supported by PEFT
        target_modules = set()
        for name, module in model.named_modules():
            # Only target leaf modules of supported types within blocks so we don't pass composite modules like Mlp
            if name.startswith('blocks.') and isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.MultiheadAttention)):
                target_modules.add(name)

        # Add some common fallback names used in transformer/ViT implementations
        fallback_names = ["blocks.0.qkv", "blocks.0.q_proj", "blocks.0.k_proj", "blocks.0.v_proj", "blocks.0.q", "blocks.0.k", "blocks.0.v", "blocks.0.fc1", "blocks.0.fc2", "blocks.0.proj", "blocks.0.out_proj"]
        if not target_modules:
            target_modules = set(fallback_names)

        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=list(target_modules),
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, config)
    
    return model.float()


def train(args):

    fix_random_seeds(args.seed)
    # Create experiment directory
    if not os.path.exists(args.exp_directory):
        os.makedirs(args.exp_directory)
    os.makedirs(f'{args.exp_directory}/{args.exp_name}', exist_ok=True)
    curr_exp_dir = f'{args.exp_directory}/{args.exp_name}'
    checkpoint_dir = f"{curr_exp_dir}/checkpoints/"
    writer = SummaryWriter(curr_exp_dir)

    # Dump args to yaml and save to experiment directory
    with open(f"{curr_exp_dir}/args.yaml", "w") as f:
        yaml.dump(vars(args), f)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Init Model
    model = get_dinoV33d_model(args)
    model = model.to(device)
    model.train(True)


    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized model with {learnable_params:,} learnable parameters out of total {total_params:,} parameters.")

    # Init base model
    base_model = get_dinoV33d_model(args, 
                                device=device, 
                                spatial_reduction=False, 
                                use_sa_ffn=False, 
                                plucker_emb_dim=0)
    base_model = base_model.to(device)
    base_model.eval()

    # Data
    batch_size = args.batch_size
    dataloader, eval_dataloader, train_dataset_size = get_dataloaders(args)

    # Dino feature map dims
    feature_size = args.image_size[0] // model.patch_size
    if args.concat_base_features:
        feature_size *= 2
    # Loss and optimizer
    correlation_module = Correlation2Displacement(feature_size=feature_size, beta=args.argsoftmax_beta)
    consistency_loss = ConsistencyLoss(correlation_module, optim_strategy=args.optim_strategy, loss_type=args.loss_type)
    eval_correlation_module = Correlation2Displacement(feature_size=feature_size, calc_soft_argmax=False)
    eval_consistency_loss = ConsistencyLoss(eval_correlation_module, optim_strategy='similarity', loss_type='L1')
    parameters = [
        {"params": model.spatial_adapters.parameters(), "lr": float(args.lr)},
        {"params": model.plucker_embed.parameters(), "lr": float(args.lr)},
    ]
    if args.use_lora:
        parameters.append({"params": [p for p in model.blocks.parameters() if p.requires_grad], "lr": float(args.lr)})


    optimizer = torch.optim.AdamW(parameters, weight_decay=args.weight_decay, eps=1e-7)
    lr_scheduler = get_scheduler(optimizer, train_dataset_size, batch_size, args)

    checkpointer = CheckPoint(checkpoint_dir)
    model, optimizer, lr_scheduler, global_step = checkpointer.load(model, optimizer, lr_scheduler)

    # If no checkpoint loaded, start from 0
    if global_step:
        first_epoch = global_step // len(dataloader)
        print(f"Resuming from global step {global_step}, epoch {first_epoch}")
    else:
        global_step = 0
        first_epoch = 0
    location_error = None

    # Training loop
    for epoch in range(first_epoch, args.num_epochs):
        if epoch % args.validation_interval == 0 and not args.skip_eval:
            location_error = 0
            cosine_similarity = 0
            print("Evaluating on validation set...")
            random_idx = random.randint(0, len(eval_dataloader) - 1)
            for eval_idx, eval_data in enumerate(tqdm(eval_dataloader)):
                with torch.no_grad():
                    eval_features, eval_base_features, eval_dense_loss, eval_sparse_loss, eval_regularization_loss = run_iter(
                        model, eval_data, eval_consistency_loss, device=device, base_model=base_model, concat_with_base=args.concat_with_base, separate_forward_regularization=args.separate_forward_regularization, dinov3=True)
                if eval_idx == random_idx:
                    if args.concat_with_base:
                        eval_features = eval_features[:, :, eval_features.shape[2] // 2:]
                    log_pca(writer, global_step, 'Eval', eval_data['images'], eval_features, eval_base_features)
                location_error += eval_dense_loss.item()
                cosine_similarity += 1 - eval_sparse_loss.item()

            location_error /= len(eval_dataloader)
            cosine_similarity /= len(eval_dataloader)

            writer.add_scalar("Eval/Location Error", location_error, global_step)
            writer.add_scalar("Eval/Cosine Similarity", cosine_similarity, global_step)

        # Save checkpoint
        if epoch % args.checkpoint_interval == 0:
            checkpointer.save(model, optimizer, lr_scheduler, global_step, epoch, location_error)


        pbar = tqdm(dataloader)
        random_idx = random.randint(0, len(dataloader) - 1)

        for train_idx, data in enumerate(pbar):
            optimizer.zero_grad()
            features, base_features, dense_loss, sparse_loss, regularization_loss = run_iter(model,
                                                                                             data,
                                                                                             consistency_loss,
                                                                                             device=device,
                                                                                             base_model=base_model,
                                                                                             concat_with_base = args.concat_with_base,
                                                                                             separate_forward_regularization=args.separate_forward_regularization,
                                                                                             dinov3=True)
            loss = args.dense_lamda * dense_loss + float(args.cl_lamda) * sparse_loss + float(args.reg_lamda) * regularization_loss
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch {epoch}/{args.num_epochs-1} - Loss: {loss.item():.4f}")
            global_step += 1

            # Log to tensorboard
            if global_step % args.log_interval == 0:
                plt_dict = {'Total loss': loss.item(),
                            'Dense loss': dense_loss.item(),
                            'Sparse loss': sparse_loss.item(),
                            'Regularization loss': regularization_loss.item(),
                            }
                for k, v in plt_dict.items():
                    writer.add_scalar(f"Train Loss/{k}", v, global_step)
                writer.add_scalar("Train Params/lr", optimizer.param_groups[0]["lr"], global_step)

            if train_idx == random_idx:
                if args.concat_with_base:
                    features = features[:, :, features.shape[2] // 2:]
                log_pca(writer, global_step, 'Train', data['images'], features, base_features)

            if lr_scheduler is not None:
                lr_scheduler.step()
            
        
        if args.shuffle_data:
            dataloader, train_dataset_size = get_train_dataloader(args)


    checkpointer.save(model, optimizer, lr_scheduler, global_step)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = ArgumentParser()
    # parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--exp_directory", default='experiments', type=str, help="Name of the experiment for logging")
    parser.add_argument("--exp_name", required=True, type=str, help="Name of the experiment for logging")
    parser.add_argument("--config_name", default='train.yaml', type=str, help="YAML Configuration file")
    parser.add_argument("--log_interval", default=1, type=int, help="Interval for logging to TensorBoard")
    parser.add_argument("--validation_interval", default=1, type=int, help="Interval for calculating validation set")
    parser.add_argument("--pca_log_interval", default=25000 , type=int, help="Interval for logging to TensorBoard")
    parser.add_argument("--checkpoint_interval", default=1, type=int, help="Interval for saving checkpoints")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--colmap_path", default='data/', type=str, help="Path to COLMAP data")
    parser.add_argument("--scene", default=None, type=str, help="Name of scene")
    parser.add_argument('--skip_eval', action='store_true', help='Skip evaluation on validation set')
    parser.add_argument('--chunk_data', action='store_true', help='Use chunked data loading')
     # Parse known args

    args, _ = parser.parse_known_args()
    # Merge cfg with args
    config_path = os.path.join('configs', args.config_name)
    cfg = load_config(config_path)
    args = SafeNamespace(**vars(args), **cfg)
    train(args)
