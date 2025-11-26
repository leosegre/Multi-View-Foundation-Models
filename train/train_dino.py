import os
import random

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

from dino3d.models.utils.utils import get_dino3d_model
from dino3d.train.train import fix_random_seeds, get_dataloaders, get_train_dataloader, run_iter, log_pca, get_scheduler, load_config, SafeNamespace

import torch.multiprocessing as mp
from dino3d.losses.CosineSimLoss import CosineSimilarityLoss



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
    model = get_dino3d_model(args)
    model = model.to(device)
    model.train(True)


    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized model with {learnable_params:,} learnable parameters out of total {total_params:,} parameters.")


    # Init base model
    base_model = DINO(
        model_name=args.model_name,
        output=args.model_output_type,
        return_multilayer=False,
        stride=args.stride,
    )
    base_model = base_model.to(device)
    base_model.eval()

    # Data
    batch_size = args.batch_size
    dataloader, eval_dataloader, train_dataset_size = get_dataloaders(args)

    # Dino feature map dims
    feature_size = 1 + (args.image_size[0] - model.patch_size) // model.stride
    if args.concat_base_features:
        feature_size *= 2
    # Loss and optimizer
    correlation_module = Correlation2Displacement(feature_size=feature_size, beta=args.argsoftmax_beta)
    consistency_loss = ConsistencyLoss(correlation_module, optim_strategy=args.optim_strategy, loss_type=args.loss_type)
    eval_correlation_module = Correlation2Displacement(feature_size=feature_size, calc_soft_argmax=False)
    eval_consistency_loss = ConsistencyLoss(eval_correlation_module, optim_strategy='similarity', loss_type='L1')
    cosine_similarity_loss = CosineSimilarityLoss()

    parameters = [
        {"params": model.spatial_adapters.parameters(), "lr": float(args.lr)},
        {"params": model.plucker_embed.parameters(), "lr": float(args.lr)},
    ]
    if args.use_lora:
        parameters.append({"params": [p for p in model.vit.parameters() if p.requires_grad], "lr": float(args.lr)})


    optimizer = torch.optim.AdamW(parameters, weight_decay=args.weight_decay)
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
            cross_plucker_similarity = 0
            print("Evaluating on validation set...")
            random_idx = random.randint(0, len(eval_dataloader) - 1)
            for eval_idx, eval_data in enumerate(tqdm(eval_dataloader)):
                with torch.no_grad():
                    eval_features, eval_base_features, eval_dense_loss, eval_sparse_loss, eval_regularization_loss, eval_cross_plucker_loss = run_iter(
                        model, eval_data, eval_consistency_loss, device=device, base_model=base_model, concat_with_base=args.concat_with_base, separate_forward_regularization=args.separate_forward_regularization, cross_plucker=args.cross_plucker,cosine_similarity_loss=cosine_similarity_loss)
                if eval_idx == random_idx:
                    if args.concat_with_base:
                        eval_features = eval_features[:, :, eval_features.shape[2] // 2:]
                    log_pca(writer, global_step, 'Eval', eval_data['images'], eval_features, eval_base_features)
                location_error += eval_dense_loss.item()
                cosine_similarity += 1 - eval_sparse_loss.item()
                cross_plucker_similarity += 1 - eval_cross_plucker_loss.item()

            location_error /= len(eval_dataloader)
            cosine_similarity /= len(eval_dataloader)
            cross_plucker_similarity /= len(eval_dataloader)

            writer.add_scalar("Eval/Location Error", location_error, global_step)
            writer.add_scalar("Eval/Cosine Similarity", cosine_similarity, global_step)
            writer.add_scalar("Eval/Cross Plucker Similarity", cross_plucker_similarity, global_step)
        # Save checkpoint
        if epoch % args.checkpoint_interval == 0:
            checkpointer.save(model, optimizer, lr_scheduler, global_step, epoch, location_error)


        pbar = tqdm(dataloader)
        random_idx = random.randint(0, len(dataloader) - 1)

        for train_idx, data in enumerate(pbar):
            optimizer.zero_grad()
            features, base_features, dense_loss, sparse_loss, regularization_loss, cross_plucker_loss = run_iter(model,
                                                                                             data,
                                                                                             consistency_loss,
                                                                                             device=device,
                                                                                             base_model=base_model,
                                                                                             concat_with_base = args.concat_with_base,
                                                                                             separate_forward_regularization=args.separate_forward_regularization,
                                                                                             cross_plucker=args.cross_plucker,
                                                                                             cosine_similarity_loss=cosine_similarity_loss
                                                                                             )
            loss = (float(args.dense_lamda) * dense_loss + float(args.cl_lamda) * sparse_loss +
                    float(args.reg_lamda) * regularization_loss + float(args.cross_plucker_loss_lamda) * cross_plucker_loss)
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
                            'Cross Plucker loss': cross_plucker_loss.item(),
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

    parsed_args, unknown = parser.parse_known_args()
    # Merge cfg with args
    config_path = os.path.join('configs', parsed_args.config_name or 'train.yaml')
    cfg = load_config(config_path)
    cfg_parser = ArgumentParser()
    for key, value in cfg.items():
        if key in vars(parsed_args):
            continue
        if isinstance(value, bool):
            cfg_parser.add_argument(f'--{key}', action='store_true', default=None)
        elif isinstance(value, int):
            cfg_parser.add_argument(f'--{key}', type=int, default=None)
        elif isinstance(value, float):
            cfg_parser.add_argument(f'--{key}', type=float, default=None)
        elif isinstance(value, str):
            cfg_parser.add_argument(f'--{key}', type=str, default=None)
        elif isinstance(value, list):
            if value and all(isinstance(x, int) for x in value):
                cfg_parser.add_argument(f'--{key}', nargs='*', type=int, default=None)
            elif value and all(isinstance(x, float) for x in value):
                cfg_parser.add_argument(f'--{key}', nargs='*', type=float, default=None)
    cfg_parser.set_defaults(**cfg)
    cfg_args = cfg_parser.parse_args(unknown)
    args = SafeNamespace(**vars(cfg_args))
    for key, value in vars(parsed_args).items():
        if value is not None and value is not False:
            setattr(args, key, value)
    print("Final parameters:")
    print(yaml.dump(vars(args), default_flow_style=False))
    train(args)
