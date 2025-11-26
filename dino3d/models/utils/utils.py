from peft import LoraConfig, get_peft_model
from dino3d.models.base_dino import DINO
from dino3d.models.dino3d import DINO3D
from dino3d.models.infalted_dino import DINO3D_V2


def get_dino3d_model(args):
    if args.inflated_attn:
        model = DINO3D_V2(
            model_name=args.model_name,
            output=args.model_output_type,
            return_multilayer=False,
            stride=args.stride,
            pe_embedding_strategy=args.pe_embedding_strategy,
            plucker_emb_dim=args.plucker_emb_dim,
            disable_plucker=args.disable_plucker,
            use_sa_cls_token=args.use_sa_cls_token,
            disable_dino_pe=args.disable_dino_pe,
            sa_parametric_spatial_conv=args.sa_parametric_spatial_conv,
            spatial_reduction=args.spatial_reduction,
            sa_before_spatial=args.sa_before_spatial,
            lora_finetune=args.lora_finetune,
            plucker_mlp=args.plucker_mlp,
            sa_layers=args.sa_layers,
        )
    else:
        model = DINO3D(
            model_name=args.model_name,
            output=args.model_output_type,
            return_multilayer=False,
            stride=args.stride,
            pe_embedding_strategy=args.pe_embedding_strategy,
            plucker_emb_dim=args.plucker_emb_dim,
            disable_plucker=args.disable_plucker,
            use_sa_cls_token=args.use_sa_cls_token,
            disable_dino_pe=args.disable_dino_pe,
            sa_parametric_spatial_conv=args.sa_parametric_spatial_conv,
            spatial_reduction=args.spatial_reduction,
            use_sa_ffn=args.use_sa_ffn,
            lora_finetune= args.lora_finetune,
            plucker_mlp=args.plucker_mlp,
            sa_layers=args.sa_layers,
            epipolar_enabled=args.use_epipolar_attention,
            num_epipolar_samples=args.num_epipolar_samples,
            disable_3d=args.disable_3d,
        )
    if args.use_lora:
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["qkv", "ffn"],
            lora_dropout=0.1,
            bias="none",
        )
        model.vit = get_peft_model(model.vit, config)

    return model


def get_lora_dino_model(args):
    model = DINO(
        model_name=args.model_name,
        output=args.model_output_type,
        return_multilayer=False,
        stride=args.stride,
    )
    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["qkv", "ffn"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model

