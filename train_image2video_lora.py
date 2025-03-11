import os
import sys
import time
import warnings
import json
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, Union
from functools import partial

warnings.filterwarnings("ignore")

import deepspeed
import torch
import torchvision
import torch.distributed as dist
from deepspeed.runtime import lr_schedules
from deepspeed.runtime.engine import DeepSpeedEngine
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from einops import rearrange

from hyvideo.config import parse_args
from hyvideo.constants import C_SCALE, PROMPT_TEMPLATE
from hyvideo.dataset.video_loader import VideoDataset
from hyvideo.diffusion import load_denoiser
from hyvideo.ds_config import get_deepspeed_config
from hyvideo.utils.train_utils import (
    prepare_model_inputs,
    load_state_dict,
    set_worker_seed_builder,
    get_module_kohya_state_dict,
    load_lora,
)
from hyvideo.modules import load_model
from hyvideo.text_encoder import TextEncoder
from hyvideo.utils.file_utils import (
    safe_dir,
    get_experiment_max_number,
    empty_logger,
    dump_args,
    dump_codes,
    resolve_resume_path,
    logger_filter,
)
from hyvideo.utils.helpers import (
    as_tuple,
    set_manual_seed,
    set_reproducibility,
    profiler_context,
    all_gather_sum,
    EventsMonitor,
)
from hyvideo.vae import load_vae
from hyvideo.constants import PRECISION_TO_TYPE

from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file


def setup_distributed_training(args):
    deepspeed.init_distributed()

    # Treat micro/global batch size as tuples for compatibility with mix-scale training.
    world_size = dist.get_world_size()
    if args.data_type == "video" and args.video_micro_batch_size is None:
        # When data_type is video and video_micro_batch_size is None, we set the value from micro_batch_size
        args.video_micro_batch_size = args.micro_batch_size

    micro_batch_size = as_tuple(args.micro_batch_size)
    video_micro_batch_size = as_tuple(args.video_micro_batch_size)
    grad_accu_steps = args.gradient_accumulation_steps
    global_batch_size = as_tuple(args.global_batch_size)
    if "video" in args.data_type:
        refer_micro_batch_size = video_micro_batch_size
    else:
        refer_micro_batch_size = micro_batch_size

    if global_batch_size[0] is None:
        # Note: Model/Pipeline parallel is not supported yet. So, data-parallel-size equals to world-size.
        global_batch_size = tuple(
            [mbs_i * world_size * grad_accu_steps for mbs_i in refer_micro_batch_size]
        )
    else:
        assert global_batch_size == [
            mbs_i * world_size * grad_accu_steps for mbs_i in refer_micro_batch_size
        ], f"Global batch size should be divisible by world size, but got {global_batch_size} and {world_size}."

    rank = dist.get_rank()  # Rank of the current process in the cluster.
    device = (
        rank % torch.cuda.device_count()
    )  # Device of the current process in current node.
    # Set current device for the current process, otherwise dist.barrier() will occupy more memory in rank 0.
    torch.cuda.set_device(device)

    # Setup seed for reproducibility or performance.
    set_manual_seed(args.global_seed)
    set_reproducibility(args.reproduce, args.global_seed)

    return (
        rank,
        device,
        world_size,
        micro_batch_size,
        video_micro_batch_size,
        grad_accu_steps,
        global_batch_size,
    )


def setup_experiment_directory(args, rank):
    output_dir = safe_dir(args.output_dir)

    # Automatically increase the experiment number.
    existed_experiments = list(output_dir.glob("*"))
    experiment_index = get_experiment_max_number(existed_experiments) + 1
    model_name = args.model.replace("/", "").replace(
        "-", "_"
    )  # Replace '/' to avoid sub-directory.
    experiment_dir = (
        output_dir / f"{experiment_index:04d}_{model_name}_{args.task_flag}"
    )
    ckpt_dir = experiment_dir / "checkpoints"

    # Makesure all processes have the same experiment directory.
    dist.barrier()

    if rank == 0:
        from loguru import logger

        logger.add(
            experiment_dir / "train.log",
            level="DEBUG",
            colorize=False,
            backtrace=True,
            diagnose=True,
            encoding="utf-8",
            filter=logger_filter("train"),
        )
        logger.add(
            experiment_dir / "val.log",
            level="DEBUG",
            colorize=False,
            backtrace=True,
            diagnose=True,
            encoding="utf-8",
            filter=logger_filter("val"),
        )
        train_logger = logger.bind(name="train")
        val_logger = logger.bind(name="val")

        ckpt_dir = safe_dir(ckpt_dir)
    else:
        val_logger = train_logger = empty_logger()

    train_logger.info(f"Experiment directory created at: {experiment_dir}")

    return experiment_dir, ckpt_dir, train_logger, val_logger


def get_trainable_params(model, args):
    if args.training_parts is None:
        params = []
        for param in model.parameters():
            if param.requires_grad == True:
                params.append(param)
    else:
        raise ValueError(f"Unknown training_parts {args.training_parts}")
    return params


@dataclass
class ScalarStates:
    rank: int = 0  # rank id
    epoch: int = 1  # Accumulated training epochs
    epoch_train_steps: int = 0  # Accumulated training steps in current epoch
    epoch_update_steps: int = 0  # Accumulated update steps in current epoch
    train_steps: int = 0  # Accumulated training steps
    update_steps: int = 0  # Accumulated update steps
    current_run_update_steps: int = 0  # Update steps in current run
    consumed_samples_total: int = 0  # Accumulated consumed samples
    consumed_video_samples_total: int = 0  # Accumulated consumed video samples
    consumed_samples_per_dp: int = (
        0  # Accumulated consumed samples per data-parallel group
    )
    consumed_video_samples_per_dp: int = (
        0  # Accumulated consumed video samples per data-parallel group
    )
    consumed_tokens_total: int = 0  # Accumulated consumed tokens
    consumed_computations_attn: int = (
        0  # Accumulated consumed computations of attention + mlp
    )
    consumed_computations_total: int = 0  # Accumulated consumed computations of total

    def add(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, getattr(self, k) + v)


@dataclass
class CycleStates:
    log_steps: int = 0
    running_loss: float = 0
    running_tokens: int = 0
    running_samples: int = 0
    running_video_samples: int = 0
    running_grad_norm: float = 0
    running_loss_dict: Dict[int, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    log_steps_dict: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, getattr(self, k) + v)

    def reset(self):
        self.log_steps = 0
        self.running_loss = 0
        self.running_tokens = 0
        self.running_samples = 0
        self.running_video_samples = 0
        self.running_grad_norm = 0
        # Must be reset to float to avoid all_reduce type error.
        self.running_loss_dict = defaultdict(float)
        self.log_steps_dict = defaultdict(int)


def save_checkpoint(
    args,
    rank: int,
    logger,
    model_engine: DeepSpeedEngine,
    ema,
    scalar_state: ScalarStates,
    ckpt_dir: Path,
):
    _ = rank  # Currently not used.

    # gather scalar state
    scalar_state_dict = dict(**asdict(scalar_state))
    gather_results_list = [None for _ in range(dist.get_world_size())]
    torch.distributed.all_gather_object(gather_results_list, scalar_state_dict)
    gather_scalar_states = {}
    for results in gather_results_list:
        gather_scalar_states[results["rank"]] = results

    client_state = {
        "args": args,
        "scalar_state": gather_scalar_states,
    }
    if ema is not None:
        client_state["ema"] = ema.state_dict()
        client_state["ema_config"] = ema.config

    def try_save(_save_name):
        checkpoint_path = ckpt_dir / _save_name
        try:
            model_engine.save_checkpoint(
                str(ckpt_dir),
                client_state=client_state,
                tag=_save_name,
            )
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            return checkpoint_path
        except Exception as e:
            logger.error(f"Saved failed to {checkpoint_path}. {type(e)}: {e}")
            return None

    update_steps = scalar_state.update_steps
    save_name = f"{update_steps:07d}"
    save_path = try_save(save_name)

    return [save_path]


def main(args):
    # ============================= Setup ==============================
    # Setup distributed training environment and reproducibility.
    (
        rank,
        device,
        world_size,
        micro_batch_size,
        video_micro_batch_size,
        grad_accu_steps,
        global_batch_size,
    ) = setup_distributed_training(args)
    # Setup experiment directory
    exp_dir, ckpt_dir, logger, val_logger = setup_experiment_directory(args, rank)
    # Load deepspeed config
    deepspeed_config = get_deepspeed_config(
        args,
        video_micro_batch_size[0],
        global_batch_size[0],
        args.output_dir,
        exp_dir.name,
    )
    # Log and dump the arguments and codes.
    logger.info(sys.argv)
    logger.info(str(args))
    if rank == 0:
        # Dump the arguments to a file.
        extra_args = {"world_size": world_size, "global_batch_size": global_batch_size}
        dump_args(args, exp_dir / "args.json", extra_args)
        # Dump codes to the experiment directory.
        dump_codes(
            exp_dir / "codes.tar.gz",
            root=Path(__file__).parent.parent,
            sub_dirs=["hymm", "jobs"],
            save_prefix=args.task_flag,
        )

    # =========================== Build main model ===========================
    logger.info("Building model...")
    factor_kwargs = {"device": device, "dtype": PRECISION_TO_TYPE[args.precision]}
    if args.i2v_mode and args.i2v_condition_type == "latent_concat":
        in_channels = args.latent_channels * 2 + 1
        image_embed_interleave = 2
    elif args.i2v_mode and args.i2v_condition_type == "token_replace":
        in_channels = args.latent_channels
        image_embed_interleave = 4
    else:
        in_channels = args.latent_channels
        image_embed_interleave = 1
    out_channels = args.latent_channels

    if args.embedded_cfg_scale:
        factor_kwargs["guidance_embed"] = True

    model = load_model(
        args,
        in_channels=in_channels,
        out_channels=out_channels,
        factor_kwargs=factor_kwargs,
    )
    model = load_state_dict(args, model, logger)

    if args.use_lora:
        for param in model.parameters():
            param.requires_grad_(False)

        target_modules = [
            "linear",
            "fc1",
            "fc2",
            "img_attn_qkv",
            "img_attn_proj",
            "txt_attn_qkv",
            "txt_attn_proj",
            "linear1",
            "linear2",
        ]

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)

        if args.lora_path != "":
            model = load_lora(model, args.lora_path, device=device)

    logger.info(model)

    if args.reproduce:
        model.enable_deterministic()

    # After model initialization, we set different seed for each process.
    if args.same_data_batch:
        set_manual_seed(args.global_seed)
    else:
        set_manual_seed(args.global_seed + rank)

    ema = None
    ss = ScalarStates(rank=rank)

    # ========================== Initialize model_engine, optimizer =========================
    if args.warmup_num_steps > 0:
        logger.info(
            f"Building scheduler with warmup_min_lr={args.warmup_min_lr}, warmup_max_lr={args.lr}, "
            f"warmup_num_steps={args.warmup_num_steps}."
        )
        lr_scheduler = partial(
            lr_schedules.WarmupLR,
            warmup_min_lr=args.warmup_min_lr,
            warmup_max_lr=args.lr,
            warmup_num_steps=args.warmup_num_steps,
        )
    else:
        lr_scheduler = None

    logger.info("Initializing optimizer (using deepspeed)...")
    model_engine, opt, _, scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=get_trainable_params(model, args),
        config_params=deepspeed_config,
        lr_scheduler=lr_scheduler,
    )

    # ====================== Build denoise scheduler ========================
    logger.info("Building denoise scheduler...")
    denoiser = load_denoiser(args)

    # ============================= Build extra models =========================
    # 2d/3d VAE
    vae, vae_path, s_ratio, t_ratio = load_vae(
        args.vae, args.vae_precision, logger=logger, device=device
    )

    # Text encoder
    text_encoder = TextEncoder(
        text_encoder_type=args.text_encoder,
        max_length=args.text_len
        + (
            PROMPT_TEMPLATE[args.prompt_template_video].get("crop_start", 0)
            if args.prompt_template_video is not None
            else PROMPT_TEMPLATE[args.prompt_template].get("crop_start", 0)
            if args.prompt_template is not None
            else 0
        ),
        text_encoder_precision=args.text_encoder_precision,
        tokenizer_type=args.tokenizer,
        i2v_mode=args.i2v_mode,
        prompt_template=(
            PROMPT_TEMPLATE[args.prompt_template]
            if args.prompt_template is not None
            else None
        ),
        prompt_template_video=(
            PROMPT_TEMPLATE[args.prompt_template_video]
            if args.prompt_template_video is not None
            else None
        ),
        hidden_state_skip_layer=args.hidden_state_skip_layer,
        apply_final_norm=args.apply_final_norm,
        reproduce=args.reproduce,
        logger=logger,
        device=device,
        image_embed_interleave=image_embed_interleave
    )
    if args.text_encoder_2 is not None:
        text_encoder_2 = TextEncoder(
            text_encoder_type=args.text_encoder_2,
            max_length=args.text_len_2,
            text_encoder_precision=args.text_encoder_precision_2,
            tokenizer_type=args.tokenizer_2,
            reproduce=args.reproduce,
            logger=logger,
            device=device,
        )
    else:
        text_encoder_2 = None

    # ================== Define dtype and forward autocast ===============
    target_dtype = None
    autocast_enabled = False
    if model_engine.bfloat16_enabled():
        target_dtype = torch.bfloat16
        autocast_enabled = True
    elif model_engine.fp16_enabled():
        target_dtype = torch.half
        autocast_enabled = True

    # ============================== Load dataset ==============================
    if "video" in args.data_type:
        video_dataset = VideoDataset(
            data_jsons_path=args.data_jsons_path,
            sample_n_frames=args.sample_n_frames,
            sample_stride=args.sample_stride,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            uncond_p=args.uncond_p,
            args=args,
            logger=logger,
        )
        video_sampler = DistributedSampler(
            video_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.global_seed,
            drop_last=False,
        )
        video_batch_sampler = None
        video_loader = DataLoader(
            video_dataset,
            batch_size=video_micro_batch_size[0],
            shuffle=False,
            sampler=video_sampler,
            batch_sampler=video_batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=None if args.num_workers == 0 else args.prefetch_factor,
            worker_init_fn=set_worker_seed_builder(rank),
            persistent_workers=True,
        )
        num_video_samples = len(video_dataset)
    else:
        video_dataset = None
        video_loader = None
        num_video_samples = 0

    loader = video_loader

    # ============================= Print key info =============================
    print(f"[{rank}] Worker ready.")
    dist.barrier()
    main_loader = video_loader

    try:
        iters_per_epoch = len(main_loader) // grad_accu_steps
    except NotImplementedError:
        iters_per_epoch = 0
    except TypeError:
        iters_per_epoch = 0

    params_count = model.params_count()
    logger.info("****************************** Running training ******************************")
    logger.info(f"  Number GPUs:               {world_size}")
    logger.info(f"  Training video samples(total):   {num_video_samples:,}")
    for k, v in params_count.items():
        logger.info(f"  Number {k} parameters:   {v:,}")
    logger.info(f"  Number trainable params:   {sum(p.numel() for p in get_trainable_params(model, args)):,}")
    logger.info("------------------------------------------------------------------------------")
    logger.info(f"  Iters per epoch:           {iters_per_epoch:,}")
    logger.info(f"  Updates per epoch:         {iters_per_epoch // grad_accu_steps:,}")

    logger.info(f"  Batch size per device:     {video_micro_batch_size}")
    logger.info(f"  Batch size all device:     {global_batch_size:}")

    logger.info(f"  Gradient Accu steps:       {args.gradient_accumulation_steps}")
    logger.info(f"  Training epochs:           {ss.epoch}/{args.epochs}")
    logger.info(f"  Training total steps:      {ss.update_steps:,}/{args.max_training_steps:,}")
    logger.info("------------------------------------------------------------------------------")

    logger.info(f"  Path type:                 {args.flow_path_type}")
    logger.info(f"  Predict type:              {args.flow_predict_type}")
    logger.info(f"  Loss weight:               {args.flow_loss_weight}")
    logger.info(f"  Flow reverse:              {args.flow_reverse}")
    logger.info(f"  Flow shift:                {args.flow_shift}")
    logger.info(f"  Train eps:                 {args.flow_train_eps}")
    logger.info(f"  Sample eps:                {args.flow_sample_eps}")
    logger.info(f"  Timestep type:             {args.flow_snr_type}")

    logger.info("------------------------------------------------------------------------------")
    logger.info(f"  Main model precision:      {args.precision}")
    logger.info("------------------------------------------------------------------------------")
    logger.info(f"  VAE:                       {args.vae} ({args.vae_precision}) - {vae_path}")
    logger.info(f"  Text encoder:              {text_encoder}")
    if text_encoder_2 is not None:
        logger.info(f"  Text encoder 2:            {text_encoder_2}")
    logger.info(f"  Experiment directory:      {ckpt_dir}")
    logger.info("*******************************************************************************")

    # ============================= Start training =============================
    model_engine.train()

    if args.init_save:
        save_checkpoint(args, rank, logger, model_engine, ema, ss, ckpt_dir)

    # Training loop
    start_epoch = ss.epoch
    finished = False

    ss.current_run_update_steps = 0
    for epoch in range(start_epoch, args.epochs):

        if video_dataset is not None:
            logger.info(f"Start video random shuffle(seed={args.global_seed + epoch})")
            video_sampler.set_epoch(epoch)  # epoch start from 1
            logger.info(f"End of video random shuffle")

        logger.info(f"Beginning epoch {epoch}...")
        with profiler_context(
            args.profile, exp_dir, worker_name=f"Rank_{rank}"
        ) as prof:
            # Define cycle states, which accumulate the training information between log_steps.
            cs = CycleStates()
            start_time = time.time()

            for batch_idx, batch in enumerate(loader):
                # broadcast a zero size tensor to indicate starting of step
                start_flag_tensor = torch.cuda.FloatTensor([])
                if torch.distributed.is_initialized():
                    torch.distributed.broadcast(start_flag_tensor, 0, async_op=True)

                # main diff
                (
                    latents,
                    model_kwargs,
                    n_tokens,
                    cond_latents,
                ) = prepare_model_inputs(
                    args,
                    batch,
                    device,
                    model,
                    vae,
                    text_encoder,
                    text_encoder_2,
                    rope_theta_rescale_factor=args.rope_theta_rescale_factor,
                    rope_interpolation_factor=args.rope_interpolation_factor,
                )
                cur_batch_size = latents.shape[0]

                cur_anchor_size = max(args.video_size)

                # A forward-backward step
                with torch.autocast(
                    device_type="cuda", dtype=target_dtype, enabled=autocast_enabled
                ):
                    _, loss_dict = denoiser.training_losses(
                        model_engine,
                        latents,
                        model_kwargs,
                        n_tokens=n_tokens,
                        i2v_mode=args.i2v_mode,
                        cond_latents=cond_latents,
                        args=args,
                    )
                loss = loss_dict["loss"].mean()
                model_engine.backward(loss)

                # Update model parameters at the step of gradient accumulation.
                model_engine.step(lr_kwargs={"last_batch_iteration": ss.update_steps})

                # Update accumulated states
                ss.add(
                    train_steps=1,
                    epoch_train_steps=1,
                    consumed_samples_per_dp=cur_batch_size,
                )
                ss.add(consumed_video_samples_per_dp=cur_batch_size)

                # We enable `is_update_step` if the current step is the gradient accumulation boundary.
                is_update_step = ss.train_steps % grad_accu_steps == 0
                if is_update_step:
                    ss.add(
                        update_steps=1, epoch_update_steps=1, current_run_update_steps=1
                    )

                if ss.update_steps >= args.max_training_steps:
                    # Enter stopping routine if max steps reached after this step.
                    finished = True

                # Log training information:
                cs.add(
                    log_steps=1,
                    running_loss=loss.item(),
                    running_samples=cur_batch_size,
                    running_tokens=cur_batch_size * n_tokens,
                    running_grad_norm=0,
                )
                cs.add(running_video_samples=cur_batch_size)

                cs.running_loss_dict[cur_anchor_size] += loss.item()
                cs.log_steps_dict[cur_anchor_size] += 1
                if is_update_step and ss.update_steps % args.log_every == 0:
                    # Reduce loss history over all processes:
                    avg_loss = (
                        all_gather_sum(cs.running_loss / cs.log_steps, device)
                        / world_size
                    )
                    avg_grad_norm = (
                        all_gather_sum(cs.running_grad_norm / cs.log_steps, device)
                        / world_size
                    )
                    cum_samples = all_gather_sum(cs.running_samples, device)
                    cum_video_samples = all_gather_sum(cs.running_video_samples, device)
                    cum_tokens = all_gather_sum(cs.running_tokens, device)
                    # Measure training speed:
                    torch.cuda.synchronize()
                    end_time = time.time()
                    steps_per_sec = (
                        cs.log_steps / (end_time - start_time) / grad_accu_steps
                    )
                    samples_per_sec = cum_samples / (end_time - start_time)
                    sec_per_step = (end_time - start_time) / cs.log_steps
                    ss.add(
                        consumed_samples_total=cum_samples,
                        consumed_video_samples_total=cum_video_samples,
                        consumed_tokens_total=cum_tokens,
                        consumed_computations_attn=6
                        * params_count["attn+mlp"]
                        * cum_tokens
                        / C_SCALE,
                        consumed_computations_total=6
                        * params_count["total"]
                        * cum_tokens
                        / C_SCALE,
                    )
                    log_events = [
                        f"Train Loss: {avg_loss:.4f}",
                        f"Grad Norm: {avg_grad_norm:.4f}",
                        f"Lr: {opt.param_groups[0]['lr']:.6g}",
                        f"Sec/Step: {sec_per_step:.2f}, "
                        f"Steps/Sec: {steps_per_sec:.2f}",
                        f"Samples/Sec: {int(samples_per_sec):d}",
                        f"Consumed Samples: {ss.consumed_samples_total:,}",
                        f"Consumed Video Samples: {ss.consumed_video_samples_total:,}",
                        f"Consumed Tokens: {ss.consumed_tokens_total:,}",
                    ]
                    summary_events = [
                        ("Train/Steps/train_loss", avg_loss, ss.update_steps),
                        ("Train/Steps/grad_norm", avg_grad_norm, ss.update_steps),
                        ("Train/Steps/steps_per_sec", steps_per_sec, ss.update_steps),
                        (
                            "Train/Steps/samples_per_sec",
                            int(samples_per_sec),
                            ss.update_steps,
                        ),
                        ("Train/Tokens/train_loss", avg_loss, ss.consumed_tokens_total),
                        (
                            "Train/ComputationsAttn/train_loss",
                            avg_loss,
                            ss.consumed_computations_attn,
                        ),
                        (
                            "Train/ComputationsTotal/train_loss",
                            avg_loss,
                            ss.consumed_computations_total,
                        ),
                    ]
                    # Log the training information to the logger.
                    logger.info(
                        f"(step={ss.update_steps:07d}) " + ", ".join(log_events)
                    )
                    if model_engine.monitor.enabled and rank == 0:
                        model_engine.monitor.write_events(summary_events)

                    # Reset monitoring variables:
                    cs.reset()
                    start_time = time.time()

                # Save checkpoint:
                if (is_update_step and ss.update_steps % args.ckpt_every == 0) or (
                    finished and args.final_save
                ):
                    if args.use_lora:
                        if rank == 0:
                            output_dir = os.path.join(
                                ckpt_dir, f"global_step{ss.update_steps}"
                            )
                            os.makedirs(output_dir, exist_ok=True)

                            lora_kohya_state_dict = get_module_kohya_state_dict(
                                model, "Hunyuan_video_I2V_lora", dtype=torch.bfloat16
                            )
                            save_file(
                                lora_kohya_state_dict,
                                f"{output_dir}/pytorch_lora_kohaya_weights.safetensors",
                            )
                    else:
                        save_checkpoint(
                            args, rank, logger, model_engine, ema, ss, ckpt_dir
                        )

                if prof:
                    prof.step()

                if finished:
                    logger.info(
                        f"Finished and breaking loop at step={ss.update_steps}."
                    )
                    break

            if finished:
                logger.info(f"Finished and breaking loop at epoch={epoch}.")
                break

            # Reset epoch states
            ss.epoch += 1
            ss.epoch_train_steps = 0
            ss.epoch_update_steps = 0

    logger.info("Training Finished!")


if __name__ == "__main__":
    main(parse_args(mode="train"))
