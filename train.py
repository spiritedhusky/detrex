import json
from typing import Optional
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
import random
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from pathlib import Path
import numpy as np
from datetime import datetime
from tqdm import tqdm
from detectron2.structures.boxes import BoxMode
from omegaconf import OmegaConf
import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator

# from detrex.data import DetrDatasetMapper
# from projects.maskDINO.data.dataset_mappers.coco_instance_lsj_aug_dataset_mapper import COCOInstanceLSJDatasetMapper, build_transform_gen

from detrex.config import get_config
from projects.maskdino.configs.models.maskdino_r50 import model

from fvcore.common.param_scheduler import MultiStepParamScheduler
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler

from projects.maskdino.modeling.meta_arch.maskdino_head import MaskDINOHead
from projects.maskdino.modeling.pixel_decoder.maskdino_encoder import MaskDINOEncoder
from projects.maskdino.modeling.transformer_decoder.maskdino_decoder import MaskDINODecoder
from projects.maskdino.modeling.criterion import SetCriterion
from projects.maskdino.modeling.matcher import HungarianMatcher
from projects.maskdino.maskdino import MaskDINO
from detectron2.data import MetadataCatalog
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detrex.modeling.backbone import ResNet, BasicStem
import logging
import os
import time
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.utils.events import (
    CommonMetricPrinter, 
    JSONWriter, 
    TensorboardXWriter
)
from detectron2.checkpoint import DetectionCheckpointer
# from detrex.checkpoint import DetectionCheckpointer

from detrex.utils import WandbWriter
from detrex.modeling import ema
from detrex.data.dataset_mappers import COCOInstanceNewBaselineDatasetMapper,coco_instance_transform_gen
from detectron2.utils.logger import setup_logger

setup_logger()

SEED = int(datetime.now().timestamp())

TRAIN_IMG_DIR = Path("../dataset/badlad/images/train")
TRAIN_COCO_PATH = Path("../dataset/badlad/labels/coco_format/train/badlad-train-coco.json")
TEST_IMG_DIR = Path("../dataset/badlad/images/test")
TEST_METADATA_PATH = Path("../dataset/badlad/badlad-test-metadata.json")

OUTPUT_DIR = Path("./output")
OUTPUT_MODEL = OUTPUT_DIR/"model_final.pth"

with TRAIN_COCO_PATH.open() as f:
    train_dict = json.load(f)

with TEST_METADATA_PATH.open() as f:
    test_dict = json.load(f)

print("#### LABELS AND METADATA LOADED ####")

def organize_coco_data(data_dict):
    thing_classes = []

    # Map Category Names to IDs
    for cat in data_dict['categories']:
        thing_classes.append(cat['name'])

    # Images
    images_metadata = data_dict['images']

    # Convert COCO annotations to detectron2 annotations format
    data_annotations = []
    for ann in data_dict['annotations']:
        # coco format -> detectron2 format
        annot_obj = {
            # Annotation ID
            "id": ann['id'],

            # Segmentation Polygon (x, y) coords
            "gt_masks": ann['segmentation'],

            # Image ID for this annotation (Which image does this annotation belong to?)
            "image_id": ann['image_id'],

            # Category Label (0: paragraph, 1: text box, 2: image, 3: table)
            "category_id": ann['category_id'],

            "x_min": ann['bbox'][0],  # left
            "y_min": ann['bbox'][1],  # top
            "x_max": ann['bbox'][0] + ann['bbox'][2],  # left+width
            "y_max": ann['bbox'][1] + ann['bbox'][3]  # top+height
        }
        data_annotations.append(annot_obj)

    return thing_classes, images_metadata, data_annotations


thing_classes, images_metadata, data_annotations = organize_coco_data(
    train_dict
)

thing_classes_test, images_metadata_test, _ = organize_coco_data(
    test_dict
)

print(thing_classes)

train_metadata = pd.DataFrame(images_metadata)
train_metadata = train_metadata[['id', 'file_name', 'width', 'height']]
train_metadata = train_metadata.rename(columns={"id": "image_id"})
print("train_metadata size=", len(train_metadata))
train_metadata.head(5)

train_annot_df = pd.DataFrame(data_annotations)
print("train_annot_df size=", len(train_annot_df))
train_annot_df.head(5)

test_metadata = pd.DataFrame(images_metadata_test)
test_metadata = test_metadata[['id', 'file_name', 'width', 'height']]
test_metadata = test_metadata.rename(columns={"id": "image_id"})
print("test_metadata size=", len(test_metadata))
test_metadata.head(5)

TRAIN_SPLIT = 0.95


n_dataset = len(train_metadata)
n_train = int(n_dataset * TRAIN_SPLIT)
print("n_dataset", n_dataset, "n_train", n_train, "n_val", n_dataset-n_train)

np.random.seed(SEED)

inds = np.random.permutation(n_dataset)
train_inds, valid_inds = inds[:n_train], inds[n_train:]


def convert_coco_to_detectron2_format(
    imgdir: Path,
    metadata_df: pd.DataFrame,
    annot_df: Optional[pd.DataFrame] = None,
    target_indices: Optional[np.ndarray] = None,
):

    dataset_dicts = []
    for _, train_meta_row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
        # Iterate over each image
        image_id, filename, width, height = train_meta_row.values

        annotations = []

        # If train/validation data, then there will be annotations
        if annot_df is not None:
            for _, ann in annot_df.query("image_id == @image_id").iterrows():
                # Get annotations of current iteration's image
                class_id = ann["category_id"]
                gt_masks = ann["gt_masks"]
                bbox_resized = [
                    float(ann["x_min"]),
                    float(ann["y_min"]),
                    float(ann["x_max"]),
                    float(ann["y_max"]),
                ]

                annotation = {
                    "bbox": bbox_resized,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": gt_masks,
                    "category_id": class_id,
                }

                annotations.append(annotation)

        # coco format -> detectron2 format dict
        record = {
            "file_name": str(imgdir/filename),
            "image_id": image_id,
            "width": width,
            "height": height,
            "annotations": annotations
        }

        dataset_dicts.append(record)

    if target_indices is not None:
        dataset_dicts = [dataset_dicts[i] for i in target_indices]

    return dataset_dicts

DATA_REGISTER_TRAINING = "badlad_train"
DATA_REGISTER_VALID    = "badlad_valid"
DATA_REGISTER_TEST     = "badlad_test"

DatasetCatalog.register(
        DATA_REGISTER_TRAINING,
        lambda: convert_coco_to_detectron2_format(
            TRAIN_IMG_DIR,
            train_metadata,
            train_annot_df,
            target_indices=train_inds,
        ),
    )

# Set Training data categories
MetadataCatalog.get(DATA_REGISTER_TRAINING).set(thing_classes=thing_classes)

dataset_dicts_train = DatasetCatalog.get(DATA_REGISTER_TRAINING)
metadata_dicts_train = MetadataCatalog.get(DATA_REGISTER_TRAINING)

print("dicts training size=", len(dataset_dicts_train))
print("################")

DatasetCatalog.register(
    DATA_REGISTER_VALID,
    lambda: convert_coco_to_detectron2_format(
        TRAIN_IMG_DIR,
        train_metadata,
        train_annot_df,
        target_indices=valid_inds,
    ),
)

# Set Validation data categories
MetadataCatalog.get(DATA_REGISTER_VALID).set(thing_classes=thing_classes)

dataset_dicts_valid = DatasetCatalog.get(DATA_REGISTER_VALID)
metadata_dicts_valid = MetadataCatalog.get(DATA_REGISTER_VALID)

print("dicts valid size=", len(dataset_dicts_valid))
print("################")


# Register Test Inference data
DatasetCatalog.register(
    DATA_REGISTER_TEST,
    lambda: convert_coco_to_detectron2_format(
        TEST_IMG_DIR,
        test_metadata,
    )
)

# Set Test data categories
MetadataCatalog.get(DATA_REGISTER_TEST).set(
    thing_classes=thing_classes_test
)

dataset_dicts_test = DatasetCatalog.get(DATA_REGISTER_TEST)
metadata_dicts_test = MetadataCatalog.get(DATA_REGISTER_TEST)

print("#### DATA REGISTERED ####")

# create data loader for badlad dataset
dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names=DATA_REGISTER_TRAINING),
    mapper=L(COCOInstanceNewBaselineDatasetMapper)(
        augmentation=L(coco_instance_transform_gen)(
            image_size=1024,
            min_scale=0.1,
            max_scale=1.0,
            random_flip="horizontal"
        ),
        is_train=True,
        image_format="RGB",
    ),
    total_batch_size=16,
    num_workers=2,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names=DATA_REGISTER_VALID, filter_empty=False),
    mapper=L(COCOInstanceNewBaselineDatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        is_train=False,
        image_format="RGB",
    ),
    num_workers=2,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)

print("#### DATALOADER CREATED ####")

# configure model
dim=256
n_class=4
dn="seg"
dec_layers = 9
input_shape={'res2': ShapeSpec(channels=256, height=None, width=None, stride=4), 'res3': ShapeSpec(channels=512, height=None, width=None, stride=8), 'res4': ShapeSpec(channels=1024, height=None, width=None, stride=16), 'res5': ShapeSpec(channels=2048, height=None, width=None, stride=32)}
model = L(MaskDINO)(
    backbone=L(ResNet)(
        stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
        stages=L(ResNet.make_default_stages)(
            depth=50,
            stride_in_1x1=False,
            norm="FrozenBN",
        ),
        out_features=["res2", "res3", "res4", "res5"],
        freeze_at=1,
    ),
    sem_seg_head=L(MaskDINOHead)(
        input_shape=input_shape,
        num_classes=n_class,
        pixel_decoder=L(MaskDINOEncoder)(
            input_shape=input_shape,
            transformer_dropout=0.0,
            transformer_nheads=8,
            transformer_dim_feedforward=2048,
            transformer_enc_layers=6,
            conv_dim=dim,
            mask_dim=dim,
            norm = 'GN',
            transformer_in_features=['res3', 'res4', 'res5'],
            common_stride=4,
            num_feature_levels=3,
            total_num_feature_levels=4,
            feature_order='low2high',
        ),
        loss_weight= 1.0,
        ignore_value= -1,
        transformer_predictor=L(MaskDINODecoder)(
            in_channels=dim,
            mask_classification=True,
            num_classes="${..num_classes}",
            hidden_dim=dim,
            num_queries=300,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=dec_layers,
            mask_dim=dim,
            enforce_input_project=False,
            two_stage=True,
            dn=dn,
            noise_scale=0.4,
            dn_num=100,
            initialize_box_type='mask2box',
            initial_pred=True,
            learn_tgt=False,
            total_num_feature_levels= "${..pixel_decoder.total_num_feature_levels}",
            dropout = 0.0,
            activation= 'relu',
            nhead= 8,
            dec_n_points= 4,
            return_intermediate_dec = True,
            query_dim= 4,
            dec_layer_share = False,
            semantic_ce_loss = False,
        ),
    ),
    criterion=L(SetCriterion)(
        num_classes="${..sem_seg_head.num_classes}",
        matcher=L(HungarianMatcher)(
            cost_class = 4.0,
            cost_mask = 5.0,
            cost_dice = 5.0,
            num_points = 12544,
            cost_box=5.0,
            cost_giou=2.0,
            panoptic_on="${..panoptic_on}",
        ),
        weight_dict=dict(),
        eos_coef=0.1,
        losses=['labels', 'masks', 'boxes'],
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        dn=dn,
        dn_losses=['labels', 'masks', 'boxes'],
        panoptic_on="${..panoptic_on}",
        semantic_ce_loss=False
    ),
    num_queries=300,
    object_mask_threshold=0.25,
    overlap_threshold=0.8,
    metadata=MetadataCatalog.get(DATA_REGISTER_TRAINING),
    size_divisibility=32,
    sem_seg_postprocess_before_inference=True,
    pixel_mean=[123.675, 116.28, 103.53],
    pixel_std=[58.395, 57.12, 57.375],
    # inference
    semantic_on=False,
    panoptic_on=False,
    instance_on=True,
    test_topk_per_image=100,
    pano_temp=0.06,
    focus_on_box = False,
    transform_eval = True,
)

# set aux loss weight dict
class_weight=4.0
mask_weight=5.0
dice_weight=5.0
box_weight=5.0
giou_weight=2.0
weight_dict = {"loss_ce": class_weight}
weight_dict.update({"loss_mask": mask_weight, "loss_dice": dice_weight})
weight_dict.update({"loss_bbox": box_weight, "loss_giou": giou_weight})
# two stage is the query selection scheme

interm_weight_dict = {}
interm_weight_dict.update({k + f'_interm': v for k, v in weight_dict.items()})
weight_dict.update(interm_weight_dict)
# denoising training

if dn == "standard":
    weight_dict.update({k + f"_dn": v for k, v in weight_dict.items() if k != "loss_mask" and k != "loss_dice"})
    dn_losses = ["labels", "boxes"]
elif dn == "seg":
    weight_dict.update({k + f"_dn": v for k, v in weight_dict.items()})
    dn_losses = ["labels", "masks", "boxes"]
else:
    dn_losses = []
# if deep_supervision:

aux_weight_dict = {}
for i in range(dec_layers):
    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
weight_dict.update(aux_weight_dict)
model.criterion.weight_dict=weight_dict


print("#### MODEL CONFIGURED ####")

# configure hyperparameters

train = get_config("common/train.py").train
# max training iterations
train.max_iter = 500
# warmup lr scheduler
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[2000, 4000],
    ),
    warmup_length=10 / train.max_iter,
    warmup_factor=1.0,
)

optimizer = get_config("common/optim.py").AdamW
# lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep

# initialize checkpoint to be loaded
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = OUTPUT_DIR/"maskdino_r50"


# run evaluation every 5000 iters
train.eval_period = 20

# log training infomation every 20 iters
train.log_period = 20

# save checkpoint every 5000 iters
train.checkpointer.period = 250

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.01
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"


# modify optimizer config
optimizer.lr = 0.0001
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# # modify dataloader config
dataloader.train.num_workers = 2
#
# # please notice that this is total batch size.
# # surpose you're using 4 gpus for training and the batch size for
# # each gpu is 16/4 = 4
dataloader.train.total_batch_size = 8

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir




print("#### MODEL PARAMETERS CONFIGURED ####")

cfg = OmegaConf.create()
cfg.model = model
cfg.optimizer = optimizer
cfg.lr_multiplier = lr_multiplier
cfg.train = train
cfg.dataloader = dataloader

# Custom Trainer Class

class Trainer(SimpleTrainer):
    """
    We've combine Simple and AMP Trainer together.
    """

    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        amp=False,
        clip_grad_params=None,
        grad_scaler=None,
    ):
        super().__init__(model=model, data_loader=dataloader, optimizer=optimizer)

        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        if amp:
            if grad_scaler is None:
                from torch.cuda.amp import GradScaler

                grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler
        
        # set True to use amp training
        self.amp = amp

        # gradient clip hyper-params
        self.clip_grad_params = clip_grad_params

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[Trainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        with autocast(enabled=self.amp):
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()

        if self.amp:
            self.grad_scaler.scale(losses).backward()
            if self.clip_grad_params is not None:
                self.grad_scaler.unscale_(self.optimizer)
                self.clip_grads(self.model.parameters())
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            losses.backward()
            if self.clip_grad_params is not None:
                self.clip_grads(self.model.parameters())
            self.optimizer.step()

        self._write_metrics(loss_dict, data_time)

    def clip_grads(self, params):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return torch.nn.utils.clip_grad_norm_(
                parameters=params,
                **self.clip_grad_params,
            )

    def state_dict(self):
        ret = super().state_dict()
        if self.grad_scaler and self.amp:
            ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if self.grad_scaler and self.amp:
            self.grad_scaler.load_state_dict(state_dict["grad_scaler"])


def do_test(cfg, model, eval_only=False):
    logger = logging.getLogger("detectron2")

    if eval_only:
        logger.info("Run evaluation under eval-only mode")
        if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
            logger.info("Run evaluation with EMA.")
        else:
            logger.info("Run evaluation without EMA.")
        if "evaluator" in cfg.dataloader:
            ret = inference_on_dataset(
                model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
            )
            print_csv_format(ret)
        return ret
    
    logger.info("Run evaluation without EMA.")
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)

        if cfg.train.model_ema.enabled:
            logger.info("Run evaluation with EMA.")
            with ema.apply_model_ema_and_restore(model):
                if "evaluator" in cfg.dataloader:
                    ema_ret = inference_on_dataset(
                        model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
                    )
                    print_csv_format(ema_ret)
                    ret.update(ema_ret)
        return ret


def do_train(resume, cfg):
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)
    
    # instantiate optimizer
    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    # build training loader
    train_loader = instantiate(cfg.dataloader.train)
    
    # create ddp model
    model = create_ddp_model(model, **cfg.train.ddp)

    # build model ema
    ema.may_build_model_ema(cfg, model)

    trainer = Trainer(
        model=model,
        dataloader=train_loader,
        optimizer=optim,
        amp=cfg.train.amp.enabled,
        clip_grad_params=cfg.train.clip_grad.params if cfg.train.clip_grad.enabled else None,
    )
    
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
        # save model ema
        **ema.may_get_ema_checkpointer(cfg, model)
    )

    if comm.is_main_process():
        # writers = default_writers(cfg.train.output_dir, cfg.train.max_iter)
        output_dir = cfg.train.output_dir
        PathManager.mkdirs(output_dir)
        writers = [
            CommonMetricPrinter(cfg.train.max_iter),
            JSONWriter(os.path.join(output_dir, "metrics.json")),
            TensorboardXWriter(output_dir),
        ]
        if cfg.train.wandb.enabled:
            PathManager.mkdirs(cfg.train.wandb.params.dir)
            writers.append(WandbWriter(cfg))

    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            ema.EMAHook(cfg, model) if cfg.train.model_ema.enabled else None,
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                writers,
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=resume)
    if resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)

def main():
    do_train(False, cfg)


# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
torch.cuda.empty_cache()

launch(
        main,
        1,
        num_machines=1,
        args=(),
    )
