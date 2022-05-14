# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# from apex import amp
# from apex.parallel import DistributedDataParallel as DDP

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import json
from torchvision import transforms
from collections import defaultdict
import pandas as pd
from torchsummary import summary


logger = logging.getLogger(__name__)

OUTPUT_DIM = {
    "resnet18":  512,
    "resnet50": 2048,
    "r18_sw-sup": 512,
}

def augmentation(key, imsize=500):
    """Using ImageNet statistics for normalization.
    """

    augment_dict = {

        "augment_train":
            transforms.Compose([
                transforms.RandomResizedCrop(
                    imsize, scale=(0.7, 1.0), ratio=(0.99, 1 / 0.99)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])
                ]),

        "augment_inference":
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])
                ])

    }

    return augment_dict[key]

class MET_database(VisionDataset):
    def __init__(
            self,
            root: str = ".",
            mini: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            im_root=None,
            vis_only: bool = True
        ) -> None:
        super().__init__(root, transform=transform,
                            target_transform=target_transform)

        fn = "MET_database.json"

        if mini:
            fn = "mini_" + fn

        with open(os.path.join(self.root, fn)) as f:
            data = json.load(f)

        samples = []
        targets = []

        # read ID and path of images from json file
        for i, e in enumerate(data):
            if i == 18800:
                break
            samples.append(e['path'])
            targets.append(int(e['id']))

        self.loader = loader
        self.samples = samples
        self.targets = targets

        assert len(self.samples) == len(self.targets)

        self.im_root = im_root


    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        if self.im_root is not None:
            path = os.path.join(self.im_root, self.samples[index])

        else:
            path = os.path.join(os.path.dirname(self.root), self.samples[index])

        target = self.targets[index]
        sample = self.loader(path)

        if self.transform is not None:
            # resize and transform image
            sample = self.transform(sample)

        if self.target_transform is not None:
            ## ASK FOR TYPES OF TARGET_TRANSFORMS
            target = self.target_transform(target)

        return sample, target


    def __len__(self) -> int:
        return len(self.samples)


class MET_queries(VisionDataset):

    def __init__(
            self,
            root: str = ".",
            test: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            im_root = None,
            csv = None
    ) -> None:
        super().__init__(root, transform=transform,
                            target_transform=target_transform)

        if test:
            fn = "testset.json"
        else:
            fn = "valset.json"

        departments = dict()

        for i, department in enumerate(csv["Department"].unique()):
            departments[department] = i

        departments["Other"] = 19 # some images do not have a department

        with open(os.path.join(self.root, fn)) as f:
            data = json.load(f)

        samples = []
        targets = []

        for e in data:
            if "MET_id" in e:
                samples.append(e['path'])
                try:
                    department = csv[csv["Object ID"] == row["id"]]["Department"].values[0]
                except:
                    department = "Other"
                targets.append(departments[department]) # see what i want to use as label
            else:
                pass


        self.loader = loader
        self.samples = samples
        self.targets = targets

        assert len(self.samples) == len(self.targets)

        self.im_root = im_root

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        if self.im_root is not None:
            path = os.path.join(self.im_root, self.samples[index])

        else:
            path = os.path.join(os.path.dirname(self.root),
                                self.samples[index])

        target = self.targets[index]

        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:

        return len(self.samples)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    num_classes = 20 if args.dataset == "MET" else 100
    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy

def extract_embeddings(net, dataloader, print_freq = None):
    net.eval()
    with torch.no_grad():

        vecs = np.zeros((768, len(dataloader.dataset)))

        for i, input in enumerate(dataloader):
            # print(net(input[0])[0])
            vecs[:, i] = net(input[0])[0].cpu().data.squeeze()

            if print_freq is not None:
                if i % print_freq == 0:
                    print("image: " + str(i))
    return vecs.T

def train(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    print("Start loading CSV...")
    data = pd.read_csv(args.csv)
    print("Finished loading CSV!")

    extraction_transform = augmentation(args.augment_type)
    train_dataset = MET_database(root = args.info_dir,mini=args.mini,
                                 transform = extraction_transform,
                                 im_root = args.im_root, csv = data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    print("Number of train images: {}".format(len(train_dataset)))

    test_dataset = MET_queries(root = args.info_dir, test = True,
                                    transform = extraction_transform, im_root = args.im_root, csv = data)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    print("Number of test images: {}".format(len(test_dataset)))

    # train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # if args.fp16:
    #     model, optimizer = amp.initialize(models=model,
    #                                       optimizers=optimizer,
    #                                       opt_level=args.fp16_opt_level)
    #     amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    # Distributed training
    # if args.local_rank != -1:
    #     model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            loss = model(x, y)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            # if args.fp16:
            #     with amp.scale_loss(loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                # if args.fp16:
                #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                # else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    accuracy = valid(args, model, writer, test_loader, global_step)
                    if best_acc < accuracy:
                        save_model(args, model)
                        best_acc = accuracy
                    model.train()

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


def create_descriptors(export_dir, net = None, mini = True, info_dir = None,
                        im_root = None, queries_only = False,
                        augment_type = "augment_inference", gpu = False,
                        vis_only = True):
    """
    :export_dir: directory to export descriptors to
    :net: which backbone to use to extract features
    :net_path: path to pretrained neural net
    :mini: use a smaller version of the dataset
    :info_dir: directory where ground truth is stored
    :im_root: directory where images are stored
    """

    # network_variant = net
    exp_name = export_dir + "/" + "ViT"

    if mini:
        exp_name += ("_mini") # use mini dataset
    if queries_only:
        exp_name += ("_queries_only")

    exp_dir = exp_name + "/"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)

    print(f"Will save descriptors in {exp_dir}")

    extraction_transform = augmentation(augment_type)

    if not queries_only:
        train_dataset = MET_database(root = info_dir, mini = mini,
                                     transform = extraction_transform,
                                     im_root = im_root, vis_only = vis_only)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        print("Number of train images: {}".format(len(train_dataset)))

    # if not vis_only:
    # try:
    #     test_dataset = MET_queries(root = info_dir, test = True,
    #                                 transform = extraction_transform, im_root = im_root)
    #     print("Number of test images: {}".format(len(test_dataset)))
    #     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    # except:
    #     pass
    #     val_dataset = MET_queries(root = info_dir, transform = extraction_transform,
    #                                     im_root = im_root)
    #     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    #     print("Number of val images: {}".format(len(val_dataset)))


    # initialization of the global descriptor extractor model
    # if there is a pretrained model

    if gpu:
        net.cuda()

    print("Starting the extraction of the descriptors")

    if not queries_only:
        train_descr = extract_embeddings(net, train_loader, print_freq = 1)
        print("Train descriptors finished...")

    # if not vis_only:
    # try:
    #     test_descr = extract_embeddings(net, test_loader, ms = scales, msp = 1.0, print_freq = 1)
    #     print("Test descriptors finished...")
    # except:
    #     pass
    #     val_descr = extract_embeddings(net,val_loader, ms = scales, msp = 1.0, print_freq = 1)
    #     print("Val descriptors finished...")

    descriptors_dict = {}

    if not queries_only:
        descriptors_dict["train_descriptors"] = np.array(train_descr).astype("float32")

    # if not vis_only:
    # try:
    #     descriptors_dict["test_descriptors"] = np.array(test_descr).astype("float32")
    # except:
    #     pass
    #     descriptors_dict["val_descriptors"] = np.array(val_descr).astype("float32")

    # save descriptors
    print("Saving descriptors...")
    try:
        with h5py.File(exp_dir + "ViT_descriptors.hdf5", "w") as f:
            for key in descriptors_dict:
                dset = f.create_dataset(key, data = data_dict[key])
            print("descriptors hdf5 file complete: {}".format(exp_dir+"descriptors.hdf5"))

    except:
        with open(exp_dir+"ViT_descriptors.pkl", 'wb') as data:
            pickle.dump(descriptors_dict, data,protocol = pickle.HIGHEST_PROTOCOL)
            print("descriptors pickle file complete: {}".format(exp_dir+"descriptors.pkl"))


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument('directory', metavar='EXPORT_DIR',help='destination where descriptors will be saved')
    parser.add_argument('--mini', action='store_true') #use the mini database
    parser.add_argument('--info_dir',default=None, type=str, help = 'directory where ground truth is stored')
    parser.add_argument('--im_root',default=None, type=str, help = 'directory where images are stored')
    parser.add_argument('--augment_type',default='augment_train', type=str, help = 'preprocessing of images')


    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=500, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=1, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=50, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    parser.add_argument("--dataset", choices=["MET", "cifar100"], default="MET",
                        help="Which downstream task.")
    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=300, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=10, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--csv', type=str, default='https://media.githubusercontent.com/media/metmuseum/openaccess/master/MetObjects.csv',
                        help="MET csv")

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)
    print(model)
    # Training
    create_descriptors(args.directory, net = model,
                              mini = args.mini, info_dir = args.info_dir, im_root = args.im_root,
                              augment_type = args.augment_type)
    # train(args, model)


if __name__ == "__main__":
    """
    python3 train.py --info_dir ../../Documents/scriptie/ground_truth --im_root ../../Documents/scriptie/images --name met_first --model_type ViT-B_16 --pretrained_dir model/ViT-B_16.npz --num_steps 300 --warmup_steps 10
    """
    main()
