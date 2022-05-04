import os
import os.path
import json
import numpy as np
import pandas as pd
import sys
import pickle
import argparse
from collections import OrderedDict
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader

import torch
from torchvision import transforms
from torch.utils.model_zoo import load_url

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.models as models

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
            if i == 18800: # personal number: larger will cause it to crash
                if vis_only:
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
            im_root = None
    ) -> None:
        super().__init__(root, transform=transform,
                            target_transform=target_transform)

        if test:
            fn = "testset.json"
        else:
            fn = "valset.json"

        with open(os.path.join(self.root, fn)) as f:
            data = json.load(f)

        samples = []
        targets = []

        for e in data:
            samples.append(e['path'])
            if "MET_id" in e:
                targets.append(int(e['MET_id']))
            else:
                # not a MET image
                targets.append(-1)

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


class siamese_network(nn.Module):
    """Network architecture for contrastive learning.
    """

    def __init__(self, backbone, pooling = "gem", pretrained = True,
                    emb_proj = False, init_emb_projector = None):
        super(siamese_network, self).__init__()

        net = Embedder(backbone, gem_p=3.0,
                       pretrained_flag=pretrained,
                       projector=emb_proj,
                       init_projector=init_emb_projector)

        self.backbone = net  # the backbone produces l2 normalized descriptors

    def forward(self, augs1, augs2):

        descriptors_left = self.backbone(augs1)
        descriptors_right = self.backbone(augs2)

        return descriptors_left, descriptors_right


class GeM(nn.Module):
    """Credits to Filip Radenovic (https://github.com/filipradenovic/cnnimageretrieval-pytorch)
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + "(" + "p=" + "{:.4f}".format(self.p.data.tolist()[0]) + ", " + "eps=" + str(self.eps) + ")"


def gem(x, p=3, eps=1e-6):
    """Credits to Filip Radenovic (https://github.com/filipradenovic/cnnimageretrieval-pytorch)
    """

    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class Embedder(nn.Module):
    """Class that implements a descriptor extractor as a (fully convolutional backbone -> pooling -> l2 normalization).
    Optionally followed by a FC layer (fully convolutional backbone -> pooling -> l2 normalization -> FC -> l2 normalization)
    that can be initialized with the result of PCAw.
    """

    def __init__(self, architecture, gem_p=3, pretrained_flag=True, projector=False, init_projector=None):
        """The FC layer is called projector.
        """

        super(Embedder, self).__init__()

        if architecture == "r18_sw-sup":
            # r18 facebook pretrained model (https://github.com/facebookresearch/semi-supervised-ImageNet1K-models)

            network = torch.hub.load(
                "facebookresearch/semi-supervised-ImageNet1K-models", "resnet18_swsl")
            self.backbone = nn.Sequential(*list(network.children())[:-2])

        else:

            # load the base model from PyTorch's pretrained models (imagenet pretrained)
            network = getattr(models, architecture)(pretrained=pretrained_flag)

            # keep only the convolutional layers, ends with relu to get non-negative descriptors
            if architecture.startswith("resnet"):
                self.backbone = nn.Sequential(*list(network.children())[:-2])

            elif architecture.startswith("alexnet"):
                self.backbone = nn.Sequential(*list(network.features.children())[:-1])

        # spatial pooling layer
        self.pool = GeM(p=gem_p)

        # normalize on the unit-hypershpere
        # self.norm = L2N()
        self.norm = F.normalize

        # information about the network
        self.meta = {
            "architecture": architecture,
            "pooling": "gem",
            # imagenet statistics for imagenet pretrained models
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "outputdim": OUTPUT_DIM[architecture],
        }

        if projector:
            print("using FC layer in the backbone")
            self.projector = nn.Linear(
                self.meta["outputdim"], self.meta["outputdim"], bias=True)

            if init_projector is not None:

                print("initialising the backbone's project layer")

                self.projector.weight.data = torch.transpose(
                    torch.Tensor(init_projector[1]), 0, 1)
                self.projector.bias.data = - \
                    torch.matmul(torch.Tensor(
                        init_projector[0]), torch.Tensor(init_projector[1]))

        else:
            self.projector = None

    def forward(self, img):
        """Output has shape: batch size x descriptor dimension
        """

        x = self.norm(self.pool(self.backbone(img))).squeeze(-1).squeeze(-1)

        if self.projector is None:
            return x

        else:
            return self.norm(self.projector(x))


def extract_ss(net, input):
    """Credits to Filip Radenovic (https://github.com/filipradenovic/cnnimageretrieval-pytorch)
    """

    return net(input).cpu().data.squeeze()


def extract_ms(net, input, ms, msp):
    """Credits to Filip Radenovic (https://github.com/filipradenovic/cnnimageretrieval-pytorch)
    """

    v = torch.zeros(net.meta["outputdim"])

    for s in ms:

        if s == 1:
            input_t = input.clone()

        else:
            input_t = nn.functional.interpolate(
                input, scale_factor=s, mode="bilinear", align_corners=False)

        v += net(input_t).pow(msp).cpu().data.squeeze()

    v /= len(ms)
    v = v.pow(1. / msp)
    v /= v.norm()

    return v


def extract_embeddings(net, dataloader, ms=[1], msp=1, print_freq=None, verbose=False):
    """Credits to Filip Radenovic (https://github.com/filipradenovic/cnnimageretrieval-pytorch)
    """

    if verbose:

        if len(ms) == 1:
            print("Singlescale extraction")
        else:
            print("Multiscale extraction at scales: " + str(ms))

    net.eval()

    with torch.no_grad():

        vecs = np.zeros((net.meta["outputdim"], len(dataloader.dataset)))

        for i, input in enumerate(dataloader):

            if len(ms) == 1 and ms[0] == 1:
                vecs[:, i] = extract_ss(net, input[0])

            else:
                vecs[:, i] = extract_ms(net, input[0], ms, msp)

            if print_freq is not None:
                if i % print_freq == 0:
                    print("image: " + str(i))

    return vecs.T


def create_descriptors(export_dir, net = "r18INgem", net_path = None,
                          mini = True, info_dir = None, im_root = None,
                          ms = False, queries_only = False,
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

    network_variant = net
    exp_name = export_dir + "/" + network_variant
    if ms:
        exp_name += ("_ms") # multiscale descriptors
    else:
        exp_name += ("_ss") # singlescale descriptors
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

    if not vis_only:
        test_dataset = MET_queries(root = info_dir, test = True,
                                        transform = extraction_transform, im_root = im_root)
        val_dataset = MET_queries(root = info_dir, transform = extraction_transform,
                                        im_root = im_root)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        print("Number of test images: {}".format(len(test_dataset)))
        print("Number of val images: {}".format(len(val_dataset)))

    if not queries_only:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        print("Number of train images: {}".format(len(train_dataset)))


    # initialization of the global descriptor extractor model
    # if there is a pretrained model
    if net_path is not None:

        if network_variant == "r18_contr_loss_gem":
            model = siamese_network("resnet18", pooling = "gem", pretrained = False)
            print("loading weights from checkpoint")
            model.load_state_dict(torch.load(net_path)["state_dict"])
            net = model.backbone

        elif network_variant == "r18_contr_loss_gem_fc":
            model = siamese_network("resnet18", pooling = "gem", pretrained = False,
                emb_proj = True)
            model.backbone.projector.bias.data = model.backbone.projector.bias.data.unsqueeze(0)
            print("loading weights from checkpoint")
            model.load_state_dict(torch.load(net_path)["state_dict"])
            net = model.backbone

        elif network_variant == "r18_contr_loss_gem_fc_swsl":
            model = siamese_network("r18_sw-sup",pooling = "gem", pretrained = False,
                emb_proj = True)
            model.backbone.projector.bias.data = model.backbone.projector.bias.data.unsqueeze(0)
            print("loading weights from checkpoint")
            model.load_state_dict(torch.load(net_path)["state_dict"])
            net = model.backbone
        else:
            raise ValueError(f"Unsupported  architecture: {net}!")
    # no pretrained net
    else:

        if network_variant == "r18INgem":
            net = Embedder("resnet18",gem_p = 3.0, pretrained_flag = True)

        elif network_variant == "r50INgem_caffe": #we use this version because it has the weights from caffe library which perform better
            net_params = {"architecture" : "resnet50", "pooling" : "gem", "pretrained" : True, "whitening" : False}
            net = init_network(net_params)

        elif network_variant == "r50INgem": #pytorch weights
            net = Embedder("resnet50", gem_p = 3.0,pretrained_flag = True, projector = False)

        elif network_variant == "r50_swav_gem":
            model = torch.hub.load("facebookresearch/swav", "resnet50")
            net = torch.nn.Sequential(*list(model.children())[:-2],GeM(),L2N())
            net.meta = {
                "architecture" : "resnet50",
                "pooling" : "gem",
                "mean" : [0.485, 0.456, 0.406], #imagenet statistics
                "std" : [0.229, 0.224, 0.225],
                "outputdim" : 2048,
            }

        elif network_variant == "r50_SIN_gem":

            model = torchvision.models.resnet50(pretrained=False)
            checkpoint = model_zoo.load_url("https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar")
            model = torch.nn.DataParallel(model)
            model.load_state_dict(checkpoint["state_dict"])
            model = model.module
            net = torch.nn.Sequential(*list(model.children())[:-2],GeM(),L2N())
            net.meta = {
                "architecture" : "resnet50",
                "pooling" : "gem",
                "mean" : [0.485, 0.456, 0.406], #imagenet statistics
                "std" : [0.229, 0.224, 0.225],
                "outputdim" : 2048,
            }

        elif network_variant == "r18_sw-sup_gem":
            model = torch.hub.load("facebookresearch/semi-supervised-ImageNet1K-models", "resnet18_swsl")
            net = torch.nn.Sequential(*list(model.children())[:-2],GeM(),L2N())
            net.meta = {
                "architecture" : "resnet18",
                "pooling" : "gem",
                "mean" : [0.485, 0.456, 0.406], #imagenet statistics
                "std" : [0.229, 0.224, 0.225],
                "outputdim" : 512,
            }

        elif network_variant == "r50_sw-sup_gem":
            model = torch.hub.load("facebookresearch/semi-supervised-ImageNet1K-models", "resnet50_swsl")
            net = torch.nn.Sequential(*list(model.children())[:-2],GeM(),L2N())
            net.meta = {
                "architecture" : "resnet50",
                "pooling" : "gem",
                "mean" : [0.485, 0.456, 0.406], #imagenet statistics
                "std" : [0.229, 0.224, 0.225],
                "outputdim" : 2048,
            }

        else:
            raise ValueError(f"Unsupported  architecture: {net}!")

    if gpu:
        net.cuda()

    if ms:
        # multi-scale case
        scales = [1, 1/np.sqrt(2), 1/2]

    else:
        # single-scale case
        scales = [1]

    print("Starting the extraction of the descriptors")

    if not queries_only:
        train_descr = extract_embeddings(net, train_loader, ms = scales, msp = 1.0, print_freq = 100)
        print("Train descriptors finished...")

    if not vis_only:
        test_descr = extract_embeddings(net, test_loader, ms = scales, msp = 1.0, print_freq = 100)
        print("Test descriptors finished...")
        val_descr = extract_embeddings(net,val_loader, ms = scales, msp = 1.0, print_freq = 100)
        print("Val descriptors finished...")

    descriptors_dict = {}

    if not queries_only:
        descriptors_dict["train_descriptors"] = np.array(train_descr).astype("float32")

    if not vis_only:
        descriptors_dict["test_descriptors"] = np.array(test_descr).astype("float32")
        descriptors_dict["val_descriptors"] = np.array(val_descr).astype("float32")

    # save descriptors
    with open(exp_dir+"descriptors.pkl", 'wb') as data:
        pickle.dump(descriptors_dict, data,protocol = pickle.HIGHEST_PROTOCOL)
        print("descriptors pickle file complete: {}".format(exp_dir+"descriptors.pkl"))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('directory', metavar='EXPORT_DIR',help='destination where descriptors will be saved')
    parser.add_argument('--gpu', action='store_true') #id of the gpu in your machine
    parser.add_argument('--net', default='r18INgem')
    parser.add_argument('--netpath', default=None) #optional
    parser.add_argument('--ms', action='store_true') #multiscale descriptors
    parser.add_argument('--mini', action='store_true') #use the mini database
    parser.add_argument('--queries_only', action='store_true')
    parser.add_argument('--trained_on_mini', action='store_true') #if your model has a classification head for the mini dataset
    parser.add_argument('--info_dir',default=None, type=str, help = 'directory where ground truth is stored')
    parser.add_argument('--im_root',default=None, type=str, help = 'directory where images are stored')
    parser.add_argument('--augment_type',default='augment_inference', type=str, help = 'preprocessing of images')

    args = parser.parse_args()

    export_dir = args.directory
    network_variant = args.net
    netpath = args.netpath
    mini = args.mini
    info_dir = args.info_dir
    im_root = args.im_root
    ms = args.ms
    queries_only = args.queries_only
    augment_type = args.augment_type
    gpu = args.gpu

    create_descriptors(export_dir, net = network_variant, net_path = netpath,
                              mini = mini, info_dir = info_dir, im_root = im_root,
                              ms = ms, queries_only = queries_only,
                              augment_type = augment_type, gpu = gpu,
                              vis_only = False)


if __name__ == '__main__':
    """
    python3 descriptor.py ../descriptors --net "r18INgem" --mini --info_dir ../ground_truth --im_root ../images
    python3 descriptor.py ../descriptors --net "r50INgem" --mini --info_dir ../ground_truth --im_root ../images
    """
    main()