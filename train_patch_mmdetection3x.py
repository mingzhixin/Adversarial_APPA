"""
Adversarial patch training
"""

import os
import time
import warnings

import pandas as pd
import torch
import wandb
from tensorboardX import SummaryWriter
from torch import autograd
from torchvision import transforms
from tqdm import tqdm

import patch_config
from load_data import *

warnings.filterwarnings("ignore")

# os.environ["CUDA_VISIBLE_DEVICES"] = '7'

# Expand to show
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)


class PatchTrainer(object):
    def __init__(self, mode):
        self.mode = mode
        self.epoch_length = 0
        self.config = patch_config.patch_configs[mode]()

        self.model = self.config.model
        self.prob_extractor = self.config.prob_extractor.cuda()
        # self.prob_extractor = MaxProbExtractor(0, 80, self.config).cuda()

        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer().cuda()

        self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """

        # img_size = self.darknet_model.height
        img_size = 800
        batch_size = self.config.batch_size
        n_epochs = 100

        # max number of bbox in txt file(namely max number of lines in label file)
        # max_lab = 13 + 1  # person
        # max_lab = 34 + 1  # yolov2
        # max_lab = 50
        max_lab = 59 + 1  # yolov5l

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate stating point
        adv_patch_cpu = self.generate_patch("gray")  # Training from gray patch
        # adv_patch_cpu = self.read_image("patches/swin2.png")  # Training from existing patch
        # adv_patch_cpu = self.read_image("patches/faster_rcnn.png")  # Training from existing patch

        adv_patch_cpu.requires_grad_(True)

        train_loader = torch.utils.data.DataLoader(
            InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size, shuffle=False),
            batch_size=batch_size,
            shuffle=True,
            num_workers=1)
        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')

        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)

        wandb.init(project="Adversarial-attack", mode="offline")
        wandb.watch_called = False  # Re-run the model without restarting the runtime, unnecessary after our next release
        wandb.watch(self.model.model, log="all")

        et0 = time.time()
        for epoch in range(n_epochs):
            ep_det_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            bt0 = time.time()
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}', total=self.epoch_length):
                with autograd.detect_anomaly():
                    img_batch = img_batch.cuda()
                    lab_batch = lab_batch.cuda()
                    # print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
                    adv_patch = adv_patch_cpu.cuda()
                    adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                    p_img_batch = F.interpolate(p_img_batch, (img_size, img_size))
                    # print(p_img_batch[0], p_img_batch[0].size())  # Tensor: torch.Size([1, 3, 800, 800])

                    img = p_img_batch[0, :, :,]
                    img = transforms.ToPILImage()(img.detach().cpu())
                    img.save(f"patch_applier/patched_image{epoch}.png")
                    patch_pil = transforms.ToPILImage()(adv_patch_cpu.detach().cpu())
                    patch_pil.save(f"patch_applier/patch{epoch}.png")
                    # adv_batch_t: [B, N, 3, H, W]
                    patch_t = adv_batch_t[0, 0].detach().cpu()  # 取第1张图第1个目标的patch
                    transforms.ToPILImage()(patch_t).save("patch_applier/patch_transformed.png")

                    output = self.model.get_predictions(p_img_batch, [img_size, img_size], self.config.conf_thres)[0]
                    extracted_prob = self.prob_extractor(output)

                    nps = self.nps_calculator(adv_patch)
                    tv = self.total_variation(adv_patch)

                    nps_loss = nps * 0.01
                    tv_loss = tv * 2.5
                    det_loss = torch.mean(extracted_prob)

                    loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())

                    ep_det_loss += det_loss.detach().cpu().numpy()
                    ep_nps_loss += nps_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    ep_loss += loss

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0, 1)  # keep patch in image range

                    bt1 = time.time()
                    if i_batch % 100 == 0:
                        iteration = self.epoch_length * epoch + i_batch

                        wandb.log({
                            "Patches": wandb.Image(adv_patch_cpu, caption="patch{}".format(iteration)),
                            "tv_loss": tv_loss,
                            "nps_loss": nps_loss,
                            "det_loss": det_loss,
                            "total_loss": loss,
                        })
                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    # if i_batch % 2 == 0:
                    ######################################################################################################################################################
                    # del img_batch, lab_batch, adv_patch, adv_batch_t, p_img_batch_cpu, data, output, max_prob, nps, tv, det_loss, p_img_batch, nps_loss, tv_loss, loss
                    del img_batch, lab_batch, adv_patch, adv_batch_t, p_img_batch_cpu, output, nps, tv, det_loss, p_img_batch, nps_loss, tv_loss, loss
                    ######################################################################################################################################################
                    torch.cuda.empty_cache()
                    bt0 = time.time()
            et1 = time.time()
            ep_det_loss = ep_det_loss / len(train_loader)
            ep_nps_loss = ep_nps_loss / len(train_loader)
            ep_tv_loss = ep_tv_loss / len(train_loader)
            ep_loss = ep_loss / len(train_loader)

            scheduler.step(ep_loss)
            if True:
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_det_loss)
                print('  NPS LOSS: ', ep_nps_loss)
                print('   TV LOSS: ', ep_tv_loss)
                print('EPOCH TIME: ', et1 - et0)
                # del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
                # del adv_patch, adv_batch_t, p_img_batch_cpu, data, output, max_prob, nps, tv, det_loss, p_img_batch, nps_loss, tv_loss, loss
                # torch.cuda.empty_cache()
            et0 = time.time()

    def generate_patch(self, type):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        if type == 'gray':
            adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_size))

        return adv_patch_cpu

    def read_image(self, path):
        """
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()

        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu


def main():
    trainer = PatchTrainer('faster-rcnnmmdet3x')
    trainer.train()


if __name__ == '__main__':
    main()

# CUDA_VISIBLE_DEVICES=0,1 python train_patch_mmdetection.py
# pip install wandb --upgrade
