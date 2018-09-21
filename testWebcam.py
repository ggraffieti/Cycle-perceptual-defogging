import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import argparse
import imutils
import time
import cv2
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torch
import numpy as np
from util import util


if __name__ == '__main__':
    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # pix2pix: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # CycleGAN: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        image = Image.open("./datasets/mine/jurassic_park.jpg").convert('RGB')
        transform_list = []
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
        transform_list += [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        totTrans = transforms.Compose(transform_list)

        tens = totTrans(image)
        if opt.direction == 'BtoA':
            input_nc = opt.output_nc
        else:
            input_nc = opt.input_nc

        if input_nc == 1:  # RGB to gray
            tmp = tens[0, ...] * 0.299 + tens[1, ...] * 0.587 + tens[2, ...] * 0.114
            tens = tmp.unsqueeze(0)
        tens = tens.unsqueeze(0)
        model.set_input(tens)
        model.test()
        outputM = model.get_current_visuals()

        output = util.tensor2im(outputM['fake_B'])
        original = util.tensor2im(outputM['real_A'])
        output = output[..., [2, 0, 1]]

        cv2.imshow("Input", original)
        cv2.imshow("Output", output)
        cv2.waitKey(0)
