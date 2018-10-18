import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import torchvision
from torchvision import models
import torch
import time
from torch import nn


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


    vgg16 = models.vgg16(pretrained=True)

    vgg16pool5 = vgg16.features
    features = list(vgg16.features.children())[:-21]  # Remove last layer
    vgg16pool2 = nn.Sequential(*features)  # Replace the model classifier

    for param in vgg16pool5.parameters():
        param.require_grad = False
    for param in vgg16pool2.parameters():
        param.require_grad = False


    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # pix2pix: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # CycleGAN: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break

        start_time = time.time()
        pool2 = vgg16pool2(data['A'])
        print(time.time() - start_time)
        start_time = time.time()
        pool5 = vgg16pool5(data['A'])
        print(time.time() - start_time)
        print("pool2 ", pool2.size())
        print("pool5 ", pool5.size())
        model.set_input(data)
        print(i, data['A_paths'])
        start_time = time.time()
        model.test()
        print(time.time() - start_time)
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    # save the website
    webpage.save()
