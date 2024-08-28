# — coding: utf-8 –
import numpy as np
from thop import profile

import torch
import torch.nn as nn

import argparse

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            print(self.postprocessor.deploy_mode)

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    print('==> Building model..')
    model = Model()
    device = torch.device('cuda')
    model.eval().to(device)
    optimal_batch_size = 1
    dummy_input = torch.randn(optimal_batch_size, 3,  512, 512, dtype=torch.float).to(device)
    size = torch.tensor([[512, 512]]).to(device)

    # flops与参数量
    flops, params = profile(model, (dummy_input, size))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

    # 模型推理速度计算
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input, size)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input, size)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn,
                                                                                         std_syn=std_syn,
                                                                                         mean_fps=mean_fps))
    print(mean_syn)

    # 吞吐量
    repetitions=100
    total_time = 0
    with torch.no_grad():
      for rep in range(repetitions):
         starter, ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
         starter.record()
         _ = model(dummy_input, size)
         ender.record()
         torch.cuda.synchronize()
         curr_time = starter.elapsed_time(ender)/1000
         total_time += curr_time
    Throughput = (repetitions*optimal_batch_size)/total_time
    print('Final Throughput:', Throughput)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str,
                        default=r'/home/kemove/PycharmProjects/RT-DETR-Mamba/rtdetr_pytorch/configs/rtdetr/rtdetr_r101vd_6x_coco.yml', )
    parser.add_argument('--resume', '-r', type=str,
                        default=r'/home/kemove/PycharmProjects/RT-DETR-Mamba/rtdetr_pytorch/tools/logs_freeze123/checkpoint0270.pth')
    parser.add_argument('--check', action='store_true', default=False, )
    parser.add_argument('--simplify', action='store_true', default=False, )

    args = parser.parse_args()

    main(args)
