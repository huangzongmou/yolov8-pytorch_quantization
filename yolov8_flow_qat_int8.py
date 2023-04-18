import argparse
from copy import deepcopy
import os
import sys
from pathlib import Path
from typing import List
import warnings
import torch
from tqdm import tqdm
import yaml



FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch.optim as optim
from torch.cuda import amp
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, check_img_size, check_yaml, file_size, colorstr, print_args, check_dataset, check_img_size, colorstr, init_seeds)
from utils.torch_utils import select_device
from ultralytics import YOLO
import py_quant_utils as quant
from yolov8_ptq_int8 import calibrate_model


def evaluate_accuracy(yolo, opt):

    res = yolo.val(data = opt.data)
    map50 = res.results_dict['metrics/mAP50(B)']
    map = res.results_dict['metrics/mAP50-95(B)']
    return map50, map


def load_model(weight, device):
    yolo = YOLO(weight)
 
    yolo.model.float()
    yolo.model.eval()

    with torch.no_grad():
        yolo.model.fuse()
    return yolo


def prepare_model(calibrator, opt, hyp, device, per_channel_quantization=True):
    # Hyperparameters
    # if isinstance(hyp, str):
    with open(hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict

    with open(opt.data, encoding='utf-8') as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
        data_dict = check_dataset(data_dict)  # check
    
    train_path = data_dict['train']
    test_path = data_dict['val']

    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    ## Initialize quantization, model and data loaders
    quant.initialize_calib_method(per_channel_quantization=per_channel_quantization, calib_method=calibrator)

    # Model
    yolo = load_model(opt.weights, device)
    quant.replace_to_quantization_module(yolo.model, ignore_policy=opt.sensitive_layer)
    yolo.model.eval()
    yolo.model.cuda()

    gs = max(int(yolo.model.stride.max()), 32)  # grid size (max stride)
    imgsz, _ = [check_img_size(x, gs) for x in [opt.imgsz, opt.imgsz]]  # verify imgsz are gs-multiples

    # Train dataloader
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              opt.batch_size,
                                              gs,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=False,
                                              rank=-1,
                                              workers=opt.workers,
                                              image_weights=False,
                                              quad=False,
                                              prefix=colorstr('train: '),
                                              shuffle=True)

    # Test dataloader
    val_loader = create_dataloader(test_path,
                                   imgsz,
                                   opt.batch_size,
                                   gs,
                                   hyp=hyp,
                                   cache=opt.cache,
                                   rect=True,
                                   rank=-1,
                                   workers=opt.workers * 2,
                                   pad=0.5,
                                   prefix=colorstr('val: '))[0]

    # Calib dataloader
    calib_loader = create_dataloader(test_path,
                                     imgsz,
                                     opt.batch_size,
                                     gs,
                                     hyp=None,
                                     cache=opt.cache,
                                     rect=True,
                                     rank=-1,
                                     workers=opt.workers * 2,
                                     pad=0.5,
                                     prefix=colorstr('calib: '))[0]

    return yolo, train_loader, val_loader, calib_loader, dataset


def export_onnx(model, onnx_filename, batch_onnx, dynamic_shape, simplify, imgsz=672, prefix=colorstr('calib: ')):

    # We have to shift to pytorch's fake quant ops before exporting the model to ONNX
    quant.quant_nn.TensorQuantizer.use_fb_fake_quant = True

    # Export ONNX for multiple batch sizes
    print("Creating ONNX file: " + onnx_filename)
    dummy_input = torch.randn(batch_onnx, 3, imgsz, imgsz)  

    try:
        import onnx
        with torch.no_grad():
            torch.onnx.export(model.cpu(), 
                            dummy_input.cpu(), 
                            onnx_filename, 
                            verbose=False, 
                            opset_version=13, 
                            input_names=['images'],
                            output_names=['output'],
                            dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'}} if dynamic_shape else None,
                            enable_onnx_checker=False, 
                            do_constant_folding=True)

        print('ONNX export success, saved as %s' % onnx_filename)

    except ValueError:
        warnings.warn(UserWarning("Per-channel quantization is not yet supported in Pytorch/ONNX RT (requires ONNX opset 13)"))
        print("Failed to export to ONNX")
        return False

    except Exception as e:
            print(f'{prefix} export failure: {e}')
    
    # Checks
    model_onnx = onnx.load(onnx_filename)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    
    # Simplify
    if simplify:
        try:
            import onnxsim

            print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(
                    model_onnx,
                    dynamic_input_shape=dynamic_shape,
                    input_shapes={'images': list(dummy_input.shape)} if dynamic_shape else None)

            assert check, 'assert check failed'
            onnx.save(model_onnx, onnx_filename)
        except Exception as e:
            print(f'{prefix} simplifier failure: {e}')

        print(f'{prefix} export success, saved as {onnx_filename} ({file_size(onnx_filename):.1f} MB)')
        print(f"{prefix} Run ONNX model inference with: 'python detect.py --weights {onnx_filename}'")
        
    # Restore the PSX/TensorRT's fake quant mechanism
    quant.quant_nn.TensorQuantizer.use_fb_fake_quant = False
    # Restore the model to train/test mode, use Detect() layer grid
    model.export = False

    return True


def train(model, epochs, train_loader, fp16=True, lrschedule=None, train_layers=None):
    torch_model = deepcopy(model).eval()
    quant.disable_quantization(torch_model).apply()

    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers

    scaler       = amp.GradScaler(enabled=fp16)
    optimizer    = optim.Adam(model.parameters(), 1e-5)
    quant_loss_fn   = torch.nn.MSELoss()
    device       = next(model.parameters()).device
    # print(lrschedule)
    # import pdb
    # pdb.set_trace()

    if lrschedule is None:
        lrschedule = {
            0: 1e-4,
            5: 1e-5,
            20: 1e-6
        }
    # 定义钩子函数，结合 `register_forward_hook` 获取每层的特征并append到l list之中
    def make_layer_forward_hook(l):
        def forward_hook(m, input, output):
            l.append(output)
        return forward_hook
    
    supervision_module_pairs = []
    for ((mname, ml), (oriname, ori)) in zip(model.named_modules(), torch_model.named_modules()):
        if isinstance(ml, quant.quant_nn.TensorQuantizer): 
            # print(ml)
            continue
        if mname not in train_layers:
            continue
        # print(mname)
        supervision_module_pairs.append([ml, ori])

    for epoch in range(epochs):  
        if epoch in lrschedule:
            learningrate = lrschedule[epoch]
            for g in optimizer.param_groups:
                g["lr"] = learningrate
        
        model_outputs  = []
        origin_outputs = []
        remove_handle  = []
        for ml, ori in supervision_module_pairs:
            # 为每层加入钩子，在进行Forward的时候会自动将每层的特征传送给model_outputs和origin_outputs
            remove_handle.append(ml.register_forward_hook(make_layer_forward_hook(model_outputs))) 
            remove_handle.append(ori.register_forward_hook(make_layer_forward_hook(origin_outputs)))

        model.train()
        pbar = tqdm(train_loader, desc="QAT")
        for batch_i, (imgs, targets, paths, shapes) in enumerate(pbar):  
            imgs = imgs.to(device, non_blocking=True).float() / 255  
            with amp.autocast(enabled=fp16):
                # QAT model forward
                model(imgs)
                # Torch model forward
                with torch.no_grad():
                    torch_model(imgs)

                quant_loss = 0
                for index, (mo, fo) in enumerate(zip(model_outputs, origin_outputs)):

                    quant_loss += quant_loss_fn(mo, fo)

                model_outputs.clear()
                origin_outputs.clear()

                # print(model_outputs)
                # import pdb
                # pdb.set_trace()

            if fp16:
                scaler.scale(quant_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                quant_loss.backward()
                optimizer.step()
            optimizer.zero_grad()

            pbar.set_description(f"QAT Finetuning {epoch + 1} / {epochs}, Loss: {quant_loss.detach().item():.5f}, LR: {learningrate:g}")

        # You must remove hooks during onnx export or torch.save
        for rm in remove_handle:
            rm.remove()
    return model


def select_layers_for_finetuning(model, stride=1):
    supervision_list = []
    train_layers = []
    for item in model.model:
        supervision_list.append(id(item))
    keep_idx = list(range(0, len(model.model) - 1, stride))
    keep_idx.append(len(model.model) - 2)
    for name, module in model.named_modules():
        if id(module) in supervision_list:
            idx = supervision_list.index(id(module))
            if idx in keep_idx:
                train_layers.append(name)
                print(f"{name} will compute loss with torch model during QAT training")
            else:
                print(f"{name} no compute loss during QAT training, that is unsupervised only and doesn't mean don't learn")
    return train_layers


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / './ultralytics/datasets/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--save-dir', type=str, default=ROOT / 'weights', help='save-dir path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / "yolov8n.pt", help='model.pt path(s)')
    parser.add_argument('--hyp', type=str, default=ROOT / './ultralytics/yolo/cfg/default.yaml', help='hyperparameters path')
    parser.add_argument('--model-name', '-m', default='yolov8n', help='model name: default yolov8s')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')

    # setting for calibration
    parser.add_argument('--calibrator', type=str, choices=["max", "histogram"], default="max")
    parser.add_argument('--calib-batch-size', type=int, default=64, help='calib batch size: default 64')
    parser.add_argument('--num-calib-batch', default=64, type=int, help='Number of batches for calibration. 0 will disable calibration. (default: 4)')
    parser.add_argument('--percentile', nargs='+', type=float, default=[99.9, 99.99, 99.999, 99.9999])
    parser.add_argument('--batch-size-onnx', type=int, default=1, help='batch size for onnx: default 1')
    parser.add_argument('--out-calib-dir', '-o', default=ROOT / 'weights/', help='output folder: default ./runs/finetune')
    parser.add_argument('--sensitive-layer', default=[], help='skip sensitive layer: default detect head')
    parser.add_argument('--dynamic', default=False, help='dynamic ONNX axes')
    parser.add_argument('--simplify', default=True, help='simplify ONNX file')
    parser.add_argument('--epochs', type=int, default=60)

    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    print_args(vars(opt))
    return opt


if __name__ == "__main__":
    init_seeds(888)
    opt = parse_opt()
    device = select_device(opt.device, batch_size=opt.batch_size)
    yolo, train_loader, val_loader, calib_loader, dataset = prepare_model(calibrator=opt.calibrator, opt=opt,hyp=opt.hyp, device=device)
    # 校准模型
    with torch.no_grad():
        calibrate_model(
            model=yolo.model,
            model_name=opt.model_name,
            data_loader=calib_loader,
            num_calib_batch=opt.num_calib_batch,
            calibrator=opt.calibrator,
            hist_percentile=opt.percentile,
            out_dir=opt.out_calib_dir,
            device=device)
    

    with torch.no_grad():
        map50_calibrated_ptq, map_calibrated_ptq = evaluate_accuracy(yolo, opt)
        print('PTQ evaluation:', "mAP@IoU=0.50:{:.5f}, mAP@IoU=0.50:0.95:{:.5f}".format(map50_calibrated_ptq, map_calibrated_ptq))

    train_layers = select_layers_for_finetuning(yolo.model)
    model = train(yolo.model, opt.epochs, train_loader, train_layers=train_layers)

    with torch.no_grad():
        map50_calibrated_qat, map_calibrated_qat = evaluate_accuracy(yolo, opt)
        print('QAT evaluation:', "mAP@IoU=0.50:{:.5f}, mAP@IoU=0.50:0.95:{:.5f}".format(map50_calibrated_qat, map_calibrated_qat))

    with torch.no_grad():
        with quant.disable_quantization(model):
            map50_Orgin, map_Orgin = evaluate_accuracy(yolo, opt)
            print('Torch evaluation:', "mAP@IoU=0.50:{:.5f}, mAP@IoU=0.50:0.95:{:.5f}".format(map50_Orgin, map_Orgin))


    onnx_filename = './weights/yolov8n_qat_detect_3.onnx'
    export_onnx(model, onnx_filename, opt.batch_size_onnx, opt.dynamic, opt.simplify)
