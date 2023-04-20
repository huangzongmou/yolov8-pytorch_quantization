import argparse
import os
import sys
from pathlib import Path
import warnings
import yaml
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics import YOLO
from utils.dataloaders import create_dataloader
from utils.general import (check_img_size, check_yaml, file_size, colorstr, check_dataset)
from utils.torch_utils import select_device

import py_quant_utils as quant



def collect_stats(model, data_loader, num_batches, device):
    """Feed data to the network and collect statistics"""
    # Enable calibrators
    model.eval()
    for name, module in model.named_modules():
        if isinstance(module, quant.quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # Feed data to the network for collecting stats
    for i, (image, targets, paths, shapes) in tqdm(enumerate(data_loader), total=num_batches):
        image = image.to(device, non_blocking=True)
        image = image.float()  # uint8 to fp16/32
        image /= 255.0  # 0 - 255 to 0.0 - 1.0
        model(image)
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant.quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant.quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, quant.calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")


def calibrate_model(model, model_name, data_loader, num_calib_batch, calibrator, hist_percentile, out_dir, device):
    """
        Feed data to the network and calibrate.
        Arguments:
            model: detection model
            model_name: name to use when creating state files
            data_loader: calibration data set
            num_calib_batch: amount of calibration passes to perform
            calibrator: type of calibration to use (max/histogram)
            hist_percentile: percentiles to be used for historgram calibration
            out_dir: dir to save state files in
    """

    if num_calib_batch > 0:
        print("Calibrating model")
        with torch.no_grad():
            collect_stats(model, data_loader, num_calib_batch, device)

        if not calibrator == "histogram":
            compute_amax(model, method="max")
            calib_output = os.path.join(out_dir, F"{model_name}-max-{num_calib_batch * data_loader.batch_size}.pth")
            torch.save(model.state_dict(), calib_output)
        else:
            for percentile in hist_percentile:
                print(F"{percentile} percentile calibration")
                compute_amax(model, method="percentile")
                calib_output = os.path.join(out_dir, F"{model_name}-percentile-{percentile}-{num_calib_batch * data_loader.batch_size}.pth")
                torch.save(model.state_dict(), calib_output)

            for method in ["mse", "entropy"]:
                print(F"{method} calibration")
                compute_amax(model, method=method)
                calib_output = os.path.join(out_dir, F"{model_name}-{method}-{num_calib_batch * data_loader.batch_size}.pth")
                torch.save(model.state_dict(), calib_output)


def load_model(weight, device):
    yolo = YOLO(weight)
 
    yolo.model.float()
    yolo.model.eval()

    with torch.no_grad():
        yolo.model.fuse()
    return yolo


def prepare_model(calibrator, opt, device):
    """
    1、Load模型
    2、插入模型的Q和DQ节点
    3、设置量化方法：per_tensor/per_channels,
    4、Dataloader的制作
    """
    with open(opt.data, encoding='utf-8') as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  
        data_dict = check_dataset(data_dict)
    calib_path = data_dict['val']

    # 初始化量化方法
    quant.initialize_calib_method(per_channel_quantization=True, calib_method=calibrator)  
    # 加载FP32的Pytorch模型
    yolo = load_model(opt.weights, device)
    # 为FP32的Torch的Q和DQ的节点
    quant.replace_to_quantization_module(yolo.model, ignore_policy=opt.sensitive_layer)
    yolo.model.eval()
    yolo.model.cuda()


    gs = max(int(yolo.model.stride.max()), 32)  # grid size (max stride)
    imgsz, _ = [check_img_size(x, gs) for x in [opt.imgsz, opt.imgsz]]  # verify imgsz are gs-multiples

    # Calib dataloader
    calib_loader = create_dataloader(calib_path,
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

    return calib_loader,yolo

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / '../ultralytics/ultralytics/datasets/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default="yolov8n.pt", help='model.pt path(s)')
    parser.add_argument('--model-name', '-m', default='yolov8n', help='model name: default yolov5s')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')

    # setting for calibration
    parser.add_argument('--calib-batch-size', type=int, default=64, help='calib batch size: default 64')
  
    parser.add_argument('--sensitive-layer', default=[], help='skip sensitive layer: default detect head')
    parser.add_argument('--num-calib-batch', default=64, type=int,
                        help='Number of batches for calibration. 0 will disable calibration. (default: 4)')
    parser.add_argument('--calibrator', type=str, choices=["max", "histogram"], default="max")
    parser.add_argument('--percentile', nargs='+', type=float, default=[99.9, 99.99, 99.999, 99.9999])
    parser.add_argument('--dynamic', default=False, help='dynamic ONNX axes')
    parser.add_argument('--simplify', default=True, help='simplify ONNX file')
    parser.add_argument('--out-dir', '-o', default=ROOT / 'weights/', help='output folder: default ./runs/finetune')
    parser.add_argument('--batch-size-onnx', type=int, default=1, help='batch size for onnx: default 1')

    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    return opt


def evaluate_accuracy(yolo, opt):

    res = yolo.val(data = opt.data)
    map50 = res.results_dict['metrics/mAP50(B)']
    map = res.results_dict['metrics/mAP50-95(B)']
    return map50, map


def sensitive_analysis(yolo, opt, data_loader, summary_file='./summary_sensitive_analysis.json'):
    summary = quant.SummaryTool(summary_file)

    map50_calibrated, map_calibrated = evaluate_accuracy(yolo, opt)
    summary.append([map50_calibrated, map_calibrated, "PTQ"])

    print("Sensitive Analysis by each layer...")
    for i in range(0, len(yolo.model.model)):
        layer = yolo.model.model[i]
        if quant.have_quantizer(layer):
            print(f"Quantization disable model.{i}")
            quant.disable_quantization(layer).apply()
            map50_calibrated, map_calibrated = evaluate_accuracy(yolo, opt)
            summary.append([map50_calibrated, map_calibrated, f"model.{i}"])
            quant.enable_quantization(layer).apply()
        else:
            print(f"ignore model.{i} because it is {type(layer)}")
    
    summary = sorted(summary.data, key=lambda x:x[0], reverse=True)
    print("Sensitive summary:")
    for n, (map_calibrated, map50_calibrated, name) in enumerate(summary):
        print(f"Top{n}: Using fp16 {name}, map_calibrated = {map_calibrated:.5f}")


if __name__ == "__main__":

    opt = parse_opt()

    device = select_device(opt.device, batch_size=opt.batch_size)
    data_loader, yolo = prepare_model(calibrator=opt.calibrator, opt=opt, device=device)

    with torch.no_grad():
        calibrate_model(
            model=yolo.model,
            model_name=opt.model_name,
            data_loader=data_loader,
            num_calib_batch=opt.num_calib_batch,
            calibrator=opt.calibrator,
            hist_percentile=opt.percentile,
            out_dir=opt.out_dir,
            device=device)

    sensitive_analysis(yolo, opt, data_loader)
