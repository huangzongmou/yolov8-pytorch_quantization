# yolov8-pytorch_quantization

    1、使用pytorch_quantization对yolov8进行量化,ptq、敏感层分析、qat。主要参考里 
    《集智书童》的yolov5量化。

# 安装yolov8

    pip install ultralytics
    注释ultralytics源码ops.py:262一下内容,避免推理超时break，导致验证值存在波动、不准确问题。
```
    if (time.time() - t) > time_limit:
        LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
        break  # time limit exceeded
```

# 运行

    自行修改ptq、qat、敏感层分析配置参数后直接运行。
    python yolov8_ptq_int8.py

# 运行结果

                Class     Images  Instances    Box(P          R      mAP50    mAP50-95
    未量化      all        128        929       0.64      0.537      0.605      0.446
    ptq         all        128        929      0.721      0.487      0.596      0.432
    跳过铭感层   all        128        929      0.676       0.51      0.606      0.435

# 后续完善内容

    1、增加每一层输出的相似度。
    2、铭感层分析更加细化，当前仅对 yolo.model.model[i]进行分析。
    
