# A brief guide to Acuitylite

Acuitylite is an end-to-end neural-network deployment tool for embedded systems.<br/>
Acuitylite support converting caffe/tflite model to TIM-VX case.
In addition, Acuitylite support asymmetric uint8 qutization.

### System Requirement
- OS:  Ubuntu Linux 20.04 LTS 64-bit (recommend)
- Python Version:  python3.8 (needed)

### Install
    pip install acuitylite

### Document
    Reference: https://verisilicon.github.io/acuitylite/

### Framework Support
- Importer:
    [Caffe](https://github.com/BVLC/caffe),
    [TFLite](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite)
- Exporter:  [TIM-VX](https://github.com/VeriSilicon/TIM-VX)

### How to run TIM-VX case
The exported TIM-VX case supports both make and cmake.<br/>
Please set environment for build and run case:<br/>
- TIM_VX_DIR=/path/to/tim-vx/build
- VIVANTE_SDK_DIR=/path/to/tim-vx/prebuilt-sdk/x86_64_linux
- LD_LIBRARY_PATH=$TIM_VX_DIR/lib:$VIVANTE_SDK_DIR/lib

Attention: The TIM_VX_DIR path should include lib and header files of TIM-VX.
You can refer [TIM-VX](https://github.com/VeriSilicon/TIM-VX) to build TIM-VX.

### Support
Create issue on github or email to ML_Support@verisilicon.com
