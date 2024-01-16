# A brief guide to Acuitylite

Acuitylite is an end-to-end neural-network deployment tool for embedded systems.<br/>
Acuitylite support converting caffe/darknet/onnx/tensorflow/tflite models to TIM-VX/TFLite cases.
In addition, Acuitylite support asymmetric uint8 and symmetric int8 quantization.<br/>

Attention: We have introduced some important changes and updated the APIs that are not compatible with the version before Acuitylite6.21.0(include).
Please read the document and demos carefully.

### System Requirement
- OS:<br/>
    Ubuntu Linux 20.04 LTS 64-bit(python3.8)<br/>
    Ubuntu Linux 22.04 LTS 64-bit(python3.10)

### Install
    1. build the recommended docker image and run a container
    2. pip install acuitylite --no-deps

### Document
    Reference: https://verisilicon.github.io/acuitylite

### Framework Support
- Importer:
    [Caffe](https://github.com/BVLC/caffe),
    [Darknet](https://github.com/pjreddie/darknet),
    [Onnx](https://github.com/onnx/onnx),
    [Tensorflow](https://github.com/tensorflow/tensorflow),
    [TFLite](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite)
- Exporter:
    [TFLite](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite),
    [TIM-VX](https://github.com/VeriSilicon/TIM-VX)

Tips: You can export a TFLite app and using [tflite-vx-delegate](https://github.com/VeriSilicon/tflite-vx-delegate)
to run on TIM-VX if the exported TIM-VX app does not meet your requirements.

### How to generate nbg and TIM-VX case
When you need generate TIM-VX case and nbg, please set the export() function's param pack_nbg_unify=True.
Such as: TimVxExporter(model).export(pack_nbg_unify=True), it will use our default SDK.
If you want to use your own SDK and licence, please set the param of export() viv_sdk, licence.
Such as: TimVxExporter(model).export(pack_nbg_unify=True, viv_sdk=your_sdk_path, licence=path_of_licence_txt)

Attention: your sdk directory structure must strictly follow the directory structure of acuitylib/vsi_sdk!!!
your sdk need satisfy the structure of "your_sdk_path/build/install", "your_sdk_path/prebuilt-sdk/x86_64_linux",
otherwise the path may have problems.
And the licence content is the device target which you want to use.

### How to run TIM-VX case
The exported TIM-VX case supports both make and cmake.<br/>
Please set environment for build and run case:<br/>
- TIM_VX_DIR=/path/to/tim-vx/build/install
- VIVANTE_SDK_DIR=/path/to/tim-vx/prebuilt-sdk/x86_64_linux
- LD_LIBRARY_PATH=$TIM_VX_DIR/lib:$VIVANTE_SDK_DIR/lib

Attention: The TIM_VX_DIR path should include lib and header files of TIM-VX.
You can refer [TIM-VX](https://github.com/VeriSilicon/TIM-VX) to build TIM-VX.

### How to generate nbg by Ovxlib
When you need generate nbg, please use OvxlibExporter class and set the export() function's param pack_nbg_only=True.
Such as: OvxlibExporter(model).export(pack_nbg_only=True), it will use our default SDK.
If you want to use your own SDK and licence, please set the "viv_sdk" and "licence" params of export() function.
Such as: OvxlibExporter(model).export(pack_nbg_only=True, viv_sdk=your_sdk_path, licence=path_of_licence_txt)

Attention: your sdk directory structure must strictly follow the directory structure of  acuitylib/vsi_sdk!!!
your sdk need satisfy the structure of "your_sdk_path/prebuilt-sdk/x86_64_linux", otherwise the path may have problems.
The content of licence is the device target which you want to use.


### Support
Create issue on github or email to ML_Support@verisilicon.com
