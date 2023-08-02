#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from acuitylib.interface.importer import OnnxLoader
from acuitylib.interface.exporter import TimVxExporter
from acuitylib.interface.exporter import TFLiteExporter
from acuitylib.interface.quantization import Quantization, QuantizerType
from acuitylib.interface.inference import Inference


# prepare
onnx_model = "model/squeezenet.onnx"
image0 = "data/813.jpg"


# data generator
def get_data():
    for image in [image0]:
        arr = tf.io.decode_image(tf.io.read_file(image)).numpy()
        arr = np.transpose(arr, [2, 0, 1])
        arr = arr.reshape(1, 3, 224, 224)
        inputs = {'data_0': np.array(arr, dtype=np.float32)}
        yield inputs


def test_onnx_squeezenet():
    # import model
    model = OnnxLoader(model=onnx_model).load(inputs=['data_0'], input_size_list=[[1, 3, 224, 224]],
                                              outputs=['softmaxout_1'])
    # inference
    infer = Inference(model)
    infer.build_session()  # build inference session
    for i, data in enumerate(get_data()):
        ins, outs = infer.run_session(data)  # run inference session
        assert np.argmax(outs[0].flatten()) == 812

    TimVxExporter(model).export('export_timvx/float16/lenet')

    # quantize
    quantizer = QuantizerType.ASYMMETRIC_AFFINE
    qtype = 'uint8'
    Quantization(model).quantize(input_generator_func=get_data, quantizer=quantizer, qtype=qtype, iterations=1)

    # inference with quantized model
    infer = Inference(model)
    infer.build_session()  # build inference session
    for i, data in enumerate(get_data()):
        ins, outs = infer.run_session(data)  # run inference session
        assert np.argmax(outs[0].flatten()) == 812

    # export tim-vx case
    TimVxExporter(model).export('export_timvx/asymu8/squeezenet')
    TFLiteExporter(model).export('export_tflite/asymu8/squeezenet.tflite')


if __name__ == '__main__':
    test_onnx_squeezenet()
