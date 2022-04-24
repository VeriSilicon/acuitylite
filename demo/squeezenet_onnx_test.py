#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import os
from acuitylib.lite.acuitymodel import AcuityModel
from acuitylib.lite.importer import OnnxLoader
from acuitylib.lite.exporter import TimVxExporter
from acuitylib.lite.exporter import TFLiteExporter
from acuitylib.lite.quantizer import QuantizeType

# prepare
onnx_model = "model/squeezenet.onnx"
image0 = "data/813.jpg"

# dataset, generator
def get_data():
    for image in [image0]:
        arr = tf.io.decode_image(tf.io.read_file(image)).numpy()
        arr = np.transpose(arr, [2, 0, 1])
        arr = arr.reshape(1, 3, 224, 224)
        yield np.array(arr, dtype=np.float32)

def test_onnx_squeezenet():
    # import model
    net = OnnxLoader(model=onnx_model,
                     inputs=['data_0'],
                     input_size_list=[[1, 3, 224, 224]],
                     outputs=['softmaxout_1']).load()
    model = AcuityModel(net)
    # inference
    data_list = [get_data()]
    results = model.inference(data_list, batch_size=1, iterations=1)
    for i, result in enumerate(results):
        for _, outs in result.items():
            assert np.argmax(outs.flatten()) == 812
    TimVxExporter(model).export('export_timvx/float16/lenet')
    # quantize
    data_list = [get_data()]
    model.quantize(data_list, quantizer=QuantizeType.asymu8, iteration=1)
    # inference with quantized model
    data_list = [get_data()]
    results = model.inference(data_list, iterations=1)
    for i, result in enumerate(results):
        for _, outs in result.items():
            assert np.argmax(outs.flatten()) == 812
    # export tim-vx case
    TimVxExporter(model).export('export_timvx/asymu8/squeezenet')
    TFLiteExporter(model).export('export_tflite/asymu8/squeezenet.tflite')

if __name__ == '__main__':
    test_onnx_squeezenet()
