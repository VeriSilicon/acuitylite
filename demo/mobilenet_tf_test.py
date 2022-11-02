#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from acuitylib.lite.acuitymodel import AcuityModel
from acuitylib.lite.importer import TensorflowLoader
from acuitylib.lite.exporter import TFLiteExporter
from acuitylib.lite.quantizer import QuantizeType

# prepare
mobilenet = "model/mobilenet_v1.pb"
image0 = "data/697.jpg"
image1 = "data/813.jpg"
labels = [697, 813]

# data generator
def get_data():
    for image in [image0, image1]:
        arr = tf.io.decode_image(tf.io.read_file(image)).numpy().reshape(1, 224, 224, 3).astype(np.float32)
        arr = (arr-128.0)/128.0  # preprocess
        yield np.array(arr, dtype=np.float32)

def test_tensorflow_mobilenet():
    # load tensorflow model
    acuitynet = TensorflowLoader(mobilenet,
                                    inputs=['input'],
                                    input_size_list=[[1, 224, 224, 3]],
                                    outputs=['MobilenetV1/Logits/SpatialSqueeze']).load()
    model = AcuityModel(acuitynet)
    data_list = [get_data()]
    model.quantize(data_list, quantizer=QuantizeType.asymu8, iteration=1)

    # quant model inference
    data_list = [get_data()]
    results = model.inference(data_list, batch_size=1, iterations=2)
    for i, result in enumerate(results):
        for _, outs in result.items():
            assert outs[0][labels[i]] > 0.9

    # export tflite quant case
    TFLiteExporter(model).export('export_tflite/asymu8/mobilenet.tflite')

if __name__ == '__main__':
    test_tensorflow_mobilenet()
