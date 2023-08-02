#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from acuitylib.interface.importer import TensorflowLoader
from acuitylib.interface.exporter import TFLiteExporter
from acuitylib.interface.quantization import Quantization, QuantizerType
from acuitylib.interface.inference import Inference


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
        inputs = {'input': np.array(arr, dtype=np.float32)}
        yield inputs


def test_tensorflow_mobilenet():
    # load tensorflow model
    model = TensorflowLoader(mobilenet).load(inputs=['input'], input_size_list=[[1, 224, 224, 3]],
                                             outputs=['MobilenetV1/Logits/SpatialSqueeze'])

    # quantization
    quantizer = QuantizerType.ASYMMETRIC_AFFINE
    qtype = 'uint8'
    Quantization(model).quantize(input_generator_func=get_data, quantizer=quantizer, qtype=qtype, iterations=1)

    # inference with quant model
    infer = Inference(model)
    infer.build_session()  # build inference session
    for i, data in enumerate(get_data()):
        ins, outs = infer.run_session(data)  # run inference session
        assert outs[0].flatten()[labels[i]] > 0.99

    # export tflite quant case
    TFLiteExporter(model).export('export_tflite/asymu8/mobilenet.tflite')


if __name__ == '__main__':
    test_tensorflow_mobilenet()
