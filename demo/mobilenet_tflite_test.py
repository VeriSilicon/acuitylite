#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from acuitylib.interface.importer import TFLiteLoader
from acuitylib.interface.exporter import TimVxExporter
from acuitylib.interface.exporter import OvxlibExporter
from acuitylib.interface.inference import Inference

# wget https://storage.googleapis.com/download.tensorflow.org/
# models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz


mobilenet = "model/mobilenet_v1_1.0_224_quant.tflite"
image0 = "data/697.jpg"
image1 = "data/813.jpg"
labels = [697, 813]


# data generator
def get_data():
    for image in [image0, image1]:
        arr = tf.io.decode_image(tf.io.read_file(image)).numpy().reshape(1, 224, 224, 3).astype(np.float32)
        arr = (arr-128.0)/128.0  # preprocess
        # quantize arr as the quantized input for quantized model(new feature begin from 6.27.0)
        arr = np.rint(arr / 0.0078125) + 128
        inputs = {'input': np.array(arr, dtype=np.uint8)}
        yield inputs


def test_tflite_mobilenet():
    # no need to quantize using acuity lite for quant model
    # load tflite quant model
    quantmodel = TFLiteLoader(mobilenet).load()

    # inference with quant model
    infer = Inference(quantmodel)
    infer.build_session()  # build inference session
    for i, data in enumerate(get_data()):
        ins, outs = infer.run_session(data)  # run inference session
        assert outs[0].flatten()[labels[i]] > 0.9

    # export tim-vx quant case
    TimVxExporter(quantmodel).export('export_timvx/quant/mobilenet')

    # export nbg
    OvxlibExporter(quantmodel).export('export_ovxlib/quant/mobilenet', pack_nbg_only=True)


if __name__ == '__main__':
    test_tflite_mobilenet()
