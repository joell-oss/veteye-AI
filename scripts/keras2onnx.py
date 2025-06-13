# -*- coding: utf-8 -*-
"""
Spyder Editor
Józef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""
import tensorflow as tf
import tf2onnx

model_path = "checkpoints/USGEquina-Pregna_v1_0.keras"
model = tf.keras.models.load_model(model_path)

spec = (tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

with open("checkpoints/USGEquina-Pregna_v1_0.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())
