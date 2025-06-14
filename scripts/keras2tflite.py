# -*- coding: utf-8 -*-
"""
Spyder Editor
Józef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""
import tensorflow as tf

"""
Konwersja modelu Keras do formatu TensorFlow Lite dla urządzeń mobilnych i wbudowanych.
Skrypt przekształca wytrenowany model sieci neuronowej do zoptymalizowanego formatu TFLite,
który jest przeznaczony do wdrażania na urządzeniach o ograniczonych zasobach obliczeniowych
takich jak smartfony, tablety czy systemy wbudowane. Proces obejmuje optymalizację rozmiaru
i szybkości działania modelu przy zachowaniu jego dokładności.
Zastosowane optymalizacje:
- Domyślne optymalizacje TensorFlow Lite (kwantyzacja, przycinanie wag)
- Wsparcie dla standardowych operacji TFLite oraz wybranych operacji TensorFlow
- Włączenie operacji elastycznych (Flex Ops) dla lepszej kompatybilności
Wynikowy model jest gotowy do implementacji w aplikacjach mobilnych.
"""

model_path = "checkpoints/USGEquina-Pregna_v1_0.keras"
model = tf.keras.models.load_model(model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Włączenie optymalizacji i wsparcia dla Flex Ops
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

tflite_model = converter.convert()

with open("checkpoints/USGEquina-Pregna_v1_0.tflite", "wb") as f:
    f.write(tflite_model)
