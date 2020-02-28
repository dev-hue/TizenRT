/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "main_functions.h"
#include "ice_maker_model_data.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#define STD 6000
#define RIGHT_END 6023
#define WORNG_END 4882
#define VALUE 1.0f
#define ARENA 5

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;
int right = 0;
float data[41] = {};
FILE *fp = nullptr;

constexpr int kTensorArenaSize = ARENA * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

bool setup() {
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(g_ice_maker_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  static tflite::ops::micro::AllOpsResolver all;

  static tflite::MicroInterpreter static_interpreter(
      model, all, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return false;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  inference_count = 0;

  fp = fopen("/rom/sensor_data3.bin", "rb");

  if (fp == nullptr) {
    return false;
  }

  return true;
}

bool loop() {

  fread((void *)data, sizeof(float), 41, fp);

  input->data.f = data;

  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    printf("Invoke fail\n");
    return false;
  }

  float y_val = output->data.f[0];

  if (y_val >= 0.5f) {
    right += 1;
  }

  // if (y_val < 0.5f) {
  //   right += 1;
  // }

  inference_count += 1;

  // if (inference_count == STD) {
  //   printf("\nResult: [  %4d / %4d  ]\n", right, STD);
  //   return false;
  // }

  if (inference_count == RIGHT_END) {
    printf("\n===> Actual Error Count=%4d | Successful Error Predction Count=%4d <===\n", RIGHT_END, right);
    printf("===> Actual None-Error Count=0 | Successfull None-Error Prediction Count=0 <===\n");

    return false;
  }

  // if (inference_count == WORNG_END) {
  //   printf("Result: [  %4d / %4d  ]\n", right, WORNG_END);
  //   return false;
  // }

  return true;
}
