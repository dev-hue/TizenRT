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

#include <tinyara/config.h>
#include <cstdio>
#include <debug.h>
#include <tinyara/init.h>
#include <time.h>

extern "C"
{
  int tfmicro_ice_maker_main(int argc, char* argv[]) {
    clock_t start, end;
    if (!setup()) {
      printf("File does not exist\n");
      return -1;
    }
    start = clock();
    while (loop()) { }
    printf("Inference Time: %f seconds\n", (double)(clock() - start) / CLOCKS_PER_SEC);
    while (1)
    {
      sleep(1);
    }
  }
}
