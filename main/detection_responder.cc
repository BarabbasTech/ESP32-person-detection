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

#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "sdkconfig.h"

#define OUTPUT_PIN GPIO_NUM_4
#define DETECTION_THRESHOLD 200
#define HISTORY_SIZE 3  // Reduced history size

static uint8_t score_history[HISTORY_SIZE];
static uint8_t history_index = 0;

#include "detection_responder.h"

// This dummy implementation writes person and no person scores to the error
// console. Real applications will want to take some custom action instead, and
// should implement their own versions of this function.
void RespondToDetection(tflite::ErrorReporter* error_reporter,
                        uint8_t person_score, uint8_t no_person_score) {
  score_history[history_index] = person_score;
  history_index = (history_index + 1) % HISTORY_SIZE;
  
  uint16_t avg_score = 0;
  for (int i = 0; i < HISTORY_SIZE; i++) {
    avg_score += score_history[i];
  }
  avg_score /= HISTORY_SIZE;

  gpio_pad_select_gpio(OUTPUT_PIN);
  gpio_set_direction(OUTPUT_PIN, GPIO_MODE_OUTPUT);
  gpio_set_level(OUTPUT_PIN, avg_score >= DETECTION_THRESHOLD ? 0 : 1);

  TF_LITE_REPORT_ERROR(error_reporter, 
                       "person score: %d, avg: %d, detected: %d",
                       person_score, avg_score, avg_score >= DETECTION_THRESHOLD);
}
