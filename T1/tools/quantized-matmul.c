#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int32_t quantize_symmetric(float value, float scale) {
  float normalized = value / scale;
  int32_t rounded = (int32_t)(normalized + (normalized >= 0 ? 0.5f : -0.5f));
  return rounded < -127 ? -127 : (rounded > 127 ? 127 : rounded);
}

void matmul(float *output, float *input, float *weights, int reduction, int outputs) {
  if (reduction <= 0 || reduction % 32) {
    fprintf(stderr, "quantized reference requires group-32 reduction\n");
    exit(1);
  }
  int8_t *activation = malloc((size_t)reduction);
  float *scales = malloc((size_t)(reduction / 32) * sizeof(float));
  if (!activation || !scales) {
    fprintf(stderr, "cannot allocate quantized reference scratch\n");
    exit(1);
  }
  for (int group = 0; group < reduction / 32; ++group) {
    float magnitude = 0;
    for (int inner = 0; inner < 32; ++inner)
      magnitude = fmaxf(magnitude, fabsf(input[group * 32 + inner]));
    scales[group] = magnitude > 0 ? magnitude / 127.0f : 1.0f;
    for (int inner = 0; inner < 32; ++inner)
      activation[group * 32 + inner] = quantize_symmetric(input[group * 32 + inner], scales[group]);
  }
#pragma omp parallel for
  for (int row = 0; row < outputs; ++row) {
    float result = 0;
    for (int group = 0; group < reduction / 32; ++group) {
      const float *block = weights + (size_t)row * reduction + group * 32;
      float magnitude = 0;
      for (int inner = 0; inner < 32; ++inner)
        magnitude = fmaxf(magnitude, fabsf(block[inner]));
      float weight_scale = magnitude > 0 ? magnitude / 127.0f : 1.0f;
      int32_t accumulator = 0;
      for (int inner = 0; inner < 32; ++inner)
        accumulator += activation[group * 32 + inner] * quantize_symmetric(block[inner], weight_scale);
      result += ((float)accumulator * scales[group]) * weight_scale;
    }
    output[row] = result;
  }
  free(scales);
  free(activation);
}
