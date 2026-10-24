void saxpy(const float *input_x, const float *input_y, float *output,
           float alpha, int length) {
  for (int index = 0; index < length; ++index)
    output[index] = alpha * input_x[index] + input_y[index];
}
