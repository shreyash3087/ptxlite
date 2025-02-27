// SAXPY: y[i] = a * x[i] + y[i]
kernel saxpy(float* x, float* y, float a, int n) {
  int i = gid();
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}
