// Math builtins showcase: sqrt, min, max on arrays.
kernel mathops(float* a, float* b, float* out_sqrt, float* out_min, float* out_max, int n) {
  int i = gid();
  if (i < n) {
    out_sqrt[i] = sqrt(a[i]);
    out_min[i] = min(a[i], b[i]);
    out_max[i] = max(a[i], b[i]);
  }
}
