// Mandelbrot iteration count per pixel.
// cx[], cy[]: real/imag coordinates; out[]: iteration counts; max_iter, n: sizes.
kernel mandelbrot(float* cx, float* cy, int* out, int max_iter, int n) {
  int i = gid();
  if (i < n) {
    float x = 0.0;
    float y = 0.0;
    float c_re = cx[i];
    float c_im = cy[i];
    int iter = 0;
    while (iter < max_iter && x * x + y * y < 4.0) {
      float xnew = x * x - y * y + c_re;
      y = 2.0 * x * y + c_im;
      x = xnew;
      iter += 1;
    }
    out[i] = iter;
  }
}
