// Row sum: each thread sums one row of a flattened matrix.
// out[i] = sum of row[i * cols .. (i+1)*cols - 1]
kernel rowsum(float* row, float* out, int cols, int nrows) {
  int i = gid();
  if (i < nrows) {
    float s = 0.0;
    int base = i * cols;
    int j = 0;
    while (j < cols) {
      s += row[base + j];
      j += 1;
    }
    out[i] = s;
  }
}
