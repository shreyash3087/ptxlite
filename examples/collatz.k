// Collatz step count: count steps to reach 1 from n (capped at 1000).
kernel collatz(int* in, int* out, int n) {
  int i = gid();
  if (i < n) {
    int x = in[i];
    int steps = 0;
    while (x != 1 && steps < 1000) {
      if (x % 2 == 0) {
        x = x / 2;
      } else {
        x = 3 * x + 1;
      }
      steps += 1;
    }
    out[i] = steps;
  }
}
