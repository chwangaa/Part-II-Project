#ifndef UTIL_H
#define UTIL_H

#include <sys/time.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <stdint.h>

static inline uint64_t timestamp_us() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return 1000000L * tv.tv_sec + tv.tv_usec;
}

#endif