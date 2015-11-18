#ifndef UTIL_H
#define UTIL_H

#include <sys/time.h>


static inline uint64_t timestamp_us() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return 1000000L * tv.tv_sec + tv.tv_usec;
}

#endif