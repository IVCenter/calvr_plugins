#include "Timing.h"

void getTime(struct timeval & time)
{
    gettimeofday(&time, NULL);
}

void printDiff(std::string s, struct timeval tStart, struct timeval tEnd)
{
#ifdef PRINT_TIMING
    std::cerr << s << (tEnd.tv_sec - tStart.tv_sec) + ((tEnd.tv_usec - tStart.tv_usec) / 1000000.0) << std::endl;
#endif
}

float getDiff(struct timeval tStart, struct timeval tEnd)
{
    return (tEnd.tv_sec - tStart.tv_sec) + ((tEnd.tv_usec - tStart.tv_usec) / 1000000.0);
}
