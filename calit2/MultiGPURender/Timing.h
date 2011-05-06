#ifndef _TIMING_H_
#define _TIMING_H_

#include <sys/time.h>
#include <string>
#include <iostream>

//#define PRINT_TIMING

void getTime(struct timeval & time);
void printDiff(std::string s, struct timeval tStart, struct timeval tEnd); 
float getDiff(struct timeval tStart, struct timeval tEnd);

#endif
