/**
 * @file Timing.h
 * Contains functions to collect and print timing values 
 *
 * @author Andrew Prudhomme (aprudhomme@ucsd.edu)
 *
 * @date 09/15/2010
 */

#ifndef _TIMING_H_
#define _TIMING_H_

#include <sys/time.h>
#include <string>
#include <iostream>

//#define PRINT_TIMING

/// get current time
/// @param time reference that receives the current time
void getTime(struct timeval & time);

/// print the difference between two timevals
/// @param s string that prefixes the printout
/// @param tStart the starting timeval
/// @param tEnd the ending timeval
void printDiff(std::string s, struct timeval tStart, struct timeval tEnd);

/// get the difference between two timevals
/// @param tStart the starting timeval
/// @param tEnd the ending timeval
/// @return the difference between the timevals in seconds 
float getDiff(struct timeval tStart, struct timeval tEnd);

#endif
