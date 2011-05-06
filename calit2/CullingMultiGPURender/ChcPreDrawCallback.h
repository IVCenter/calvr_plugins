#ifndef CHCPREDRAWCALLBACK_H
#define CHCPREDRAWCALLBACK_H

#include "PreDrawHook.h"
#include "Geometry.h"
#include <map>
#include <vector>
#include <OpenThreads/Barrier>
#include "BufferManager.h"

#include <osg/Camera>

class ChcAnimate;

/** @file ChcPreCallback.h
 *  ChcPreDrawCallback class is used for managing the mulitple
 *  contexts and rendering sequence.
 *  Derived from PreDrawCallback class.
 */
class ChcPreDrawCallback : public PreDrawCallback
{
public:
     ChcPreDrawCallback(int gpuNum, int totalGpuNum, ChcAnimate * chcAnimate);
     void preDrawCallback(osg::Camera * cam = NULL);

protected:
     ChcPreDrawCallback();
     ~ChcPreDrawCallback();

     // pointer to data for copying
     std::vector<int> * _parts;
     std::map<int, Geometry* > * _pmap;
     int _gpuNum;
     int _totalGpuNum;
     bool _initGlew;
     osg::Camera * _camera;
     ChcAnimate * _chcAnimate; 
     static OpenThreads::Barrier _occlusionBarrier;

     BufferManager * _bufferManager;

     bool _cudaCopy;
     bool _cudaInit;
     bool _bufferManInit;
};

#endif

