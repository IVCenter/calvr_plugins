#include "CudaCopyCallback.h"
#include "Timing.h"

#include "MultiGPUDrawable.h"

#include <iostream>

using namespace std;
using namespace osg;

bool CudaCopyCallback::_copy = false;

CudaCopyCallback::CudaCopyCallback(int dev, MultiGPUDrawable * drawable)
{
    _dev = dev;
    _drawable = drawable;
    _firstFrame = true;
}

CudaCopyCallback::~CudaCopyCallback()
{
}

void CudaCopyCallback::setCopy(bool b)
{
    _copy = b;
}

void CudaCopyCallback::operator()(osg::RenderInfo & ri) const
{
    if(_firstFrame)
    {
	cudaSetDevice(_dev);
	_firstFrame = false;
    }
    //std::cerr << "CudaCallback" << std::endl;
    if(_copy)
    {
	cudaStream_t stream;
	//cudaStreamCreate(&stream);
	cudaThreadSynchronize();

#ifdef PRINT_TIMING
	struct timeval bcopy, acopy;
	getTime(bcopy);
#endif
	_drawable->cudaLoadNextVBOs(_dev,stream);
#ifdef PRINT_TIMING
	cudaThreadSynchronize();
	//cudaStreamSynchronize(stream);
	getTime(acopy);
	printDiff("Cuda Copy Time: ",bcopy,acopy);
#endif
	//cudaStreamDestroy(stream);
    }
}
