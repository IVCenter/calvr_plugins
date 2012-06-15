#ifndef MGPU_CUDA_COPY_CALLBACK
#define MGPU_CUDA_COPY_CALLBACK

#include <osg/Camera>

#include "CudaHelper.h"

class MultiGPUDrawable;
class CudaCopyCallback : public osg::Camera::Camera::DrawCallback
{
    public:
        CudaCopyCallback(int dev, MultiGPUDrawable * drawable);
        ~CudaCopyCallback();

        static void setCopy(bool b);

        virtual void operator()(osg::RenderInfo & ri) const;
    protected:
        static bool _copy;
        int _dev;
        MultiGPUDrawable * _drawable;
        mutable bool _firstFrame;
};

#endif
