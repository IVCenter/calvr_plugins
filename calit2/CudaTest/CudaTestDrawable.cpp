#include "GL/glew.h"
#include "CudaTestDrawable.h"

#include "CudaHelper.h"
#include "Timing.h"

#include <iostream>

#define VBO_SIZE 4
#define NUM_VBOS 1500

using namespace osg;

CudaTestDrawable::CudaTestDrawable()
{
    setUseDisplayList(false);
    _init = false;
}

CudaTestDrawable::CudaTestDrawable(const CudaTestDrawable&,const osg::CopyOp& copyop)
{
}

CudaTestDrawable::~CudaTestDrawable()
{
}

void CudaTestDrawable::drawImplementation(osg::RenderInfo& ri) const
{
    if(ri.getContextID())
    {
	return;
    }
    //std::cerr << "Draw context: " << ri.getContextID() << std::endl;
    if(!_init)
    {
	init();
	_init = true;
    }

    glFinish();

    struct timeval start,end;
    getTime(start);

    copyData();

    getTime(end);
    printDiff("Copy Time: ",start,end);
}

osg::BoundingBox CudaTestDrawable::computeBound() const
{
    Vec3 size2(10000, 10000, 10000);
    _boundingBox.init();
    _boundingBox.set(-size2[0], -size2[1], -size2[2], size2[0], size2[1], size2[2]);
    return _boundingBox;
}

void CudaTestDrawable::updateBoundingBox()
{
    computeBound();
    dirtyBound();
}

void CudaTestDrawable::init() const
{
    std::cerr << "CudaTestDrawable init" << std::endl;
    glewInit();
    GLuint buffer;
    for(int i = 0; i < NUM_VBOS; i++)
    {
	glGenBuffersARB(1,&buffer);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, buffer);
	glBufferDataARB(GL_ARRAY_BUFFER_ARB, VBO_SIZE*sizeof(float), 0, GL_STREAM_COPY_ARB);
	_vbolist.push_back(buffer);
	checkRegBufferObj(buffer);
    }
    glBindBufferARB(GL_ARRAY_BUFFER_ARB,0);
}

void CudaTestDrawable::copyData() const
{
    //std::cerr << "Copy Data" << std::endl;
    char * vert;
    float * data = new float[VBO_SIZE];
    for(int i = 0; i < _vbolist.size(); i++)
    {
	checkMapBufferObj((void**)&(vert), _vbolist[i]);
	cudaMemcpyAsync(vert, data, VBO_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	checkUnmapBufferObj(_vbolist[i]);
    }
    cudaThreadSynchronize();
    delete[] data;
}
