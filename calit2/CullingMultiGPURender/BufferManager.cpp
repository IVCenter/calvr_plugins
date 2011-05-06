#include <GL/glew.h>

#include "BufferManager.h"
#include "Geometry.h"
#include "CudaHelper.h"

#include <iostream>

/// defined maximum points a vbo can hold
#define MAX_BUFFER_SIZE 1000000

/// driver hint to use when creating vbos
#define VBO_TYPE GL_STREAM_COPY_ARB

/**
 * @param partMap list of part objects indexed by part number
 * @param partList list of parts assigned to this gpu
 * @param context gpu assigned to this object
 * @param cudaCopy should cuda be used to copy data to the gpu
 */
BufferManager::BufferManager(std::map<int,Geometry *> & partMap, std::vector<int> & partList, int context, bool cudaCopy)
{
    _partMap = partMap;
    _partList = partList;
    _context = context;
    _cudaCopy = cudaCopy;
    _frame = -2;
    _nextFrame = -2;
}

BufferManager::~BufferManager()
{
}

/**
 * Request a location in the buffers for the current frame to hold the vertex data for a geometry.
 * @param geo geometry that is requesting space in the buffers
 * @return pointer to location in the buffers to load data
 */
char * BufferManager::requestBuffer(Geometry * geo)
{
    int verts = geo->getNumVerts();

    // see if part has too many points
    if(verts > MAX_BUFFER_SIZE)
    {
	std::cerr << "ERROR: part has too many points, MAX: " << MAX_BUFFER_SIZE << " got: " << verts << std::endl;
	exit(0);
    }

    int size = verts * 3 * sizeof(float);

    int i;

    // see if any buffers have room for the vertex data
    for(i = 0; i < _frameRemList.size(); i++)
    {
	if(size <= _frameRemList[i])
	{
	    break;
	}
    }

    // if no room, add a new buffer
    if(i == _frameRemList.size())
    {
	addMemoryBuffer();
    }

    // set buffer location in geometry
    geo->setVertBuffer(_context, _vboList[i]);
    geo->setVBOOffset(_context, _frameOffsetList[i]);

    char * memOffset = _frameMemList[i] + _frameOffsetList[i];

    //std::cerr << "Buffer requsted for part " << geo->getPartNumber() << " using index " << i << " vbo " << _vboList[i] << " offset " << _frameOffsetList[i] << std::endl;

    // update buffer status
    _frameRemList[i] -= size;
    _frameOffsetList[i] += size;

    return memOffset;
}

/**
 * Request a location in the buffers for the next frame to hold the vertex data for a geometry.
 * @param geo geometry that is requesting space in the buffers
 * @return pointer to location in the buffers to load data
 */
char * BufferManager::requestNextBuffer(Geometry * geo)
{
    int verts = geo->getNumVerts();

    // see if part has too many points
    if(verts > MAX_BUFFER_SIZE)
    {
	std::cerr << "ERROR: part has too many points, MAX: " << MAX_BUFFER_SIZE << " got: " << verts << std::endl;
	exit(0);
    }

    int size = verts * 3 * sizeof(float);

    int i;

    // see if any buffers have room for vertex data
    for(i = 0; i < _nextFrameRemList.size(); i++)
    {
	if(size <= _nextFrameRemList[i])
	{
	    break;
	}
    }

    // if no room, add another buffer
    if(i == _nextFrameRemList.size())
    {
	addMemoryBuffer();
    }

    // set next buffer location in geometry
    geo->setNextVertBuffer(_vboList[i]);
    geo->setNextVBOOffset(_nextFrameOffsetList[i]);

    char * memOffset = _nextFrameMemList[i] + _nextFrameOffsetList[i];

    //std::cerr << "Next Buffer requsted for part " << geo->getPartNumber() << " using index " << i << " vbo " << _vboList[i] << " offset " << _nextFrameOffsetList[i] << std::endl;
    // update buffer status
    _nextFrameRemList[i] -= size;
    _nextFrameOffsetList[i] += size;

    return memOffset;
}

/**
 * Load needed vertex data from ram buffers onto gpu
 * @param predraw are we doing this call in predraw (cudacopy can only be used in predraw)
 */
void BufferManager::loadFrameData(bool predraw)
{
    for(int i = 0; i < _frameMemList.size(); i++)
    {
	// see if whole buffer is already on the gpu
	if(_frameCopiedList[i] < _frameOffsetList[i])
	{
	    // copy using cuda or gl calls
	    if(predraw && _cudaCopy)
	    {
		char * vert;

		checkMapBufferObj((void**)&vert,_vboList[i]);

		cudaMemcpyAsync(vert + _frameCopiedList[i], _frameMemList[i] + _frameCopiedList[i], _frameOffsetList[i] - _frameCopiedList[i], cudaMemcpyHostToDevice);

		checkUnmapBufferObj(_vboList[i]);
	    }
	    else
	    {
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, _vboList[i]);

		glBufferSubDataARB(GL_ARRAY_BUFFER_ARB, _frameCopiedList[i], _frameOffsetList[i] - _frameCopiedList[i], (void*)(_frameMemList[i] + _frameCopiedList[i]) );

		glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
	    }

	    _frameCopiedList[i] = _frameOffsetList[i];
	}
    }
}

/// reset ram buffer status data for current frame
void BufferManager::resetFrame()
{
    for(int i = 0; i < _frameMemList.size(); i++)
    {
	_frameRemList[i] = MAX_BUFFER_SIZE * 3 * sizeof(float);
	_frameCopiedList[i] = 0;
	_frameOffsetList[i] = 0;
    }
}

/// reset ram buffer status data for next frame
void BufferManager::resetNextFrame()
{
    for(int i = 0; i < _nextFrameMemList.size(); i++)
    {
	_nextFrameRemList[i] = MAX_BUFFER_SIZE * 3 * sizeof(float);
	_nextFrameCopiedList[i] = 0;
	_nextFrameOffsetList[i] = 0;
    }
}

/// swap buffers and status data for current and next frame (advance to next frame)
void BufferManager::swapFrame()
{
    std::vector<char *> memscratch;

    std::vector<int> iscratch;

    memscratch = _frameMemList;
    _frameMemList = _nextFrameMemList;
    _nextFrameMemList = memscratch;

    iscratch = _frameRemList;
    _frameRemList = _nextFrameRemList;
    _nextFrameRemList = iscratch;

    iscratch = _frameCopiedList;
    _frameCopiedList = _nextFrameCopiedList;
    _nextFrameCopiedList = iscratch;

    iscratch = _frameOffsetList;
    _frameOffsetList = _nextFrameOffsetList;
    _nextFrameOffsetList = iscratch;

    int f = _frame;

    _frame = _nextFrame;
    _nextFrame = f;

    
}
/**
 * Set the frame number for the current frame, swap buffers if needed.
 * @param frame set current frame to this frame
 */
void BufferManager::setFrame(int frame)
{
    if(_frame == frame)
    {
	return;
    }

    if(frame == _nextFrame)
    {
	swapFrame();
    }
    else
    {
	resetFrame();
    }

    _frame = frame;
}

/**
 * Set the frame number for the next frame.
 * @param frame set next frame to this frame
 */
void BufferManager::setNextFrame(int frame)
{
    if(frame == _nextFrame)
    {
	return;
    }

    resetNextFrame();

    _nextFrame = frame;
}

/**
 * Find the memory pointer that holds the vertex data in context 0 for a given geometry, used for paused memcpy.
 * @param geo geometry to search with for pointer
 * @return address in ram of vertex data
 */
char * BufferManager::getBuffer0Pointer(Geometry * geo)
{
    // find buffer containing vert data
    GLuint buffer = geo->getVertBuffer(0);
    int i;
    for(int i = 0; i < _vboList.size(); i++)
    {
	if(buffer == _vboList[i])
	{
	    break;
	}
    }

    if(i == _vboList.size())
    {
	std::cerr << "Error: unable to find vbo in context 0 list." << std::endl;
	exit(0);
    }

    // return pointer location
    return _frameMemList[i] + geo->getVBOOffset(0);
}

/**
 * Add another buffer in ram to hold vertex data for current and next frame, add another VBO, init buffer status data.
 */
void BufferManager::addMemoryBuffer()
{
    char * buffer;

    // allocate new buffer for current frame, with cuda if needed
    if(!_cudaCopy)
    {
	buffer = (char *) new float[3 * MAX_BUFFER_SIZE];
    }
    else
    {
	checkHostAlloc((void**)&buffer,3*MAX_BUFFER_SIZE*sizeof(float),cudaHostAllocPortable);
    }

    _frameMemList.push_back(buffer);

    // allocate new buffer for next frame, with cuda if needed
    if(!_cudaCopy)
    {
	buffer = (char *) new float[3 * MAX_BUFFER_SIZE];
    }
    else
    {
	checkHostAlloc((void**)&buffer,3*MAX_BUFFER_SIZE*sizeof(float),cudaHostAllocPortable);
    }

    _nextFrameMemList.push_back(buffer);

    // init buffer status info for new buffer
    _frameRemList.push_back(MAX_BUFFER_SIZE * 3 * sizeof(float));
    _nextFrameRemList.push_back(MAX_BUFFER_SIZE * 3 * sizeof(float));

    _frameCopiedList.push_back(0);
    _nextFrameCopiedList.push_back(0);

    _frameOffsetList.push_back(0);
    _nextFrameOffsetList.push_back(0);

    // create new vbo
    GLuint vbo;
    glGenBuffersARB(1,&vbo);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);

    glBufferDataARB(GL_ARRAY_BUFFER_ARB, 3 *  sizeof(float) * MAX_BUFFER_SIZE, 0, VBO_TYPE);

    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

    // register wih cuda if needed
    if(_cudaCopy)
    {
	 checkRegBufferObj(vbo);
    }

    _vboList.push_back(vbo);
}
