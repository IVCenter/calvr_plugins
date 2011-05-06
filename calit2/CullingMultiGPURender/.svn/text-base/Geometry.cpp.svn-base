#include <GL/glew.h>
#include "Geometry.h"
#include "FetchQueue.h"
#include "stdio.h"
#include <math.h>
#include <cstring>
#include "CudaHelper.h"

//#define VBO_TYPE GL_STREAM_COPY_ARB

int Geometry::totalNumberOfBytes = 0;
int Geometry::totalPreFetch = 0;
int Geometry::totalLoadOnDemand = 0;
int Geometry::totalNumGeometry = 0;
double Geometry::totalCopyTime = 0;

struct ColorVal Geometry::makeColor(float f)
{
    if(f < 0)
    {
        f = 0;
    }
    else if(f > 1.0)
    {
        f = 1.0;
    }

    ColorVal color;

    if(f <= 0.33)
    {
        float part = f / 0.33;
        float part2 = 1.0 - part;

        color.r = part2;
        color.g = part;
        color.b = 0;
    }
    else if(f <= 0.66)
    {
        f = f - 0.33;
        float part = f / 0.33;
        float part2 = 1.0 - part;

        color.r = 0;
        color.g = part2;
        color.b = part;
    }
    else if(f <= 1.0)
    {
        f = f - 0.66;
        float part = f / 0.33;
        float part2 = 1.0 - part;

        color.r = part;
        color.g = 0;
        color.b = part2;
    }

    return color;
}


Geometry::Geometry(int partnum, std::string &name, int numindices, unsigned int * indexs, bool cudaCopy)
{
	// initalize frame to be -1
	number = partnum;
	_frame = -2;
	_nextFrame = -2;
	buffernum = -2;
	basename = name;
	shared = false;
	isVisible = false;
	drawVisible = false;
	_drawn = false;
	_postDrawn = false;

	//std::cerr << "Num ind " << numindices << std::endl;

	//set color
	color = makeColor( (float) (partnum % 29) / 31.0);

	initalizedBuffers = false;
	initalized0Buffers = false;

	numIndices = numindices;
	indices = indexs;
	
	vertices = NULL;
	numVertexs = 0;
	offset = 0;
	newframe = -2;
	_cudaCopy = cudaCopy;    
}

Geometry::~Geometry()
{
    if(!_cudaCopy)
    {
	if (vertices)
	    delete[] vertices;
	    vertices = NULL;
    }
    else
    {
	if(vertices)
	{
	    cudaFreeHost(vertices);
	}
    }
}

void Geometry::SetVisible(bool visible)
{
    if(!isVisible && visible)
    {
	drawVisible = true;
    }
    isVisible = visible;

    /*if(_postDrawn)
    {
	std::cerr << "Setting part: " << getPartNumber() << " to " << visible << std::endl;
    }*/
}

void Geometry::SetVertices(int numvertices, int vertOffset)
{
	numVertexs = numvertices;
	offset = vertOffset;
	return;
	if(!_cudaCopy)
	{
	    vertices = new float[ 3 * numVertexs];
	}
	else
	{
	    checkHostAlloc((void**)&vertices,3*numVertexs*sizeof(float),cudaHostAllocPortable);
	}
}


Geometry::Geometry(int partnum, std::string &name, bool cudaCopy)
{
	// initalize frame to be -1
	number = partnum;
	_frame = -2;
	_nextFrame = -2;
	buffernum = -2;
	newframe = -2;
	basename = name;
	shared = false;
	isVisible = false;
	drawVisible = false;
	_drawn = false;
	initalizedBuffers = false;
	initalized0Buffers = false;
	_postDrawn = false;

	_frameStatus = NOT_LOADING;
	_nextFrameStatus = NOT_LOADING;

	//set color
	color = makeColor( (float) (partnum % 29) / 31.0);

	_cudaCopy = cudaCopy;

	//find and read in indice data
	std::string filename(basename);
        filename.append("Indices.bin");

	int numberParts = 0;
	int partNumber = 0;
        std::ifstream in(filename.c_str(), std::ios::binary | std::ios::in);
	in.read((char*)&numberParts, sizeof(int));
	for(int i = 0; i < numberParts; i++)
	{	
                // calculate where description data is kept in file
                int offseti = sizeof(int) + (sizeof(int) * 3 * i);
                in.seekg(offseti);

		in.read((char*) &partNumber, sizeof(int));

		if(partNumber == number)
		{
		    // read out size and then offset
		    in.read((char*) &numIndices, sizeof(int));
		    indices  = new unsigned int[numIndices];
		    in.read((char*) &offseti, sizeof(int));

		    // get part number and number of vertices from file, calc fileoffset first
		    in.seekg(offseti);
		    in.read((char*)indices, sizeof(int) * numIndices);

		    break;
		}
	}
	in.close();

	//reset name
	filename.clear();
	filename.append(basename).append("0data.bin");

        in.open(filename.c_str(), std::ios::binary | std::ios::in);
	in.read((char*)&numberParts, sizeof(int));
	for(int i = 0; i < numberParts; i++)
	{	
                // calculate where description data is kept in file
                int offseti = sizeof(int) + (sizeof(int) * 3 * i);
                in.seekg(offseti);

		in.read((char*) &partNumber, sizeof(int));

		if(partNumber == number)
		{
		    // read out size and then offset
		    in.read((char*) &numVertexs, sizeof(int));
		    vertices  = new float[3 * numVertexs];
		    in.read((char*) &offset, sizeof(int));
		    break;
		}
	}
	in.close();
}

void Geometry::InitalizeBuffers(int context)
{
    if(context == 0)
    {
	// check if buffers have been initalized
	if( initalized0Buffers )
	    return;


	// set indices and copy data and then delete indices from memory
	glGenBuffersARB(1, &vbo0Id[1]);
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, vbo0Id[1]);
	glBufferDataARB(GL_ELEMENT_ARRAY_BUFFER_ARB, sizeof(unsigned int) * numIndices, 0, GL_STATIC_DRAW_ARB);
	glBufferSubDataARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0, sizeof(unsigned int) * numIndices, (void*)indices);


	/*
	// initalize vertex buffers
	glGenBuffersARB(1, &vbo0Id[0]);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo0Id[0]);
	glBufferDataARB(GL_ARRAY_BUFFER_ARB, 3 *  sizeof(float) * numVertexs, 0, VBO_TYPE);
	*/

	glBindBufferARB(GL_ARRAY_BUFFER_ARB,0);
	
	//printf("Intialized index buffers %d\n", number);

	/*if(_cudaCopy && numVertexs)
	{
	    checkRegBufferObj(vbo0Id[0]);
	}*/

	initalized0Buffers = true;
    }
    else
    {

	// check if buffers have been initalized
	if( initalizedBuffers )
	    return;


	// set indices and copy data and then delete indices from memory
	glGenBuffersARB(1, &vboId[1]);
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, vboId[1]);
	glBufferDataARB(GL_ELEMENT_ARRAY_BUFFER_ARB, sizeof(unsigned int) * numIndices, 0, GL_STATIC_DRAW_ARB);
	glBufferSubDataARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0, sizeof(unsigned int) * numIndices, (void*)indices);

	/*
	// initalize vertex buffers
	glGenBuffersARB(1, &vboId[0]);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, vboId[0]);
	glBufferDataARB(GL_ARRAY_BUFFER_ARB, 3 *  sizeof(float) * numVertexs, 0, VBO_TYPE);

	*/

	glBindBufferARB(GL_ARRAY_BUFFER_ARB,0);

	/*if(_cudaCopy && numVertexs)
	{
	    checkRegBufferObj(vboId[0]);
	}*/

	initalizedBuffers = true;
    }

    if(initalized0Buffers && initalizedBuffers)
    {
	    // can delete indices array since copied to video card memory
	    delete[] indices;
	    indices = NULL;
    }
}

//need to create vbo buffers in first context for all parts (so can do occlusion query draw if required)
void Geometry::Render()
{
	SetVisible(true);
	if(!numVertexs || _drawn)
	{
	    //std::cerr << "Failed to render, verts: " << numVertexs << " part: " << getPartNum() << std::endl;
	      return;
	}

	//std::cerr << "Render part " << getPartNumber() << std::endl;

	//std::cerr << "Doing post render" << std::endl;
	//std::cerr << "I am ... " << isOn() << std::endl;
	SetFrame(0,false);

	

	_drawn = true;

	bool textureOn = glIsEnabled(GL_TEXTURE_COORD_ARRAY);
	bool vertexOn = glIsEnabled(GL_VERTEX_ARRAY);

	if(textureOn)
	{
	    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	}

	if(!vertexOn)
	{
	    glEnableClientState(GL_VERTEX_ARRAY);
	}

	//check if frame data needs to be loaded
        glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo0Id[0]);
        glVertexPointer(3, GL_FLOAT, 0, (GLvoid *)_vbo0Offset);
	glColor3f(color.r, color.g, color.b);
        glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, vbo0Id[1]);
        glDrawElements(GL_QUADS, numIndices, GL_UNSIGNED_INT, 0);
        glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);
        glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

	if(!vertexOn)
	{
	    glDisableClientState(GL_VERTEX_ARRAY);
	}

	if(textureOn)
	{
	    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	}

	_postDrawn = true;
	// turn off after rendered first time
	//drawVisible = false;
}

void Geometry::ResetStats()
{
	totalNumberOfBytes = 0;
        totalPreFetch = 0;
        totalLoadOnDemand = 0;
	totalNumGeometry = 0;
	totalCopyTime = 0;
}

double Geometry::GetTotalCopyTime()
{
	return totalCopyTime;
}

int Geometry::GetTotalLoadedBytes()
{
	return totalNumberOfBytes;
}

int Geometry::GetTotalPreFetched()
{
        return totalPreFetch;
}

int Geometry::GetTotalLoadedOnDemand()
{
        return totalLoadOnDemand;
}

int Geometry::GetTotalNumGeometry()
{
        return totalNumGeometry;
}

void Geometry::SetDiffuseColor(float diffuseR, float diffuseG, float diffuseB)
{
	mDiffuseColor[0] = diffuseR;
	mDiffuseColor[1] = diffuseG;
	mDiffuseColor[2] = diffuseB;
}

void Geometry::SetShared(bool share)
{
	shared = share;
}

bool Geometry::IsShared()
{
	return shared;
}
	
const AABox &Geometry::GetBoundingVolume()
{
	return mBoundingBox;
}



void Geometry::SetLastRendered(int lastRendered)
{
	mLastRendered = lastRendered;
}


int Geometry::GetLastRendered()
{
	return mLastRendered;
}

void Geometry::SetBuffer(int framenum, char * buffer, bool updateStatus)
{
    _mutex.lock();
    _statusMutex.lock();
    if(framenum != _frame && framenum != _nextFrame)
    {
	std::cerr << "Load Frame number " << framenum << " does not match this frame: " << _frame << " or next: " << _nextFrame << std::endl;
	_statusMutex.unlock();
	_mutex.unlock();
	return;
    }
    _statusMutex.unlock();
	//if(buffernum != framenum)
	//{
		timeval start, end;
		getTime(start);
		std::string filename(basename);
                std::stringstream ss;
                ss << framenum;
                filename.append(ss.str()).append("data.bin");

		//printf("Reading data\n");
                std::ifstream in(filename.c_str(), std::ios::binary | std::ios::in);

		// offset read in when geometry constructed
                in.seekg(offset);

		if(buffer)
		{
		    in.read((char*)buffer, sizeof(float) * 3 * numVertexs);
		}
		else
		{
		    // get part number and number of vertices from file, calc fileoffset first
		    in.read((char*)vertices, sizeof(float) * 3 * numVertexs);
		}

                in.close();

		buffernum = framenum;

		getTime(end);
		loadTime = getDiff(start, end);
		dataSize = sizeof(float) * 3 * numVertexs;
	//}
	_statusMutex.lock();
	if(updateStatus)
	{
	    //std::cerr << "Part finish load " << getPartNumber() << std::endl;
	    if(framenum == _frame)
	    {
		//std::cerr << "Setting frame done." << std::endl;
		_frameStatus = LOADED;
	    }
	    else if(framenum == _nextFrame)
	    {
		//std::cerr << "Setting next frame done." << std::endl;
		_nextFrameStatus = LOADED;
	    }
	}
	_statusMutex.unlock();
	_mutex.unlock();
}

// indicating what frame to copy next
void Geometry::SetFrameToCopy(int framenum)
{
	newframe = framenum;
}

void Geometry::setFrameNumbers(int context, int thisFrame, int nextFrame)
{
    _statusMutex.lock();

    setFrameNumber(context,thisFrame);
    setNextFrameNumber(nextFrame);

    _statusMutex.unlock();
}

void Geometry::setFrameNumber(int context, int fnum)
{
    //std::cerr << "Set frame part " << getPartNumber() << " this " << _frame << " next " << _nextFrame << " in " << fnum << " context " << context << std::endl;
    if(fnum == _frame)
    {
	return;
    }


    if(fnum == _nextFrame)
    {
	//std::cerr << "SwapFrames." << std::endl;
	swapFrames(context);
    }
    else
    {
	_frameStatus = NOT_LOADING;
    }

    _frame = fnum;
}

void Geometry::setNextFrameNumber(int fnum)
{
    if(fnum == _nextFrame)
    {
	return;
    }

    _nextFrameStatus = NOT_LOADING;

    _nextFrame = fnum;
}

void Geometry::SetFrame(int context, bool predraw)
{
	// add metrix to determine time for copy
	//timeval start, end;
	//getTime(start);

	// only copy buffer data in if different
	//if(frame != newframe)
	//{
	    if(predraw)
	    {
		//std::cerr << "Predraw SetFrame part: " << getPartNumber() << std::endl;
		_statusMutex.lock();
		if(_frameStatus == NOT_LOADING)
		{
		    _statusMutex.unlock();
		    std::cerr << "Next Frame is not preloading!?!? part: " << getPartNumber() << std::endl;
		    SetBuffer(_frame, _bufferMan->requestBuffer(this));
		}
		_statusMutex.unlock();

		//std::cerr << "waiting for part load." << std::endl;
		while(1)
		{
		    _statusMutex.lock();

		    if(_frameStatus == LOADED)
		    {
			_statusMutex.unlock();
			break;
		    }
		    _statusMutex.unlock();
		    //std::cerr << "FrameStatus: " << _frameStatus << std::endl;
		    //sleep(1);
		}
		//std::cerr << "load done." << std::endl;
	    }
	    else
	    {
		SetBuffer(_frame, _bufferMan0->requestBuffer(this), _bufferMan == _bufferMan0);
		_bufferMan0->loadFrameData(false);
	    }

		// set current frame
		//frame = newframe;

		/*
        	// loaddata into buffers
		if( context )
		    glBindBufferARB(GL_ARRAY_BUFFER_ARB, vboId[0]);
		else
		    glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo0Id[0]);

        	glBufferSubDataARB(GL_ARRAY_BUFFER_ARB, 0, sizeof(float) * 3 * numVertexs, (void*)vertices );
		glBindBuffer(GL_ARRAY_BUFFER_ARB,  0);

		getTime(end);
		copyTime = getDiff(start, end);

		// since copy is completed add next frame to queue
		FetchQueue::getInstance()->addRequest(frame + 1, this);	
		*/
		//totalNumGeometry++;	
	//}
}

void Geometry::SetFrame(int context, cudaStream_t & stream, bool predraw)
{
	// add metrix to determine time for copy
	/*timeval start, end;
	getTime(start);

	// only copy buffer data in if different
	if(frame != newframe)
	{
		SetBuffer(newframe);

		// set current frame
		frame = newframe;

        	// loaddata into buffers
		//long starttime = getTime();
		if(numVertexs)
		{
		    char * vert;
		    if(context)
		    {
			//checkMapBufferObjAsync((void**)&vert,vboId[0],stream);
			checkMapBufferObj((void**)&vert,vboId[0]);
		    }
		    else
		    {
			checkMapBufferObj((void**)&vert,vbo0Id[0]);
		    }

		    cudaMemcpyAsync(vert, vertices, sizeof(float) * 3 * numVertexs, cudaMemcpyHostToDevice);

		    if(context)
		    {
			//checkUnmapBufferObjAsync(vboId[0],stream);
			checkUnmapBufferObj(vboId[0]);
		    }
		    else
		    {
			checkUnmapBufferObj(vbo0Id[0]);
		    }
		    cudaThreadSynchronize();

		}
        	//glBindBufferARB(GL_ARRAY_BUFFER_ARB, vboId[0]);
        	//glBufferSubDataARB(GL_ARRAY_BUFFER_ARB, 0, sizeof(float) * 3 * numVertexs, (void*)vertices );
		//totalCopyTime += timeDiff(starttime, getTime());

		getTime(end);
		copyTime = getDiff(start, end);

		// since copy is completed add next frame to queue
		FetchQueue::getInstance()->addRequest(frame + 1, this);	

		//totalNumGeometry++;	
	}*/
}

void Geometry::setBufferManager0(BufferManager * bufferManager)
{
    _bufferMan0 = bufferManager;
}

void Geometry::setBufferManager(BufferManager * bufferManager)
{
    _bufferMan = bufferManager;
}

void Geometry::setVertBuffer(int context, GLuint buffer)
{
    if(context)
    {
	vboId[0] = buffer;
    }
    else
    {
	vbo0Id[0] = buffer;
    }
}

void Geometry::startNextFetch()
{
    _statusMutex.lock();
    //std::cerr << "Start next fetch frame: " << _frame << " next: " << _nextFrame << std::endl;
    if(_nextFrameStatus == NOT_LOADING)
    {
	//std::cerr << "Starting to load next part: " << getPartNumber() << std::endl;
	FetchQueue::getInstance()->addRequest(_nextFrame, this, _bufferMan->requestNextBuffer(this));
	_nextFrameStatus = LOADING;
    }
    else
    {
	//std::cerr << "next part " << getPartNumber() << " is already loading." << std::endl;
    }
    _statusMutex.unlock();
}

void Geometry::swapFrames(int context)
{
    if(context)
    {
	vboId[0] = _nextVertVBO;
	_vboOffset = _nextVBOOffset;
    }
    else
    {
	//std::cerr << "Swap part " << getPartNumber() << " this " << vbo0Id[0] << " next " << _nextVertVBO << " offset this " << _vbo0Offset << " next " << _nextVBOOffset << std::endl;
	vbo0Id[0] = _nextVertVBO;
	_vbo0Offset = _nextVBOOffset;
    }
    _frameStatus = _nextFrameStatus;
}

void Geometry::processPostDrawn(bool paused)
{
    if(!_postDrawn)
    {
	return;
    }

    //std::cerr << "Doing thread post drawn. part: " << getPartNumber() << std::endl;
    
    if(paused)
    {
	if(_bufferMan != _bufferMan0)
	{
	    //memcpy context zero data into correct buffer
	    memcpy(_bufferMan->requestBuffer(this),_bufferMan0->getBuffer0Pointer(this),sizeof(float) * 3 * numVertexs);
	    _statusMutex.lock();
	    _frameStatus = LOADED;
	    _statusMutex.unlock();
	}
    }

    //std::cerr << "I am ... " << isOn() << std::endl;

    startNextFetch();

    _postDrawn = false;
}
