/*
 * Description: MultiGPURenderer - Class that handles the parallel draw over 
 *              multiple video cards and combines them into a single image
 */
/**
 * @file MultiGPURenderer.cpp
 * @author Andrew Prudhomme (aprudhomme@ucsd.edu)
 * @date 09/15/2010
 */

#include <GL/glew.h>
#include "MultiGPURenderer.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/CVRViewer.h>
#include <OpenThreads/ScopedLock>

#include "GLHelper.h"
#include "CudaHelper.h"
#include "Timing.h"

#include <iostream>
#include <string>
#include <sstream>

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xresource.h>

/// macro to cast an int to a pointer
#define BUFFER_OFFSET(i) ((char *)NULL + (i))

/// max colors to use in shader
#define MAX_COLORS 32

// does a simple draw test instead of drawing parts
//#define SIMPLE_TEST

/// in case not defined
#ifndef GL_GEOMETRY_SHADER
#define GL_GEOMETRY_SHADER 0x8DD9
#endif

#ifndef GL_TEXTURE_RECTANGLE
#define GL_TEXTURE_RECTANGLE 0x84F5
#endif

using namespace cvr;

/// function that takes a value 0-1 and returns a color
struct ColorVal makeColor(float f)
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

    //std::cerr << "Color x: " << color.x() << " y: " << color.y() << " z: " << color.z() << std::endl;

    return color;
}

/// debug draw function, draws a quad of different size, location and 
/// color in each gpu
void testDraw(int gpu)
{
    glDisable(GL_LIGHTING);
    if(gpu == 1)
    {
	glBegin(GL_QUADS);
	    glNormal3f(0,-1,0);
	    glColor3f(0.7,0.0,0.0);
	    glVertex3f(-200,0,100);
	    glNormal3f(0,-1,0);
	    glColor3f(0.7,0.0,0.0);
	    glVertex3f(200,0,100);
	    glNormal3f(0,-1,0);
	    glColor3f(0.7,0.0,0.0);
	    glVertex3f(200,0,-100);
	    glNormal3f(0,-1,0);
	    glColor3f(0.7,0.0,0.0);
	    glVertex3f(-200,0,-100);
	glEnd();
    }
    else if(gpu == 0)
    {
	glBegin(GL_QUADS);
	    glNormal3f(0,-1,0);
	    glColor3f(0.5,0.1,0.0);
	    glVertex3f(200,0,100);
	    glNormal3f(0,-1,0);
	    glColor3f(0.5,0.1,0.0);
	    glVertex3f(400,0,100);
	    glNormal3f(0,-1,0);
	    glColor3f(0.5,0.1,0.0);
	    glVertex3f(400,0,-100);
	    glNormal3f(0,-1,0);
	    glColor3f(0.5,0.1,0.0);
	    glVertex3f(200,0,-100);
	glEnd();
    }
    else if(gpu == 2)
    {
	glBegin(GL_QUADS);
	    glNormal3f(0,-1,0);
	    glColor3f(0.2,0.1,0.0);
	    glVertex3f(200,0,300);
	    glNormal3f(0,-1,0);
	    glColor3f(0.2,0.1,0.0);
	    glVertex3f(400,0,300);
	    glNormal3f(0,-1,0);
	    glColor3f(0.2,0.1,0.0);
	    glVertex3f(400,0,100);
	    glNormal3f(0,-1,0);
	    glColor3f(0.2,0.1,0.0);
	    glVertex3f(200,0,100);
	glEnd();
    }
    else if(gpu == 3)
    {
	glBegin(GL_QUADS);
	    glNormal3f(0,-1,0);
	    glColor3f(0.0,0.0,0.0);
	    glVertex3f(-200,0,250);
	    glNormal3f(0,-1,0);
	    glColor3f(0.0,0.0,0.0);
	    glVertex3f(200,0,250);
	    glNormal3f(0,-1,0);
	    glColor3f(0.0,0.0,0.0);
	    glVertex3f(200,0,100);
	    glNormal3f(0,-1,0);
	    glColor3f(0.0,0.0,0.0);
	    glVertex3f(-200,0,100);
	glEnd();
    }
    glEnable(GL_LIGHTING);
}

/**
 * Init class, determine number of gpus used to rendering
 * 
 * @param width width of graphics window
 * @param height height of graphics window
 * @param geoShader set to use geometry shader for normals
 * @param copyType set how to perform texture copy for multi-gpu
 * @param dbits set how many bits to use for depth textures
 */
MultiGPURenderer::MultiGPURenderer(int width, int height, bool geoShader, TextureCopyType copyType, DepthBits dbits)
{
    _width = width;
    _height = height;
    _useGeoShader = geoShader;
    //_usePBOs = pbos;
    //_cudaCopy = cudaCopy;
    //_cudaCopy = true;
    switch(copyType)
    {
	case READ_PIX:
	    _usePBOs = false;
	    _cudaCopy = false;
	    break;
	case PBOS:
	    _usePBOs = true;
	    _cudaCopy = false;
	    break;
	case CUDA_COPY:
	    if(dbits == D32)
	    {
		std::cerr << "Unable to use cuda texture copy with 32 bit depth buffer, switching to read pixels copy." << std::endl;
		_cudaCopy = false;
	    }
	    else
	    {
		_cudaCopy = true;
	    }
	    _usePBOs = false;
	    std::cerr << "Using cuda texture copy." << std::endl;
	    break;
    }

    _depth = dbits;

    // look in config file for number of gpus to use
    _numGPUs = ConfigManager::getInt("Plugin.CullingMultiGPURender.NumberOfGPUs",0);
    _shaderDir = ConfigManager::getEntry("Plugin.CullingMultiGPURender.ShaderDir");

    // use X library to determine the number of X screens, use this for number of gpus
    if(!_numGPUs)
    {
	std::cerr << "Finding number of X Screens." << std::endl;
	Display *dpy;

	dpy = XOpenDisplay(NULL);
	if(!dpy && !(dpy = XOpenDisplay(":0.0")))
	{
	    std::cerr << "Unable to open X display." << std::endl;
	    _numGPUs = 0;
	}
	else
	{
	    _numGPUs = XScreenCount(dpy);
	}
	std::cerr << "Found " << _numGPUs << " screen(s)." << std::endl;
    }

    threadSyncBlock = new bool[_numGPUs];
    for(int i = 0; i < _numGPUs; i++)
    {
	threadSyncBlock[i] = false;
	_init[i] = false;
    }
    _errorState = false;
    _partMap = NULL;
    _updateColors = false;
    
}

/**
 * clean up class resources
 */
MultiGPURenderer::~MultiGPURenderer()
{
    // TODO: add cleanup
}

/**
 * Does draw operation for given gpu, copies textures and combines
 * them into final image.
 *
 * @param gpu gpu to draw
 * @param cam used to gather timing data
 */
void MultiGPURenderer::draw(int gpu, osg::Camera * cam)
{

    // skip draw if gpu is not used or global parts map is not set
#ifndef SIMPLE_TEST
    if(!_partMap || gpu >= _numGPUs)
#else
    if(gpu >= _numGPUs)
#endif
    {
	return;
    }

    //std::cerr << "MGR draw gpu: " << gpu << std::endl;

    // determine if stats can be collected
    osg::Stats * stats = NULL;
    if(cam)
    {
	stats = cam->getStats();
    }

    if(stats && !stats->collectStats("mgpu"))
    {
	stats = NULL;
    }

    double drawStart, drawEnd;

    if(stats)
    {
	glFinish();
	drawStart = osg::Timer::instance()->delta_s(CVRViewer::instance()->getStartTick(), osg::Timer::instance()->tick());
    }

    std::stringstream contextss;
    contextss << gpu << ": ";

    // init opengl objects during first draw
    if(!_init[gpu])
    {
	// use lock to serialize buffer initialization
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_initMutex);
	if(!initBuffers(gpu) || !initShaders(gpu))
	{
	    std::cerr << "Error init for gpu: " << gpu << std::endl;
	    _errorState = true;
	}
	_init[gpu] = true;
    }

    if(_errorState)
    {
	return;
    }

#ifdef PRINT_TIMING

    glFinish();
    struct timeval vboDrawStart, vboDrawEnd;
    getTime(vboDrawStart);

#endif

    // bind the gpus frame buffer object
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, _frameBufferMap[gpu]);

    GLenum buffers[] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_COLOR_ATTACHMENT2_EXT};
    GLenum buffer = GL_COLOR_ATTACHMENT0_EXT;

    // set which fbo textures to draw to in the shader
    // the first is for the color, any others are used for custom depth textures
    if(_depth == D16)
    {
        glDrawBuffers(2, buffers);
    }
    else if(_depth == D24)
    {
        glDrawBuffers(3, buffers);
    }

    // clear textures
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // bind draw shader and do draw for gpu
    glUseProgram(_drawProg[gpu]);
    {
#ifndef SIMPLE_TEST
	drawGPU(gpu);
#else
	testDraw(gpu);
#endif

    }
    glUseProgram(0);

    // reset to standard draw buffer
    glDrawBuffers(1, &buffer);

    glFinish();
#ifdef PRINT_TIMING
    getTime(vboDrawEnd);
    printDiff(contextss.str() + "VBO Draw: ",vboDrawStart,vboDrawEnd);
#endif

    if(stats)
    {
	drawEnd = osg::Timer::instance()->delta_s(CVRViewer::instance()->getStartTick(), osg::Timer::instance()->tick());
	stats->setAttribute(CVRViewer::instance()->getViewerFrameStamp()->getFrameNumber(), "MDraw traversal begin time", drawStart);
        stats->setAttribute(CVRViewer::instance()->getViewerFrameStamp()->getFrameNumber(), "MDraw traversal end time", drawEnd);
        stats->setAttribute(CVRViewer::instance()->getViewerFrameStamp()->getFrameNumber(), "MDraw traversal time taken", drawEnd-drawStart);
    }

    // all non primary gpus copy their textures to memory, primary gpu copies these from memory
    if(gpu != 0)
    {

#ifdef PRINT_TIMING
	glFinish();
	timeval sstore,estore;
	getTime(sstore);
#endif

	double copyDownStart, copyDownEnd;

	if(stats)
	{
	    copyDownStart = osg::Timer::instance()->delta_s(CVRViewer::instance()->getStartTick(), osg::Timer::instance()->tick());
	}

	if(!_cudaCopy)
	{
	    glPixelStorei(GL_PACK_ALIGNMENT,1);

	    // set read buffer to fbo color texture
	    glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
	}

	// read color texture to memory using pixel buffer or standard read call
	if(_usePBOs)
	{
	    // bind pbo
	    glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, _colorBufferMap[gpu]);
	    // read pixels to pbo
	    glReadPixels(0,0,_width,_height,GL_RG_INTEGER,GL_UNSIGNED_BYTE,BUFFER_OFFSET(0)); 

	    // map pbo and copy to memeory
	    void * mapped = glMapBuffer(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY);
	    memcpy(_colorDataMap[gpu],mapped,_width*_height*2);

	    // unmap and unbind
	    glUnmapBuffer(GL_PIXEL_PACK_BUFFER_ARB);
	    glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB,0);
	}
	else if(_cudaCopy)
	{
	    _cudaColorImage[gpu]->setMapFlags(cudaGraphicsMapFlagsReadOnly);
	    _cudaColorImage[gpu]->map();
	    cudaMemcpyFromArrayAsync(_colorDataMap[gpu],_cudaColorImage[gpu]->getPointer(),0,0,_width*_height*2,cudaMemcpyDeviceToHost);
	    _cudaColorImage[gpu]->unmap();
	}
	else
	{
	    // read pixels to memory
	    glReadPixels(0,0,_width,_height,GL_RG_INTEGER,GL_UNSIGNED_BYTE,_colorDataMap[gpu]);
	}

	// do a similar read for depth texture(s) depending on number of bits used
	if(_depth == D32)
	{
	    if(_usePBOs)
	    {
		glReadBuffer(GL_DEPTH_ATTACHMENT_EXT);
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, _depthBufferMap[gpu]);
		glReadPixels(0,0,_width,_height,GL_DEPTH_COMPONENT,GL_FLOAT,BUFFER_OFFSET(0));

		void * mapped = glMapBuffer(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY);
		memcpy(_depthDataMap[gpu],mapped,_width*_height*4);
		glUnmapBuffer(GL_PIXEL_PACK_BUFFER_ARB);
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB,0);
	    }
	    else
	    {
		glReadPixels(0,0,_width,_height,GL_DEPTH_COMPONENT,GL_FLOAT,_depthDataMap[gpu]);
	    }
	}
	else if(_depth == D16)
	{
	    if(!_cudaCopy)
	    {
		glReadBuffer(GL_COLOR_ATTACHMENT1_EXT);
	    }

	    if(_usePBOs)
	    {
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, _depthBufferMap[gpu]);
		glReadPixels(0,0,_width,_height,GL_RED_INTEGER,GL_UNSIGNED_SHORT,BUFFER_OFFSET(0));

		void * mapped = glMapBuffer(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY);
		memcpy(_depthDataMap[gpu],mapped,_width*_height*2);
		glUnmapBuffer(GL_PIXEL_PACK_BUFFER_ARB);
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB,0);
	    }
	    else if(_cudaCopy)
	    {
		_cudaDepth16Image[gpu]->setMapFlags(cudaGraphicsMapFlagsReadOnly);
		_cudaDepth16Image[gpu]->map();
		cudaMemcpyFromArrayAsync(_depthDataMap[gpu],_cudaDepth16Image[gpu]->getPointer(),0,0,_width*_height*2,cudaMemcpyDeviceToHost);
		_cudaDepth16Image[gpu]->unmap();
	    }
	    else
	    {
		glReadPixels(0,0,_width,_height,GL_RED_INTEGER,GL_UNSIGNED_SHORT,_depthDataMap[gpu]);
	    }
	}
	else if(_depth == D24)
	{
	    if(!_cudaCopy)
	    {
		glReadBuffer(GL_COLOR_ATTACHMENT1_EXT);
	    }

	    if(_usePBOs)
	    {
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, _depthBufferMap[gpu]);
		glReadPixels(0,0,_width,_height,GL_RED_INTEGER,GL_UNSIGNED_SHORT,BUFFER_OFFSET(0));

		void * mapped = glMapBuffer(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY);
		memcpy(_depthDataMap[gpu],mapped,_width*_height*2);
		glUnmapBuffer(GL_PIXEL_PACK_BUFFER_ARB);
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB,0);
	    }
	    else if(_cudaCopy)
	    {
		_cudaDepth16Image[gpu]->setMapFlags(cudaGraphicsMapFlagsReadOnly);
		_cudaDepth16Image[gpu]->map();
		cudaMemcpyFromArrayAsync(_depthDataMap[gpu],_cudaDepth16Image[gpu]->getPointer(),0,0,_width*_height*2,cudaMemcpyDeviceToHost);
		_cudaDepth16Image[gpu]->unmap();
	    }
	    else
	    {
		glReadPixels(0,0,_width,_height,GL_RED_INTEGER,GL_UNSIGNED_SHORT,_depthDataMap[gpu]);
	    }
	    
	    if(!_cudaCopy)
	    {
		glReadBuffer(GL_COLOR_ATTACHMENT2_EXT);
	    }

	    if(_usePBOs)
	    {
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, _depth8BufferMap[gpu]);
		glReadPixels(0,0,_width,_height,GL_RED_INTEGER,GL_UNSIGNED_BYTE,BUFFER_OFFSET(0));

		void * mapped = glMapBuffer(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY);
		memcpy(_depth8DataMap[gpu],mapped,_width*_height);
		glUnmapBuffer(GL_PIXEL_PACK_BUFFER_ARB);
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB,0);
	    }
	    else if(_cudaCopy)
	    {
		_cudaDepth8Image[gpu]->setMapFlags(cudaGraphicsMapFlagsReadOnly);
		_cudaDepth8Image[gpu]->map();
		cudaMemcpyFromArrayAsync(_depth8DataMap[gpu],_cudaDepth8Image[gpu]->getPointer(),0,0,_width*_height,cudaMemcpyDeviceToHost);
		_cudaDepth8Image[gpu]->unmap();
	    }
	    else
	    {
		glReadPixels(0,0,_width,_height,GL_RED_INTEGER,GL_UNSIGNED_BYTE,_depth8DataMap[gpu]);
	    }
	}


	if(_cudaCopy)
	{
	    cudaThreadSynchronize();
	}

	glFinish();
#ifdef PRINT_TIMING
	getTime(estore);
	printDiff(contextss.str() + "Store Time: ",sstore,estore);
#endif

	if(stats)
	{
	    std::stringstream ss;
	    ss << "CopyDown" << gpu;
	    copyDownEnd = osg::Timer::instance()->delta_s(CVRViewer::instance()->getStartTick(), osg::Timer::instance()->tick());
	    stats->setAttribute(CVRViewer::instance()->getViewerFrameStamp()->getFrameNumber(), ss.str() + " begin time", copyDownStart);
	    stats->setAttribute(CVRViewer::instance()->getViewerFrameStamp()->getFrameNumber(), ss.str() + " end time", copyDownEnd);
	    stats->setAttribute(CVRViewer::instance()->getViewerFrameStamp()->getFrameNumber(), ss.str() + " time taken", copyDownEnd-copyDownStart);
	}

	// signal primary gpu that copy is done
	threadSyncBlock[gpu] = true;
    }
    else
    {
	// update shader color list if needed
	if(_updateColors)
	{
	    updateColors();
	    _updateColors = false;
	}

	// wait for other threads to finish copy
	
	bool copyDone[_numGPUs];
	for(int i = 1; i < _numGPUs; i++)
	{
	    copyDone[i] = false;
	}

	glPixelStorei(GL_PACK_ALIGNMENT,1);
	// loop until all other gpu textures are copied
	while(1)
	{
	    // check if finished
	    bool syncDone = true;
	    for(int i = 1; i < _numGPUs; i++)
	    {
		if(!copyDone[i])
		{
		    syncDone = false;
		    break;
		}
	    }
	    if(syncDone)
	    {
		//std::cerr << "Sync Done." << std::endl;
		break;
	    }

	    // for each non primary gpu
	    for(int i = 1; i < _numGPUs; i++)
	    {
		// if it has not already been copied and it has signaled it is done
		if(!copyDone[i] && threadSyncBlock[i])
		{
#ifdef PRINT_TIMING
		    glFinish();
		    struct timeval cbstart,cbend;
		    getTime(cbstart);
#endif
		    double copyBackStart, copyBackEnd;
		    if(stats)
		    {
			if(!_cudaCopy)
			{
			    glFinish();
			}
			copyBackStart = osg::Timer::instance()->delta_s(CVRViewer::instance()->getStartTick(), osg::Timer::instance()->tick());
		    }

		    if(!_cudaCopy)
		    {
			// bind texture to copy color data to
			glBindTexture(GL_TEXTURE_RECTANGLE_ARB,_colorCopyTextureMap[i]);
		    }

		    // copy texture from memeory using pixel buffer or not
		    if(_usePBOs)
		    {
			// bind pbo
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, _colorCopyBufferMap[i]);

			// map buffer and copy data
			void* buffer = glMapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY); 
			memcpy(buffer,_colorDataMap[i],_width*_height*2);

			// unmap and unbind objects
			glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB);
			glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, _width, _height, GL_RG_INTEGER, GL_UNSIGNED_BYTE, BUFFER_OFFSET(0));
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		    }
		    else if(_cudaCopy)
		    {
			_cudaCBColorImage[i]->setMapFlags(cudaGraphicsMapFlagsWriteDiscard);
			_cudaCBColorImage[i]->map();
			cudaMemcpyToArrayAsync(_cudaCBColorImage[i]->getPointer(),0,0,_colorDataMap[i],_width*_height*2,cudaMemcpyHostToDevice);
			_cudaCBColorImage[i]->unmap();
		    }
		    else
		    {
			// opengl texture load command
			glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, _width, _height, GL_RG_INTEGER, GL_UNSIGNED_BYTE, _colorDataMap[i]);
		    }

		    // do similar load for depth texture(s)
		    if(_depth == D32)
		    {
			if(!_cudaCopy)
			{
			    glBindTexture(GL_TEXTURE_RECTANGLE_ARB,_depthCopyTextureMap[i]);
			}

			if(_usePBOs)
			{
			    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, _depthCopyBufferMap[i]);
			    void* buffer = glMapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY); 
			    memcpy(buffer,_depthDataMap[i],_width*_height*4);
			    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB);
			    glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, _width, _height, GL_DEPTH_COMPONENT, GL_FLOAT, BUFFER_OFFSET(0));
			    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
			}
			else
			{
			    glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, _width, _height, GL_DEPTH_COMPONENT, GL_FLOAT, _depthDataMap[i]);
			}
		    }
		    else if(_depth == D16)
		    {
			if(!_cudaCopy)
			{
			    glBindTexture(GL_TEXTURE_RECTANGLE_ARB,_depthCopyTextureMap[i]);
			}
			
			if(_usePBOs)
			{
			    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, _depthCopyBufferMap[i]);
			    void* buffer = glMapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY); 
			    memcpy(buffer,_depthDataMap[i],_width*_height*2);
			    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB);
			    glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, _width, _height, GL_RED_INTEGER, GL_UNSIGNED_SHORT, BUFFER_OFFSET(0));
			    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
			}
			else if(_cudaCopy)
			{
			    _cudaCBDepth16Image[i]->setMapFlags(cudaGraphicsMapFlagsWriteDiscard);
			    _cudaCBDepth16Image[i]->map();
			    cudaMemcpyToArrayAsync(_cudaCBDepth16Image[i]->getPointer(),0,0,_depthDataMap[i],_width*_height*2,cudaMemcpyHostToDevice);
			    _cudaCBDepth16Image[i]->unmap();
			}
			else
			{
			    glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, _width, _height, GL_RED_INTEGER, GL_UNSIGNED_SHORT, _depthDataMap[i]);
			}
		    }
		    else if(_depth == D24)
		    {
			if(!_cudaCopy)
			{
			    glBindTexture(GL_TEXTURE_RECTANGLE_ARB,_depthCopyTextureMap[i]);\
			}

			if(_usePBOs)
			{
			    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, _depthCopyBufferMap[i]);
			    void* buffer = glMapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY); 
			    memcpy(buffer,_depthDataMap[i],_width*_height*2);
			    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB);
			    glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, _width, _height, GL_RED_INTEGER, GL_UNSIGNED_SHORT, BUFFER_OFFSET(0));
			    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
			}
			else if(_cudaCopy)
			{
			    _cudaCBDepth16Image[i]->setMapFlags(cudaGraphicsMapFlagsWriteDiscard);
			    _cudaCBDepth16Image[i]->map();
			    cudaMemcpyToArrayAsync(_cudaCBDepth16Image[i]->getPointer(),0,0,_depthDataMap[i],_width*_height*2,cudaMemcpyHostToDevice);
			    _cudaCBDepth16Image[i]->unmap();
			}
			else
			{
			    glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, _width, _height, GL_RED_INTEGER, GL_UNSIGNED_SHORT, _depthDataMap[i]);
			}

			if(!_cudaCopy)
			{
			    glBindTexture(GL_TEXTURE_RECTANGLE_ARB,_depth8CopyTextureMap[i]);
			}

			if(_usePBOs)
			{
			    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, _depth8CopyBufferMap[i]);
			    void* buffer = glMapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY); 
			    memcpy(buffer,_depth8DataMap[i],_width*_height);
			    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB);
			    glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, _width, _height, GL_RED_INTEGER, GL_UNSIGNED_BYTE, BUFFER_OFFSET(0));
			    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
			}
			else if(_cudaCopy)
			{
			    _cudaCBDepth8Image[i]->setMapFlags(cudaGraphicsMapFlagsWriteDiscard);
			    _cudaCBDepth8Image[i]->map();
			    cudaMemcpyToArrayAsync(_cudaCBDepth8Image[i]->getPointer(),0,0,_depth8DataMap[i],_width*_height,cudaMemcpyHostToDevice);
			    _cudaCBDepth8Image[i]->unmap();
			}
			else
			{
			    glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, _width, _height, GL_RED_INTEGER, GL_UNSIGNED_BYTE, _depth8DataMap[i]);
			}
		    }

		    if(!_cudaCopy)
		    {
			glBindTexture(GL_TEXTURE_RECTANGLE_ARB,0);
		    }

		    // flag this gpu as copied to primary gpu
		    threadSyncBlock[i] = false;
		    copyDone[i] = true;

		    if(stats)
		    {
			if(!_cudaCopy)
			{
			    glFinish();
			}
			std::stringstream ss;
			ss << "CopyBack" << i;
			copyBackEnd = osg::Timer::instance()->delta_s(CVRViewer::instance()->getStartTick(), osg::Timer::instance()->tick());
			stats->setAttribute(CVRViewer::instance()->getViewerFrameStamp()->getFrameNumber(), ss.str() + " begin time", copyBackStart);
			stats->setAttribute(CVRViewer::instance()->getViewerFrameStamp()->getFrameNumber(), ss.str() + " end time", copyBackEnd);
			stats->setAttribute(CVRViewer::instance()->getViewerFrameStamp()->getFrameNumber(), ss.str() + " time taken", copyBackEnd-copyBackStart);
		    }

#ifdef PRINT_TIMING
		    glFinish();
		    getTime(cbend);
		    std::stringstream cbss;
		    cbss << i << ": ";
		    printDiff(cbss.str() + "CopyBackTime: ", cbstart,cbend);
#endif
		}
	    }
	    //std::cerr << "Sync not Done." << std::endl;
	    //OpenThreads::Thread::YieldCurrentThread();
	}

	if(_cudaCopy)
	{
	    cudaThreadSynchronize();
	}

#ifdef PRINT_TIMING
	struct timeval shaderstart,shaderend;
	getTime(shaderstart);
#endif

	double shaderStart, shaderEnd;

	if(stats)
	{
	    glFinish();
	    shaderStart = osg::Timer::instance()->delta_s(CVRViewer::instance()->getStartTick(), osg::Timer::instance()->tick());
	}

	// bind default frame buffer
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,0);

	// draw screen filling quad with shader to combine all rendered images
	drawScreen();

#ifdef PRINT_TIMING
	glFinish();
	getTime(shaderend);
	printDiff(contextss.str() + "shader time: ",shaderstart,shaderend);
#endif
	if(stats)
	{
	    glFinish();
	    shaderEnd = osg::Timer::instance()->delta_s(CVRViewer::instance()->getStartTick(), osg::Timer::instance()->tick());
	    stats->setAttribute(CVRViewer::instance()->getViewerFrameStamp()->getFrameNumber(), "Shader begin time", shaderStart);
	    stats->setAttribute(CVRViewer::instance()->getViewerFrameStamp()->getFrameNumber(), "Shader end time", shaderEnd);
	    stats->setAttribute(CVRViewer::instance()->getViewerFrameStamp()->getFrameNumber(), "Shader time taken", shaderEnd-shaderStart);
	}
    }

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,0);

    if(stats)
    {
	glFinish();	
    }

}

/// @return number of gpus being used by parallel rendering
int MultiGPURenderer::getNumGPUs()
{
    return _numGPUs;
}

// set global parts map for renderer
/// @param pmap pointer to map of part number to part object
void MultiGPURenderer::setPartMap(std::map<int,PartInfo*> * pmap)
{
    _partMap = pmap;
}

// set list of parts drawn by a gpu
/// @param gpu gpu/context id number
/// @param plist list of part numbers assigned to that gpu
void MultiGPURenderer::setPartList(int gpu, std::vector<int> plist)
{
    _partList[gpu] = plist;
}

// set a new color mapping for the final combination shader
/// @param colors reference to color list to use for mapping in the shader
void MultiGPURenderer::setColorMapping(std::vector<ColorVal> & colors)
{
    _colorMap = colors;
    _updateColors = true;
}

/**
 * @param gpu graphics card to init opengl buffers for
 * @return true if no error occurs
 */
 /* Description: Creates and initializes the opengl objects used for
 *              parallel rendering
 */
bool MultiGPURenderer::initBuffers(int gpu)
{
    glewInit();

    // generate textures for frame buffer objects
    GLuint buffer;
    glGenFramebuffersEXT(1,&buffer);
    _frameBufferMap[gpu] = buffer;

    glGenTextures(1,&buffer);
    _colorTextureMap[gpu] = buffer;

    glGenTextures(1,&buffer);
    _depthTextureMap[gpu] = buffer;

    if(_depth < D32)
    {
	glGenTextures(1,&buffer);
	_depth16TextureMap[gpu] = buffer;
    }
    if(_depth == D24)
    {
	glGenTextures(1,&buffer);
	_depth8TextureMap[gpu] = buffer;
    }

    //std::cerr << "frame: " << _frameBufferMap[context] << " color: " << _colorTextureMap[context] << " depth: " << _depthTextureMap[context] << std::endl;

    // init depth texture
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB,_depthTextureMap[gpu]);
    glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_DEPTH_COMPONENT32, _width, _height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
    //glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB,0);


    // init color texture
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, _colorTextureMap[gpu]);
    glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RG8UI, _width, _height, 0, GL_RG_INTEGER, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);

    if(_cudaCopy && gpu)
    {
	_cudaColorImage[gpu] = new CudaGLImage(_colorTextureMap[gpu],GL_TEXTURE_RECTANGLE);
	_cudaColorImage[gpu]->registerImage(cudaGraphicsMapFlagsNone);
    }

    // init additional depth textures if needed
    if(_depth < D32)
    {
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, _depth16TextureMap[gpu]);
	glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_R16UI, _width, _height, 0, GL_RED_INTEGER, GL_UNSIGNED_SHORT, NULL);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);

	if(_cudaCopy && gpu)
	{
	    _cudaDepth16Image[gpu] = new CudaGLImage(_depth16TextureMap[gpu],GL_TEXTURE_RECTANGLE);
	    _cudaDepth16Image[gpu]->registerImage(cudaGraphicsMapFlagsNone);
	}
    }
    if(_depth == D24)
    {
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, _depth8TextureMap[gpu]);
	glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_R8UI, _width, _height, 0, GL_RED_INTEGER, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);

	if(_cudaCopy && gpu)
	{
	    _cudaDepth8Image[gpu] = new CudaGLImage(_depth8TextureMap[gpu],GL_TEXTURE_RECTANGLE);
	    _cudaDepth8Image[gpu]->registerImage(cudaGraphicsMapFlagsNone);
	}
    }

    // bind textures to frame buffer
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, _frameBufferMap[gpu]);
    
    // attach the required textures to the frame buffer object
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_RECTANGLE_ARB, _colorTextureMap[gpu], 0);
    if(_depth < D32)
    {
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_TEXTURE_RECTANGLE_ARB, _depth16TextureMap[gpu], 0);
    }
    if(_depth == D24)
    {
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT2_EXT, GL_TEXTURE_RECTANGLE_ARB, _depth8TextureMap[gpu], 0);
    }
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_RECTANGLE_ARB, _depthTextureMap[gpu], 0);

    checkFramebuffer();
    // bind default frame buffer
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,0);

    // if not the primary gpu
    if(gpu > 0)
    {
	// if using pixel buffers, set them up
	if(_usePBOs)
	{
	    // generate buffer
	    glGenBuffers(1,&buffer);
	    _colorBufferMap[gpu] = buffer;

	    // init with size info
	    glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, _colorBufferMap[gpu]);
	    glBufferData(GL_PIXEL_PACK_BUFFER_ARB,_width*_height*2,NULL,GL_STREAM_READ);
	    glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);

	    // repeat for depth buffer pbo(s)
	    glGenBuffers(1,&buffer);
	    _depthBufferMap[gpu] = buffer;

	    if(_depth < D32)
	    {
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, _depthBufferMap[gpu]);
		glBufferData(GL_PIXEL_PACK_BUFFER_ARB,_width*_height*2,NULL,GL_STREAM_READ);
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
	    }

	    if(_depth == D32)
	    {
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, _depthBufferMap[gpu]);
		glBufferData(GL_PIXEL_PACK_BUFFER_ARB,_width*_height*4,NULL,GL_STREAM_READ);
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
	    }

	    if(_depth == D24)
	    {
		glGenBuffers(1,&buffer);
		_depth8BufferMap[gpu] = buffer;

		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, _depth8BufferMap[gpu]);
		glBufferData(GL_PIXEL_PACK_BUFFER_ARB,_width*_height,NULL,GL_STREAM_READ);
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
	    }
	}

	// create array in memory to copy textures to for card exchange
	
	if(!_cudaCopy)
	{
	    _colorDataMap[gpu] = new char[_width*_height*sizeof(short)];

	    if(_depth < D32)
	    {
		_depthDataMap[gpu] = new char[_width*_height*sizeof(short)];
	    }
	    else
	    {
		_depthDataMap[gpu] = new char[_width*_height*sizeof(float)];
	    }

	    if(_depth == D24)
	    {
		_depth8DataMap[gpu] = new char[_width*_height];
	    }
	}
	else
	{
	    checkHostAlloc((void**)&_colorDataMap[gpu],_width*_height*sizeof(short),cudaHostAllocPortable | cudaHostAllocWriteCombined);

	    if(_depth < D32)
	    {
		checkHostAlloc((void**)&_depthDataMap[gpu],_width*_height*sizeof(short),cudaHostAllocPortable | cudaHostAllocWriteCombined);
	    }
	    else
	    {
		checkHostAlloc((void**)&_depthDataMap[gpu],_width*_height*sizeof(float),cudaHostAllocPortable | cudaHostAllocWriteCombined);
	    }

	    if(_depth == D24)
	    {
		checkHostAlloc((void**)&_depth8DataMap[gpu],_width*_height,cudaHostAllocPortable | cudaHostAllocWriteCombined);
	    }
	}
    }
    else // if primary gpu
    {
	// setup copy back buffers for each other gpu
	for(int i = 1; i < _numGPUs; i++)
	{
	    // create texture for local depth buffer(s)
	    glGenTextures(1,&buffer);
	    _depthCopyTextureMap[i] = buffer;
	    if(_depth == D32)
	    {
		// init texture and set parameters
		glBindTexture(GL_TEXTURE_RECTANGLE_ARB,_depthCopyTextureMap[i]);
		glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_DEPTH_COMPONENT32, _width, _height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
		//glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE);
	    }
	    else
	    {
		glBindTexture(GL_TEXTURE_RECTANGLE_ARB,_depthCopyTextureMap[i]);
		glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_R16UI, _width, _height, 0, GL_RED_INTEGER, GL_UNSIGNED_SHORT, NULL);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		if(_cudaCopy)
		{
		    _cudaCBDepth16Image[i] = new CudaGLImage(_depthCopyTextureMap[i],GL_TEXTURE_RECTANGLE);
		    _cudaCBDepth16Image[i]->registerImage(cudaGraphicsMapFlagsNone);
		}
	    }

	    // create additional texture if needed
	    if(_depth == D24)
	    {
		glGenTextures(1,&buffer);
		_depth8CopyTextureMap[i] = buffer;
		glBindTexture(GL_TEXTURE_RECTANGLE_ARB,_depth8CopyTextureMap[i]);
		glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_R8UI, _width, _height, 0, GL_RED_INTEGER, GL_UNSIGNED_BYTE, NULL);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		if(_cudaCopy)
		{
		    _cudaCBDepth8Image[i] = new CudaGLImage(_depth8CopyTextureMap[i],GL_TEXTURE_RECTANGLE);
		    _cudaCBDepth8Image[i]->registerImage(cudaGraphicsMapFlagsNone);
		}
	    }


	    glBindTexture(GL_TEXTURE_RECTANGLE_ARB,0);

	    // create local texture for color buffer from other cards
	    glGenTextures(1,&buffer);
	    _colorCopyTextureMap[i] = buffer; 

	    // init texture and set parameters
	    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, _colorCopyTextureMap[i]);
	    glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RG8UI, _width, _height, 0, GL_RG_INTEGER, GL_UNSIGNED_BYTE, NULL);
	    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);

	    if(_cudaCopy)
	    {
		_cudaCBColorImage[i] = new CudaGLImage(_colorCopyTextureMap[i],GL_TEXTURE_RECTANGLE);
		_cudaCBColorImage[i]->registerImage(cudaGraphicsMapFlagsNone);
	    }

	    // create pixel buffers used to copy into these textures, if required
	    if(_usePBOs)
	    {
		// generate needed pbos
		glGenBuffers(1,&buffer);
		_colorCopyBufferMap[i] = buffer;
		glGenBuffers(1,&buffer);
		_depthCopyBufferMap[i] = buffer;
		if(_depth == D24)
		{
		    glGenBuffers(1,&buffer);
		    _depth8CopyBufferMap[i] = buffer;
		}

		// init buffers with proper sizes
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, _colorCopyBufferMap[i]);
		glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB,_width*_height*2, NULL, GL_STREAM_DRAW);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

		if(_depth == D16 || _depth == D24)
		{
		    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, _depthCopyBufferMap[i]);
		    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB,_width*_height*2, NULL, GL_STREAM_DRAW);
		    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		}
		if(_depth == D32)
		{
		    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, _depthCopyBufferMap[i]);
		    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB,_width*_height*4, NULL, GL_STREAM_DRAW);
		    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		}
		if(_depth == D24)
		{
		    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, _depth8CopyBufferMap[i]);
		    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB,_width*_height, NULL, GL_STREAM_DRAW);
		    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		}
	    }
	}

	// create and init vertex buffer for the full screen geometry used in the final combine draw
	glGenBuffers(1,&_screenArray);
	glBindBuffer(GL_ARRAY_BUFFER, _screenArray);
	// these points are in clip coordinates, so the resulting draw is full screen
	float points[8] = {-1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0};

	glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(float), points, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
    }
    return checkGL();
}

/*
 * Function: initShaders(int)
 *
 * Description: Loads and initializes the shader programs used for the draws and 
 *              the final image combination
 */
/**
 * @param gpu gpu to load the shaders for
 * @return true if no error occurs
 */
bool MultiGPURenderer::initShaders(int gpu)
{
    // load the combination shader on primary gpu
    if(gpu == 0)
    {
	// arrays to hold addresses of shader uniform variables
	_redLookupUni = new GLint[32];
	_greenLookupUni = new GLint[32];
	_blueLookupUni = new GLint[32];

	_colorsUni = new GLint[_numGPUs];
	_depthUni = new GLint[_numGPUs];
	_depth8Uni = new GLint[_numGPUs];

	std::stringstream cvert, cfrag;
	cvert << _shaderDir << "/combine.vert";

	// load the correct fragment shader for the number of depth bits
	cfrag << _shaderDir << "/combine";
	if(_numGPUs > 1)
	{
	    cfrag << _numGPUs;
	}

	switch(_depth)
	{
	    case D32:
		cfrag << "-32";
		break;
	    case D24:
		cfrag << "-24";
		break;
	    case D16:
		cfrag << "-16";
		break;
	}

	cfrag << ".frag";

	// helper functions to load shader from file
	createShader(cvert.str(), GL_VERTEX_SHADER, _comVert);
	createShader(cfrag.str(), GL_FRAGMENT_SHADER, _comFrag);

	// create and link shader program
	createProgram(_comProg, _comVert, _comFrag);

	// get address of number of gpus and number of lookup colors varables
	_texturesUni = glGetUniformLocation(_comProg,"textures");
	_numColorsUni = glGetUniformLocation(_comProg,"ncolors");

	// set shaders number of gpu variable
	glUseProgram(_comProg);
	{
	    glUniform1i(_texturesUni,_numGPUs);
	}
	glUseProgram(0);

	// addresses of color texture index variables
	for(int i = 0; i < _numGPUs; i++)
	{
	    std::stringstream ss;
	    ss << "colors[" << i << "]";
	    _colorsUni[i] = glGetUniformLocation(_comProg,ss.str().c_str());
	}

	// get addresses of depth texture index varibles
	for(int i = 0; i < _numGPUs; i++)
	{
	    std::stringstream ss;
	    ss << "depth[" << i << "]";
	    _depthUni[i] = glGetUniformLocation(_comProg,ss.str().c_str());
	}

	if(_depth == D24)
	{
	    for(int i = 0; i < _numGPUs; i++)
	    {
		std::stringstream ss;
		ss << "depthR8[" << i << "]";
		_depth8Uni[i] = glGetUniformLocation(_comProg,ss.str().c_str());
	    }
	}

	// get addresses for color lookup variables
	for(int i = 0; i < 32; i++)
	{
	    std::stringstream ss;
	    ss << "redLookup[" << i << "]";
	    _redLookupUni[i] = glGetUniformLocation(_comProg,ss.str().c_str());
	    std::stringstream ss2;
	    ss2 << "greenLookup[" << i << "]";
	    _greenLookupUni[i] = glGetUniformLocation(_comProg,ss2.str().c_str());
	    std::stringstream ss3;
	    ss3 << "blueLookup[" << i << "]";
	    _blueLookupUni[i] = glGetUniformLocation(_comProg,ss3.str().c_str());
	}

	loadDefaultColors();
    }

    std::string fragFile, geoFile, vertFile;
    
    /*if(_numGPUs == 1)
    {
	fragFile = _shaderDir + "/draw1.frag";
	if(_useGeoShader)
	{
	    vertFile = _shaderDir + "/drawGeo1.vert";
	    geoFile = _shaderDir + "/draw1.geom";
	}
	else
	{
	    vertFile = _shaderDir + "/draw1.vert";
	}
    }*/
    //else
    // determine what shader to use when drawing parts
    {
	if(_useGeoShader)
	{
	    vertFile = _shaderDir + "/drawGeo.vert";
	    geoFile = _shaderDir + "/draw.geom";
	}
	else
	{
	    vertFile = _shaderDir + "/draw.vert";
	}

	if(_depth == D32)
	{
	    fragFile = _shaderDir + "/draw32.frag";
	}
	else if(_depth == D16)
	{
	    fragFile = _shaderDir + "/draw16.frag";
	}
	else if(_depth == D24)
	{
	    fragFile = _shaderDir + "/draw24.frag";
	}
    }

    // helper functions to load shader from file
    createShader(vertFile, GL_VERTEX_SHADER, _drawVert[gpu]);
    createShader(fragFile, GL_FRAGMENT_SHADER, _drawFrag[gpu]);

    // load geometry shader, if needed, and link the program
    if(_useGeoShader)
    {
	createShader(geoFile, GL_GEOMETRY_SHADER, _drawGeo[gpu]);
	return createProgram(_drawProg[gpu], _drawVert[gpu], _drawFrag[gpu], _drawGeo[gpu], GL_TRIANGLES, GL_TRIANGLE_STRIP, 3);
    }
    else
    {
	return createProgram(_drawProg[gpu], _drawVert[gpu], _drawFrag[gpu]);
    }
}

/*
 * Function: drawGPU(int)
 * Description: Draw the parts assigned to this gpu that are currently on
 */
/**
 * @param gpu gpu to draw parts for
 */
void MultiGPURenderer::drawGPU(int gpu)
{
    // save current client state variable to work around a bug in 
    // the osg stat graph viewer
    bool textureOn = glIsEnabled(GL_TEXTURE_COORD_ARRAY);
    bool vertexOn = glIsEnabled(GL_VERTEX_ARRAY);

    // enable the required array types to draw
    if(textureOn)
    {
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    }

    checkGL();

    if(!vertexOn)
    {
	glEnableClientState(GL_VERTEX_ARRAY);
    }

    if(!_useGeoShader)
    {
	glEnableClientState(GL_NORMAL_ARRAY);
    }

    int pdraw = 0, pndraw = 0;

    for(int i = 0; i < _partList[gpu].size(); i++)
    {
	// skip parts that are off
	if(!(*_partMap)[_partList[gpu][i]]->isOn())
	{
	    pndraw++;
	    continue;
	}
	pdraw++;
        //std::cerr << "Part: " << (*_partMap)[_partList[gpu][i]]->getPartNumber() << std::endl;
	
	// set color based on the modded part number
	glColor3f(((float)((*_partMap)[_partList[gpu][i]]->getPartNumber()%29))/28.0,0.0,0.0);
	
	// bind part buffers and draw

	if(!_useGeoShader)
	{
	    glBindBufferARB(GL_ARRAY_BUFFER_ARB,(*_partMap)[_partList[gpu][i]]->getNormalBuffer());
	    glNormalPointer(GL_FLOAT, 0, 0);
	}

	//std::cerr << "Multi drawing part " << (*_partMap)[_partList[gpu][i]]->getPartNumber() << " vbo " << (*_partMap)[_partList[gpu][i]]->getVertBuffer(gpu) << " offset " << (*_partMap)[_partList[gpu][i]]->getVBOOffset(gpu) << std::endl;

	glBindBufferARB(GL_ARRAY_BUFFER_ARB,(*_partMap)[_partList[gpu][i]]->getVertBuffer(gpu));
	glVertexPointer(3, GL_FLOAT, 0, (GLvoid *)(*_partMap)[_partList[gpu][i]]->getVBOOffset(gpu) );

	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, (*_partMap)[_partList[gpu][i]]->getIndBuffer(gpu));

	//std::cerr << "NumPoints: " << (*_partMap)[_partList[gpu][i]]->getNumPoints() << std::endl;
	//std::cerr << "Ind buffer: " << (*_partMap)[_partList[gpu][i]]->getIndBuffer(gpu) << " vert buffer: " << (*_partMap)[_partList[gpu][i]]->getVertBuffer(gpu) << std::endl;
	//std::cerr << "Offset: " << (*_partMap)[_partList[gpu][i]]->getVBOOffset(gpu) << std::endl;

	/*if(glIsEnabled(GL_COLOR_ARRAY) || glIsEnabled(GL_EDGE_FLAG_ARRAY) || glIsEnabled(GL_FOG_COORD_ARRAY) || glIsEnabled(GL_INDEX_ARRAY) || glIsEnabled(GL_NORMAL_ARRAY) || glIsEnabled(GL_SECONDARY_COLOR_ARRAY) || glIsEnabled(GL_TEXTURE_COORD_ARRAY))

	{
	    std::cerr << "Something bad is turned on." << std::endl;
	}*/

	glDrawElements(GL_QUADS, (*_partMap)[_partList[gpu][i]]->getNumPoints(), GL_UNSIGNED_INT, 0);
    }

    //std::cerr << "Parts Drawn: " << pdraw << " Parts not Drawn: " << pndraw << std::endl;

    // reset client state

    if(!_useGeoShader)
    {
	glDisableClientState(GL_NORMAL_ARRAY);
    }

    if(!vertexOn)
    {
	glDisableClientState(GL_VERTEX_ARRAY);
    }

    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);

    if(textureOn)
    {
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    }
}

/**
 * Setup for combination shader, draw screen filling geometry with
 *	shader to get final image
 */
void MultiGPURenderer::drawScreen()
{
    // bind combination program
    glUseProgram(_comProg);
    {
	// go through the color and depth textures and bind them to GL_TEXTURE0 + (i)
	// then set the (i) value into the shader so it can be used for texture lookup
	int index = 1;
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB,_colorTextureMap[0]);
	glUniform1i(_colorsUni[0],0);
	
	for(int i = 1; i < _numGPUs; i++)
	{
	    glActiveTexture(GL_TEXTURE0 + index);
	    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, _colorCopyTextureMap[i]);
	    glUniform1i(_colorsUni[i],index);
	    index++;
	}

	glActiveTexture(GL_TEXTURE0 + index);
	if(_depth == D32)
	{
	    glBindTexture(GL_TEXTURE_RECTANGLE_ARB,_depthTextureMap[0]);
	}
	else
	{
	    glBindTexture(GL_TEXTURE_RECTANGLE_ARB,_depth16TextureMap[0]);
	}
	glUniform1i(_depthUni[0],index);
	index++;

	for(int i = 1; i < _numGPUs; i++)
	{
	    glActiveTexture(GL_TEXTURE0 + index);
	    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, _depthCopyTextureMap[i]);
	    glUniform1i(_depthUni[i],index);
	    index++;
	}


	if(_depth == D24)
	{
	    glActiveTexture(GL_TEXTURE0 + index);
	    glBindTexture(GL_TEXTURE_RECTANGLE_ARB,_depth8TextureMap[0]);
	    glUniform1i(_depth8Uni[0],index);
	    index++;

	    for(int i = 1; i < _numGPUs; i++)
	    {
		glActiveTexture(GL_TEXTURE0 + index);
		glBindTexture(GL_TEXTURE_RECTANGLE_ARB, _depth8CopyTextureMap[i]);
		glUniform1i(_depth8Uni[i],index);
		index++;
	    }
	}

	glActiveTexture(GL_TEXTURE0);

	// bind the vertex array of screen filling points
	glBindBuffer(GL_ARRAY_BUFFER, _screenArray);

	// draw screen filling geometry
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, 0, 0, NULL);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glDisableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// unbind the textures from GL_TEXTURE0 + (i)
	index--;
	for(;index >= 0; index--)
	{
	    glActiveTexture(GL_TEXTURE0 + index);
	    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
	}
    }
    glUseProgram(0);
}

/// load a set of default colors into the combination shader for final lookup
void MultiGPURenderer::loadDefaultColors()
{
    glUseProgram(_comProg);
    glUniform1f(_numColorsUni,28.0);
    for(int i = 0; i < 29; i++)
    {
	ColorVal myColor = makeColor(((float)i)/31.0);
	glUniform1f(_redLookupUni[i],myColor.r);
	glUniform1f(_greenLookupUni[i],myColor.g);
	glUniform1f(_blueLookupUni[i],myColor.b);
	_colorMap.push_back(myColor);
    }
    glUseProgram(0);
}

/// update the color lookup in the shader with a custom color list
void MultiGPURenderer::updateColors()
{
    int numColors = std::min(MAX_COLORS, ((int)_colorMap.size()));
    glUseProgram(_comProg);
    glUniform1f(_numColorsUni,((float)(numColors-1)));
    for(int i = 0; i < numColors; i++)
    {
	glUniform1f(_redLookupUni[i],_colorMap[i].r);
	glUniform1f(_greenLookupUni[i],_colorMap[i].g);
	glUniform1f(_blueLookupUni[i],_colorMap[i].b);
    }
    glUseProgram(0);
}
