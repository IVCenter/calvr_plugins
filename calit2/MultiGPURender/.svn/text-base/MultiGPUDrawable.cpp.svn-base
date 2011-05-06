#include <GL/glew.h>

#include "MultiGPUDrawable.h"
#include "AnimationManager.h"
#include "Timing.h"
#include "CudaHelper.h"

#include <config/ConfigManager.h>
#include <kernel/CVRViewer.h>

#include <osgDB/ReadFile>
#include <osg/Material>
#include <OpenThreads/Thread>

#include <cstdio>
#include <fcntl.h>
#include <iostream>
#include <sstream>

#include <sys/syscall.h>
#include <sys/stat.h>

#ifndef GL_GEOMETRY_SHADER
#define GL_GEOMETRY_SHADER 0x8DD9
#endif

using namespace osg;
using namespace cvr;

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

#define VBO_TYPE GL_STREAM_COPY_ARB

#define MAX_VBO_SIZE 300000

osg::Vec3 makeColor(float f)
{
    if(f < 0)
    {
        f = 0;
    }
    else if(f > 1.0)
    {
        f = 1.0;
    }

    osg::Vec3 color;

    if(f <= 0.33)
    {
        float part = f / 0.33;
        float part2 = 1.0 - part;

        color.x() = part2;
        color.y() = part;
        color.z() = 0;
    }
    else if(f <= 0.66)
    {
        f = f - 0.33;
        float part = f / 0.33;
        float part2 = 1.0 - part;

        color.x() = 0;
        color.y() = part2;
        color.z() = part;
    }
    else if(f <= 1.0)
    {
        f = f - 0.66;
        float part = f / 0.33;
        float part2 = 1.0 - part;

        color.x() = part;
        color.y() = 0;
        color.z() = part2;
    }

    std::cerr << "Color x: " << color.x() << " y: " << color.y() << " z: " << color.z() << std::endl;

    return color;
}

int il_check_shader_log(GLuint shader)
{
    GLchar *log = NULL;
    GLint   val = 0;
    GLint   len = 0;

    /* Check the shader compile status.  If failed, print the log. */

    glGetShaderiv(shader, GL_COMPILE_STATUS,  &val);
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);

    if (val == 0)
    {
        if ((log = (GLchar *) calloc(len + 1, 1)))
        {
            glGetShaderInfoLog(shader, len, NULL, log);

            fprintf(stderr, "OpenGL Shader Error:\n%s", log);
            free(log);
        }
        return 0;
    }
    return 1;
}

int il_check_program_log(GLuint program)
{
    GLchar *log = NULL;
    GLint   val = 0;
    GLint   len = 0;

    /* Check the program link status.  If failed, print the log. */

    glGetProgramiv(program, GL_LINK_STATUS,     &val);
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);

    if (val == 0)
    {
        if ((log = (GLchar *) calloc(len + 1, 1)))
        {
            glGetProgramInfoLog(program, len, NULL, log);

            fprintf(stderr, "OpenGL Program Error:\n%s", log);
            free(log);
        }
        return 0;
    }
    return 1;
}

int il_check_framebuffer(void)
{
    const char *s = NULL;

    switch (glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT))
    {
    case    GL_FRAMEBUFFER_COMPLETE_EXT:
        return 1;

    case    GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
        s = "Framebuffer incomplete attachment";
        break;
    case    GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
        s = "Framebuffer incomplete missing attachment";
        break;
    case    GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
        s = "Framebuffer incomplete dimensions";
        break;
    case    GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
        s = "Framebuffer incomplete formats";
        break;
    case    GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
        s = "Framebuffer incomplete draw buffer";
        break;
    case    GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
        s = "Framebuffer incomplete read buffer";
        break;
    case    GL_FRAMEBUFFER_UNSUPPORTED_EXT:
        s = "Framebuffer unsupported";
        break;
    default:
        s = "Framebuffer error";
        break;
    }

    fprintf(stderr, "OpenGL Error: %s\n", s);

    return 0;
}

int il_check_error(void)
{
    return 0;
    const char *s = NULL;

    switch (glGetError())
    {
    case  GL_NO_ERROR:
        return 1;

    case  GL_INVALID_ENUM:
        s = "Invalid enumerant";
        break;
    case  GL_INVALID_VALUE:
        s = "Invalid value";
        break;
    case  GL_INVALID_OPERATION:
        s = "Invalid operation";
        break;
    case  GL_STACK_OVERFLOW:
        s = "Stack overflow";
        break;
    case  GL_OUT_OF_MEMORY:
        s = "Out of memory";
        break;
    case  GL_TABLE_TOO_LARGE:
        s = "Table too large";
        break;
    default:
        s = "Unknown";
        break;
    }

    fprintf(stderr, "OpenGL Error: %s\n", s);

    return 0;
}

MultiGPUDrawable::MultiGPUDrawable(std::string vertFile, std::string fragFile)
{
    _shaderDir = ConfigManager::getEntry("Plugin.MultiGPURender.ShaderDir");
    _width = ConfigManager::getInt("Plugin.MultiGPURender.Width");
    _height = ConfigManager::getInt("Plugin.MultiGPURender.Height");
    _gpus = ConfigManager::getInt("Plugin.MultiGPURender.NumberOfGPUs", 2);

    _depth = ConfigManager::getInt("Plugin.MultiGPURender.DepthBuffer",24);

    _cudaCopy = ConfigManager::getBool("Plugin.MultiGPURender.CudaCopy",false);

    _usePBOs = ConfigManager::getBool("Plugin.MultiGPURender.PBOs",false);

    _vertFile = _shaderDir + "/" + vertFile;
    _fragFile = _shaderDir + "/" + fragFile;
    _drawGeoFile = _shaderDir + "/draw.geom";
    _drawVertFile = _shaderDir + "/draw.vert";

    if(_depth == 32)
    {
	_drawFragFile = _shaderDir + "/draw32.frag";
    }
    else if(_depth == 16)
    {
	_drawFragFile = _shaderDir + "/draw16.frag";
    }
    else if(_depth == 24)
    {
	_drawFragFile = _shaderDir + "/draw24.frag";
    }

    if(_gpus == 1)
    {
	_drawVertFile = _shaderDir + "/draw1.vert";
	_drawFragFile = _shaderDir + "/draw1.frag";
    }

    threadSyncBlock = new bool[_gpus-1];
    colorCopyBuffers = new GLuint[_gpus-1];
    depthCopyBuffers = new GLuint[_gpus-1];
    if(_depth == 24)
    {
	depthR8CopyBuffers = new GLuint[_gpus-1];
    }
    colorTextures = new GLuint[_gpus-1];
    depthTextures = new GLuint[_gpus-1];
    depthRTextures = new GLuint[_gpus-1];
    depthR8Textures = new GLuint[_gpus-1];

    _useDrawShader = true;
    _useGeometryShader = ConfigManager::getBool("Plugin.MultiGPURender.GeometryShader",false);

    _drawBlock = new OpenThreads::BlockCount(_gpus);

    if(_useGeometryShader)
    {
	if(_gpus == 1)
	{
	    _drawVertFile = _shaderDir + "/drawGeo1.vert";
	    _drawGeoFile = _shaderDir + "/draw1.geom";
	    //_drawFragFile = _shaderDir + "/drawGeo1.frag";
	}
	else
	{
	    _drawVertFile = _shaderDir + "/drawGeo.vert";
	    //_drawFragFile = _shaderDir + "/drawGeo.frag";
	}
    }

    _drawLines = ConfigManager::getBool("Plugin.MultiGPURender.DrawLines",true);

    _currentFrame = NULL;

    for(int i = 0; i < _gpus -1; i++)
    {
	threadSyncBlock[i] = false;
    }

    for(int i = 0; i < _gpus; i++)
    {
	_makeVBOs[i] = false;
	_madeVBOs[i] = false;
	_loadVBOs[i] = false;
	_getTimings[i] = true;
	_nextFrameLoadDone[i] = false;
	_lineVBOs[i] = std::vector<GLuint>();
	_lineData[i] = std::vector<float *>();
	_lineSize[i] = std::vector<unsigned int>();
	_lineNextSize[i] = std::vector<unsigned int>();
	_quadVBOs[i] = std::vector<std::pair<std::vector<GLuint>,std::vector<GLuint> > >();
	_quadData[i] = std::vector<std::pair<float*,float*> >();
	_quadSize[i] = std::vector<std::vector<unsigned int> >();
	_quadNextSize[i] = std::vector<std::vector<unsigned int> >();
	_colorMap[i] = std::vector<osg::Vec4>();
	_triData[i] = std::vector<std::pair<float*,float*> >();
	_triSize[i] = std::vector<unsigned int>();
	_triNextSize[i] = std::vector<unsigned int>();
	_triVBOs[i] = std::vector<std::pair<GLuint,GLuint> >();
    }

    setUseDisplayList(false);
}

MultiGPUDrawable::MultiGPUDrawable(const MultiGPUDrawable& mgd,const osg::CopyOp& copyop) : Drawable(mgd,copyop)
{
    setUseDisplayList(false);
}

MultiGPUDrawable::~MultiGPUDrawable()
{
}

BoundingBox MultiGPUDrawable::computeBound() const
{
    Vec3 size2(10000, 10000, 10000);
    _boundingBox.init();
    _boundingBox.set(-size2[0], -size2[1], -size2[2], size2[0], size2[1], size2[2]);
    return _boundingBox;
}

void MultiGPUDrawable::updateBoundingBox()
{
    computeBound();
    dirtyBound();
}

void MultiGPUDrawable::addArray(int gpu, GLenum type, float * data, float * normals, unsigned int size, osg::Vec4 color)
{
    if(type == GL_LINES)
    {
	_lineData[gpu].push_back(data);
	_lineSize[gpu].push_back(size);
    }
    else if(type == GL_QUADS)
    {
	std::cerr << "Size: " << size << std::endl;
	_quadData[gpu].push_back(std::pair<float*,float*>(data,normals));
	//_quadSize[gpu].push_back(size);
	std::vector<unsigned int> sizevec;
	unsigned int tsize = size;
	while(tsize > MAX_VBO_SIZE)
	{
	    sizevec.push_back((unsigned int)MAX_VBO_SIZE);
	    tsize -= MAX_VBO_SIZE;
	}
	sizevec.push_back(tsize);
	_quadSize[gpu].push_back(sizevec);
	//TODO set better
	_colorMap[gpu].push_back(color);
    }
    else if(type == GL_TRIANGLES)
    {
	_triData[gpu].push_back(std::pair<float*,float*>(data,normals));
	_triSize[gpu].push_back(size);
    }
    //_colorMap[gpu].push_back(color);
    _makeVBOs[gpu] = true;
}

void MultiGPUDrawable::loadVBOs(int context) const
{
    //std::cerr << "Starting vbo init." << std::endl;
    //std::cerr << "Frame " << _currentFrame->frameNum << std::endl;
    
    if(!_madeVBOs[context])
    {
	// thread lock buffer creation, just in case
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);

	GLuint buffer;
	GLuint buffer2;

	for(int i = 0; i < _currentFrame->gpuPartsMap[context].size(); i++)
	{
	    //std::cerr << "Loop" << std::endl;
	    _lineSize[context].push_back(_currentFrame->maxLineSize[_currentFrame->gpuPartsMap[context][i]]);
	    if(_currentFrame->lineSize[_currentFrame->gpuPartsMap[context][i]])
	    {
		glGenBuffersARB(1,&buffer);
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, buffer);
		glBufferDataARB(GL_ARRAY_BUFFER_ARB, _currentFrame->lineSize[_currentFrame->gpuPartsMap[context][i]]*2*3*sizeof(float), 0, VBO_TYPE);
		_lineVBOs[context].push_back(buffer);
		
		if(_cudaCopy)
		{
		    checkRegBufferObj(buffer);
		}

		glGenBuffersARB(1,&buffer);
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, buffer);
		glBufferDataARB(GL_ARRAY_BUFFER_ARB, _currentFrame->lineSize[_currentFrame->gpuPartsMap[context][i]]*2*3*sizeof(float), 0, VBO_TYPE);
		_lineNextVBOs[context].push_back(buffer);

		if(_cudaCopy)
		{
		    checkRegBufferObj(buffer);
		}
	    }
	    else
	    {
		_lineVBOs[context].push_back(0);
		_lineNextVBOs[context].push_back(0);
	    }

	    std::vector<unsigned int> sizevec;
	    unsigned int tsize = _currentFrame->maxQuadSize[_currentFrame->gpuPartsMap[context][i]];
	    while(tsize > MAX_VBO_SIZE)
	    {
		sizevec.push_back((unsigned int)MAX_VBO_SIZE);
		tsize -= MAX_VBO_SIZE;
	    }
	    sizevec.push_back(tsize);
	    _quadSize[context].push_back(sizevec);

	    std::vector<GLuint> vert;
	    std::vector<GLuint> norm;
	    std::vector<GLuint> vertNext;
	    std::vector<GLuint> normNext;
	    if(!_currentFrame->quadSize[_currentFrame->gpuPartsMap[context][i]])
	    {
		sizevec.clear();
	    }
	    for(int j = 0; j < sizevec.size(); j++)
	    {
		glGenBuffersARB(1,&buffer);
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, buffer);
		glBufferDataARB(GL_ARRAY_BUFFER_ARB, _quadSize[context][i][j]*4*3*sizeof(float), 0, VBO_TYPE);
		vert.push_back(buffer);

		if(_cudaCopy)
		{
		    checkRegBufferObj(buffer);
		}

		if(!_useGeometryShader)
		{
		    glGenBuffersARB(1,&buffer2);
		    glBindBufferARB(GL_ARRAY_BUFFER_ARB, buffer2);
		    glBufferDataARB(GL_ARRAY_BUFFER_ARB, _quadSize[context][i][j]*3*4*sizeof(float), 0, VBO_TYPE);
		    norm.push_back(buffer2);

		    if(_cudaCopy)
		    {
			checkRegBufferObj(buffer2);
		    }
		}

		glGenBuffersARB(1,&buffer);
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, buffer);
		glBufferDataARB(GL_ARRAY_BUFFER_ARB, _quadSize[context][i][j]*4*3*sizeof(float), 0, VBO_TYPE);
		vertNext.push_back(buffer);

		if(_cudaCopy)
		{
		    checkRegBufferObj(buffer);
		}

		if(!_useGeometryShader)
		{
		    glGenBuffersARB(1,&buffer2);
		    glBindBufferARB(GL_ARRAY_BUFFER_ARB, buffer2);
		    glBufferDataARB(GL_ARRAY_BUFFER_ARB, _quadSize[context][i][j]*3*4*sizeof(float), 0, VBO_TYPE);
		    normNext.push_back(buffer2);

		    if(_cudaCopy)
		    {
			checkRegBufferObj(buffer2);
		    }
		}
	    }
	    _quadVBOs[context].push_back(std::pair<std::vector<GLuint>,std::vector<GLuint> >(vert,norm));
	    _quadNextVBOs[context].push_back(std::pair<std::vector<GLuint>,std::vector<GLuint> >(vertNext,normNext));

	    _triSize[context].push_back(_currentFrame->maxTriSize[_currentFrame->gpuPartsMap[context][i]]);

	    if(_triSize[context][i])
	    {
		glGenBuffersARB(1,&buffer);
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, buffer);
		glBufferDataARB(GL_ARRAY_BUFFER_ARB, _triSize[context][i]*3*3*sizeof(float), 0, VBO_TYPE);

		if(_cudaCopy)
		{
		    checkRegBufferObj(buffer);
		}

		if(!_useGeometryShader)
		{
		    glGenBuffersARB(1,&buffer2);
		    glBindBufferARB(GL_ARRAY_BUFFER_ARB, buffer2);
		    glBufferDataARB(GL_ARRAY_BUFFER_ARB, _triSize[context][i]*3*3*sizeof(float), 0, VBO_TYPE);

		    if(_cudaCopy)
		    {
			checkRegBufferObj(buffer2);
		    }
		}
		_triVBOs[context].push_back(std::pair<GLuint,GLuint>(buffer,buffer2));

		glGenBuffersARB(1,&buffer);
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, buffer);
		glBufferDataARB(GL_ARRAY_BUFFER_ARB, _triSize[context][i]*3*3*sizeof(float), 0, VBO_TYPE);

		if(_cudaCopy)
		{
		    checkRegBufferObj(buffer);
		}

		if(!_useGeometryShader)
		{
		    glGenBuffersARB(1,&buffer2);
		    glBindBufferARB(GL_ARRAY_BUFFER_ARB, buffer2);
		    glBufferDataARB(GL_ARRAY_BUFFER_ARB, _triSize[context][i]*3*3*sizeof(float), 0, VBO_TYPE);

		    if(_cudaCopy)
		    {
			checkRegBufferObj(buffer2);
		    }
		}
		_triNextVBOs[context].push_back(std::pair<GLuint,GLuint>(buffer,buffer2));
	    }
	    else
	    {
		_triVBOs[context].push_back(std::pair<GLuint,GLuint>(0,0));
		_triNextVBOs[context].push_back(std::pair<GLuint,GLuint>(0,0));
	    }
	}
	_madeVBOs[context] = true;
    }

    _lineSize[context].clear();
    _quadSize[context].clear();
    _triSize[context].clear();
    for(int i = 0; i < _currentFrame->gpuPartsMap[context].size(); i++)
    {
	//std::cerr << "Loop" << std::endl;
	_lineSize[context].push_back(_currentFrame->lineSize[_currentFrame->gpuPartsMap[context][i]]);

	std::vector<unsigned int> sizevec;
	unsigned int tsize = _currentFrame->quadSize[_currentFrame->gpuPartsMap[context][i]];
	while(tsize > MAX_VBO_SIZE)
	{
	    sizevec.push_back((unsigned int)MAX_VBO_SIZE);
	    tsize -= MAX_VBO_SIZE;
	}
	sizevec.push_back(tsize);
	_quadSize[context].push_back(sizevec);

	_triSize[context].push_back(_currentFrame->triSize[_currentFrame->gpuPartsMap[context][i]]);
    }

    int totalSize = 0;
    struct timeval start, end;
    if(_getTimings[context])
    {
	glFinish();
	getTime(start);
    }

    for(int i = 0; i < _currentFrame->gpuPartsMap[context].size(); i++)
    {

	if(_lineSize[context][i])
	{
	    glBindBufferARB(GL_ARRAY_BUFFER_ARB, _lineVBOs[context][i]);
	    //glBufferDataARB(GL_ARRAY_BUFFER_ARB, _lineSize[context][i]*2*3*sizeof(float), 0, GL_STATIC_DRAW_ARB);
	    glBufferSubDataARB(GL_ARRAY_BUFFER_ARB, 0, _lineSize[context][i]*2*3*sizeof(float), _currentFrame->lineData[_currentFrame->gpuPartsMap[context][i]]);
	    //std::cerr << "Creating line vbo of size: " << _lineSize[context][i]*2*3 << std::endl;
	    totalSize += _lineSize[context][i]*2*3*sizeof(float);
	}

	//glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

	for(int j = 0; j < _quadSize[context][i].size(); j++)
	{
	    if(!_quadSize[context][i][j])
	    {
		continue;
	    }
	    glBindBufferARB(GL_ARRAY_BUFFER_ARB, _quadVBOs[context][i].first[j]);
	    glBufferSubDataARB(GL_ARRAY_BUFFER_ARB, 0, _quadSize[context][i][j]*3*4*sizeof(float), _currentFrame->quadData[_currentFrame->gpuPartsMap[context][i]].first + (j * MAX_VBO_SIZE * 4 * 3));
	    //std::cerr << "Creating quad vbo of size: " << _quadSize[context][i]*4*3 << std::endl;
	    if(!_useGeometryShader)
	    {
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, _quadVBOs[context][i].second[j]);
		glBufferSubDataARB(GL_ARRAY_BUFFER_ARB, 0, _quadSize[context][i][j]*3*4*sizeof(float), _currentFrame->quadData[_currentFrame->gpuPartsMap[context][i]].second + (j * MAX_VBO_SIZE * 4 * 3));
		totalSize += _quadSize[context][i][j]*3*4*sizeof(float)*2;
	    }
	    else
	    {
		totalSize += _quadSize[context][i][j]*3*4*sizeof(float);
	    }
	}

	if(_triSize[context][i])
	{
	    glBindBufferARB(GL_ARRAY_BUFFER_ARB, _triVBOs[context][i].first);
	    glBufferSubDataARB(GL_ARRAY_BUFFER_ARB, 0, _triSize[context][i]*3*3*sizeof(float), _currentFrame->triData[_currentFrame->gpuPartsMap[context][i]].first);
	    if(!_useGeometryShader)
	    {
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, _triVBOs[context][i].second);
		glBufferSubDataARB(GL_ARRAY_BUFFER_ARB, 0, _triSize[context][i]*3*3*sizeof(float), _currentFrame->triData[_currentFrame->gpuPartsMap[context][i]].second);
		//std::cerr << "Creating triangle vbo of size: " << _triSize[context][i]*3*3 << std::endl;
		totalSize += _triSize[context][i]*3*3*sizeof(float)*2;
	    }
	    else
	    {
		totalSize += _triSize[context][i]*3*3*sizeof(float);
	    }
	}
    }

    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
    
    if(_getTimings[context])
    {
	glFinish();
	getTime(end);
	_dataLoadTime[context] = getDiff(start,end) / ((float)totalSize);
    }

    _loadVBOs[context] = false;
    //std::cerr << "vbo init finished" << std::endl;
}

void MultiGPUDrawable::loadNextVBOs(int context) const
{
    unsigned int maxBytesToLoad = _nextFrameBytes;
    if(_loadNextFrameSize[context])
    {
	_lineNextSize[context].clear();
	_quadNextSize[context].clear();
	_triNextSize[context].clear();
	for(int i = 0; i < _nextFrame->gpuPartsMap[context].size(); i++)
	{
	    //std::cerr << "Loop" << std::endl;
	    _lineNextSize[context].push_back(_nextFrame->lineSize[_nextFrame->gpuPartsMap[context][i]]);

	    std::vector<unsigned int> sizevec;
	    unsigned int tsize = _nextFrame->quadSize[_nextFrame->gpuPartsMap[context][i]];
	    while(tsize > MAX_VBO_SIZE)
	    {
		sizevec.push_back((unsigned int)MAX_VBO_SIZE);
		tsize -= MAX_VBO_SIZE;
	    }
	    sizevec.push_back(tsize);
	    _quadNextSize[context].push_back(sizevec);

	    _triNextSize[context].push_back(_nextFrame->triSize[_nextFrame->gpuPartsMap[context][i]]);
	    //std::cerr << "Tri size: " << _triNextSize[context][i] << std::endl;
	}

	_loadNextFrameSize[context] = false;
    }

    for(; _prgParts[context] < _nextFrame->gpuPartsMap[context].size(); _prgParts[context]++)
    {

	if(_prgStage[context] == LINE)
	{
	    if(_lineNextSize[context][_prgParts[context]])
	    {
		if(!maxBytesToLoad)
		{
		    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
		    return;  
		}

		int size = _lineNextSize[context][_prgParts[context]]*2*3*sizeof(float);
		bool partialLoad = false;
		size = size - _prgOffset[context];
		if(size > maxBytesToLoad)
		{
		    size = maxBytesToLoad;
		    partialLoad = true;
		}
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, _lineNextVBOs[context][_prgParts[context]]);
		glBufferSubDataARB(GL_ARRAY_BUFFER_ARB, _prgOffset[context], size, ((char*)(_nextFrame->lineData[_nextFrame->gpuPartsMap[context][_prgParts[context]]])) + _prgOffset[context]);
		maxBytesToLoad -= size;
		if(partialLoad)
		{
		    _prgOffset[context] += size;
		    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
		    return;
		}
		else
		{
		    _prgOffset[context] = 0;
		}
		//std::cerr << "Creating line vbo of size: " << _lineSize[context][i]*2*3 << std::endl;
	    }
	    _prgStage[context] = QUAD_VERT;
	}

	//glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

	for(; _prgQuadNum[context] < _quadNextSize[context][_prgParts[context]].size(); _prgQuadNum[context]++)
	{
	    if(!_quadNextSize[context][_prgParts[context]][_prgQuadNum[context]])
	    {
		continue;
	    }
	    if(_prgStage[context] == QUAD_VERT)
	    {
		if(!maxBytesToLoad)
		{
		    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
		    return;  
		}
		int size = _quadNextSize[context][_prgParts[context]][_prgQuadNum[context]]*3*4*sizeof(float);
		bool partialLoad = false;
		size = size - _prgOffset[context];
		if(size > maxBytesToLoad)
		{
		    size = maxBytesToLoad;
		    partialLoad = true;
		}
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, _quadNextVBOs[context][_prgParts[context]].first[_prgQuadNum[context]]);
		glBufferSubDataARB(GL_ARRAY_BUFFER_ARB, _prgOffset[context], size, ((char*)(_nextFrame->quadData[_nextFrame->gpuPartsMap[context][_prgParts[context]]].first + (_prgQuadNum[context] * MAX_VBO_SIZE * 4 * 3))) + _prgOffset[context]);

		maxBytesToLoad -= size;
		if(partialLoad)
		{
		    _prgOffset[context] += size;
		    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
		    return;
		}
		else
		{
		    _prgOffset[context] = 0;
		}

		_prgStage[context] = QUAD_NORM;
	    }

	    if(!maxBytesToLoad)
	    {
	        glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
	        return;  
	    }

	    if(!_useGeometryShader)
	    {
		int size = _quadNextSize[context][_prgParts[context]][_prgQuadNum[context]]*3*4*sizeof(float);
		bool partialLoad = false;
		size = size - _prgOffset[context];
		if(size > maxBytesToLoad)
		{
		    size = maxBytesToLoad;
		    partialLoad = true;
		}
		//std::cerr << "Creating quad vbo of size: " << _quadSize[context][i]*4*3 << std::endl;
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, _quadNextVBOs[context][_prgParts[context]].second[_prgQuadNum[context]]);
		glBufferSubDataARB(GL_ARRAY_BUFFER_ARB, _prgOffset[context], size, ((char*)(_nextFrame->quadData[_nextFrame->gpuPartsMap[context][_prgParts[context]]].second + (_prgQuadNum[context] * MAX_VBO_SIZE * 4 * 3))) + _prgOffset[context]);

		maxBytesToLoad -= size;
		if(partialLoad)
		{
		    _prgOffset[context] += size;
		    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
		    return;
		}
		else
		{
		    _prgOffset[context] = 0;
		}

	    }

	    _prgStage[context] = QUAD_VERT;
	}

	if(_prgStage[context] == QUAD_VERT)
	{
	    _prgStage[context] = TRI_VERT;
	}

	if(_triNextSize[context][_prgParts[context]])
	{
	    if(_prgStage[context] == TRI_VERT)
	    {
		if(!maxBytesToLoad)
		{
		    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
		    return;  
		}
		int size = _triNextSize[context][_prgParts[context]]*3*3*sizeof(float);
		bool partialLoad = false;
		size = size - _prgOffset[context];
		if(size > maxBytesToLoad)
		{
		    size = maxBytesToLoad;
		    partialLoad = true;
		}

		//std::cerr << "Loading Triangles." << std::endl;
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, _triNextVBOs[context][_prgParts[context]].first);
		glBufferSubDataARB(GL_ARRAY_BUFFER_ARB, _prgOffset[context], size, ((char*)(_nextFrame->triData[_nextFrame->gpuPartsMap[context][_prgParts[context]]].first)) + _prgOffset[context]);

		maxBytesToLoad -= size;
		if(partialLoad)
		{
		    _prgOffset[context] += size;
		    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
		    return;
		}
		else
		{
		    _prgOffset[context] = 0;
		}

	        _prgStage[context] = TRI_NORM;
	    }

	    if(!maxBytesToLoad)
	    {
	        glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
	        return;  
	    }

	    if(!_useGeometryShader)
	    {
		int size = _triNextSize[context][_prgParts[context]]*3*3*sizeof(float);
		bool partialLoad = false;
		size = size - _prgOffset[context];
		if(size > maxBytesToLoad)
		{
		    size = maxBytesToLoad;
		    partialLoad = true;
		}

		glBindBufferARB(GL_ARRAY_BUFFER_ARB, _triNextVBOs[context][_prgParts[context]].second);
		glBufferSubDataARB(GL_ARRAY_BUFFER_ARB, _prgOffset[context], size, ((char*)(_nextFrame->triData[_nextFrame->gpuPartsMap[context][_prgParts[context]]].second)) + _prgOffset[context]);

		maxBytesToLoad -= size;
		if(partialLoad)
		{
		    _prgOffset[context] += size;
		    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
		    return;
		}
		else
		{
		    _prgOffset[context] = 0;
		}

	    }

	    _prgStage[context] = TRI_VERT;
	    //std::cerr << "Creating triangle vbo of size: " << _triSize[context][i]*3*3 << std::endl;
	}

	_prgQuadNum[context] = 0;
	_prgStage[context] = LINE;
    }

    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

    _loadVBOs[context] = false;
    _nextFrameLoadDone[context] = true;
    //std::cerr << "vbo init finished" << std::endl;
}

void MultiGPUDrawable::cudaLoadNextVBOs(int context, cudaStream_t & stream)
{
    //std::cerr << "cudaLoad" << std::endl;
    unsigned int maxBytesToLoad = _nextFrameBytes;
    if(_loadNextFrameSize[context])
    {
	_lineNextSize[context].clear();
	_quadNextSize[context].clear();
	_triNextSize[context].clear();
	for(int i = 0; i < _nextFrame->gpuPartsMap[context].size(); i++)
	{
	    //std::cerr << "Loop" << std::endl;
	    _lineNextSize[context].push_back(_nextFrame->lineSize[_nextFrame->gpuPartsMap[context][i]]);

	    std::vector<unsigned int> sizevec;
	    unsigned int tsize = _nextFrame->quadSize[_nextFrame->gpuPartsMap[context][i]];
	    while(tsize > MAX_VBO_SIZE)
	    {
		sizevec.push_back((unsigned int)MAX_VBO_SIZE);
		tsize -= MAX_VBO_SIZE;
	    }
	    sizevec.push_back(tsize);
	    _quadNextSize[context].push_back(sizevec);

	    _triNextSize[context].push_back(_nextFrame->triSize[_nextFrame->gpuPartsMap[context][i]]);
	    //std::cerr << "Tri size: " << _triNextSize[context][i] << std::endl;
	}

	_loadNextFrameSize[context] = false;
    }

    for(; _prgParts[context] < _nextFrame->gpuPartsMap[context].size(); _prgParts[context]++)
    {

	if(_prgStage[context] == LINE)
	{
	    if(_lineNextSize[context][_prgParts[context]])
	    {
		if(!maxBytesToLoad)
		{
		    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
		    return;  
		}

		int size = _lineNextSize[context][_prgParts[context]]*2*3*sizeof(float);
		bool partialLoad = false;
		size = size - _prgOffset[context];
		if(size > maxBytesToLoad)
		{
		    size = maxBytesToLoad;
		    partialLoad = true;
		}

		char * vert;
		checkMapBufferObj((void**)&(vert), _lineNextVBOs[context][_prgParts[context]]);
		cudaMemcpyAsync(vert + _prgOffset[context], ((char*)(_nextFrame->lineData[_nextFrame->gpuPartsMap[context][_prgParts[context]]])) + _prgOffset[context], size, cudaMemcpyHostToDevice);
		checkUnmapBufferObj(_lineNextVBOs[context][_prgParts[context]]);
		//glBindBufferARB(GL_ARRAY_BUFFER_ARB, _lineNextVBOs[context][_prgParts[context]]);
		//glBufferSubDataARB(GL_ARRAY_BUFFER_ARB, _prgOffset[context], size, ((char*)(_nextFrame->lineData[_nextFrame->gpuPartsMap[context][_prgParts[context]]])) + _prgOffset[context]);
		maxBytesToLoad -= size;
		if(partialLoad)
		{
		    _prgOffset[context] += size;
		    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
		    return;
		}
		else
		{
		    _prgOffset[context] = 0;
		}
		//std::cerr << "Creating line vbo of size: " << _lineSize[context][i]*2*3 << std::endl;
	    }
	    _prgStage[context] = QUAD_VERT;
	}

	//glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

	for(; _prgQuadNum[context] < _quadNextSize[context][_prgParts[context]].size(); _prgQuadNum[context]++)
	{
	    if(!_quadNextSize[context][_prgParts[context]][_prgQuadNum[context]])
	    {
		continue;
	    }
	    if(_prgStage[context] == QUAD_VERT)
	    {
		if(!maxBytesToLoad)
		{
		    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
		    return;  
		}
		int size = _quadNextSize[context][_prgParts[context]][_prgQuadNum[context]]*3*4*sizeof(float);
		bool partialLoad = false;
		size = size - _prgOffset[context];
		if(size > maxBytesToLoad)
		{
		    size = maxBytesToLoad;
		    partialLoad = true;
		}
		char * vert;
		checkMapBufferObj((void**)&(vert), _quadNextVBOs[context][_prgParts[context]].first[_prgQuadNum[context]]);
		cudaMemcpyAsync(vert + _prgOffset[context], ((char*)(_nextFrame->quadData[_nextFrame->gpuPartsMap[context][_prgParts[context]]].first + (_prgQuadNum[context] * MAX_VBO_SIZE * 4 * 3))) + _prgOffset[context], size, cudaMemcpyHostToDevice);
		checkUnmapBufferObj(_quadNextVBOs[context][_prgParts[context]].first[_prgQuadNum[context]]);
		//glBindBufferARB(GL_ARRAY_BUFFER_ARB, _quadNextVBOs[context][_prgParts[context]].first[_prgQuadNum[context]]);
		//glBufferSubDataARB(GL_ARRAY_BUFFER_ARB, _prgOffset[context], size, ((char*)(_nextFrame->quadData[_nextFrame->gpuPartsMap[context][_prgParts[context]]].first + (_prgQuadNum[context] * MAX_VBO_SIZE * 4 * 3))) + _prgOffset[context]);

		maxBytesToLoad -= size;
		if(partialLoad)
		{
		    _prgOffset[context] += size;
		    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
		    return;
		}
		else
		{
		    _prgOffset[context] = 0;
		}

		_prgStage[context] = QUAD_NORM;
	    }

	    if(!maxBytesToLoad)
	    {
	        glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
	        return;  
	    }

	    if(!_useGeometryShader)
	    {
		int size = _quadNextSize[context][_prgParts[context]][_prgQuadNum[context]]*3*4*sizeof(float);
		bool partialLoad = false;
		size = size - _prgOffset[context];
		if(size > maxBytesToLoad)
		{
		    size = maxBytesToLoad;
		    partialLoad = true;
		}
		char * vert;
		checkMapBufferObj((void**)&(vert), _quadNextVBOs[context][_prgParts[context]].second[_prgQuadNum[context]]);
		cudaMemcpyAsync(vert + _prgOffset[context], ((char*)(_nextFrame->quadData[_nextFrame->gpuPartsMap[context][_prgParts[context]]].second + (_prgQuadNum[context] * MAX_VBO_SIZE * 4 * 3))) + _prgOffset[context], size, cudaMemcpyHostToDevice);
		checkUnmapBufferObj(_quadNextVBOs[context][_prgParts[context]].second[_prgQuadNum[context]]);
		//std::cerr << "Creating quad vbo of size: " << _quadSize[context][i]*4*3 << std::endl;
		//glBindBufferARB(GL_ARRAY_BUFFER_ARB, _quadNextVBOs[context][_prgParts[context]].second[_prgQuadNum[context]]);
		//glBufferSubDataARB(GL_ARRAY_BUFFER_ARB, _prgOffset[context], size, ((char*)(_nextFrame->quadData[_nextFrame->gpuPartsMap[context][_prgParts[context]]].second + (_prgQuadNum[context] * MAX_VBO_SIZE * 4 * 3))) + _prgOffset[context]);

		maxBytesToLoad -= size;
		if(partialLoad)
		{
		    _prgOffset[context] += size;
		    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
		    return;
		}
		else
		{
		    _prgOffset[context] = 0;
		}

	    }

	    _prgStage[context] = QUAD_VERT;
	}

	if(_prgStage[context] == QUAD_VERT)
	{
	    _prgStage[context] = TRI_VERT;
	}

	if(_triNextSize[context][_prgParts[context]])
	{
	    if(_prgStage[context] == TRI_VERT)
	    {
		if(!maxBytesToLoad)
		{
		    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
		    return;  
		}
		int size = _triNextSize[context][_prgParts[context]]*3*3*sizeof(float);
		bool partialLoad = false;
		size = size - _prgOffset[context];
		if(size > maxBytesToLoad)
		{
		    size = maxBytesToLoad;
		    partialLoad = true;
		}

		char * vert;
		checkMapBufferObj((void**)&(vert), _triNextVBOs[context][_prgParts[context]].first);
		cudaMemcpyAsync(vert + _prgOffset[context], ((char*)(_nextFrame->triData[_nextFrame->gpuPartsMap[context][_prgParts[context]]].first)) + _prgOffset[context], size, cudaMemcpyHostToDevice);
		checkUnmapBufferObj(_triNextVBOs[context][_prgParts[context]].first);
		//std::cerr << "Loading Triangles." << std::endl;
		//glBindBufferARB(GL_ARRAY_BUFFER_ARB, _triNextVBOs[context][_prgParts[context]].first);
		//glBufferSubDataARB(GL_ARRAY_BUFFER_ARB, _prgOffset[context], size, ((char*)(_nextFrame->triData[_nextFrame->gpuPartsMap[context][_prgParts[context]]].first)) + _prgOffset[context]);

		maxBytesToLoad -= size;
		if(partialLoad)
		{
		    _prgOffset[context] += size;
		    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
		    return;
		}
		else
		{
		    _prgOffset[context] = 0;
		}

	        _prgStage[context] = TRI_NORM;
	    }

	    if(!maxBytesToLoad)
	    {
	        glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
	        return;  
	    }

	    if(!_useGeometryShader)
	    {
		int size = _triNextSize[context][_prgParts[context]]*3*3*sizeof(float);
		bool partialLoad = false;
		size = size - _prgOffset[context];
		if(size > maxBytesToLoad)
		{
		    size = maxBytesToLoad;
		    partialLoad = true;
		}

		char * vert;
		checkMapBufferObj((void**)&(vert), _triNextVBOs[context][_prgParts[context]].second);
		cudaMemcpyAsync(vert + _prgOffset[context], ((char*)(_nextFrame->triData[_nextFrame->gpuPartsMap[context][_prgParts[context]]].second)) + _prgOffset[context], size, cudaMemcpyHostToDevice);
		checkUnmapBufferObj(_triNextVBOs[context][_prgParts[context]].second);
		//glBindBufferARB(GL_ARRAY_BUFFER_ARB, _triNextVBOs[context][_prgParts[context]].second);
		//glBufferSubDataARB(GL_ARRAY_BUFFER_ARB, _prgOffset[context], size, ((char*)(_nextFrame->triData[_nextFrame->gpuPartsMap[context][_prgParts[context]]].second)) + _prgOffset[context]);

		maxBytesToLoad -= size;
		if(partialLoad)
		{
		    _prgOffset[context] += size;
		    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
		    return;
		}
		else
		{
		    _prgOffset[context] = 0;
		}

	    }

	    _prgStage[context] = TRI_VERT;
	    //std::cerr << "Creating triangle vbo of size: " << _triSize[context][i]*3*3 << std::endl;
	}

	_prgQuadNum[context] = 0;
	_prgStage[context] = LINE;
    }

    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

    _loadVBOs[context] = false;
    _nextFrameLoadDone[context] = true;
    //std::cerr << "vbo init finished" << std::endl;
}

void MultiGPUDrawable::initVBOs(int context) const
{
    std::cerr << "Starting vbo init." << std::endl;
    {
	// thread lock buffer creation, just in case
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_vboMutex);

	GLuint buffer;
	GLuint buffer2;

	for(int i = 0; i < _lineData[context].size(); i++)
	{
	    glGenBuffersARB(1,&buffer);
	    _lineVBOs[context].push_back(buffer);
	}
	for(int i = 0; i < _quadData[context].size(); i++)
	{
	    std::vector<GLuint> vert;
	    std::vector<GLuint> norm;
	    for(int j = 0; j < _quadSize[context][i].size(); j++)
	    {
		glGenBuffersARB(1,&buffer);
		glGenBuffersARB(1,&buffer2);
		vert.push_back(buffer);
		norm.push_back(buffer2);
	    }

	    _quadVBOs[context].push_back(std::pair<std::vector<GLuint>,std::vector<GLuint> >(vert,norm));
	}
	for(int i = 0; i < _triData[context].size(); i++)
	{
	    glGenBuffersARB(1,&buffer);
	    glGenBuffersARB(1,&buffer2);
	    _triVBOs[context].push_back(std::pair<GLuint,GLuint>(buffer,buffer2));
	}
    }

    for(int i = 0; i < _lineData[context].size(); i++)
    {
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, _lineVBOs[context][i]);
	//glBufferDataARB(GL_ARRAY_BUFFER_ARB, _lineSize[context][i]*2*3*sizeof(float), 0, GL_STATIC_DRAW_ARB);
	glBufferDataARB(GL_ARRAY_BUFFER_ARB, _lineSize[context][i]*2*3*sizeof(float), _lineData[context][i], VBO_TYPE);
	std::cerr << "Creating line vbo of size: " << _lineSize[context][i]*2*3 << std::endl;
    }

    //glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

    for(int i = 0; i < _quadData[context].size(); i++)
    {
	for(int j = 0; j < _quadSize[context][i].size(); j++)
	{
	    glBindBufferARB(GL_ARRAY_BUFFER_ARB, _quadVBOs[context][i].first[j]);
	    glBufferDataARB(GL_ARRAY_BUFFER_ARB, _quadSize[context][i][j]*4*3*sizeof(float), _quadData[context][i].first + (j * MAX_VBO_SIZE * 4 * 3), VBO_TYPE);
	    //std::cerr << "Creating quad vbo of size: " << _quadSize[context][i]*4*3 << std::endl;
	    glBindBufferARB(GL_ARRAY_BUFFER_ARB, _quadVBOs[context][i].second[j]);
	    glBufferDataARB(GL_ARRAY_BUFFER_ARB, _quadSize[context][i][j]*3*4*sizeof(float), _quadData[context][i].second + (j * MAX_VBO_SIZE * 4 * 3), VBO_TYPE);
	}
    }

    for(int i = 0; i < _triData[context].size(); i++)
    {
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, _triVBOs[context][i].first);
	glBufferDataARB(GL_ARRAY_BUFFER_ARB, _triSize[context][i]*3*3*sizeof(float), _triData[context][i].first, VBO_TYPE);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, _triVBOs[context][i].second);
	glBufferDataARB(GL_ARRAY_BUFFER_ARB, _triSize[context][i]*3*3*sizeof(float), _triData[context][i].second, VBO_TYPE);
	std::cerr << "Creating triangle vbo of size: " << _triSize[context][i]*3*3 << std::endl;
    }

    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
    
    std::cerr << "vbo init finished" << std::endl;
}

void MultiGPUDrawable::drawVBOs(int context) const
{
    //glFinish();
    //struct timeval start,end;
    //getTime(start);
    bool textureOn = glIsEnabled(GL_TEXTURE_COORD_ARRAY);
    bool vertexOn = glIsEnabled(GL_VERTEX_ARRAY);
    //bool normalOn = glIsEnabled(GL_NORMAL_ARRAY);
    

    if(textureOn)
    {
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    }

    if(!vertexOn)
    {
	glEnableClientState(GL_VERTEX_ARRAY);
    }
    glEnableClientState(GL_NORMAL_ARRAY);
    
    for(int i = 0; i < _quadVBOs[context].size(); i++)
    {
	//continue;
	glColor3f(_colorMap[context][i].x(),_colorMap[context][i].y(),_colorMap[context][i].z());
	for(int j = 0; j < _quadSize[context][i].size(); j++)
	{
	    glBindBufferARB(GL_ARRAY_BUFFER_ARB, _quadVBOs[context][i].second[j]);
	    glNormalPointer(GL_FLOAT, 0, 0);
	    glBindBufferARB(GL_ARRAY_BUFFER_ARB, _quadVBOs[context][i].first[j]);
	    glVertexPointer(3, GL_FLOAT, 0, 0);
	    glDrawArrays(GL_QUADS, 0, _quadSize[context][i][j] * 4);
	}
	//std::cerr << "drawing " << _quadSize[context][i] << " quads" << std::endl;
    }

    for(int i = 0; i < _triVBOs[context].size(); i++)
    {
	glColor3f(_colorMap[context][i].x(),_colorMap[context][i].y(),_colorMap[context][i].z());
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, _triVBOs[context][i].second);
	glNormalPointer(GL_FLOAT, 0, 0);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, _triVBOs[context][i].first);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glDrawArrays(GL_TRIANGLES, 0, _triSize[context][i] * 3);
    }

    glDisableClientState(GL_NORMAL_ARRAY);
    /*for(int i = 0; i < _quadVBOs[context].size(); i++)
    {
	glColor3f(_colorMap[context][i].x(),_colorMap[context][i].y(),_colorMap[context][i].z());
	int dataStep = 0;
	for(int j = 0; j < _quadSize[context][i]; j++)
	{
	    glBegin(GL_QUADS);
	    glVertex3f(_quadData[context][i].first[dataStep++],_quadData[context][i].first[dataStep++],_quadData[context][i].first[dataStep++]);
	    glVertex3f(_quadData[context][i].first[dataStep++],_quadData[context][i].first[dataStep++],_quadData[context][i].first[dataStep++]);
	    glVertex3f(_quadData[context][i].first[dataStep++],_quadData[context][i].first[dataStep++],_quadData[context][i].first[dataStep++]);
	    glVertex3f(_quadData[context][i].first[dataStep++],_quadData[context][i].first[dataStep++],_quadData[context][i].first[dataStep++]);
	    glEnd();
	}
    }*/

    glDisable(GL_LIGHTING);
    for(int i = 0; i < _lineVBOs[context].size(); i++)
    {
	if(!_drawLines)
	{
	    continue;
	}
	glColor3f(_colorMap[context][i].x(),_colorMap[context][i].y(),_colorMap[context][i].z());
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, _lineVBOs[context][i]);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glDrawArrays(GL_LINES, 0, _lineSize[context][i]*2);
    }


    if(!vertexOn)
    {
	glDisableClientState(GL_VERTEX_ARRAY);
    }
    //glDisableClientState(GL_NORMAL_ARRAY);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
    glEnable(GL_LIGHTING);

    if(textureOn)
    {
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    }
    //glFinish();
    //getTime(end);
    //printDiff("Get State: ",start,end);
}

void MultiGPUDrawable::drawFrameVBOs(int context) const
{
    if(_useDrawShader)
    {
	if(_useGeometryShader)
	{
	    glUseProgram(_drawGeoProgMap[context]);
	}
	else
	{
	    glUseProgram(_drawShaderProgMap[context]);
	}
    }

    bool textureOn = glIsEnabled(GL_TEXTURE_COORD_ARRAY);
    bool vertexOn = glIsEnabled(GL_VERTEX_ARRAY);
    //bool normalOn = glIsEnabled(GL_NORMAL_ARRAY);
    

    if(textureOn)
    {
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    }

    if(!vertexOn)
    {
	glEnableClientState(GL_VERTEX_ARRAY);
    }

    if(!_useGeometryShader)
    {
	glEnableClientState(GL_NORMAL_ARRAY);
    }
    
    for(int i = 0; i < _quadVBOs[context].size(); i++)
    {
	if(_useDrawShader && _gpus > 1)
	{
	    glColor3f(((float)_currentFrame->gpuPartsMap[context][i])/29.0,0.0,0.0);
	}
	else
	{
	    glColor3f(_currentFrame->colorList[_currentFrame->gpuPartsMap[context][i]].x(),_currentFrame->colorList[_currentFrame->gpuPartsMap[context][i]].y(),_currentFrame->colorList[_currentFrame->gpuPartsMap[context][i]].z());
	}
	for(int j = 0; j < _quadSize[context][i].size(); j++)
	{
	    if(!_quadVBOs[context][i].first[j])
	    {
		continue;
	    }
	    if(!_useGeometryShader)
	    {
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, _quadVBOs[context][i].second[j]);
		glNormalPointer(GL_FLOAT, 0, 0);
	    }
	    glBindBufferARB(GL_ARRAY_BUFFER_ARB, _quadVBOs[context][i].first[j]);
	    glVertexPointer(3, GL_FLOAT, 0, 0);
	    glDrawArrays(GL_QUADS, 0, _quadSize[context][i][j] * 4);
	}
	//std::cerr << "drawing " << _quadSize[context][i] << " quads" << std::endl;
    }

    for(int i = 0; i < _triVBOs[context].size(); i++)
    {
	if(_useDrawShader && _gpus > 1)
	{
	    glColor3f(((float)_currentFrame->gpuPartsMap[context][i])/29.0,0.0,0.0);
	}
	else
	{
	    glColor3f(_currentFrame->colorList[_currentFrame->gpuPartsMap[context][i]].x(),_currentFrame->colorList[_currentFrame->gpuPartsMap[context][i]].y(),_currentFrame->colorList[_currentFrame->gpuPartsMap[context][i]].z());
	}
	if(!_triVBOs[context][i].first)
	{
	    continue;
	}
	if(!_useGeometryShader)
	{
	    glBindBufferARB(GL_ARRAY_BUFFER_ARB, _triVBOs[context][i].second);
	    glNormalPointer(GL_FLOAT, 0, 0);
	}
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, _triVBOs[context][i].first);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glDrawArrays(GL_TRIANGLES, 0, _triSize[context][i] * 3);
    }

    if(!_useGeometryShader)
    {
	glDisableClientState(GL_NORMAL_ARRAY);
    }
    /*for(int i = 0; i < _quadVBOs[context].size(); i++)
    {
	glColor3f(_colorMap[context][i].x(),_colorMap[context][i].y(),_colorMap[context][i].z());
	int dataStep = 0;
	for(int j = 0; j < _quadSize[context][i]; j++)
	{
	    glBegin(GL_QUADS);
	    glVertex3f(_quadData[context][i].first[dataStep++],_quadData[context][i].first[dataStep++],_quadData[context][i].first[dataStep++]);
	    glVertex3f(_quadData[context][i].first[dataStep++],_quadData[context][i].first[dataStep++],_quadData[context][i].first[dataStep++]);
	    glVertex3f(_quadData[context][i].first[dataStep++],_quadData[context][i].first[dataStep++],_quadData[context][i].first[dataStep++]);
	    glVertex3f(_quadData[context][i].first[dataStep++],_quadData[context][i].first[dataStep++],_quadData[context][i].first[dataStep++]);
	    glEnd();
	}
    }*/

    if(_useDrawShader && _useGeometryShader)
    {
	glUseProgram(_drawShaderProgMap[context]);
    }

    glDisable(GL_LIGHTING);
    for(int i = 0; i < _lineVBOs[context].size(); i++)
    {
	if(!_drawLines)
	{
	    continue;
	}
	if(_useDrawShader && _gpus > 1)
	{
	    glColor3f(((float)_currentFrame->gpuPartsMap[context][i])/29.0,1.0,0.0);
	}
	else
	{
	    glColor3f(_currentFrame->colorList[_currentFrame->gpuPartsMap[context][i]].x(),_currentFrame->colorList[_currentFrame->gpuPartsMap[context][i]].y(),_currentFrame->colorList[_currentFrame->gpuPartsMap[context][i]].z());
	}
	if(!_lineVBOs[context][i])
	{
	    continue;
	}
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, _lineVBOs[context][i]);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glDrawArrays(GL_LINES, 0, _lineSize[context][i]*2);
    }


    if(!vertexOn)
    {
	glDisableClientState(GL_VERTEX_ARRAY);
    }
    //glDisableClientState(GL_NORMAL_ARRAY);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
    glEnable(GL_LIGHTING);

    if(textureOn)
    {
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    }

    if(_useDrawShader)
    {
	glUseProgram(0);
    }
}

void MultiGPUDrawable::drawImplementation(RenderInfo & ri) const
{
    //struct timeval sini, eini;
    //std::cerr << "Draw start." << std::endl;
    //getTime(sini);
    //glFinish();
    //getTime(eini);
    //printDiff("Init Finish: ",sini,eini);

    /*GLint maxt;
    glGetIntegerv(GL_MAX_TEXTURE_COORDS,&maxt);
    std::cerr << "Max Tex Coords: " << maxt << std::endl;
    glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS,&maxt);
    std::cerr << "Max Comb Tex Coords: " << maxt << std::endl;*/

    //struct timeval drawStart, drawEnd;
    //getTime(drawStart);

    //printDiff("DrawStart : ",_preFrameTime,drawStart);

    //glPushAttrib(GL_ALL_ATTRIB_BITS);

    int context = ri.getContextID();

    if(context >= _gpus)
    {
	return;
    }

    if(CVRViewer::instance()->done() && _cudaCopy)
    {
	cudaCleanup(context);
	return;
    }

    std::stringstream contextss;
    contextss << context << ": ";

    if(_getTimings[context])
    {
	glFinish();
    }

    if(!_initMap[context])
    {
	initBuffers(context);
	initShaders(context);
	//loadGeometry(context);
	_initMap[context] = true;
    }

    if(_makeVBOs[context])
    {
	initVBOs(context);
	_makeVBOs[context] = false;
    }

    struct timeval loadstart,loadend;

    if(_loadVBOs[context])
    {
#ifdef PRINT_TIMING
	glFinish();
	getTime(loadstart);
#endif
	if(!_madeVBOs[context])
	{
	    //std::cerr << "Loading Initial VBO" << std::endl;
	    loadVBOs(context);
#ifdef PRINT_TIMING
	    glFinish();
	    getTime(loadend);
	    printDiff(contextss.str() + "Load time: ",loadstart,loadend);
#endif
	}
	else if(!_cudaCopy && context == 0)
	{
	    //OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_vboMutex);
	    //std::cerr << "Loading next VBO" << std::endl;
	    loadNextVBOs(context);
#ifdef PRINT_TIMING
	    glFinish();
	    getTime(loadend);
	    printDiff(contextss.str() + "Load time: ",loadstart,loadend);
#endif
	}
    }

    //_geometryMap[context]->drawImplementation(ri);

    //pid_t tid = (pid_t) syscall(SYS_gettid);
    //std::cerr << "Thread ID: " << tid << std::endl;

    //return;

    if(!_madeVBOs[context])
    {
	return;
    }

    //glFinish();
    //_drawBlock->completed();
    //_drawBlock->block();
    //return;

    if(_gpus > 1)
    {
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, _frameBufferMap[context]);
    }

    //drawContext(context);
#ifdef PRINT_TIMING

    glFinish();
    struct timeval vboDrawStart, vboDrawEnd;
    getTime(vboDrawStart);

#endif

    GLenum buffers[] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_COLOR_ATTACHMENT2_EXT};
    GLenum buffer = GL_COLOR_ATTACHMENT0_EXT;

    if(_useDrawShader)
    {
	if(_depth == 16)
	{
	    glDrawBuffers(2, buffers);
	}
	else if(_depth == 24)
	{
	    glDrawBuffers(3, buffers);
	}
	//glUseProgram(_drawShaderProgMap[context]);
    }

    if(_gpus > 1)
    {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    if(_currentFrame)
    {
	//OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_vboMutex);
	drawFrameVBOs(context);
    }
    else
    {
	drawVBOs(context);
    }

    if(_useDrawShader)
    {
	//glUseProgram(0);
	glDrawBuffers(1, &buffer);
    }
    
    glFinish();
#ifdef PRINT_TIMING
    getTime(vboDrawEnd);
    printDiff(contextss.str() + "VBO Draw: ",vboDrawStart,vboDrawEnd);
#endif

    if(_gpus == 1)
    {
	if(_getTimings[context])
	{
	    _getTimings[context] = false;
	}
	return;
    }
    //return;
    /*glFinish();
    getTime(drawEnd);
    printDiff("Draw end : ",_preFrameTime,drawEnd);
    printDiff("Draw : ",drawStart,drawEnd);
    return;*/

    //struct timeval srem, erem;
    //getTime(srem);

    //getTime(drawEnd);
    //rintDiff("Draw : ",drawStart,drawEnd);
    //glPopAttrib();
    //return;
    /*if(context == 0)
    {
	_geometryMap[context]->drawImplementation(ri);
    }
    else if(context == 1)
    {
	glPushMatrix();
	glRotatef(180.0,0.0,0.0,1.0);
	_geometryMap[context]->drawImplementation(ri);
	glPopMatrix();
    }*/

    //bool _usePBOs = true;

    if(context != 0)
    {
	/*glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
	glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, _colorBufferMap[context]);
	glBufferData(GL_PIXEL_PACK_BUFFER_ARB,_width*_height*4,NULL,GL_STREAM_READ);
	glReadPixels(0,0,_width,_height,GL_RGBA,GL_UNSIGNED_BYTE,BUFFER_OFFSET(0));

	void * mapped = glMapBuffer(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY);
	memcpy(_colorDataMap[context],mapped,_width*_height*4);
	glUnmapBuffer(GL_PIXEL_PACK_BUFFER_ARB);*/

	/*glReadBuffer(GL_DEPTH_ATTACHMENT_EXT);
	glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, _depthBufferMap[context]);
	glBufferData(GL_PIXEL_PACK_BUFFER_ARB,_width*_height*sizeof(GLint),NULL,GL_STREAM_READ);
	glReadPixels(0,0,_width,_height,GL_DEPTH_COMPONENT24,GL_INT,BUFFER_OFFSET(0));

	mapped = glMapBuffer(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY);
	memcpy(_depthDataMap[context],mapped,_width*_height*sizeof(GLint));
	glUnmapBuffer(GL_PIXEL_PACK_BUFFER_ARB);*/

	//glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB,0);

#ifdef PRINT_TIMING
	glFinish();
	timeval sstore,estore;
	getTime(sstore);
#endif

	

	glPixelStorei(GL_PACK_ALIGNMENT,1);
	glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);

	if(_usePBOs)
	{
	    //std::cerr << "PBO copy" << std::endl;
	    glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, _colorBufferMap[context]);
	    //glBufferData(GL_PIXEL_PACK_BUFFER_ARB,_width*_height*2,NULL,GL_STREAM_READ);
	    glReadPixels(0,0,_width,_height,GL_RG,GL_UNSIGNED_BYTE,BUFFER_OFFSET(0)); 

	    void * mapped = glMapBuffer(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY);
	    memcpy(_colorDataMap[context],mapped,_width*_height*2);
	    glUnmapBuffer(GL_PIXEL_PACK_BUFFER_ARB);
	    glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB,0);
	}
	else
	{
	    //glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
	    glReadPixels(0,0,_width,_height,GL_RG,GL_UNSIGNED_BYTE,_colorDataMap[context]);
	}

	if(_depth == 32)
	{
	    if(_usePBOs)
	    {
		glReadBuffer(GL_DEPTH_ATTACHMENT_EXT);
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, _depthBufferMap[context]);
		//glBufferData(GL_PIXEL_PACK_BUFFER_ARB,_width*_height*sizeof(GLint),NULL,GL_STREAM_READ);
		glReadPixels(0,0,_width,_height,GL_DEPTH_COMPONENT,GL_FLOAT,BUFFER_OFFSET(0));

		void * mapped = glMapBuffer(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY);
		memcpy(_depthDataMap[context],mapped,_width*_height*4);
		glUnmapBuffer(GL_PIXEL_PACK_BUFFER_ARB);
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB,0);
	    }
	    else
	    {
		glReadPixels(0,0,_width,_height,GL_DEPTH_COMPONENT,GL_FLOAT,_depthDataMap[context]);
	    }
	}
	else if(_depth == 16)
	{
	    glReadBuffer(GL_COLOR_ATTACHMENT1_EXT);
	    if(_usePBOs)
	    {
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, _depthBufferMap[context]);
		//glBufferData(GL_PIXEL_PACK_BUFFER_ARB,_width*_height*sizeof(GLint),NULL,GL_STREAM_READ);
		glReadPixels(0,0,_width,_height,GL_RED,GL_UNSIGNED_SHORT,BUFFER_OFFSET(0));

		void * mapped = glMapBuffer(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY);
		memcpy(_depthDataMap[context],mapped,_width*_height*2);
		glUnmapBuffer(GL_PIXEL_PACK_BUFFER_ARB);
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB,0);
	    }
	    else
	    {
		glReadPixels(0,0,_width,_height,GL_RED,GL_UNSIGNED_SHORT,_depthDataMap[context]);
	    }
	}
	else if(_depth == 24)
	{
	    glReadBuffer(GL_COLOR_ATTACHMENT1_EXT);
	    if(_usePBOs)
	    {
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, _depthBufferMap[context]);
		//glBufferData(GL_PIXEL_PACK_BUFFER_ARB,_width*_height*sizeof(GLint),NULL,GL_STREAM_READ);
		glReadPixels(0,0,_width,_height,GL_RED,GL_UNSIGNED_SHORT,BUFFER_OFFSET(0));

		void * mapped = glMapBuffer(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY);
		memcpy(_depthDataMap[context],mapped,_width*_height*2);
		glUnmapBuffer(GL_PIXEL_PACK_BUFFER_ARB);
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB,0);
	    }
	    else
	    {
		glReadPixels(0,0,_width,_height,GL_RED,GL_UNSIGNED_SHORT,_depthDataMap[context]);
	    }
	    
	    glReadBuffer(GL_COLOR_ATTACHMENT2_EXT);
	    if(_usePBOs)
	    {
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, _depthR8BufferMap[context]);
		//glBufferData(GL_PIXEL_PACK_BUFFER_ARB,_width*_height*sizeof(GLint),NULL,GL_STREAM_READ);
		glReadPixels(0,0,_width,_height,GL_RED,GL_UNSIGNED_BYTE,BUFFER_OFFSET(0));

		void * mapped = glMapBuffer(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY);
		memcpy(_depthR8DataMap[context],mapped,_width*_height);
		glUnmapBuffer(GL_PIXEL_PACK_BUFFER_ARB);
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB,0);
	    }
	    else
	    {
		glReadPixels(0,0,_width,_height,GL_RED,GL_UNSIGNED_BYTE,_depthR8DataMap[context]);
	    }
	}

	//char * temp = (char*)_colorDataMap[context];
	//std::cerr << "buffer " << (int)temp[0] << " " << (int)temp[1] << " " << (int)temp[2] << " " << (int)temp[3] << std::endl;

	glFinish();
#ifdef PRINT_TIMING
	getTime(estore);
	printDiff(contextss.str() + "Store Time: ",sstore,estore);
#endif

	threadSyncBlock[context-1] = true;
	//std::cerr << "Context " << context << " done." << std::endl;
    }
    else
    {
	//timeval swait,ewait;
	//getTime(swait);
	//std::cerr << "Wait for threads." << std::endl;
	
	struct timeval drawStart,drawEnd;
	if(_getTimings[context])
	{
	    glFinish();
	    getTime(drawStart);
	}
	
	// wait for other threads to finish copy
	
	bool copyDone[_gpus-1];
	for(int i = 0; i < _gpus - 1; i++)
	{
	    copyDone[i] = false;
	}

	glPixelStorei(GL_PACK_ALIGNMENT,1);
	while(1)
	{
	    bool syncDone = true;
	    for(int i = 0; i < _gpus - 1; i++)
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
	    for(int i = 0; i < _gpus - 1; i++)
	    {
		if(!copyDone[i] && threadSyncBlock[i])
		{
#ifdef PRINT_TIMING
		    glFinish();
		    struct timeval cbstart,cbend;
		    getTime(cbstart);
#endif
		    glBindTexture(GL_TEXTURE_RECTANGLE_ARB,colorTextures[i]);
		    if(_usePBOs)
		    {
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, colorCopyBuffers[i]);
			//glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB,_width*_height*4, NULL, GL_STREAM_DRAW);
			void* buffer = glMapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY); 
			memcpy(buffer,_colorDataMap[i+1],_width*_height*2);
			glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB);
			glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, _width, _height, GL_RG, GL_UNSIGNED_BYTE, BUFFER_OFFSET(0));
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		    }
		    else
		    {
			glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, _width, _height, GL_RG, GL_UNSIGNED_BYTE, _colorDataMap[i+1]);
		    }

		    if(_depth == 32)
		    {
			glBindTexture(GL_TEXTURE_RECTANGLE_ARB,depthTextures[i]);

			if(_usePBOs)
			{
			    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, depthCopyBuffers[i]);
			    //glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB,_width*_height*4, NULL, GL_STREAM_DRAW);
			    void* buffer = glMapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY); 
			    memcpy(buffer,_depthDataMap[i+1],_width*_height*4);
			    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB);
			    //glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, _width, _height, GL_RG, GL_UNSIGNED_BYTE, BUFFER_OFFSET(0));
			    glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, _width, _height, GL_DEPTH_COMPONENT, GL_FLOAT, BUFFER_OFFSET(0));
			    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
			}
			else
			{
			    glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, _width, _height, GL_DEPTH_COMPONENT, GL_FLOAT, _depthDataMap[i+1]);
			}
		    }
		    else if(_depth == 16)
		    {
			glBindTexture(GL_TEXTURE_RECTANGLE_ARB,depthRTextures[i]);
			
			if(_usePBOs)
			{
			    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, depthCopyBuffers[i]);
			    //glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB,_width*_height*4, NULL, GL_STREAM_DRAW);
			    void* buffer = glMapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY); 
			    memcpy(buffer,_depthDataMap[i+1],_width*_height*2);
			    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB);
			    glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, _width, _height, GL_RED, GL_UNSIGNED_SHORT, BUFFER_OFFSET(0));
			    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
			}
			else
			{
			    glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, _width, _height, GL_RED, GL_UNSIGNED_SHORT, _depthDataMap[i+1]);
			}
		    }
		    else if(_depth == 24)
		    {
			glBindTexture(GL_TEXTURE_RECTANGLE_ARB,depthRTextures[i]);
			if(_usePBOs)
			{
			    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, depthCopyBuffers[i]);
			    //glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB,_width*_height*4, NULL, GL_STREAM_DRAW);
			    void* buffer = glMapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY); 
			    memcpy(buffer,_depthDataMap[i+1],_width*_height*2);
			    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB);
			    glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, _width, _height, GL_RED, GL_UNSIGNED_SHORT, BUFFER_OFFSET(0));
			    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
			}
			else
			{
			    glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, _width, _height, GL_RED, GL_UNSIGNED_SHORT, _depthDataMap[i+1]);
			}

			glBindTexture(GL_TEXTURE_RECTANGLE_ARB,depthR8Textures[i]);
			if(_usePBOs)
			{
			    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, depthR8CopyBuffers[i]);
			    //glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB,_width*_height*4, NULL, GL_STREAM_DRAW);
			    void* buffer = glMapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY); 
			    memcpy(buffer,_depthR8DataMap[i+1],_width*_height);
			    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB);
			    glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, _width, _height, GL_RED, GL_UNSIGNED_BYTE, BUFFER_OFFSET(0));
			    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
			}
			else
			{
			    glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, _width, _height, GL_RED, GL_UNSIGNED_BYTE, _depthR8DataMap[i+1]);
			}
		    }
		    /*glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, depthCopyBuffers[i]);
		      glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB,_width*_height*sizeof(GLint), NULL, GL_STREAM_DRAW);
		      buffer = glMapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY); 
		      memcpy(buffer,_depthDataMap[i+1],_width*_height*sizeof(GLint));
		      glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB);
		      glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, _width, _height, GL_DEPTH_COMPONENT24, GL_INT, BUFFER_OFFSET(0));
		      glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);*/
		    glBindTexture(GL_TEXTURE_RECTANGLE_ARB,0);

		    threadSyncBlock[i] = false;
		    copyDone[i] = true;
#ifdef PRINT_TIMING
		    glFinish();
		    getTime(cbend);
		    std::stringstream cbss;
		    cbss << i+1 << ": ";
		    printDiff(cbss.str() + "CopyBackTime: ", cbstart,cbend);
#endif
		}
	    }
	    //std::cerr << "Sync not Done." << std::endl;
	    //OpenThreads::Thread::YieldCurrentThread();
	}

#ifdef PRINT_TIMING
	struct timeval shaderstart,shaderend;
	getTime(shaderstart);
#endif

	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,0);
	drawScreen();

#ifdef PRINT_TIMING
	glFinish();
	getTime(shaderend);
	printDiff(contextss.str() + "shader time: ",shaderstart,shaderend);
#endif
	//struct timespec stime;
	//stime.tv_sec = 1;
	//stime.tv_nsec = 0;
	//nanosleep(&stime,NULL);
	if(_getTimings[context])
	{
	    glFinish();
	    getTime(drawEnd);
	    _drawTime = getDiff(drawStart,drawEnd);
	}
    }

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,0);

    if(!_cudaCopy && _loadVBOs[context])
    {
	if(context != 0)
	{
#ifdef PRINT_TIMING
	    glFinish();
	    getTime(loadstart);
#endif

	    loadNextVBOs(context);

#ifdef PRINT_TIMING
	    glFinish();
	    getTime(loadend);
	    printDiff(contextss.str() + "Load time: ",loadstart,loadend);
#endif
	}
    }

    if(_getTimings[context])
    {
	_getTimings[context] = false;
    }

    //glPopAttrib();
    //std::cerr << "Draw done." << std::endl;
    //il_check_error();
    //
    //glFinish();

    //getTime(erem);
    //printDiff("Remainder: ",srem,erem);

    //glFinish();
    //getTime(drawEnd);
    //printDiff("Draw end : ",_preFrameTime,drawEnd);
    //printDiff("Draw : ",drawStart,drawEnd);
}

void MultiGPUDrawable::cudaCleanup(int context) const
{
    std::cerr << "Cuda Cleanup" << std::endl;
    for(int i = 0; i < _lineVBOs[context].size(); i++)
    {
	if(_lineVBOs[context][i])
	{
	    checkUnregBufferObj(_lineVBOs[context][i]);
	}
    }
    for(int i = 0; i < _lineNextVBOs[context].size(); i++)
    {
	if(_lineNextVBOs[context][i])
	{
	    checkUnregBufferObj(_lineNextVBOs[context][i]);
	}
    }

    for(int i = 0; i < _quadVBOs[context].size(); i++)
    {
	for(int j = 0; j < _quadVBOs[context][i].first.size(); j++)
	{
	}
    }

    cudaThreadExit();
}

void MultiGPUDrawable::setFrame(struct AFrame * frame)
{
    _currentFrame = frame;
    for(int i = 0; i < _gpus; i++)
    {
	_loadVBOs[i] = true;
    }
}

void MultiGPUDrawable::setNextFrame(AFrame * frame, unsigned int bytes)
{
    _nextFrame = frame;
    _nextFrameBytes = bytes;
    for(int i = 0; i < _gpus; i++)
    {
	_loadVBOs[i] = true;
	_loadNextFrameSize[i] = true;
	_nextFrameLoadDone[i] = false;
	_prgParts[i] = 0;
	_prgOffset[i] = 0;
	_prgStage[i] = LINE;
	_prgQuadNum[i] = 0;
    }
}

bool MultiGPUDrawable::nextFrameLoadDone()
{
    bool done = true;
    for(int i = 0; i < _gpus; i++)
    {
	if(!_nextFrameLoadDone[i])
	{
	    done = false;
	}
    }
    return done;
}


void MultiGPUDrawable::swapFrames()
{
    std::map<int,std::vector<GLuint> > tempLine;
    std::map<int,std::vector<std::pair<std::vector<GLuint>,std::vector<GLuint> > > > tempQuad;
    std::map<int,std::vector<std::pair<GLuint,GLuint> > > tempTri;

    tempLine = _lineVBOs;
    tempQuad = _quadVBOs;
    tempTri = _triVBOs;

    _lineVBOs = _lineNextVBOs;
    _quadVBOs = _quadNextVBOs;
    _triVBOs = _triNextVBOs;

    _lineNextVBOs = tempLine;
    _quadNextVBOs = tempQuad;
    _triNextVBOs = tempTri;

    _lineSize = _lineNextSize;
    _quadSize = _quadNextSize;
    _triSize = _triNextSize;

    _currentFrame = _nextFrame;
    _nextFrame = NULL;
}

void MultiGPUDrawable::drawContext(int context) const
{
    glDisable(GL_LIGHTING);
    if(context == 1)
    {
	glBegin(GL_QUADS);
	    glNormal3f(0,-1,0);
	    glColor3f(1.0,0.0,0.0);
	    glVertex3f(-200,0,100);
	    glNormal3f(0,-1,0);
	    glColor3f(1.0,0.0,0.0);
	    glVertex3f(200,0,100);
	    glNormal3f(0,-1,0);
	    glColor3f(1.0,0.0,0.0);
	    glVertex3f(200,0,-100);
	    glNormal3f(0,-1,0);
	    glColor3f(1.0,0.0,0.0);
	    glVertex3f(-200,0,-100);
	glEnd();
    }
    else if(context == 0)
    {
	glBegin(GL_QUADS);
	    glNormal3f(0,-1,0);
	    glColor3f(0.0,1.0,0.0);
	    glVertex3f(200,0,100);
	    glNormal3f(0,-1,0);
	    glColor3f(0.0,1.0,0.0);
	    glVertex3f(400,0,100);
	    glNormal3f(0,-1,0);
	    glColor3f(0.0,1.0,0.0);
	    glVertex3f(400,0,-100);
	    glNormal3f(0,-1,0);
	    glColor3f(0.0,1.0,0.0);
	    glVertex3f(200,0,-100);
	glEnd();
    }
    glEnable(GL_LIGHTING);
    /*if(context == 0)
    {
	_geometryMap[context]->drawImplementation();
    }
    else if(context == 1)
    {
	glPushMatrix();
	glRotatef(180.0,0.0,0.0,1.0);
	_geometryMap[context]->drawImplementation();
	glPopMatrix();
    }*/
}



void MultiGPUDrawable::initShaders(int context) const
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    if(context == 0)
    {

	redLookupUni[context] = new GLint[29];
	greenLookupUni[context] = new GLint[29];
	blueLookupUni[context] = new GLint[29];

	colorsUni = new GLint[_gpus];
	depthUni = new GLint[_gpus];
	depthR8Uni = new GLint[_gpus];

	struct stat st;
	if(stat(_fragFile.c_str(),&st) != 0)
	{
	    std::cerr << "Error stating shader file: " << _fragFile << std::endl;
	    return;
	}
	char * fileBuffer;
	int file;
	file = open(_fragFile.c_str(),O_RDONLY);
	if(!file)
	{
	    std::cerr << "Error opening shader file: " << _fragFile << std::endl;
	    return;
	}
	fileBuffer = new char[st.st_size+1];
	fileBuffer[st.st_size] = '\0';
	read(file,fileBuffer,st.st_size);

	close(file);

	_fragShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(_fragShader, 1, (const GLchar **) &fileBuffer, NULL);
	glCompileShader(_fragShader);
	
	if(il_check_shader_log(_fragShader))
	{
	    std::cerr << "Shader load ok." << std::endl;
	}
	else
	{
	    return;
	}

	delete[] fileBuffer;


	if(stat(_vertFile.c_str(),&st) != 0)
	{
	    std::cerr << "Error stating shader file: " << _vertFile << std::endl;
	    return;
	}
	file = open(_vertFile.c_str(),O_RDONLY);
	if(!file)
	{
	    std::cerr << "Error opening shader file: " << _vertFile << std::endl;
	    return;
	}
	fileBuffer = new char[st.st_size+1];
	fileBuffer[st.st_size] = '\0';
	read(file,fileBuffer,st.st_size);

	close(file);

	_vertShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(_vertShader, 1, (const GLchar **) &fileBuffer, NULL);
	glCompileShader(_vertShader);
	
	if(il_check_shader_log(_vertShader))
	{
	    std::cerr << "Shader load ok." << std::endl;
	}
	else
	{
	    return;
	}

	delete[] fileBuffer;

	_shaderProgram = glCreateProgram();
	glAttachShader(_shaderProgram,_vertShader);
	glAttachShader(_shaderProgram,_fragShader);

	glLinkProgram(_shaderProgram);

	if(il_check_program_log(_shaderProgram))
	{
	    std::cerr << "Shader Program link ok." << std::endl;
	}

	texturesUni = glGetUniformLocation(_shaderProgram,"textures");

	glUseProgram(_shaderProgram);
	glUniform1i(texturesUni,_gpus);
	glUseProgram(0);

	for(int i = 0; i < _gpus; i++)
	{
	    std::stringstream ss;
	    ss << "colors[" << i << "]";
	    colorsUni[i] = glGetUniformLocation(_shaderProgram,ss.str().c_str());
	}

	for(int i = 0; i < _gpus; i++)
	{
	    std::stringstream ss;
	    ss << "depth[" << i << "]";
	    depthUni[i] = glGetUniformLocation(_shaderProgram,ss.str().c_str());
	}

	if(_depth == 24)
	{
	    for(int i = 0; i < _gpus; i++)
	    {
		std::stringstream ss;
		ss << "depthR8[" << i << "]";
		depthR8Uni[i] = glGetUniformLocation(_shaderProgram,ss.str().c_str());
	    }
	}

	glUseProgram(_shaderProgram);

	for(int i = 0; i < 29; i++)
	{
	    std::stringstream ss;
	    ss << "redLookup[" << i << "]";
	    redLookupUni[context][i] = glGetUniformLocation(_shaderProgram,ss.str().c_str());
	    std::stringstream ss2;
	    ss2 << "greenLookup[" << i << "]";
	    greenLookupUni[context][i] = glGetUniformLocation(_shaderProgram,ss2.str().c_str());
	    std::stringstream ss3;
	    ss3 << "blueLookup[" << i << "]";
	    blueLookupUni[context][i] = glGetUniformLocation(_shaderProgram,ss3.str().c_str());
	}

	for(int i = 0; i < 29; i++)
	{
	    osg::Vec3 myColor = makeColor(((float)i)/31.0);
	    glUniform1f(redLookupUni[context][i],myColor.x());
	    glUniform1f(greenLookupUni[context][i],myColor.y());
	    glUniform1f(blueLookupUni[context][i],myColor.z());
	}
	glUseProgram(0);
    }
    {
	
	struct stat st;
	if(stat(_drawVertFile.c_str(),&st) != 0)
	{
	    std::cerr << "Error stating shader file: " << _drawVertFile << std::endl;
	    return;
	}
	char * fileBuffer;
	int file;
	file = open(_drawVertFile.c_str(),O_RDONLY);
	if(!file)
	{
	    std::cerr << "Error opening shader file: " << _drawVertFile << std::endl;
	    return;
	}
	fileBuffer = new char[st.st_size+1];
	fileBuffer[st.st_size] = '\0';
	read(file,fileBuffer,st.st_size);

	close(file);

	_drawVertShaderMap[context] = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(_drawVertShaderMap[context], 1, (const GLchar **) &fileBuffer, NULL);
	glCompileShader(_drawVertShaderMap[context]);
	
	if(il_check_shader_log(_drawVertShaderMap[context]))
	{
	    std::cerr << "Shader load ok." << std::endl;
	}
	else
	{
	    return;
	}

	delete[] fileBuffer;

	if(_depth == 24 || _depth == 16)
	{
	    if(stat(_drawFragFile.c_str(),&st) != 0)
	    {
		std::cerr << "Error stating shader file: " << _drawFragFile << std::endl;
		return;
	    }
	    file = open(_drawFragFile.c_str(),O_RDONLY);
	    if(!file)
	    {
		std::cerr << "Error opening shader file: " << _drawFragFile << std::endl;
		return;
	    }
	    fileBuffer = new char[st.st_size+1];
	    fileBuffer[st.st_size] = '\0';
	    read(file,fileBuffer,st.st_size);

	    close(file);

	    _drawFragShaderMap[context] = glCreateShader(GL_FRAGMENT_SHADER);
	    glShaderSource(_drawFragShaderMap[context], 1, (const GLchar **) &fileBuffer, NULL);
	    glCompileShader(_drawFragShaderMap[context]);

	    if(il_check_shader_log(_drawFragShaderMap[context]))
	    {
		std::cerr << "Shader load ok." << std::endl;
	    }
	    else
	    {
		return;
	    }

	    delete[] fileBuffer;
	}

	_drawShaderProgMap[context] = glCreateProgram();
	glAttachShader(_drawShaderProgMap[context],_drawVertShaderMap[context]);
	if(_depth == 16 || _depth == 24)
	{
	    glAttachShader(_drawShaderProgMap[context],_drawFragShaderMap[context]);
	}

	glLinkProgram(_drawShaderProgMap[context]);

	if(il_check_program_log(_drawShaderProgMap[context]))
	{
	    std::cerr << "Draw Shader Program link ok." << std::endl;
	}

	if(_useGeometryShader)
	{
	    if(stat(_drawGeoFile.c_str(),&st) != 0)
	    {
		std::cerr << "Error stating shader file: " << _drawGeoFile << std::endl;
		return;
	    }
	    file = open(_drawGeoFile.c_str(),O_RDONLY);
	    if(!file)
	    {
		std::cerr << "Error opening shader file: " << _drawGeoFile << std::endl;
		return;
	    }
	    fileBuffer = new char[st.st_size+1];
	    fileBuffer[st.st_size] = '\0';
	    read(file,fileBuffer,st.st_size);

	    close(file);

	    _drawGeoShaderMap[context] = glCreateShader(GL_GEOMETRY_SHADER);
	    glShaderSource(_drawGeoShaderMap[context], 1, (const GLchar **) &fileBuffer, NULL);
	    glCompileShader(_drawGeoShaderMap[context]);

	    if(il_check_shader_log(_drawGeoShaderMap[context]))
	    {
		std::cerr << "Geo Shader load ok." << std::endl;
	    }

	    delete[] fileBuffer;

	    _drawGeoProgMap[context] = glCreateProgram();
	    glAttachShader(_drawGeoProgMap[context],_drawVertShaderMap[context]);
	    if(_depth == 16 || _depth == 24)
	    {
		glAttachShader(_drawGeoProgMap[context],_drawFragShaderMap[context]);
	    }
	    glAttachShader(_drawGeoProgMap[context],_drawGeoShaderMap[context]);

	    glProgramParameteriEXT(_drawGeoProgMap[context], GL_GEOMETRY_INPUT_TYPE_EXT, GL_TRIANGLES);
	    glProgramParameteriEXT(_drawGeoProgMap[context], GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);
	    glProgramParameteriEXT(_drawGeoProgMap[context],GL_GEOMETRY_VERTICES_OUT_EXT,3);

	    glLinkProgram(_drawGeoProgMap[context]);

	    if(il_check_program_log(_drawGeoProgMap[context]))
	    {
		std::cerr << "Draw Geo Shader Program link ok." << std::endl;
	    }
	}
    }
}

void MultiGPUDrawable::initBuffers(int context) const
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);

    glewInit();

    GLuint buffer;
    glGenFramebuffersEXT(1,&buffer);
    _frameBufferMap[context] = buffer;

    glGenTextures(1,&buffer);
    _colorTextureMap[context] = buffer;

    glGenTextures(1,&buffer);
    _depthTextureMap[context] = buffer;

    if(_depth == 16)
    {
	glGenTextures(1,&buffer);
	_depthRTextureMap[context] = buffer;
    }
    else if(_depth == 24)
    {
	glGenTextures(1,&buffer);
	_depthRTextureMap[context] = buffer;
	glGenTextures(1,&buffer);
	_depthR8TextureMap[context] = buffer;
    }

    std::cerr << "frame: " << _frameBufferMap[context] << " color: " << _colorTextureMap[context] << " depth: " << _depthTextureMap[context] << std::endl;

    // init depth texture
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB,_depthTextureMap[context]);
    glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_DEPTH_COMPONENT32, _width, _height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
    glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB,0);

    // init color texture
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, _colorTextureMap[context]);
    glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RG8, _width, _height, 0, GL_RG, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);

    if(_depth == 16)
    {
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, _depthRTextureMap[context]);
	glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_R16, _width, _height, 0, GL_RED, GL_UNSIGNED_SHORT, NULL);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
    }
    else if(_depth == 24)
    {
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, _depthRTextureMap[context]);
	glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_R16, _width, _height, 0, GL_RED, GL_UNSIGNED_SHORT, NULL);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);

	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, _depthR8TextureMap[context]);
	glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_R8, _width, _height, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
    }

    // bind textures to frame buffer
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, _frameBufferMap[context]);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_RECTANGLE_ARB, _colorTextureMap[context], 0);
    if(_depth == 16)
    {
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_TEXTURE_RECTANGLE_ARB, _depthRTextureMap[context], 0);
    }
    else if(_depth == 24)
    {
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_TEXTURE_RECTANGLE_ARB, _depthRTextureMap[context], 0);
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT2_EXT, GL_TEXTURE_RECTANGLE_ARB, _depthR8TextureMap[context], 0);
    }

    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_RECTANGLE_ARB, _depthTextureMap[context], 0);
    il_check_framebuffer();
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,0);

    if(context > 0)
    {
	if(_usePBOs)
	{
	    glGenBuffers(1,&buffer);
	    _colorBufferMap[context] = buffer;

	    glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, _colorBufferMap[context]);
	    glBufferData(GL_PIXEL_PACK_BUFFER_ARB,_width*_height*2,NULL,GL_STREAM_READ);
	    glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);

	    glGenBuffers(1,&buffer);
	    _depthBufferMap[context] = buffer;

	    if(_depth == 16 || _depth == 24)
	    {
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, _depthBufferMap[context]);
		glBufferData(GL_PIXEL_PACK_BUFFER_ARB,_width*_height*2,NULL,GL_STREAM_READ);
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
	    }

	    if(_depth == 32)
	    {
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, _depthBufferMap[context]);
		glBufferData(GL_PIXEL_PACK_BUFFER_ARB,_width*_height*4,NULL,GL_STREAM_READ);
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
	    }

	    if(_depth == 24)
	    {
		glGenBuffers(1,&buffer);
		_depthR8BufferMap[context] = buffer;

		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, _depthR8BufferMap[context]);
		glBufferData(GL_PIXEL_PACK_BUFFER_ARB,_width*_height,NULL,GL_STREAM_READ);
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
	    }
	}

	_colorDataMap[context] = new GLint[_width*_height];
	_depthDataMap[context] = new GLint[_width*_height];

	if(_depth == 24)
	{
	    _depthR8DataMap[context] = new unsigned char[_width*_height];
	}
    }
    else
    {
	if(_usePBOs)
	{
	    glGenBuffers(_gpus-1,colorCopyBuffers);
	    glGenBuffers(_gpus-1,depthCopyBuffers);
	    if(_depth == 24)
	    {
		glGenBuffers(_gpus-1,depthR8CopyBuffers);
	    }
	}

	glGenTextures(_gpus-1,colorTextures);
	if(_depth == 32)
	{
	    glGenTextures(_gpus-1,depthTextures);
	}
	else if(_depth == 16)
	{
	    glGenTextures(_gpus-1,depthRTextures);
	}
	else if(_depth == 24)
	{
	    glGenTextures(_gpus-1,depthRTextures);
	    glGenTextures(_gpus-1,depthR8Textures);
	}

	for(int i = 0; i < _gpus-1; i++)
	{
	    if(_depth == 32)
	    {
		glBindTexture(GL_TEXTURE_RECTANGLE_ARB,depthTextures[i]);
		glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_DEPTH_COMPONENT32, _width, _height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
		glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE);
	    }
	    else if(_depth == 16)
	    {
		glBindTexture(GL_TEXTURE_RECTANGLE_ARB,depthRTextures[i]);
		glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_R16, _width, _height, 0, GL_RED, GL_UNSIGNED_SHORT, NULL);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	    }
	    else if(_depth == 24)
	    {
		glBindTexture(GL_TEXTURE_RECTANGLE_ARB,depthRTextures[i]);
		glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_R16, _width, _height, 0, GL_RED, GL_UNSIGNED_SHORT, NULL);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glBindTexture(GL_TEXTURE_RECTANGLE_ARB,depthR8Textures[i]);
		glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_R8, _width, _height, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	    }


	    glBindTexture(GL_TEXTURE_RECTANGLE_ARB,0);

	    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, colorTextures[i]);
	    glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RG8, _width, _height, 0, GL_RG, GL_UNSIGNED_BYTE, NULL);
	    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);

	    if(_usePBOs)
	    {
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, colorCopyBuffers[i]);
		glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB,_width*_height*2, NULL, GL_STREAM_DRAW);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

		if(_depth == 16 || _depth == 24)
		{
		    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, depthCopyBuffers[i]);
		    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB,_width*_height*2, NULL, GL_STREAM_DRAW);
		    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		}
		if(_depth == 32)
		{
		    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, depthCopyBuffers[i]);
		    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB,_width*_height*4, NULL, GL_STREAM_DRAW);
		    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		}
		if(_depth == 24)
		{
		    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, depthR8CopyBuffers[i]);
		    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB,_width*_height, NULL, GL_STREAM_DRAW);
		    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		}
	    }
	}

	glGenBuffers(1,&_screenArray);
	glBindBuffer(GL_ARRAY_BUFFER, _screenArray);
	float points[8] = {-1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0};

	glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(float), points, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
    }
    il_check_error();
}

void MultiGPUDrawable::loadGeometry(int context) const
{
    osg::Node * node = osgDB::readNodeFile("/home/aprudhom/data/honda/FramePerCellNormals.ive");
    std::cerr << "Node: " << node << std::endl;
    osg::Geode * geode = dynamic_cast<osg::Geode*>(node);
    _geometryMap[context] = geode->getDrawable(1);
    /*_geometryMap[context] = new osg::Geometry();

    osg::Geometry * geo = _geometryMap[context];

    osg::Vec3 pos = osg::Vec3(0,0,0);
    float width = 200;
    float height = 100;

    osg::Vec3Array* verts = new osg::Vec3Array();
    verts->push_back( pos + osg::Vec3(-width,0,height));
    verts->push_back( pos + osg::Vec3(width, 0, height) );
    verts->push_back( pos + osg::Vec3(width, 0, -height) );
    verts->push_back( pos + osg::Vec3(-width, 0, -height) );

    geo->setVertexArray( verts );

    osg::DrawElementsUInt * ele = new osg::DrawElementsUInt(osg::PrimitiveSet::QUADS, 0);

    ele->push_back(0);
    ele->push_back(1);
    ele->push_back(2);
    ele->push_back(3);
    geo->addPrimitiveSet(ele);

    osg::Vec4Array* colors = new osg::Vec4Array;
    colors->push_back( osg::Vec4(1,0,0,1) );


    geo->setColorArray(colors);
    geo->setColorBinding(osg::Geometry::BIND_OVERALL);

    Vec3Array * normals = new Vec3Array(1);
    (*normals)[0].set(0,1,0);
    geo->setNormalArray(normals);
    geo->setNormalBinding(osg::Geometry::BIND_OVERALL);

    osg::StateSet * stateset = geo->getOrCreateStateSet();
    osg::Material * mat = new osg::Material();

    stateset->setAttributeAndModes(mat, StateAttribute::ON);
    */
}

void MultiGPUDrawable::drawScreen() const
{
    // setup shader

    //std::cerr << "doing screen draw." << std::endl;

    glUseProgram(_shaderProgram);
    {
	int index = 1;
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB,_colorTextureMap[0]);
	glUniform1i(colorsUni[0],0);
	//glBindTexture(GL_TEXTURE_RECTANGLE_ARB, colorTextures[0]);
	//glBindTexture(GL_TEXTURE_RECTANGLE_ARB,_colorTextureMap[0]);
	//glBindTexture(GL_TEXTURE_RECTANGLE_ARB,_depthTextureMap[0]);
	//glBindTexture(GL_TEXTURE_RECTANGLE_ARB, depthTextures[0]);
	
	for(int i = 0; i < _gpus-1; i++)
	{
	    glActiveTexture(GL_TEXTURE0 + index);
	    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, colorTextures[i]);
	    glUniform1i(colorsUni[i+1],index);
	    index++;
	}

	glActiveTexture(GL_TEXTURE0 + index);
	if(_depth == 32)
	{
	    glBindTexture(GL_TEXTURE_RECTANGLE_ARB,_depthTextureMap[0]);
	}
	else if(_depth == 24 || _depth == 16)
	{
	    glBindTexture(GL_TEXTURE_RECTANGLE_ARB,_depthRTextureMap[0]);
	}
	glUniform1i(depthUni[0],index);
	index++;

	for(int i = 0; i < _gpus-1; i++)
	{
	    glActiveTexture(GL_TEXTURE0 + index);
	    if(_depth == 32)
	    {
		glBindTexture(GL_TEXTURE_RECTANGLE_ARB, depthTextures[i]);
	    }
	    else if(_depth == 16 || _depth == 24)
	    {
		glBindTexture(GL_TEXTURE_RECTANGLE_ARB, depthRTextures[i]);
	    }
	    glUniform1i(depthUni[i+1],index);
	    index++;
	}


	if(_depth == 24)
	{
	    glActiveTexture(GL_TEXTURE0 + index);
	    glBindTexture(GL_TEXTURE_RECTANGLE_ARB,_depthR8TextureMap[0]);
	    glUniform1i(depthR8Uni[0],index);
	    index++;

	    for(int i = 0; i < _gpus-1; i++)
	    {
		glActiveTexture(GL_TEXTURE0 + index);
		glBindTexture(GL_TEXTURE_RECTANGLE_ARB, depthR8Textures[i]);
		glUniform1i(depthR8Uni[i+1],index);
		index++;
	    }
	}

	glActiveTexture(GL_TEXTURE0);

	glBindBuffer(GL_ARRAY_BUFFER, _screenArray);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, 0, 0, NULL);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glDisableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	for(int i = (2*_gpus)-1; i >= 0; i--)
	{
	    glActiveTexture(GL_TEXTURE0 + i);
	    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
	}
    }
    glUseProgram(0);
}


void MultiGPUDrawable::setPreFrameTime(struct timeval & time)
{
    _preFrameTime = time;
    _drawBlock->reset();
}
