#include <GL/glew.h>

#ifndef GL_DOUBLE_MAT3x2
#undef GL_ARB_gpu_shader_fp64
#endif

#include "FlowPagedRenderer.h"
#include "GLHelper.h"

#ifdef WITH_CUDA_LIB
#include <cuda.h>
#include <cudaGL.h>
#include "CudaHelper.h"
#endif

#include <iostream>
#include <cstring>

#include <sys/time.h>

#include "VisModes/NormalVisMode.h"
#include "VisModes/IsoSurfaceVisMode.h"
#include "VisModes/PlaneVisMode.h"
#include "VisModes/VecPlaneVisMode.h"
#include "VisModes/VortexCoresVisMode.h"
#include "VisModes/SepAttLineVisMode.h"
#include "VisModes/LicCudaVisMode.h"

pthread_mutex_t FlowPagedRenderer::_glewInitLock = PTHREAD_MUTEX_INITIALIZER;
std::map<int,bool> FlowPagedRenderer::_glewInitMap;

pthread_mutex_t FlowPagedRenderer::_cudaInitLock = PTHREAD_MUTEX_INITIALIZER;
std::map<int,bool> FlowPagedRenderer::_cudaInitMap;
std::map<int,std::pair<int,int> > FlowPagedRenderer::_cudaInitInfo;
std::map<int,int> FlowPagedRenderer::_contextRenderCountMap;

pthread_mutex_t FlowPagedRenderer::_colorTableInitLock = PTHREAD_MUTEX_INITIALIZER;
std::map<int,GLuint> FlowPagedRenderer::_colorTableMap;

FlowPagedRenderer::FlowPagedRenderer(PagedDataSet * set, int frame, FlowVisType type, std::string attribute, int cacheSize)
{
    _set = set;
    _currentFrame = frame;
    _nextFrame = frame;
    _type = type;
    _attribute = attribute;

    pthread_mutex_init(&_frameReadyLock,NULL);

    initUniData();

    _cache = new VBOCache(cacheSize);

    _visModeMap[FVT_NONE] = new NormalVisMode();
    _visModeMap[FVT_NONE]->_renderer = this;
    _visModeMap[FVT_ISO_SURFACE] = new IsoSurfaceVisMode();
    _visModeMap[FVT_ISO_SURFACE]->_renderer = this;
    _visModeMap[FVT_PLANE] = new PlaneVisMode();
    _visModeMap[FVT_PLANE]->_renderer = this;
    _visModeMap[FVT_PLANE_VEC] = new VecPlaneVisMode();
    _visModeMap[FVT_PLANE_VEC]->_renderer = this;
    _visModeMap[FVT_VORTEX_CORES] = new VortexCoresVisMode();
    _visModeMap[FVT_VORTEX_CORES]->_renderer = this;
    _visModeMap[FVT_SEP_ATT_LINES] = new SepAttLineVisMode();
    _visModeMap[FVT_SEP_ATT_LINES]->_renderer = this;

    // dummy implementation
    _visModeMap[FVT_VOLUME_CUDA] = new VisMode();
    _visModeMap[FVT_VOLUME_CUDA]->_renderer = this;

    _visModeMap[FVT_LIC_CUDA] = new LicCudaVisMode();
    _visModeMap[FVT_LIC_CUDA]->_renderer = this;

    for(int i = 0; i < _cudaInitInfo.size(); ++i)
    {
	operation * op = new operation;
	op->op = operation::INIT_OP;
	op->visMode = _visModeMap[_type];
	op->context = i;
	_opQueue[i].push(op);
    }
}

FlowPagedRenderer::~FlowPagedRenderer()
{
    for(std::map<int,std::queue<operation*> >::iterator it = _opQueue.begin(); it != _opQueue.end(); ++it)
    {
	while(it->second.size())
	{
	    delete it->second.front();
	    it->second.pop();
	}
    }

    for(std::map<FlowVisType,VisMode*>::iterator it = _visModeMap.begin(); it != _visModeMap.end(); ++it)
    {
	delete it->second;
    }

    delete _cache;

    for(std::map<std::string,struct UniData>::iterator it = _uniDataMap.begin(); it != _uniDataMap.end(); ++it)
    {
	deleteUniData(it->second);
    }

    pthread_mutex_destroy(&_frameReadyLock);
}

void FlowPagedRenderer::frameStart(int context)
{
    _visModeMap[_type]->frameStart(context);
}

void FlowPagedRenderer::preFrame()
{
    //std::cerr << "PreFrame" << std::endl;

#ifdef PRINT_TIMING
    struct timeval start, end;
    gettimeofday(&start,NULL);
#endif

    _visModeMap[_type]->preFrame();

#ifdef PRINT_TIMING
    gettimeofday(&end,NULL);
    std::cerr << "FlowPagedRenderer preframe: " << (end.tv_sec - start.tv_sec) + ((end.tv_usec - start.tv_usec)/1000000.0) << std::endl;
#endif
}

void FlowPagedRenderer::preDraw(int context)
{
    _visModeMap[_type]->preDraw(context);
}

void FlowPagedRenderer::draw(int context)
{
    checkGlewInit(context);
    checkCudaInit(context);
    checkColorTableInit(context);

    _cache->update(context);

    while(_opQueue[context].size())
    {
	_opQueue[context].front()->runOp();
	delete _opQueue[context].front();
	_opQueue[context].pop();
    }

    _visModeMap[_type]->draw(context);
}

void FlowPagedRenderer::postFrame()
{
    _visModeMap[_type]->postFrame();
    _cache->advanceTime();
}

void FlowPagedRenderer::setType(FlowVisType type, std::string attribute)
{
    if(type != _type)
    {
	for(int i = 0; i < _cudaInitInfo.size(); ++i)
	{
	    operation * op = new operation;
	    op->op = operation::UINIT_OP;
	    op->visMode = _visModeMap[_type];
	    op->context = i;
	    _opQueue[i].push(op);
	}

	for(int i = 0; i < _cudaInitInfo.size(); ++i)
	{
	    operation * op = new operation;
	    op->op = operation::INIT_OP;
	    op->visMode = _visModeMap[type];
	    op->context = i;
	    _opQueue[i].push(op);
	}
    }

    _type = type;
    _attribute = attribute;
    pthread_mutex_lock(&_frameReadyLock);

    for(std::map<int,bool>::iterator it = _nextFrameReady.begin(); it != _nextFrameReady.end(); ++it)
    {
	it->second = false;
    }

    pthread_mutex_unlock(&_frameReadyLock);
}

FlowVisType FlowPagedRenderer::getType()
{
    return _type;
}

void FlowPagedRenderer::setNextFrame(int frame)
{
    _nextFrame = frame;
    pthread_mutex_lock(&_frameReadyLock);

    for(std::map<int,bool>::iterator it = _nextFrameReady.begin(); it != _nextFrameReady.end(); ++it)
    {
	it->second = false;
    }

    pthread_mutex_unlock(&_frameReadyLock);
}

bool FlowPagedRenderer::advance()
{
    if(_currentFrame != _nextFrame)
    {
	pthread_mutex_lock(&_frameReadyLock);
	bool advance = true;
	for(std::map<int,bool>::iterator it = _nextFrameReady.begin(); it != _nextFrameReady.end(); ++it)
	{
	    if(!it->second)
	    {
		advance = false;
	    }
	}

	if(advance)
	{
	    _currentFrame = _nextFrame;

	    for(std::map<int,bool>::iterator it = _nextFrameReady.begin(); it != _nextFrameReady.end(); ++it)
	    {
		it->second = false;
	    }
	}

	pthread_mutex_unlock(&_frameReadyLock);

	return advance;
    }
    return true;
}

bool FlowPagedRenderer::canAdvance()
{
    if(_currentFrame != _nextFrame)
    {
	pthread_mutex_lock(&_frameReadyLock);
	bool advance = true;
	for(std::map<int,bool>::iterator it = _nextFrameReady.begin(); it != _nextFrameReady.end(); ++it)
	{
	    if(!it->second)
	    {
		advance = false;
	    }
	}

	pthread_mutex_unlock(&_frameReadyLock);

	return advance;
    }
    return true;
}

void FlowPagedRenderer::setNextFrameReady(int context, bool ready)
{
    pthread_mutex_lock(&_frameReadyLock);
    _nextFrameReady[context] = ready;
    pthread_mutex_unlock(&_frameReadyLock);
}

void FlowPagedRenderer::setUniData(std::string key, struct UniData & data)
{
    if(_uniDataMap.find(key) != _uniDataMap.end())
    {
	deleteUniData(_uniDataMap[key]);
    }
    _uniDataMap[key] = data;
}

bool FlowPagedRenderer::getUniData(std::string key, struct UniData & data)
{
    if(_uniDataMap.find(key) != _uniDataMap.end())
    {
	data = _uniDataMap[key];
	return true;
    }
    return false;
}

GLuint FlowPagedRenderer::getColorTableID(int context)
{
    GLuint id;
    pthread_mutex_lock(&_colorTableInitLock);
    id = _colorTableMap[context];
    pthread_mutex_unlock(&_colorTableInitLock);
    return id;
}

void FlowPagedRenderer::freeResources(int context)
{
    _cache->update(context);
    _cache->freeResources(context);

    while(_opQueue[context].size())
    {
	_opQueue[context].front()->runOp();
	delete _opQueue[context].front();
	_opQueue[context].pop();
    }

    _visModeMap[_type]->uinitContext(context);
}

bool FlowPagedRenderer::freeDone()
{
    return _cache->freeDone();
}

void FlowPagedRenderer::setCudaInitInfo(std::map<int,std::pair<int,int> > & initInfo)
{
    _cudaInitInfo = initInfo;
}

void FlowPagedRenderer::setContextRenderCount(std::map<int,int> & contextRenderCountMap)
{
    _contextRenderCountMap = contextRenderCountMap;
}

// create all uniform data here, so it doesn't need to be checked for every time
void FlowPagedRenderer::initUniData()
{
    _uniDataMap["minf"].type = UNI_FLOAT;
    _uniDataMap["minf"].data = new float[1];
    _uniDataMap["maxf"].type = UNI_FLOAT;
    _uniDataMap["maxf"].data = new float[1];
    _uniDataMap["mini"].type = UNI_INT;
    _uniDataMap["mini"].data = new int[1];
    _uniDataMap["maxi"].type = UNI_INT;
    _uniDataMap["maxi"].data = new int[1];

    _uniDataMap["isoMax"].type = UNI_FLOAT;
    _uniDataMap["isoMax"].data = new float[1];

    _uniDataMap["planePoint"].type = UNI_FLOAT3;
    _uniDataMap["planePoint"].data = new float[3];
    _uniDataMap["planeNormal"].type = UNI_FLOAT3;
    _uniDataMap["planeNormal"].data = new float[3];
    _uniDataMap["alpha"].type = UNI_FLOAT;
    _uniDataMap["alpha"].data = new float[1];

    _uniDataMap["planeUp"].type = UNI_FLOAT3;
    _uniDataMap["planeUp"].data = new float[3];
    _uniDataMap["planeRight"].type = UNI_FLOAT3;
    _uniDataMap["planeRight"].data = new float[3];

    _uniDataMap["planeUpNorm"].type = UNI_FLOAT3;
    _uniDataMap["planeUpNorm"].data = new float[3];
    _uniDataMap["planeRightNorm"].type = UNI_FLOAT3;
    _uniDataMap["planeRightNorm"].data = new float[3];
    _uniDataMap["planeBasisLength"].type = UNI_FLOAT;
    _uniDataMap["planeBasisLength"].data = new float[1];

    _uniDataMap["planeBasisXMin"].type = UNI_FLOAT;
    _uniDataMap["planeBasisXMin"].data = new float[1];
    _uniDataMap["planeBasisXMax"].type = UNI_FLOAT;
    _uniDataMap["planeBasisXMax"].data = new float[1];
    _uniDataMap["planeBasisYMin"].type = UNI_FLOAT;
    _uniDataMap["planeBasisYMin"].data = new float[1];
    _uniDataMap["planeBasisYMax"].type = UNI_FLOAT;
    _uniDataMap["planeBasisYMax"].data = new float[1];

    _uniDataMap["vmin"].type = UNI_FLOAT;
    _uniDataMap["vmin"].data = new float[1];
    _uniDataMap["vmax"].type = UNI_FLOAT;
    _uniDataMap["vmax"].data = new float[1];
}

void FlowPagedRenderer::checkGlewInit(int context)
{
    pthread_mutex_lock(&_glewInitLock);

    if(!_glewInitMap[context])
    {
	glewInit();
	_glewInitMap[context] = true;
    }

    pthread_mutex_unlock(&_glewInitLock);

    pthread_mutex_lock(&_frameReadyLock);

    _nextFrameReady[context] = false;

    pthread_mutex_unlock(&_frameReadyLock);
}

void FlowPagedRenderer::checkCudaInit(int context)
{
    pthread_mutex_lock(&_cudaInitLock);

    if(!_cudaInitMap[context])
    {
#ifdef WITH_CUDA_LIB
	if(_cudaInitInfo.find(context) != _cudaInitInfo.end())
	{
	    if(_cudaInitInfo[context].second > 1)
	    {
		//std::cerr << "Multi context per device init" << std::endl;
		CUdevice device;
		cuDeviceGet(&device,_cudaInitInfo[context].first);
		CUcontext cudaContext;

		cuGLCtxCreate(&cudaContext, 0, device);
		cuGLInit();
		cuCtxSetCurrent(cudaContext);
	    }
	    else
	    {
		//std::cerr << "Single context per device init" << std::endl;
		cudaGLSetGLDevice(_cudaInitInfo[context].first);
		cudaSetDevice(_cudaInitInfo[context].first);
	    }
	}
	else
	{
	    std::cerr << "Warning: no cuda init info for context: " << context << std::endl;
	}
#endif
	_cudaInitMap[context] = true;
    }

    pthread_mutex_unlock(&_cudaInitLock);
}

void FlowPagedRenderer::checkColorTableInit(int context)
{
    pthread_mutex_lock(&_colorTableInitLock);

    if(!_colorTableMap[context])
    {
	glGenTextures(1,&_colorTableMap[context]);
	int size = 32;
	std::vector<float> colorR;
	std::vector<float> colorG;
	std::vector<float> colorB;
	colorR.push_back(0.0);
	colorR.push_back(0.7);
	colorR.push_back(0.7);
	colorG.push_back(0.0);
	colorG.push_back(0.7);
	colorG.push_back(0.0);
	colorB.push_back(0.7);
	colorB.push_back(0.7);
	colorB.push_back(0.0);

	unsigned char * data = new unsigned char[size*3];

	for(int i = 0; i < size; ++i)
	{
	    float pos = ((float)i) / ((float)(size-1));
	    pos = fmax(pos,0.0);
	    pos = fmin(pos,1.0);
	    pos = pos * ((float)(colorR.size()-1));
	    int topIndex = ceil(pos);
	    if(topIndex >= colorR.size())
	    {
		topIndex = colorR.size() - 1;
	    }
	    int bottomIndex = floor(pos);
	    if(bottomIndex < 0)
	    {
		bottomIndex = 0;
	    }

	    float ratio = pos - floor(pos);
	    data[(3*i)+0] = (unsigned char)((colorR[bottomIndex] * (1.0 - ratio) + colorR[topIndex] * ratio) * 255.0);
	    data[(3*i)+1] = (unsigned char)((colorG[bottomIndex] * (1.0 - ratio) + colorG[topIndex] * ratio) * 255.0);
	    data[(3*i)+2] = (unsigned char)((colorB[bottomIndex] * (1.0 - ratio) + colorB[topIndex] * ratio) * 255.0);
	    //std::cerr << "color: " << (int)data[(3*i)+0] << " " << (int)data[(3*i)+1] << " " << (int)data[(3*i)+2] << std::endl;
	}

	glBindTexture(GL_TEXTURE_1D,_colorTableMap[context]);

	glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

	glTexImage1D(GL_TEXTURE_1D,0,3,size,0,GL_RGB,GL_UNSIGNED_BYTE,data);

	glBindTexture(GL_TEXTURE_1D,0);

	delete[] data;
    }

    pthread_mutex_unlock(&_colorTableInitLock);
}

void FlowPagedRenderer::deleteUniData(UniData & data)
{
    if(!data.data)
    {
	return;
    }

    switch(data.type)
    {
	case UNI_FLOAT:
	case UNI_FLOAT3:
	case UNI_MAT3:
	    delete[] (float*)data.data;
	    break;
	case UNI_INT:
	    delete[] (int*)data.data;
	    break;
	case UNI_UINT:
	    delete[] (unsigned int*)data.data;
	    break;
	default:
	    std::cerr << "Warning: trying to delete data for unknown uniform type." << std::endl;
	    break;
    }
}

void FlowPagedRenderer::loadUniform(UniformBinding & uni)
{
    switch(uni.type)
    {
	case UNI_FLOAT:
	    glUniform1fv(uni.location,1,(GLfloat*)uni.data);
	    break;
	case UNI_FLOAT3:
	    glUniform3fv(uni.location,1,(GLfloat*)uni.data);
	    break;
	case UNI_MAT3:
	    glUniformMatrix3fv(uni.location,1,false,(GLfloat*)uni.data);
	    break;
	case UNI_INT:
	    glUniform1iv(uni.location,1,(GLint*)uni.data);
	    break;
	case UNI_UINT:
	    glUniform1uiv(uni.location,1,(GLuint*)uni.data);
	    break;
	default:
	    std::cerr << "Warning: trying to load uniform of unknown type." << std::endl;
	    break;
    }
}

void FlowPagedRenderer::drawElements(GLenum mode, GLsizei count, GLenum type, GLuint indVBO, GLuint vertsVBO, std::vector<float> & color, std::vector<FlowPagedRenderer::AttribBinding> & attribBinding, GLuint program, std::vector<TextureBinding> & textureBinding, std::vector<UniformBinding> & uniBinding)
{
    glEnableClientState(GL_VERTEX_ARRAY);

    if(color.size() < 4)
    {
	glColor4f(1.0,1.0,1.0,1.0);
    }
    else
    {
	glColor4f(color[0],color[1],color[2],color[3]);
    }

    glBindBuffer(GL_ARRAY_BUFFER,vertsVBO);
    glVertexPointer(3,GL_FLOAT,0,0);

    for(int i = 0; i < attribBinding.size(); ++i)
    {
	glEnableVertexAttribArray(attribBinding[i].index);
	glBindBuffer(GL_ARRAY_BUFFER,attribBinding[i].buffer);
	if(attribBinding[i].type == GL_FLOAT)
	{
	    glVertexAttribPointer(attribBinding[i].index,attribBinding[i].size,attribBinding[i].type,GL_FALSE,0,0);
	}
	else
	{
	    glVertexAttribIPointer(attribBinding[i].index,attribBinding[i].size,attribBinding[i].type,0,0);
	}
    }

    glBindBuffer(GL_ARRAY_BUFFER,0);

    for(int i = 0; i < textureBinding.size(); ++i)
    {
	//std::cerr << "binding texture unit: " << textureBinding[i].unit << " id: " << textureBinding[i].id << std::endl;
	glActiveTexture(GL_TEXTURE0 + textureBinding[i].unit);
	glBindTexture(textureBinding[i].type,textureBinding[i].id);
    }

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,indVBO);
    glUseProgram(program);

    for(int i = 0; i < uniBinding.size(); ++i)
    {
	loadUniform(uniBinding[i]);
    }

    glDrawElements(mode,count,type,0);
    glUseProgram(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0);

    for(int i = 0; i < textureBinding.size(); ++i)
    {
	glActiveTexture(GL_TEXTURE0 + textureBinding[i].unit);
	glBindTexture(GL_TEXTURE_1D,0);
    }
    if(textureBinding.size())
    {
	glActiveTexture(GL_TEXTURE0);
    }

    for(int i = 0; i < attribBinding.size(); ++i)
    {
	glDisableVertexAttribArray(attribBinding[i].index);
    }

    glDisableClientState(GL_VERTEX_ARRAY);
}

void FlowPagedRenderer::drawArrays(GLenum mode, GLint first, GLsizei count, GLuint vertsVBO, std::vector<float> & color, std::vector<AttribBinding> & attribBinding, GLuint program, std::vector<TextureBinding> & textureBinding, std::vector<UniformBinding> & uniBinding)
{
    glEnableClientState(GL_VERTEX_ARRAY);

    if(color.size() < 4)
    {
	glColor4f(1.0,1.0,1.0,1.0);
    }
    else
    {
	glColor4f(color[0],color[1],color[2],color[3]);
    }

    glBindBuffer(GL_ARRAY_BUFFER,vertsVBO);
    glVertexPointer(3,GL_FLOAT,0,0);

    for(int i = 0; i < attribBinding.size(); ++i)
    {
	glEnableVertexAttribArray(attribBinding[i].index);
	glBindBuffer(GL_ARRAY_BUFFER,attribBinding[i].buffer);
	if(attribBinding[i].type == GL_FLOAT)
	{
	    glVertexAttribPointer(attribBinding[i].index,attribBinding[i].size,attribBinding[i].type,GL_FALSE,0,0);
	}
	else
	{
	    glVertexAttribIPointer(attribBinding[i].index,attribBinding[i].size,attribBinding[i].type,0,0);
	}
    }

    glBindBuffer(GL_ARRAY_BUFFER,0);

    for(int i = 0; i < textureBinding.size(); ++i)
    {
	//std::cerr << "binding texture unit: " << textureBinding[i].unit << " id: " << textureBinding[i].id << std::endl;
	glActiveTexture(GL_TEXTURE0 + textureBinding[i].unit);
	glBindTexture(textureBinding[i].type,textureBinding[i].id);
    }

    glUseProgram(program);

    for(int i = 0; i < uniBinding.size(); ++i)
    {
	loadUniform(uniBinding[i]);
    }

    glDrawArrays(mode,first,count);
    glUseProgram(0);

    for(int i = 0; i < textureBinding.size(); ++i)
    {
	glActiveTexture(GL_TEXTURE0 + textureBinding[i].unit);
	glBindTexture(GL_TEXTURE_1D,0);
    }
    if(textureBinding.size())
    {
	glActiveTexture(GL_TEXTURE0);
    }

    for(int i = 0; i < attribBinding.size(); ++i)
    {
	glDisableVertexAttribArray(attribBinding[i].index);
    }

    glDisableClientState(GL_VERTEX_ARRAY);
}

void FlowPagedRenderer::operation::runOp()
{
    switch(op)
    {
	case INIT_OP:
	    visMode->initContext(context);
	    break;
	case UINIT_OP:
	    visMode->uinitContext(context);
	    break;
	default:
	    break;
    }
}
