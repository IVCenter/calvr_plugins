#include <GL/glew.h>

#ifndef GL_DOUBLE_MAT3x2
#undef GL_ARB_gpu_shader_fp64
#endif

#include "LicCudaVisMode.h"

#include "../FlowPagedRenderer.h"
#include "../GLHelper.h"

#ifdef WITH_CUDA_LIB
#include <cuda.h>
#include <cudaGL.h>
#include "../CudaHelper.h"
#include "CudaLIC.h"
#ifdef WIN32
//#pragma comment(lib, "cuda.lib")
#endif
#endif

// namespace scope to fix compile issues
namespace licCudaVisMode
{
#include "../glsl/NormalShader.h"
#include "../glsl/LicShaders.h"
}

using namespace licCudaVisMode;

LicCudaVisMode::LicCudaVisMode()
{
    _licStarted = false;
    _licOutputValid = false;

    pthread_mutex_init(&_shaderInitLock,NULL);
    pthread_mutex_init(&_licLock,NULL);
}

LicCudaVisMode::~LicCudaVisMode()
{
    pthread_mutex_destroy(&_shaderInitLock);
    pthread_mutex_destroy(&_licLock);
}

void LicCudaVisMode::initContext(int context)
{
    pthread_mutex_lock(&_shaderInitLock);

    if(!_shaderInitMap[context])
    {
	GLuint verts, frags, geoms;
	createShaderFromSrc(normalVertSrc,GL_VERTEX_SHADER,verts,"NormalVert");
	createShaderFromSrc(normalGeomSrc,GL_GEOMETRY_SHADER,geoms,"NormalGeom");
	createShaderFromSrc(normalFragSrc,GL_FRAGMENT_SHADER,frags,"NormalFrag");
	createProgram(_normalProgram[context],verts,frags,geoms,GL_TRIANGLES,GL_TRIANGLE_STRIP,3);

	glDeleteShader(verts);
	glDeleteShader(geoms);
	glDeleteShader(frags);

	createShaderFromSrc(normalFloatVertSrc,GL_VERTEX_SHADER,verts,"NormalFloatVert");
	createShaderFromSrc(normalFloatGeomSrc,GL_GEOMETRY_SHADER,geoms,"NormalFloatGeom");
	createShaderFromSrc(normalFloatFragSrc,GL_FRAGMENT_SHADER,frags,"NormalFloatFrag");
	createProgram(_normalFloatProgram[context],verts,frags,geoms,GL_TRIANGLES,GL_TRIANGLE_STRIP,3);

	glDeleteShader(verts);
	glDeleteShader(geoms);
	glDeleteShader(frags);

	_normalFloatMinUni[context] = glGetUniformLocation(_normalFloatProgram[context],"min");
	_normalFloatMaxUni[context] = glGetUniformLocation(_normalFloatProgram[context],"max");

	createShaderFromSrc(normalIntVertSrc,GL_VERTEX_SHADER,verts,"NormalIntVert");
	createShaderFromSrc(normalFloatGeomSrc,GL_GEOMETRY_SHADER,geoms,"NormalFloatGeom");
	createShaderFromSrc(normalFloatFragSrc,GL_FRAGMENT_SHADER,frags,"NormalFloatFrag");
	createProgram(_normalIntProgram[context],verts,frags,geoms,GL_TRIANGLES,GL_TRIANGLE_STRIP,3);

	glDeleteShader(verts);
	glDeleteShader(geoms);
	glDeleteShader(frags);

	_normalIntMinUni[context] = glGetUniformLocation(_normalIntProgram[context],"min");
	_normalIntMaxUni[context] = glGetUniformLocation(_normalIntProgram[context],"max");

	createShaderFromSrc(normalVecVertSrc,GL_VERTEX_SHADER,verts,"NormalVecVert");
	createShaderFromSrc(normalFloatGeomSrc,GL_GEOMETRY_SHADER,geoms,"NormalFloatGeom");
	createShaderFromSrc(normalFloatFragSrc,GL_FRAGMENT_SHADER,frags,"NormalFloatFrag");
	createProgram(_normalVecProgram[context],verts,frags,geoms,GL_TRIANGLES,GL_TRIANGLE_STRIP,3);

	glDeleteShader(verts);
	glDeleteShader(geoms);
	glDeleteShader(frags);

	_normalVecMinUni[context] = glGetUniformLocation(_normalVecProgram[context],"min");
	_normalVecMaxUni[context] = glGetUniformLocation(_normalVecProgram[context],"max");

	createShaderFromSrc(licVertSrc,GL_VERTEX_SHADER,verts,"licVert");
	createShaderFromSrc(licFragSrc,GL_FRAGMENT_SHADER,frags,"licFrag");
	createProgram(_licRenderProgram[context],verts,frags);
	_licRenderAlphaUni[context] = glGetUniformLocation(_licRenderProgram[context],"alpha");

	glDeleteShader(verts);
	glDeleteShader(frags);

	_shaderInitMap[context] = true;
    }

    pthread_mutex_unlock(&_shaderInitLock);


    pthread_mutex_lock(&_licLock);

    if(!_licInit[context])
    {
	// temp seed for the moment
	srand(451651);
	glGenTextures(1,&_licNoiseTex[context]);

	float * data = new float[LIC_TEXTURE_SIZE*LIC_TEXTURE_SIZE];

	for(int i = 0; i < LIC_TEXTURE_SIZE*LIC_TEXTURE_SIZE; ++i)
	{
	    data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
	}

	glBindTexture(GL_TEXTURE_2D,_licNoiseTex[context]);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glTexImage2D(GL_TEXTURE_2D,0,GL_R32F,LIC_TEXTURE_SIZE,LIC_TEXTURE_SIZE,0,GL_RED,GL_FLOAT,data);

	delete[] data;

#ifdef WITH_CUDA_LIB
	_licCudaNoiseImage[context] = new CudaGLImage(_licNoiseTex[context],GL_TEXTURE_2D);
	_licCudaNoiseImage[context]->registerImage(cudaGraphicsRegisterFlagsReadOnly);
#endif
	
	glGenTextures(1,&_licVelTex[context]);
	glBindTexture(GL_TEXTURE_2D,_licVelTex[context]);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glTexImage2D(GL_TEXTURE_2D,0,GL_RG16F,LIC_TEXTURE_SIZE,LIC_TEXTURE_SIZE,0,GL_RG,GL_UNSIGNED_SHORT,NULL);

#ifdef WITH_CUDA_LIB
	_licCudaVelImage[context] = new CudaGLImage(_licVelTex[context],GL_TEXTURE_2D);
	_licCudaVelImage[context]->registerImage(cudaGraphicsRegisterFlagsSurfaceLoadStore);
#endif
	
	glGenTextures(1,&_licOutputTex[context]);
	glBindTexture(GL_TEXTURE_2D,_licOutputTex[context]);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glTexImage2D(GL_TEXTURE_2D,0,GL_RG16F,LIC_TEXTURE_SIZE,LIC_TEXTURE_SIZE,0,GL_RG,GL_UNSIGNED_SHORT,NULL);

#ifdef WITH_CUDA_LIB
	_licCudaOutputImage[context] = new CudaGLImage(_licOutputTex[context],GL_TEXTURE_2D);
	_licCudaOutputImage[context]->registerImage(cudaGraphicsRegisterFlagsSurfaceLoadStore);
#endif

	glGenTextures(1,&_licNextOutputTex[context]);
	glBindTexture(GL_TEXTURE_2D,_licNextOutputTex[context]);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glTexImage2D(GL_TEXTURE_2D,0,GL_RG16F,LIC_TEXTURE_SIZE,LIC_TEXTURE_SIZE,0,GL_RG,GL_UNSIGNED_SHORT,NULL);

#ifdef WITH_CUDA_LIB
	_licCudaNextOutputImage[context] = new CudaGLImage(_licNextOutputTex[context],GL_TEXTURE_2D);
	_licCudaNextOutputImage[context]->registerImage(cudaGraphicsRegisterFlagsSurfaceLoadStore);
#endif

	glBindTexture(GL_TEXTURE_2D,0);

	int draws = _renderer->getContextRenderCountMap()[context];

	if(draws <= 0)
	{
	    _renderer->getContextRenderCountMap()[context] = 1;
	    _licContextRenderCount[context] = 1;
	}
	else
	{
	    _licContextRenderCount[context] = draws;
	}

	_licInit[context] = true;
	_licFinished[context] = false;

	printCudaErr();
    }

    pthread_mutex_unlock(&_licLock);
}

void LicCudaVisMode::uinitContext(int context)
{
    pthread_mutex_lock(&_shaderInitLock);

    if(_shaderInitMap[context])
    {
	glDeleteProgram(_normalProgram[context]);
	glDeleteProgram(_normalFloatProgram[context]);
	glDeleteProgram(_normalIntProgram[context]);
	glDeleteProgram(_normalVecProgram[context]);
	glDeleteProgram(_licRenderProgram[context]);

	_shaderInitMap[context] = false;
    }

    pthread_mutex_unlock(&_shaderInitLock);

    pthread_mutex_lock(&_licLock);

    if(!_licInit[context])
    {
#ifdef WITH_CUDA_LIB
	_licCudaNoiseImage[context]->unregisterImage();
	delete _licCudaNoiseImage[context];
	_licCudaVelImage[context]->unregisterImage();
	delete _licCudaVelImage[context];
	_licCudaOutputImage[context]->unregisterImage();
	delete _licCudaOutputImage[context];
	_licCudaNextOutputImage[context]->unregisterImage();
	delete _licCudaNextOutputImage[context];
#endif

	glDeleteTextures(1,&_licNoiseTex[context]);
	glDeleteTextures(1,&_licVelTex[context]);
	glDeleteTextures(1,&_licOutputTex[context]);
	glDeleteTextures(1,&_licNextOutputTex[context]);

	_licInit[context] = false;
    }

    pthread_mutex_unlock(&_licLock);
}

void LicCudaVisMode::frameStart(int context)
{
    VBOCache * cache = _renderer->getCache();
    PagedDataSet * set = _renderer->getSet();
    int currentFrame = _renderer->getCurrentFrame();
    int nextFrame = _renderer->getNextFrame();
    std::map<std::string,struct UniData> & uniDataMap = _renderer->getUniDataMap();

#ifdef WITH_CUDA_LIB
    int fileID = cache->getFileID(set->frameFiles[currentFrame]);
    // get needed buffers
    GLuint indVBO, vertsVBO, velVBO = 0;

    indVBO = cache->getOrRequestBuffer(context,fileID,set->frameList[currentFrame]->indices.second,set->frameList[currentFrame]->indices.first*sizeof(unsigned int),GL_ELEMENT_ARRAY_BUFFER,true);
    for(int i = 0; i < set->frameList[currentFrame]->pointData.size(); ++i)
    {
	if(set->frameList[currentFrame]->pointData[i]->name == "Velocity")
	{
	    velVBO = cache->getOrRequestBuffer(context,fileID,set->frameList[currentFrame]->pointData[i]->offset,set->frameList[currentFrame]->verts.first*3*sizeof(float),GL_ARRAY_BUFFER,true);
	    break;
	}
    }

    vertsVBO = cache->getOrRequestBuffer(context,fileID,set->frameList[currentFrame]->verts.second,set->frameList[currentFrame]->verts.first*3*sizeof(float),GL_ARRAY_BUFFER,true);

    //std::cerr << "Frame start context: " << context << " frame: " << _currentFrame << " indVBO: " << indVBO << " vertVBO: " << vertsVBO << " velVBO: " << velVBO << std::endl;

    if(!_licStarted)
    {
	return;
    }

    // if all needed buffers are loaded, do LIC
    if(indVBO && vertsVBO && velVBO)
    {
	setPlaneConsts(ccPlanePoint,ccPlaneNormal,ccPlaneRight,ccPlaneUp,ccPlaneRightNorm,ccPlaneUpNorm,&ccPlaneBasisLength);
	setTexConsts(&ccTexXMin,&ccTexXMax,&ccTexYMin,&ccTexYMax);

	// map buffers
	CUdeviceptr d_indVBO, d_vertsVBO, d_velVBO;
	checkMapBufferObj((void**)&d_indVBO,indVBO);
	checkMapBufferObj((void**)&d_vertsVBO,vertsVBO);
	checkMapBufferObj((void**)&d_velVBO,velVBO);

#ifdef PRINT_TIMING	
	cudaThreadSynchronize();
	struct timeval tliststart,tlistend;
	gettimeofday(&tliststart,NULL);
#endif

	// form a list of tets on the plane
	void * d_tetList;
	cudaMalloc(&d_tetList,(set->frameList[currentFrame]->indices.first/4)*sizeof(unsigned int));

	void * d_numTets;
	cudaMalloc(&d_numTets,sizeof(unsigned int));
	cudaMemset(d_numTets,0,sizeof(unsigned int));

	launchMakeTetList((unsigned int*)d_tetList,(unsigned int*)d_numTets,set->frameList[currentFrame]->indices.first/4,(uint4*)d_indVBO,(float3*)d_vertsVBO);
	cudaThreadSynchronize();

	unsigned int h_numTets;
	cudaMemcpy(&h_numTets,d_numTets,sizeof(unsigned int),cudaMemcpyDeviceToHost);

#ifdef PRINT_TIMING
	gettimeofday(&tlistend,NULL);
	std::cerr << "TetList time: " << (tlistend.tv_sec - tliststart.tv_sec) + ((tlistend.tv_usec - tliststart.tv_usec)/1000000.0) << std::endl;
#endif

	//std::cerr << "TotalTets: " << set->frameList[currentFrame]->indices.first/4 << " tets on plane: " << h_numTets << std::endl;

	// map velocity texture
	_licCudaVelImage[context]->setMapFlags(cudaGraphicsMapFlagsWriteDiscard);
	_licCudaVelImage[context]->map();

	// bind array to surface
	setVelSurfaceRef(_licCudaVelImage[context]->getPointer());

	//std::cerr << "Active tets: " << h_numTets << std::endl;

#ifdef PRINT_TIMING
	struct timeval velstart,velend;
	gettimeofday(&velstart,NULL);
#endif

	// populate texture
	launchVel((uint4*)d_indVBO,(float3*)d_vertsVBO,(float3*)d_velVBO,(unsigned int*)d_tetList,h_numTets,LIC_TEXTURE_SIZE,LIC_TEXTURE_SIZE);
	cudaThreadSynchronize();

#ifdef PRINT_TIMING
	gettimeofday(&velend,NULL);
	std::cerr << "VelKernel time: " << (velend.tv_sec - velstart.tv_sec) + ((velend.tv_usec - velstart.tv_usec)/1000000.0) << std::endl;
#endif

	cudaFree(d_tetList);
	cudaFree(d_numTets);

	// unmap buffers
	checkUnmapBufferObj(indVBO);
	checkUnmapBufferObj(vertsVBO);
	checkUnmapBufferObj(velVBO);

	// map output texture
	_licCudaNextOutputImage[context]->setMapFlags(cudaGraphicsMapFlagsWriteDiscard);
	_licCudaNextOutputImage[context]->map();

	// map noise texture
	_licCudaNoiseImage[context]->setMapFlags(cudaGraphicsMapFlagsReadOnly);
	_licCudaNoiseImage[context]->map();

	// bind array to surface
	setOutSurfaceRef(_licCudaNextOutputImage[context]->getPointer());

#ifdef PRINT_TIMING
	struct timeval licstart,licend;
	gettimeofday(&licstart,NULL);
#endif

	// run LIC kernel
	launchLIC(LIC_TEXTURE_SIZE,LIC_TEXTURE_SIZE,*((float*)uniDataMap["licLength"].data),_licCudaNoiseImage[context]->getPointer());

#ifdef PRINT_TIMING
	gettimeofday(&licend,NULL);
	std::cerr << "LIC time: " << (licend.tv_sec - licstart.tv_sec) + ((licend.tv_usec - licstart.tv_usec)/1000000.0) << std::endl;
#endif

	// unmap textures
	_licCudaVelImage[context]->unmap();
	_licCudaNextOutputImage[context]->unmap();
	_licCudaNoiseImage[context]->unmap();

	pthread_mutex_lock(&_licLock);
	_licFinished[context] = true;
	pthread_mutex_unlock(&_licLock);
    }

#endif
}

void LicCudaVisMode::draw(int context)
{
    VBOCache * cache = _renderer->getCache();
    PagedDataSet * set = _renderer->getSet();
    int currentFrame = _renderer->getCurrentFrame();
    int nextFrame = _renderer->getNextFrame();
    int fileID = cache->getFileID(set->frameFiles[currentFrame]);
    std::map<std::string,struct UniData> & uniDataMap = _renderer->getUniDataMap();

    GLuint surfVBO, vertsVBO, attribVBO;

    // will need these buffers for cuda calc
    cache->getOrRequestBuffer(context,fileID,set->frameList[currentFrame]->indices.second,set->frameList[currentFrame]->indices.first*sizeof(unsigned int),GL_ELEMENT_ARRAY_BUFFER,true);
    for(int i = 0; i < set->frameList[currentFrame]->pointData.size(); ++i)
    {
	if(set->frameList[currentFrame]->pointData[i]->name == "Velocity")
	{
	    cache->getOrRequestBuffer(context,fileID,set->frameList[currentFrame]->pointData[i]->offset,set->frameList[currentFrame]->verts.first*3*sizeof(float),GL_ARRAY_BUFFER,true);
	    break;
	}
    }

    surfVBO = cache->getOrRequestBuffer(context,fileID,set->frameList[currentFrame]->surfaceInd.second,set->frameList[currentFrame]->surfaceInd.first*sizeof(unsigned int),GL_ELEMENT_ARRAY_BUFFER);
    vertsVBO = cache->getOrRequestBuffer(context,fileID,set->frameList[currentFrame]->verts.second,set->frameList[currentFrame]->verts.first*3*sizeof(float),GL_ARRAY_BUFFER,true);

    PagedDataAttrib * attrib = NULL;
    for(int i = 0; i < set->frameList[currentFrame]->pointData.size(); ++i)
    {
	if(set->frameList[currentFrame]->pointData[i]->name == _renderer->getAttribute())
	{
	    attrib = set->frameList[currentFrame]->pointData[i];
	    break;
	}
    }

    std::vector<FlowPagedRenderer::AttribBinding> binding;
    std::vector<FlowPagedRenderer::TextureBinding> texBinding;
    std::vector<FlowPagedRenderer::UniformBinding> uniBinding;

    GLuint prog = 0;
    GLuint texture = 0;

    unsigned int unitsize;
    if(attrib)
    {
	FlowPagedRenderer::AttribBinding ab;
	FlowPagedRenderer::UniformBinding ub;
	if(attrib->attribType == VAT_VECTORS)
	{
	    ab.size = 3;
	    ab.type = GL_FLOAT;
	    unitsize = 3*sizeof(float);
	    prog = _normalVecProgram[context];

	    ub.location = _normalVecMinUni[context];
	    ub.type = uniDataMap["minf"].type;
	    ub.data = uniDataMap["minf"].data;
	    uniBinding.push_back(ub);
	    ub.location = _normalVecMaxUni[context];
	    ub.type = uniDataMap["maxf"].type;
	    ub.data = uniDataMap["maxf"].data;
	    uniBinding.push_back(ub);
	}
	else if(attrib->dataType == VDT_INT)
	{
	    ab.size = 1;
	    ab.type = GL_UNSIGNED_INT;
	    unitsize = sizeof(int);
	    prog = _normalIntProgram[context];

	    ub.location = _normalIntMinUni[context];
	    ub.type = uniDataMap["mini"].type;
	    ub.data = uniDataMap["mini"].data;
	    uniBinding.push_back(ub);
	    ub.location = _normalIntMaxUni[context];
	    ub.type = uniDataMap["maxi"].type;
	    ub.data = uniDataMap["maxi"].data;
	    uniBinding.push_back(ub);
	}
	else
	{
	    ab.size = 1;
	    ab.type = GL_FLOAT;
	    unitsize = sizeof(float);
	    prog = _normalFloatProgram[context];

	    ub.location = _normalFloatMinUni[context];
	    ub.type = uniDataMap["minf"].type;
	    ub.data = uniDataMap["minf"].data;
	    uniBinding.push_back(ub);
	    ub.location = _normalFloatMaxUni[context];
	    ub.type = uniDataMap["maxf"].type;
	    ub.data = uniDataMap["maxf"].data;
	    uniBinding.push_back(ub);
	}

	ab.index = 4;
	attribVBO = cache->getOrRequestBuffer(context,fileID,attrib->offset,set->frameList[currentFrame]->verts.first*unitsize,GL_ARRAY_BUFFER);
	ab.buffer = attribVBO;
	binding.push_back(ab);

	FlowPagedRenderer::TextureBinding tb;

	tb.id = _renderer->getColorTableID(context);

	tb.unit = 0;
	tb.type = GL_TEXTURE_1D;
	texBinding.push_back(tb);
    }
    else
    {
	attribVBO = 0;
	prog = _normalProgram[context];
    }

    //std::cerr << "CurrentFrame: " << _currentFrame << " nextFrame: " << _nextFrame << std::endl;
    if(surfVBO && vertsVBO && (!attrib || attribVBO))
    {
	std::vector<float> color(4);
	color[0] = 1.0;
	color[1] = 1.0;
	color[2] = 1.0;
	color[3] = 1.0;
	//std::cerr << "drawn" << std::endl;
	
	if(set->revCullFace)
	{
	    glCullFace(GL_FRONT);
	}
	else
	{
	    glCullFace(GL_BACK);
	}
	
	glEnable(GL_CULL_FACE);
	_renderer->drawElements(GL_TRIANGLES,set->frameList[currentFrame]->surfaceInd.first,GL_UNSIGNED_INT,surfVBO,vertsVBO,color,binding,prog,texBinding,uniBinding);
	glDisable(GL_CULL_FACE);
    }
    else
    {
	//std::cerr << "not drawn" << std::endl;
    }

    pthread_mutex_lock(&_licLock);

    //std::cerr << "Context: " << context << " Count: " << _licContextRenderCount[context] << " total: " << _contextRenderCountMap[context] << std::endl;

    // only check this on the first draw in the context
    if(_licContextRenderCount[context] == _renderer->getContextRenderCountMap()[context])
    {
	//std::cerr << "Checking for finish" << std::endl;
	bool finished = true;

	for(std::map<int,bool>::iterator it = _licFinished.begin(); it != _licFinished.end(); ++it)
	{
	    if(!it->second)
	    {
		finished = false;
		break;
	    }
	}

	if(finished)
	{
	    _licOutputPoints = _licNextOutputPoints;
	    //swap next output into current output
	    GLuint tempid = _licOutputTex[context];
	    _licOutputTex[context] = _licNextOutputTex[context];
	    _licNextOutputTex[context] = tempid;

	    CudaGLImage * tempptr = _licCudaOutputImage[context];
	    _licCudaOutputImage[context] = _licCudaNextOutputImage[context];
	    _licCudaNextOutputImage[context] = tempptr;

	    _licStarted = false;
	    if(!_licOutputValid)
	    {
		_licOutputValid = true;
	    }
	}

    }

    pthread_mutex_unlock(&_licLock);

    if(_licOutputValid)
    {
	//std::cerr << "Draw outputValid context: " << context << " texture: " << _licOutputTex[context] << std::endl;
	//glBindTexture(GL_TEXTURE_2D,_licNoiseTex[context]);
	glBindTexture(GL_TEXTURE_2D,_licOutputTex[context]);
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_BLEND);

	glUseProgram(_licRenderProgram[context]);
	glUniform1f(_licRenderAlphaUni[context],*((float*)uniDataMap["alpha"].data));

	glBegin(GL_QUADS);
	glTexCoord2f(0.0,0.0);
	glVertex3fv(&_licOutputPoints[0]);
	glTexCoord2f(1.0,0.0);
	glVertex3fv(&_licOutputPoints[3]);
	glTexCoord2f(1.0,1.0);
	glVertex3fv(&_licOutputPoints[6]);
	glTexCoord2f(0.0,1.0);
	glVertex3fv(&_licOutputPoints[9]);
	glEnd();

	glUseProgram(0);

	glDisable(GL_BLEND);
	glDisable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D,0);
    }

    pthread_mutex_lock(&_licLock);

    // only check for frame advance on last context draw
    _licContextRenderCount[context]--;

    if(currentFrame != nextFrame && !_licStarted && _licContextRenderCount[context] == 0)
    {
	//std::cerr << "Checking frame advance" << std::endl;
	pthread_mutex_unlock(&_licLock);
	int nextfileID = cache->getFileID(set->frameFiles[nextFrame]);
	GLuint ibuf, vbuf, abuf = 0, fullibuf, velbuf = 0;
	ibuf = cache->getOrRequestBuffer(context,nextfileID,set->frameList[nextFrame]->surfaceInd.second,set->frameList[nextFrame]->surfaceInd.first*sizeof(unsigned int),GL_ELEMENT_ARRAY_BUFFER);
	vbuf = cache->getOrRequestBuffer(context,nextfileID,set->frameList[nextFrame]->verts.second,set->frameList[nextFrame]->verts.first*3*sizeof(float),GL_ARRAY_BUFFER);

	PagedDataAttrib * nextattrib = NULL;
	for(int i = 0; i < set->frameList[nextFrame]->pointData.size(); ++i)
	{
	    if(set->frameList[nextFrame]->pointData[i]->name == _renderer->getAttribute())
	    {
		nextattrib = set->frameList[nextFrame]->pointData[i];
		break;
	    }
	}

	if(nextattrib)
	{
	    abuf = cache->getOrRequestBuffer(context,nextfileID,nextattrib->offset,set->frameList[nextFrame]->verts.first*unitsize,GL_ARRAY_BUFFER);
	}

	fullibuf = cache->getOrRequestBuffer(context,nextfileID,set->frameList[nextFrame]->indices.second,set->frameList[nextFrame]->indices.first*sizeof(unsigned int),GL_ELEMENT_ARRAY_BUFFER,true);
	for(int i = 0; i < set->frameList[nextFrame]->pointData.size(); ++i)
	{
	    if(set->frameList[nextFrame]->pointData[i]->name == "Velocity")
	    {
		velbuf = cache->getOrRequestBuffer(context,nextfileID,set->frameList[nextFrame]->pointData[i]->offset,set->frameList[nextFrame]->verts.first*3*sizeof(float),GL_ARRAY_BUFFER,true);
		break;
	    }
	}

	bool nextReady;
	if(ibuf && vbuf && (!nextattrib || abuf) && fullibuf && velbuf)
	{
	    nextReady = true;
	}
	else
	{
	    nextReady = false;
	}
	_renderer->setNextFrameReady(context,nextReady);
    }
    else
    {
	pthread_mutex_unlock(&_licLock);
    }
}

void LicCudaVisMode::postFrame()
{
    if(!_licStarted)
    {
	std::map<std::string,struct UniData> & uniDataMap = _renderer->getUniDataMap();

	_licNextOutputPoints = std::vector<float>(12);
	float * planePoint = (float*)uniDataMap["planePoint"].data;
	float * planeNormal = (float*)uniDataMap["planeNormal"].data;
	float * planeRight = (float*)uniDataMap["planeRight"].data;
	float * planeUp = (float*)uniDataMap["planeUp"].data;

#ifdef WITH_CUDA_LIB

#ifdef PRINT_TIMING
	struct timeval cstart,cend;
	gettimeofday(&cstart,NULL);
#endif

	// copy consts into isolated variables
	memcpy(ccPlanePoint,planePoint,3*sizeof(float));
	memcpy(ccPlaneNormal,planeNormal,3*sizeof(float));
	memcpy(ccPlaneRight,planeRight,3*sizeof(float));
	memcpy(ccPlaneUp,planeUp,3*sizeof(float));
	memcpy(ccPlaneRightNorm,uniDataMap["planeRightNorm"].data,3*sizeof(float));
	memcpy(ccPlaneUpNorm,uniDataMap["planeUpNorm"].data,3*sizeof(float));
	memcpy(&ccPlaneBasisLength,uniDataMap["planeBasisLength"].data,sizeof(float));
	memcpy(&ccTexXMin,uniDataMap["planeBasisXMin"].data,sizeof(float));
	memcpy(&ccTexXMax,uniDataMap["planeBasisXMax"].data,sizeof(float));
	memcpy(&ccTexYMin,uniDataMap["planeBasisYMin"].data,sizeof(float));
	memcpy(&ccTexYMax,uniDataMap["planeBasisYMax"].data,sizeof(float));

#ifdef PRINT_TIMING
	gettimeofday(&cend,NULL);
	std::cerr << "FlowPagedRenderer preframe LIC cuda: " << (cend.tv_sec - cstart.tv_sec) + ((cend.tv_usec - cstart.tv_usec)/1000000.0) << std::endl;
#endif

#endif

	float right[3];
	float up[3];

	right[0] = (LIC_TEXTURE_SIZE/2.0) * planeRight[0];
	right[1] = (LIC_TEXTURE_SIZE/2.0) * planeRight[1];
	right[2] = (LIC_TEXTURE_SIZE/2.0) * planeRight[2];

	up[0] = (LIC_TEXTURE_SIZE/2.0) * planeUp[0];
	up[1] = (LIC_TEXTURE_SIZE/2.0) * planeUp[1];
	up[2] = (LIC_TEXTURE_SIZE/2.0) * planeUp[2];

	_licNextOutputPoints[0] = planePoint[0] - right[0] - up[0];
	_licNextOutputPoints[1] = planePoint[1] - right[1] - up[1];
	_licNextOutputPoints[2] = planePoint[2] - right[2] - up[2];
	_licNextOutputPoints[3] = planePoint[0] + right[0] - up[0];
	_licNextOutputPoints[4] = planePoint[1] + right[1] - up[1];
	_licNextOutputPoints[5] = planePoint[2] + right[2] - up[2];
	_licNextOutputPoints[6] = planePoint[0] + right[0] + up[0];
	_licNextOutputPoints[7] = planePoint[1] + right[1] + up[1];
	_licNextOutputPoints[8] = planePoint[2] + right[2] + up[2];
	_licNextOutputPoints[9] = planePoint[0] - right[0] + up[0];
	_licNextOutputPoints[10] = planePoint[1] - right[1] + up[1];
	_licNextOutputPoints[11] = planePoint[2] - right[2] + up[2];


	//std::cerr << "Setting _licStarted to true" << std::endl;
	_licStarted = true;

	for(std::map<int,bool>::iterator it = _licFinished.begin(); it != _licFinished.end(); ++it)
	{
	    it->second = false;
	}
    }

    // no lock should be needed, threads are blocked waiting for frame start
    for(std::map<int,int>::iterator it = _licContextRenderCount.begin(); it != _licContextRenderCount.end(); ++it)
    {
	int draws = _renderer->getContextRenderCountMap()[it->first];
	if(draws <= 0)
	{
	    _renderer->getContextRenderCountMap()[it->first] = 1;
	    it->second = 1;
	}
	else
	{
	    it->second = draws;
	}
    }
}
