#include <GL/glew.h>
#include "FlowPagedRenderer.h"
#include "GLHelper.h"

#include <iostream>

#include <sys/time.h>

namespace fpr
{
#include "NormalShader.h"
}

using namespace fpr;

pthread_mutex_t FlowPagedRenderer::_glewInitLock = PTHREAD_MUTEX_INITIALIZER;
std::map<int,bool> FlowPagedRenderer::_glewInitMap;

pthread_mutex_t FlowPagedRenderer::_colorTableInitLock = PTHREAD_MUTEX_INITIALIZER;
std::map<int,GLuint> FlowPagedRenderer::_colorTableMap;

FlowPagedRenderer::FlowPagedRenderer(PagedDataSet * set, int frame, FlowVisType type, std::string attribute)
{
    _set = set;
    _currentFrame = frame;
    _nextFrame = frame;
    _type = type;
    _attribute = attribute;

    pthread_mutex_init(&_shaderInitLock,NULL);
    pthread_mutex_init(&_frameReadyLock,NULL);

    initUniData();

    //TODO: read from config
    // 1GB gpu cache
    _cache = new VBOCache(1048576);
}

FlowPagedRenderer::~FlowPagedRenderer()
{
    delete _cache;
}

void FlowPagedRenderer::frameStart(int context)
{
}

void FlowPagedRenderer::preFrame()
{
}

void FlowPagedRenderer::preDraw(int context)
{
}

void FlowPagedRenderer::draw(int context)
{
    checkGlewInit(context);
    checkShaderInit(context);
    checkColorTableInit(context);


    _cache->update(context);

    int fileID = _cache->getFileID(_set->frameFiles[_currentFrame]);

    switch(_type)
    {
	case FVT_NONE:
	{
	    GLuint surfVBO, vertsVBO, attribVBO;
	    
	    surfVBO = _cache->getOrRequestBuffer(context,fileID,_set->frameList[_currentFrame]->surfaceInd.second,_set->frameList[_currentFrame]->surfaceInd.first*sizeof(unsigned int),GL_ELEMENT_ARRAY_BUFFER);
	    vertsVBO = _cache->getOrRequestBuffer(context,fileID,_set->frameList[_currentFrame]->verts.second,_set->frameList[_currentFrame]->verts.first*3*sizeof(float),GL_ARRAY_BUFFER);

	    PagedDataAttrib * attrib = NULL;
	    for(int i = 0; i < _set->frameList[_currentFrame]->pointData.size(); ++i)
	    {
		if(_set->frameList[_currentFrame]->pointData[i]->name == _attribute)
		{
		    attrib = _set->frameList[_currentFrame]->pointData[i];
		    break;
		}
	    }

	    std::vector<AttribBinding> binding;
	    std::vector<TextureBinding> texBinding;
	    std::vector<UniformBinding> uniBinding;

	    GLuint prog = 0;
	    GLuint texture = 0;

	    unsigned int unitsize;
	    if(attrib)
	    {
		AttribBinding ab;
		UniformBinding ub;
		if(attrib->attribType == VAT_VECTORS)
		{
		    ab.size = 3;
		    ab.type = GL_FLOAT;
		    unitsize = 3*sizeof(float);
		    prog = _normalVecProgram[context];

		    ub.location = _normalVecMinUni[context];
		    ub.type = _uniDataMap["minf"].type;
		    ub.data = _uniDataMap["minf"].data;
		    uniBinding.push_back(ub);
		    ub.location = _normalVecMaxUni[context];
		    ub.type = _uniDataMap["maxf"].type;
		    ub.data = _uniDataMap["maxf"].data;
		    uniBinding.push_back(ub);
		}
		else if(attrib->dataType == VDT_INT)
		{
		    ab.size = 1;
		    ab.type = GL_UNSIGNED_INT;
		    unitsize = sizeof(int);
		    prog = _normalIntProgram[context];

		    ub.location = _normalIntMinUni[context];
		    ub.type = _uniDataMap["mini"].type;
		    ub.data = _uniDataMap["mini"].data;
		    uniBinding.push_back(ub);
		    ub.location = _normalIntMaxUni[context];
		    ub.type = _uniDataMap["maxi"].type;
		    ub.data = _uniDataMap["maxi"].data;
		    uniBinding.push_back(ub);
		}
		else
		{
		    ab.size = 1;
		    ab.type = GL_FLOAT;
		    unitsize = sizeof(float);
		    prog = _normalFloatProgram[context];

		    ub.location = _normalFloatMinUni[context];
		    ub.type = _uniDataMap["minf"].type;
		    ub.data = _uniDataMap["minf"].data;
		    uniBinding.push_back(ub);
		    ub.location = _normalFloatMaxUni[context];
		    ub.type = _uniDataMap["maxf"].type;
		    ub.data = _uniDataMap["maxf"].data;
		    uniBinding.push_back(ub);
		}

		ab.index = 4;
		attribVBO = _cache->getOrRequestBuffer(context,fileID,attrib->offset,_set->frameList[_currentFrame]->verts.first*unitsize,GL_ARRAY_BUFFER);
		ab.buffer = attribVBO;
		binding.push_back(ab);

		TextureBinding tb;

		pthread_mutex_lock(&_colorTableInitLock);
		tb.id = _colorTableMap[context];
		pthread_mutex_unlock(&_colorTableInitLock);

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
		glEnable(GL_CULL_FACE);
		drawElements(GL_TRIANGLES,_set->frameList[_currentFrame]->surfaceInd.first,GL_UNSIGNED_INT,surfVBO,vertsVBO,color,binding,prog,texBinding,uniBinding);
		glDisable(GL_CULL_FACE);
	    }
	    else
	    {
		//std::cerr << "not drawn" << std::endl;
	    }

	    if(_currentFrame != _nextFrame)
	    {
		int nextfileID = _cache->getFileID(_set->frameFiles[_nextFrame]);
		GLuint ibuf, vbuf, abuf = 0;
		ibuf = _cache->getOrRequestBuffer(context,nextfileID,_set->frameList[_nextFrame]->surfaceInd.second,_set->frameList[_nextFrame]->surfaceInd.first*sizeof(unsigned int),GL_ELEMENT_ARRAY_BUFFER);
		vbuf = _cache->getOrRequestBuffer(context,nextfileID,_set->frameList[_nextFrame]->verts.second,_set->frameList[_nextFrame]->verts.first*3*sizeof(float),GL_ARRAY_BUFFER);
		if(attrib)
		{
		    abuf = _cache->getOrRequestBuffer(context,nextfileID,attrib->offset,_set->frameList[_nextFrame]->verts.first*unitsize,GL_ARRAY_BUFFER);
		}


		pthread_mutex_lock(&_frameReadyLock);
		if(ibuf && vbuf && (!attrib || abuf))
		{
		    _nextFrameReady[context] = true;
		}
		else
		{
		    _nextFrameReady[context] = false;
		}
		pthread_mutex_unlock(&_frameReadyLock);
	    }

	    break;
	}
	case FVT_ISO_SURFACE:
	{
	    GLuint indVBO, surfVBO, vertsVBO, attribVBO;
	    
	    indVBO = _cache->getOrRequestBuffer(context,fileID,_set->frameList[_currentFrame]->indices.second,_set->frameList[_currentFrame]->indices.first*sizeof(unsigned int),GL_ELEMENT_ARRAY_BUFFER);
	    surfVBO = _cache->getOrRequestBuffer(context,fileID,_set->frameList[_currentFrame]->surfaceInd.second,_set->frameList[_currentFrame]->surfaceInd.first*sizeof(unsigned int),GL_ELEMENT_ARRAY_BUFFER);
	    vertsVBO = _cache->getOrRequestBuffer(context,fileID,_set->frameList[_currentFrame]->verts.second,_set->frameList[_currentFrame]->verts.first*3*sizeof(float),GL_ARRAY_BUFFER);

	    PagedDataAttrib * attrib = NULL;
	    for(int i = 0; i < _set->frameList[_currentFrame]->pointData.size(); ++i)
	    {
		if(_set->frameList[_currentFrame]->pointData[i]->name == _attribute)
		{
		    attrib = _set->frameList[_currentFrame]->pointData[i];
		    break;
		}
	    }

	    std::vector<AttribBinding> surfAttribBinding;
	    std::vector<TextureBinding> surfTexBinding;
	    std::vector<UniformBinding> surfUniBinding;

	    std::vector<AttribBinding> meshAttribBinding;
	    std::vector<TextureBinding> meshTexBinding;
	    std::vector<UniformBinding> meshUniBinding;

	    GLuint surfProg = 0;
	    GLuint meshProg = 0;

	    bool drawMesh = false;

	    unsigned int unitsize;
	    if(attrib)
	    {
		AttribBinding ab;
		UniformBinding ub;
		if(attrib->attribType == VAT_VECTORS)
		{
		    ab.size = 3;
		    ab.type = GL_FLOAT;
		    unitsize = 3*sizeof(float);
		    surfProg = _normalVecProgram[context];
		    meshProg = _isoVecProgram[context];

		    ub.location = _isoVecMaxUni[context];
		    ub.type = _uniDataMap["isoMax"].type;
		    ub.data = _uniDataMap["isoMax"].data;
		    meshUniBinding.push_back(ub);

		    ub.location = _normalVecMinUni[context];
		    ub.type = _uniDataMap["minf"].type;
		    ub.data = _uniDataMap["minf"].data;
		    surfUniBinding.push_back(ub);
		    ub.location = _normalVecMaxUni[context];
		    ub.type = _uniDataMap["maxf"].type;
		    ub.data = _uniDataMap["maxf"].data;
		    surfUniBinding.push_back(ub);

		    drawMesh = true;
		}
		else if(attrib->dataType == VDT_INT)
		{
		    ab.size = 1;
		    ab.type = GL_UNSIGNED_INT;
		    unitsize = sizeof(int);
		    surfProg = _normalIntProgram[context];

		    ub.location = _normalIntMinUni[context];
		    ub.type = _uniDataMap["mini"].type;
		    ub.data = _uniDataMap["mini"].data;
		    surfUniBinding.push_back(ub);
		    ub.location = _normalIntMaxUni[context];
		    ub.type = _uniDataMap["maxi"].type;
		    ub.data = _uniDataMap["maxi"].data;
		    surfUniBinding.push_back(ub);

		    // no mesh for int attributes
		    drawMesh = false;
		}
		else
		{
		    ab.size = 1;
		    ab.type = GL_FLOAT;
		    unitsize = sizeof(float);
		    surfProg = _normalFloatProgram[context];
		    meshProg = _isoProgram[context];

		    ub.location = _isoMaxUni[context];
		    ub.type = _uniDataMap["isoMax"].type;
		    ub.data = _uniDataMap["isoMax"].data;
		    meshUniBinding.push_back(ub);

		    ub.location = _normalFloatMinUni[context];
		    ub.type = _uniDataMap["minf"].type;
		    ub.data = _uniDataMap["minf"].data;
		    surfUniBinding.push_back(ub);
		    ub.location = _normalFloatMaxUni[context];
		    ub.type = _uniDataMap["maxf"].type;
		    ub.data = _uniDataMap["maxf"].data;
		    surfUniBinding.push_back(ub);

		    drawMesh = true;
		}

		ab.index = 4;
		attribVBO = _cache->getOrRequestBuffer(context,fileID,attrib->offset,_set->frameList[_currentFrame]->verts.first*unitsize,GL_ARRAY_BUFFER);
		ab.buffer = attribVBO;
		meshAttribBinding.push_back(ab);
		surfAttribBinding.push_back(ab);

		TextureBinding tb;

		pthread_mutex_lock(&_colorTableInitLock);
		tb.id = _colorTableMap[context];
		pthread_mutex_unlock(&_colorTableInitLock);

		tb.unit = 0;
		tb.type = GL_TEXTURE_1D;
		surfTexBinding.push_back(tb);

	    }
	    else
	    {
		attribVBO = 0;
		surfProg = _normalProgram[context];
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
		glEnable(GL_CULL_FACE);
		drawElements(GL_TRIANGLES,_set->frameList[_currentFrame]->surfaceInd.first,GL_UNSIGNED_INT,surfVBO,vertsVBO,color,surfAttribBinding,surfProg,surfTexBinding,surfUniBinding);
		glDisable(GL_CULL_FACE);
	    }
	    else
	    {
		//std::cerr << "not drawn" << std::endl;
	    }

	    if(drawMesh && attribVBO && indVBO)
	    {
		std::vector<float> color(4);
		color[0] = 0.0;
		color[1] = 0.0;
		color[2] = 1.0;
		color[3] = 1.0;
		drawElements(GL_LINES_ADJACENCY,_set->frameList[_currentFrame]->indices.first,GL_UNSIGNED_INT,indVBO,vertsVBO,color,meshAttribBinding,meshProg,meshTexBinding,meshUniBinding);
	    }

	    if(_currentFrame != _nextFrame)
	    {
		int nextfileID = _cache->getFileID(_set->frameFiles[_nextFrame]);
		GLuint fullibuf, ibuf, vbuf, abuf = 0;
		fullibuf = _cache->getOrRequestBuffer(context,nextfileID,_set->frameList[_nextFrame]->indices.second,_set->frameList[_nextFrame]->indices.first*sizeof(unsigned int),GL_ELEMENT_ARRAY_BUFFER);
		ibuf = _cache->getOrRequestBuffer(context,nextfileID,_set->frameList[_nextFrame]->surfaceInd.second,_set->frameList[_nextFrame]->surfaceInd.first*sizeof(unsigned int),GL_ELEMENT_ARRAY_BUFFER);
		vbuf = _cache->getOrRequestBuffer(context,nextfileID,_set->frameList[_nextFrame]->verts.second,_set->frameList[_nextFrame]->verts.first*3*sizeof(float),GL_ARRAY_BUFFER);
		if(attrib)
		{
		    abuf = _cache->getOrRequestBuffer(context,nextfileID,attrib->offset,_set->frameList[_nextFrame]->verts.first*unitsize,GL_ARRAY_BUFFER);
		}


		pthread_mutex_lock(&_frameReadyLock);
		if(fullibuf && ibuf && vbuf && (!attrib || abuf))
		{
		    _nextFrameReady[context] = true;
		}
		else
		{
		    _nextFrameReady[context] = false;
		}
		pthread_mutex_unlock(&_frameReadyLock);
	    }

	    break;
	}
	default:
	    break;
    }
}

void FlowPagedRenderer::postFrame()
{
    _cache->advanceTime();
}

void FlowPagedRenderer::setType(FlowVisType type, std::string attribute)
{
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

std::string FlowPagedRenderer::getAttribute()
{
    return _attribute;
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

void FlowPagedRenderer::freeResources(int context)
{
    _cache->update(context);
    _cache->freeResources(context);
}

bool FlowPagedRenderer::freeDone()
{
    return _cache->freeDone();
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

void FlowPagedRenderer::checkShaderInit(int context)
{
    pthread_mutex_lock(&_shaderInitLock);

    if(!_shaderInitMap[context])
    {
	GLuint verts, frags, geoms;
	createShaderFromSrc(normalVertSrc,GL_VERTEX_SHADER,verts,"NormalVert");
	createShaderFromSrc(normalGeomSrc,GL_GEOMETRY_SHADER,geoms,"NormalGeom");
	createShaderFromSrc(normalFragSrc,GL_FRAGMENT_SHADER,frags,"NormalFrag");
	createProgram(_normalProgram[context],verts,frags,geoms,GL_TRIANGLES,GL_TRIANGLE_STRIP,3);

	createShaderFromSrc(normalFloatVertSrc,GL_VERTEX_SHADER,verts,"NormalFloatVert");
	createShaderFromSrc(normalFloatGeomSrc,GL_GEOMETRY_SHADER,geoms,"NormalFloatGeom");
	createShaderFromSrc(normalFloatFragSrc,GL_FRAGMENT_SHADER,frags,"NormalFloatFrag");
	createProgram(_normalFloatProgram[context],verts,frags,geoms,GL_TRIANGLES,GL_TRIANGLE_STRIP,3);

	_normalFloatMinUni[context] = glGetUniformLocation(_normalFloatProgram[context],"min");
	_normalFloatMaxUni[context] = glGetUniformLocation(_normalFloatProgram[context],"max");

	createShaderFromSrc(normalIntVertSrc,GL_VERTEX_SHADER,verts,"NormalIntVert");
	createShaderFromSrc(normalFloatGeomSrc,GL_GEOMETRY_SHADER,geoms,"NormalFloatGeom");
	createShaderFromSrc(normalFloatFragSrc,GL_FRAGMENT_SHADER,frags,"NormalFloatFrag");
	createProgram(_normalIntProgram[context],verts,frags,geoms,GL_TRIANGLES,GL_TRIANGLE_STRIP,3);

	_normalIntMinUni[context] = glGetUniformLocation(_normalIntProgram[context],"min");
	_normalIntMaxUni[context] = glGetUniformLocation(_normalIntProgram[context],"max");

	createShaderFromSrc(normalVecVertSrc,GL_VERTEX_SHADER,verts,"NormalVecVert");
	createShaderFromSrc(normalFloatGeomSrc,GL_GEOMETRY_SHADER,geoms,"NormalFloatGeom");
	createShaderFromSrc(normalFloatFragSrc,GL_FRAGMENT_SHADER,frags,"NormalFloatFrag");
	createProgram(_normalVecProgram[context],verts,frags,geoms,GL_TRIANGLES,GL_TRIANGLE_STRIP,3);

	_normalVecMinUni[context] = glGetUniformLocation(_normalVecProgram[context],"min");
	_normalVecMaxUni[context] = glGetUniformLocation(_normalVecProgram[context],"max");

	createShaderFromSrc(isoFloatVertSrc,GL_VERTEX_SHADER,verts,"isoFloatVert");
	createShaderFromSrc(isoGeomSrc,GL_GEOMETRY_SHADER,geoms,"isoGeom");
	createShaderFromSrc(isoFragSrc,GL_FRAGMENT_SHADER,frags,"isoFrag");
	createProgram(_isoProgram[context],verts,frags,geoms,GL_LINES_ADJACENCY,GL_TRIANGLE_STRIP,4);

	_isoMaxUni[context] = glGetUniformLocation(_isoProgram[context],"isoMax");

	createShaderFromSrc(isoVecVertSrc,GL_VERTEX_SHADER,verts,"isoVecVert");
	createShaderFromSrc(isoGeomSrc,GL_GEOMETRY_SHADER,geoms,"isoGeom");
	createShaderFromSrc(isoFragSrc,GL_FRAGMENT_SHADER,frags,"isoFrag");
	createProgram(_isoVecProgram[context],verts,frags,geoms,GL_LINES_ADJACENCY,GL_TRIANGLE_STRIP,4);

	_isoVecMaxUni[context] = glGetUniformLocation(_isoVecProgram[context],"isoMax");

	_shaderInitMap[context] = true;
    }

    pthread_mutex_unlock(&_shaderInitLock);
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
    switch(data.type)
    {
	case UNI_FLOAT:
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
    glDisable(GL_CULL_FACE);
}
