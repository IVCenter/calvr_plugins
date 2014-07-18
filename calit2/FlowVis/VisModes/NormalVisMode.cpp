#include <GL/glew.h>

#ifndef GL_DOUBLE_MAT3x2
#undef GL_ARB_gpu_shader_fp64
#endif

#include "NormalVisMode.h"

#include "../FlowPagedRenderer.h"
#include "../GLHelper.h"

// namespace scope to fix compile issues
namespace normVisMode
{
#include "../glsl/NormalShader.h"
}

using namespace normVisMode;

NormalVisMode::NormalVisMode()
{
    pthread_mutex_init(&_shaderInitLock,NULL);
}

NormalVisMode::~NormalVisMode()
{
    pthread_mutex_destroy(&_shaderInitLock);
}

void NormalVisMode::initContext(int context)
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

	_shaderInitMap[context] = true;
    }

    pthread_mutex_unlock(&_shaderInitLock);
}

void NormalVisMode::uinitContext(int context)
{
    pthread_mutex_lock(&_shaderInitLock);

    if(_shaderInitMap[context])
    {
	glDeleteProgram(_normalProgram[context]);
	glDeleteProgram(_normalFloatProgram[context]);
	glDeleteProgram(_normalIntProgram[context]);
	glDeleteProgram(_normalVecProgram[context]);

	_shaderInitMap[context] = false;
    }

    pthread_mutex_unlock(&_shaderInitLock);
}

void NormalVisMode::draw(int context)
{
    GLuint surfVBO, vertsVBO, attribVBO;

    VBOCache * cache = _renderer->getCache();
    PagedDataSet * set = _renderer->getSet();
    int currentFrame = _renderer->getCurrentFrame();
    int nextFrame = _renderer->getNextFrame();
    int fileID = cache->getFileID(set->frameFiles[currentFrame]);
    std::map<std::string,struct UniData> & uniDataMap = _renderer->getUniDataMap();

    surfVBO = _renderer->getCache()->getOrRequestBuffer(context,fileID,set->frameList[currentFrame]->surfaceInd.second,set->frameList[currentFrame]->surfaceInd.first*sizeof(unsigned int),GL_ELEMENT_ARRAY_BUFFER);
    vertsVBO = cache->getOrRequestBuffer(context,fileID,set->frameList[currentFrame]->verts.second,set->frameList[currentFrame]->verts.first*3*sizeof(float),GL_ARRAY_BUFFER);

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

    if(currentFrame != nextFrame)
    {
	int nextfileID = cache->getFileID(set->frameFiles[nextFrame]);
	GLuint ibuf, vbuf, abuf = 0;
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

	bool nextReady;
	if(ibuf && vbuf && (!attrib || abuf))
	{
	    nextReady = true;
	}
	else
	{
	    nextReady = false;
	}
	_renderer->setNextFrameReady(context,nextReady);
    }
}
