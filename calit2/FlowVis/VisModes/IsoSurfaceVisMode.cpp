#include <GL/glew.h>

#ifndef GL_DOUBLE_MAT3x2
#undef GL_ARB_gpu_shader_fp64
#endif

#include "IsoSurfaceVisMode.h"

#include "../FlowPagedRenderer.h"
#include "../GLHelper.h"

// namespace scope to fix compile issues
namespace isoVisMode
{
#include "../glsl/NormalShader.h"
}

using namespace isoVisMode;

IsoSurfaceVisMode::IsoSurfaceVisMode()
{
    pthread_mutex_init(&_shaderInitLock,NULL);
}

IsoSurfaceVisMode::~IsoSurfaceVisMode()
{
    pthread_mutex_destroy(&_shaderInitLock);
}

void IsoSurfaceVisMode::initContext(int context)
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

	createShaderFromSrc(isoFloatVertSrc,GL_VERTEX_SHADER,verts,"isoFloatVert");
	createShaderFromSrc(isoGeomSrc,GL_GEOMETRY_SHADER,geoms,"isoGeom");
	createShaderFromSrc(isoFragSrc,GL_FRAGMENT_SHADER,frags,"isoFrag");
	createProgram(_isoProgram[context],verts,frags,geoms,GL_LINES_ADJACENCY,GL_TRIANGLE_STRIP,4);

	glDeleteShader(verts);
	glDeleteShader(geoms);
	glDeleteShader(frags);

	_isoMaxUni[context] = glGetUniformLocation(_isoProgram[context],"isoMax");

	createShaderFromSrc(isoVecVertSrc,GL_VERTEX_SHADER,verts,"isoVecVert");
	createShaderFromSrc(isoGeomSrc,GL_GEOMETRY_SHADER,geoms,"isoGeom");
	createShaderFromSrc(isoFragSrc,GL_FRAGMENT_SHADER,frags,"isoFrag");
	createProgram(_isoVecProgram[context],verts,frags,geoms,GL_LINES_ADJACENCY,GL_TRIANGLE_STRIP,4);

	glDeleteShader(verts);
	glDeleteShader(geoms);
	glDeleteShader(frags);

	_isoVecMaxUni[context] = glGetUniformLocation(_isoVecProgram[context],"isoMax");

	_shaderInitMap[context] = true;
    }

    pthread_mutex_unlock(&_shaderInitLock);
}

void IsoSurfaceVisMode::uinitContext(int context)
{
    pthread_mutex_lock(&_shaderInitLock);

    if(_shaderInitMap[context])
    {
	glDeleteProgram(_normalProgram[context]);
	glDeleteProgram(_normalFloatProgram[context]);
	glDeleteProgram(_normalIntProgram[context]);
	glDeleteProgram(_normalVecProgram[context]);
	glDeleteProgram(_isoProgram[context]);
	glDeleteProgram(_isoVecProgram[context]);

	_shaderInitMap[context] = false;
    }

    pthread_mutex_unlock(&_shaderInitLock);
}

void IsoSurfaceVisMode::draw(int context)
{
    VBOCache * cache = _renderer->getCache();
    PagedDataSet * set = _renderer->getSet();
    int currentFrame = _renderer->getCurrentFrame();
    int nextFrame = _renderer->getNextFrame();
    int fileID = cache->getFileID(set->frameFiles[currentFrame]);
    std::map<std::string,struct UniData> & uniDataMap = _renderer->getUniDataMap();

    GLuint indVBO, surfVBO, vertsVBO, attribVBO;

    indVBO = cache->getOrRequestBuffer(context,fileID,set->frameList[currentFrame]->indices.second,set->frameList[currentFrame]->indices.first*sizeof(unsigned int),GL_ELEMENT_ARRAY_BUFFER);
    surfVBO = cache->getOrRequestBuffer(context,fileID,set->frameList[currentFrame]->surfaceInd.second,set->frameList[currentFrame]->surfaceInd.first*sizeof(unsigned int),GL_ELEMENT_ARRAY_BUFFER);
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

    std::vector<FlowPagedRenderer::AttribBinding> surfAttribBinding;
    std::vector<FlowPagedRenderer::TextureBinding> surfTexBinding;
    std::vector<FlowPagedRenderer::UniformBinding> surfUniBinding;

    std::vector<FlowPagedRenderer::AttribBinding> meshAttribBinding;
    std::vector<FlowPagedRenderer::TextureBinding> meshTexBinding;
    std::vector<FlowPagedRenderer::UniformBinding> meshUniBinding;

    GLuint surfProg = 0;
    GLuint meshProg = 0;

    bool drawMesh = false;

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
	    surfProg = _normalVecProgram[context];
	    meshProg = _isoVecProgram[context];

	    ub.location = _isoVecMaxUni[context];
	    ub.type = uniDataMap["isoMax"].type;
	    ub.data = uniDataMap["isoMax"].data;
	    meshUniBinding.push_back(ub);

	    ub.location = _normalVecMinUni[context];
	    ub.type = uniDataMap["minf"].type;
	    ub.data = uniDataMap["minf"].data;
	    surfUniBinding.push_back(ub);
	    ub.location = _normalVecMaxUni[context];
	    ub.type = uniDataMap["maxf"].type;
	    ub.data = uniDataMap["maxf"].data;
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
	    ub.type = uniDataMap["mini"].type;
	    ub.data = uniDataMap["mini"].data;
	    surfUniBinding.push_back(ub);
	    ub.location = _normalIntMaxUni[context];
	    ub.type = uniDataMap["maxi"].type;
	    ub.data = uniDataMap["maxi"].data;
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
	    ub.type = uniDataMap["isoMax"].type;
	    ub.data = uniDataMap["isoMax"].data;
	    meshUniBinding.push_back(ub);

	    ub.location = _normalFloatMinUni[context];
	    ub.type = uniDataMap["minf"].type;
	    ub.data = uniDataMap["minf"].data;
	    surfUniBinding.push_back(ub);
	    ub.location = _normalFloatMaxUni[context];
	    ub.type = uniDataMap["maxf"].type;
	    ub.data = uniDataMap["maxf"].data;
	    surfUniBinding.push_back(ub);

	    drawMesh = true;
	}

	ab.index = 4;
	attribVBO = cache->getOrRequestBuffer(context,fileID,attrib->offset,set->frameList[currentFrame]->verts.first*unitsize,GL_ARRAY_BUFFER);
	ab.buffer = attribVBO;
	meshAttribBinding.push_back(ab);
	surfAttribBinding.push_back(ab);

	FlowPagedRenderer::TextureBinding tb;

	tb.id = _renderer->getColorTableID(context);

	tb.unit = 0;
	tb.type = GL_TEXTURE_1D;
	surfTexBinding.push_back(tb);

    }
    else
    {
	attribVBO = 0;
	surfProg = _normalProgram[context];
    }

    //std::cerr << "context: " << context << " CurrentFrame: " << _currentFrame << " nextFrame: " << _nextFrame << std::endl;
    //std::cerr << "context: " << context << " fullind: " << indVBO << " surfind: " << surfVBO << " vert: " << vertsVBO << " attrib: " << attribVBO << std::endl;
    if(surfVBO && vertsVBO && (!attrib || attribVBO))
    {
	std::vector<float> color(4);
	color[0] = 1.0;
	color[1] = 1.0;
	color[2] = 1.0;
	color[3] = 1.0;

	if(set->revCullFace)
	{
	    glCullFace(GL_FRONT);
	}
	else
	{
	    glCullFace(GL_BACK);
	}

	glEnable(GL_CULL_FACE);
	_renderer->drawElements(GL_TRIANGLES,set->frameList[currentFrame]->surfaceInd.first,GL_UNSIGNED_INT,surfVBO,vertsVBO,color,surfAttribBinding,surfProg,surfTexBinding,surfUniBinding);
	glDisable(GL_CULL_FACE);
    }
    else
    {
	//std::cerr << "not drawn surfind: " << surfVBO << " vert: " << vertsVBO << " attrib: " << attribVBO << std::endl;
    }

    if(drawMesh && attribVBO && indVBO)
    {
	std::vector<float> color(4);
	color[0] = 0.0;
	color[1] = 0.0;
	color[2] = 1.0;
	color[3] = 1.0;
	_renderer->drawElements(GL_LINES_ADJACENCY,set->frameList[currentFrame]->indices.first,GL_UNSIGNED_INT,indVBO,vertsVBO,color,meshAttribBinding,meshProg,meshTexBinding,meshUniBinding);
    }

    if(currentFrame != nextFrame)
    {
	int nextfileID = cache->getFileID(set->frameFiles[nextFrame]);
	GLuint fullibuf, ibuf, vbuf, abuf = 0;
	fullibuf = cache->getOrRequestBuffer(context,nextfileID,set->frameList[nextFrame]->indices.second,set->frameList[nextFrame]->indices.first*sizeof(unsigned int),GL_ELEMENT_ARRAY_BUFFER);
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
	if(fullibuf && ibuf && vbuf && (!nextattrib || abuf))
	{
	    //std::cerr << "next frame ready. surf: " << ibuf << " vert: " << vbuf << " attrib: " << abuf << std::endl;
	    nextReady = true;
	}
	else
	{
	    nextReady = false;
	}
	_renderer->setNextFrameReady(context,nextReady);
    }
}
