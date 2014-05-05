#include <GL/glew.h>

#ifndef GL_DOUBLE_MAT3x2
#undef GL_ARB_gpu_shader_fp64
#endif

#include "CallbackDrawable.h"

struct MyGLClientState
{
    GLint abuffer;
    GLint ebuffer;
    GLboolean carray;
    GLboolean efarray;
    GLboolean farray;
    GLboolean iarray;
    GLboolean narray;
    GLboolean scarray;
    GLboolean tarray;
    GLboolean varray;
};

void pushClientState(MyGLClientState & state)
{
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&state.abuffer);
    glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING,&state.ebuffer);
    state.carray = glIsEnabled(GL_COLOR_ARRAY);
    state.efarray = glIsEnabled(GL_EDGE_FLAG_ARRAY);
    state.farray = glIsEnabled(GL_FOG_COORD_ARRAY);
    state.iarray = glIsEnabled(GL_INDEX_ARRAY);
    state.narray = glIsEnabled(GL_NORMAL_ARRAY);
    state.scarray = glIsEnabled(GL_SECONDARY_COLOR_ARRAY);
    state.tarray = glIsEnabled(GL_TEXTURE_COORD_ARRAY);
    state.varray = glIsEnabled(GL_VERTEX_ARRAY);

    if(state.carray)
    {
	glDisableClientState(GL_COLOR_ARRAY);
    }
    if(state.efarray)
    {
	glDisableClientState(GL_EDGE_FLAG_ARRAY);
    }
    if(state.farray)
    {
	glDisableClientState(GL_FOG_COORD_ARRAY);
    }
    if(state.iarray)
    {
	glDisableClientState(GL_INDEX_ARRAY);
    }
    if(state.narray)
    {
	glDisableClientState(GL_NORMAL_ARRAY);
    }
    if(state.scarray)
    {
	glDisableClientState(GL_SECONDARY_COLOR_ARRAY);
    }
    if(state.tarray)
    {
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    }
    if(state.varray)
    {
	glDisableClientState(GL_VERTEX_ARRAY);
    }
}

void popClientState(MyGLClientState & state)
{
    if(state.carray)
    {
	glEnableClientState(GL_COLOR_ARRAY);
    }
    if(state.efarray)
    {
	glEnableClientState(GL_EDGE_FLAG_ARRAY);
    }
    if(state.farray)
    {
	glEnableClientState(GL_FOG_COORD_ARRAY);
    }
    if(state.iarray)
    {
	glEnableClientState(GL_INDEX_ARRAY);
    }
    if(state.narray)
    {
	glEnableClientState(GL_NORMAL_ARRAY);
    }
    if(state.scarray)
    {
	glEnableClientState(GL_SECONDARY_COLOR_ARRAY);
    }
    if(state.tarray)
    {
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    }
    if(state.varray)
    {
	glEnableClientState(GL_VERTEX_ARRAY);
    }

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, state.ebuffer);
    glBindBuffer(GL_ARRAY_BUFFER, state.abuffer);
}

void printGLState()
{
    std::cerr << "Draw State:" << std::endl;
    std::cerr << "ColorArray: " << (int)glIsEnabled(GL_COLOR_ARRAY) << std::endl;
    std::cerr << "EdgeFlag: " << (int)glIsEnabled(GL_EDGE_FLAG_ARRAY) << std::endl;
    std::cerr << "Fog: " << (int)glIsEnabled(GL_FOG_COORD_ARRAY) << std::endl;
    std::cerr << "Index: " << (int)glIsEnabled(GL_INDEX_ARRAY) << std::endl;
    std::cerr << "Normal: " << (int)glIsEnabled(GL_NORMAL_ARRAY) << std::endl;
    std::cerr << "Sec Color: " << (int)glIsEnabled(GL_SECONDARY_COLOR_ARRAY) << std::endl;
    std::cerr << "Texture: " << (int)glIsEnabled(GL_TEXTURE_COORD_ARRAY) << std::endl;
    std::cerr << "Vertex: " << (int)glIsEnabled(GL_VERTEX_ARRAY) << std::endl;
}

void clearGLClientState()
{
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_EDGE_FLAG_ARRAY);
    glDisableClientState(GL_FOG_COORD_ARRAY);
    glDisableClientState(GL_INDEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_SECONDARY_COLOR_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
}

CallbackDrawable::CallbackDrawable(FlowPagedRenderer * renderer, osg::BoundingBox bounds)
{
    _bbox = bounds;
    _renderer = renderer;
    setUseDisplayList(false);
    setUseVertexBufferObjects(true);
}

CallbackDrawable::CallbackDrawable(const CallbackDrawable&,const osg::CopyOp& copyop)
{
}

CallbackDrawable::~CallbackDrawable()
{
}

void CallbackDrawable::drawImplementation(osg::RenderInfo& ri) const
{
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_TEXTURE_1D);

    MyGLClientState state;
    //printGLState();
    pushClientState(state);
    _renderer->draw(ri.getContextID());
    popClientState(state);

    glPopAttrib();
}

osg::BoundingBox CallbackDrawable::computeBound() const
{
    return _bbox;
}

void CallbackDrawable::updateBoundingBox()
{
    computeBound();
    dirtyBound();
}
