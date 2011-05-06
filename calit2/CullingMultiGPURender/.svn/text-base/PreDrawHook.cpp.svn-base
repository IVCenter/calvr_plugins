#include <GL/glew.h>
#include "PreDrawHook.h"
#include <osg/RenderInfo>

#include <iostream>

/**
 * @param callback a pointer to a class that implements the PreDrawCallback interface
 */
PreDrawHook::PreDrawHook(PreDrawCallback * callback)
{
    _callback = callback;
}

PreDrawHook::~PreDrawHook()
{
}

/**
 * Function called every frame before the draw.
 * Does a callback to our PreDrawCallback class.
 *
 * @param ri the render state of the osg context
 */
void PreDrawHook::operator()(osg::RenderInfo & ri) const
{
    //std::cerr << "PreDraw" << std::endl;
    if(_callback)
    {
	_callback->preDrawCallback(ri.getCurrentCamera());
	
    }
}
