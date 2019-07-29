#include "GeometryShape.h"

#include <sstream>
#include <string>
#include <iostream>

//#include "../Variables.h"

using namespace std;

GeometryShape::GeometryShape()
{
    getOrCreateVertexBufferObject()->setUsage(GL_STREAM_DRAW);
    setUseDisplayList(false);
    setUseVertexBufferObjects(true);
    setUpdateCallback(new GeometryUpdateCallback());
}

osg::MatrixTransform* GeometryShape::getMatrixParent()
{
	return (osg::Geometry::getParent(0)->asGeode())->osg::Node::getParent(0)->asTransform()->asMatrixTransform();
}

osg::Geode* GeometryShape::getParent()
{
    return osg::Geometry::getParent(0)->asGeode();
}

osg::Drawable* GeometryShape::asDrawable()
{
    return dynamic_cast<osg::Drawable*>(this);
}

GeometryShape::~GeometryShape()
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    setUpdateCallback(NULL);
    _dirty = false;
}
