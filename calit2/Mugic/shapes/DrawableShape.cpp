#include "DrawableShape.h"

#include <sstream>
#include <string>
#include <iostream>

using namespace std;

DrawableShape::DrawableShape()
{

	setUpdateCallback(new DrawableUpdateCallback());

}

osg::MatrixTransform* DrawableShape::getMatrixParent()
{
	return (osg::ShapeDrawable::getParent(0)->asGeode())->osg::Node::getParent(0)->asTransform()->asMatrixTransform();
}

osg::Geode* DrawableShape::getParent()
{
	return osg::ShapeDrawable::getParent(0)->asGeode();
}

osg::Drawable* DrawableShape::asDrawable()
{
	return dynamic_cast<osg::Drawable*>(this);
}

DrawableShape::~DrawableShape()
{
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
	setUpdateCallback(NULL);
	_dirty = false;
}
