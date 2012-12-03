#include "PointShape.h"

#include <osg/Geometry>

#include <string>
#include <vector>
#include <iostream>

PointShape::PointShape(std::string command, std::string name) 
{
    _type = SimpleShape::POINT;

    setName(name);
    
    _point = new osg::Point();
    
    _vertices = new osg::Vec3Array(1);
    _colors = new osg::Vec4Array(1);
    
    setPosition(osg::Vec3(0.0, 0.0, 0.0));
    setColor(osg::Vec4(1.0, 1.0, 1.0, 1.0));
    setSize(1.0);
    update(command);

    setVertexArray(_vertices); 
    setColorArray(_colors); 
    setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS,0,1));
}

PointShape::~PointShape()
{
}

void PointShape::setPosition(osg::Vec3 p0)
{
    (*_vertices)[0].set(p0[0], p0[1], p0[2]);    
}

void PointShape::setColor(osg::Vec4 c0)
{
    (*_colors)[0].set(c0[0], c0[1], c0[2], c0[3]);

    if( c0[3] != 1.0)
    {
        getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::ON);
        getOrCreateStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    }
    else
    {
        getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::OFF);
        getOrCreateStateSet()->setRenderingHint(osg::StateSet::OPAQUE_BIN);
    }
}

void PointShape::setSize(float size)
{
    _point->setSize(size);    
}

void PointShape::update(std::string command)
{
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    _dirty = true;

    // check for changed values
    addParameter(command, "x");
    addParameter(command, "y");
    addParameter(command, "z");
    addParameter(command, "r");
    addParameter(command, "g");
    addParameter(command, "b");
    addParameter(command, "a");
    addParameter(command, "size");
}

void PointShape::update()
{
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    if( !_dirty )
        return;

    osg::Vec3 p((*_vertices)[0]);
    osg::Vec4 c((*_colors)[0]);
    
    setParameter("x", p.x()); 
    setParameter("y", p.y()); 
    setParameter("z", p.z()); 
    setParameter("r", c.r()); 
    setParameter("g", c.b()); 
    setParameter("b", c.g()); 
    setParameter("a", c.a());
        
    float size = _point->getSize();
    setParameter("size", size);
    _point->setSize(size);
    setPosition(p);
    setColor(c);

	_colors->dirty();
	_vertices->dirty();
    dirtyBound();

	// reset flag
    _dirty = false;
}
