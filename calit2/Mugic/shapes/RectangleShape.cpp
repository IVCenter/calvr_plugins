#include "RectangleShape.h"

#include <osg/Geometry>
#include <osg/Material>

#include <string>
#include <vector>
#include <iostream>

RectangleShape::RectangleShape(std::string command, std::string name) 
{
    _type = SimpleShape::RECTANGLE;

    setName(name);
    
    _vertices = new osg::Vec3Array(4);
    _colors = new osg::Vec4Array(4);
    
    setPosition(osg::Vec3(0.0, 0.0, 0.0), 1.0, 1.0);
    setColor(osg::Vec4(1.0, 0.0, 0.0, 1.0));
    update(command);
    
    setVertexArray(_vertices); 
    setColorArray(_colors); 
    setColorBinding(osg::Geometry::BIND_OVERALL);
    addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,4));

    osg::StateSet* state = getOrCreateStateSet();
    state->setMode(GL_BLEND, osg::StateAttribute::ON);
    osg::Material* mat = new osg::Material();
    mat->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    state->setAttributeAndModes(mat, osg::StateAttribute::ON);
}

RectangleShape::~RectangleShape()
{
}

void RectangleShape::setPosition(osg::Vec3 p, float width, float height)
{
    (*_vertices)[0].set(p[0], p[1], p[2]);
	(*_vertices)[1].set(p[0] + width, p[1], p[2]);
	(*_vertices)[2].set(p[0] + width, p[1], p[2] + height);
	(*_vertices)[3].set(p[0], p[1], p[2] + height);
}

void RectangleShape::setColor(osg::Vec4 c0)
{
    (*_colors)[0].set(c0[0], c0[1], c0[2], c0[3]);
}

void RectangleShape::update(std::string command)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    _dirty = true;

    // check for changed values
    addParameter(command, "x");
    addParameter(command, "y");
    addParameter(command, "z");
    addParameter(command, "width");
    addParameter(command, "height");
    addParameter(command, "r");
    addParameter(command, "g");
    addParameter(command, "b");
    addParameter(command, "a");
}

void RectangleShape::update()
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    if( !_dirty )
        return;

    osg::Vec3 p1((*_vertices)[0]);
    osg::Vec4 c1((*_colors)[0]);
    float width = (*_vertices)[1].x() - (*_vertices)[0].x();
    float height = (*_vertices)[2].z() - (*_vertices)[1].z();

    setParameter("x", p1.x()); 
    setParameter("y", p1.y()); 
    setParameter("z", p1.z()); 
    setParameter("width", width); 
    setParameter("height", height); 
    setParameter("r", c1.r()); 
    setParameter("g", c1.g()); 
    setParameter("b", c1.b()); 
    setParameter("a", c1.a()); 

    setPosition(p1, width, height);
    setColor(c1);
    _vertices->dirty();
    _colors->dirty();
    dirtyBound();

    // reset flag
    _dirty = false;
}

