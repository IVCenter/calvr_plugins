#include "CircleShape.h"

#include <osg/Geometry>
#include <osg/Material>

#include <string>
#include <vector>
#include <iostream>

CircleShape::CircleShape(std::string command, std::string name) 
{
    _type = SimpleShape::CIRCLE;

    setName(name);
    _numFaces = 20;
    
    _vertices = new osg::Vec3Array(_numFaces + 2);
    _colors = new osg::Vec4Array(_numFaces + 2);
    
    setPosition(osg::Vec3(0.0, 0.0, 0.0), 1.0);
    setColor(osg::Vec4(1.0, 0.0, 0.0, 1.0),osg::Vec4(0.0, 1.0, 0.0, 1.0));
    update(command);
    
    setVertexArray(_vertices); 
    setColorArray(_colors); 
    setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_FAN,0,_numFaces + 2));

    osg::StateSet* state = getOrCreateStateSet();
    state->setMode(GL_BLEND, osg::StateAttribute::ON);
    state->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    //osg::Material* mat = new osg::Material();
    //mat->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    //state->setAttributeAndModes(mat, osg::StateAttribute::ON);
}

CircleShape::~CircleShape()
{
}

void CircleShape::setPosition(osg::Vec3 p, float radius)
{
    // first point center
    (*_vertices)[0].set(p[0], p[1], p[2]);
    
    // compute exterior points anti-clockwise
    float portion = -osg::PI * 2 / _numFaces;
    osg::Vec3d pos;

    for (int i = 1; i < _numFaces + 2; i++) 
    {
        pos = p;
        pos.x()+= cos(portion * i) * radius;
        pos.z()+= sin(portion * i) * radius;
        (*_vertices)[i].set(pos.x(), pos.y(), pos.z());
    }
}

void CircleShape::setColor(osg::Vec4 c0, osg::Vec4 c1)
{
    (*_colors)[0].set(c0[0], c0[1], c0[2], c0[3]);
    
    for(int i = 1; i < (int)_colors->size(); i++)
    {
        (*_colors)[i].set(c1[0], c1[1], c1[2], c1[3]);
    }

    if( (c0[3] != 1.0) || (c1[3] != 1.0))
        getOrCreateStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    else
        getOrCreateStateSet()->setRenderingHint(osg::StateSet::DEFAULT_BIN);
}

void CircleShape::update(std::string command)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    _dirty = true;

    if( !command.empty() )
    {
        // check for changed values
        addParameter(command, "x");
        addParameter(command, "y");
        addParameter(command, "z");
        addParameter(command, "r1");
        addParameter(command, "g1");
        addParameter(command, "b1");
        addParameter(command, "a1");
        addParameter(command, "r2");
        addParameter(command, "g2");
        addParameter(command, "b2");
        addParameter(command, "a2");
        addParameter(command, "radius");
    }
}

void CircleShape::update()
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    if( !_dirty )
        return;

    osg::Vec3 p1((*_vertices)[0]);
    osg::Vec4 c1((*_colors)[0]);
    osg::Vec4 c2((*_colors)[1]);
    float radius = (*_vertices)[1].x() - (*_vertices)[0].x();

    setParameter("x", p1.x()); 
    setParameter("y", p1.y()); 
    setParameter("z", p1.z()); 
    setParameter("radius", radius); 
    setParameter("r1", c1.r()); 
    setParameter("g1", c1.g()); 
    setParameter("b1", c1.b()); 
    setParameter("a1", c1.a()); 
    setParameter("r2", c2.r()); 
    setParameter("g2", c2.g()); 
    setParameter("b2", c2.b()); 
    setParameter("a2", c2.a()); 

    setPosition(p1, radius);
    setColor(c1, c2);
    _vertices->dirty();
    _colors->dirty();
    dirtyBound();

    // reset flag
    _dirty = false;
}
