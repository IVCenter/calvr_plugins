#include "CircleShape.h"

#include <osg/Geometry>
#include <osg/Material>

#include <string>
#include <vector>
#include <iostream>

CircleShape::CircleShape(std::string command, std::string name) 
{
    // check for changed values
    createParameter("pos", new Vec3Type());
    createParameter("color", new Vec4Type());
    createParameter("radius", new FloatType());

    _type = SimpleShape::CIRCLE;

    BasicShape::setName(name);
    _numFaces = 20;
    
    _vertices = new osg::Vec3Array(_numFaces + 2);
    _colors = new osg::Vec4Array(1);
    
    setPosition(osg::Vec3(0.0, 0.0, 0.0), 1.0);
    setColor(osg::Vec4(1.0, 0.0, 0.0, 1.0));
    
    update(command);

    setVertexArray(_vertices); 
    setColorArray(_colors); 
    setColorBinding(osg::Geometry::BIND_OVERALL);
    addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_FAN,0,_numFaces + 2));

    osg::StateSet* state = getOrCreateStateSet();
    //state->setMode(GL_BLEND, osg::StateAttribute::ON);
    state->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
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

        if( Globals::G_ROTAXIS )
        {
            pos.x()+= cos(portion * i) * radius;
            pos.y()+= sin(portion * i) * radius;
        }
        else
        {
            pos.x()+= cos(portion * i) * radius;
            pos.z()+= sin(portion * i) * radius;
        }
        (*_vertices)[i].set(pos.x(), pos.y(), pos.z());
    }
}

void CircleShape::setColor(osg::Vec4 c)
{
    (*_colors)[0].set(c[0], c[1], c[2], c[3]);
    
    if( c[3] != 1.0 )
        getOrCreateStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    else
        getOrCreateStateSet()->setRenderingHint(osg::StateSet::DEFAULT_BIN);
}

// map update
void CircleShape::update(std::string command)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    _dirty = true;

    if( !command.empty() )
    {
        // check for changed values
        setParameter(command, "pos");
        setParameter(command, "color");
        setParameter(command, "radius");
    }
}

// render update
void CircleShape::update()
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    if( !_dirty )
        return;

    osg::Vec3 p1((*_vertices)[0]);
    osg::Vec4 c((*_colors)[0]);
    float radius = (*_vertices)[1].x() - (*_vertices)[0].x();

    // get values out
    p1 = getParameter("pos")->asVec3Type()->getValue();
    c = getParameter("color")->asVec4Type()->getValue();
    radius = getParameter("radius")->asFloatType()->getValue();

    setPosition(p1, radius);
    setColor(c);
    _vertices->dirty();
    _colors->dirty();
    dirtyBound();

    // reset flag
    _dirty = false;
}
