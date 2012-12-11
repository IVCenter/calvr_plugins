#include "TriangleShape.h"

#include <osg/Geometry>
#include <osg/Material>

#include <string>
#include <vector>

TriangleShape::TriangleShape(std::string command, std::string name) 
{
    _type = SimpleShape::TRIANGLE;

    BasicShape::setName(name);
    
    _vertices = new osg::Vec3Array(3);
    _colors = new osg::Vec4Array(3);
   
    setPosition(osg::Vec3(0.0, 0.0, 0.0), osg::Vec3(1.0, 0.0, 0.0), osg::Vec3(1.0, 0.0, 1.0));
    setColor(osg::Vec4(1.0, 0.0, 0.0, 1.0),osg::Vec4(0.0, 1.0, 0.0, 1.0),osg::Vec4(0.0, 0.0, 1.0, 1.0));
    update(command);

    setVertexArray(_vertices); 
    setColorArray(_colors); 
    setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES,0,3));

    osg::StateSet* state = getOrCreateStateSet();
    osg::Material* mat = new osg::Material();
    state->setMode(GL_BLEND, osg::StateAttribute::ON);
    mat->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    state->setAttributeAndModes(mat, osg::StateAttribute::ON);
}

TriangleShape::~TriangleShape()
{
}

void TriangleShape::setPosition(osg::Vec3 p0, osg::Vec3 p1, osg::Vec3 p2)
{
    (*_vertices)[0].set(p0[0], p0[1], p0[2]);    
    (*_vertices)[1].set(p1[0], p1[1], p1[2]);    
    (*_vertices)[2].set(p2[0], p2[1], p2[2]);    
}

void TriangleShape::setColor(osg::Vec4 c0, osg::Vec4 c1, osg::Vec4 c2)
{
    (*_colors)[0].set(c0[0], c0[1], c0[2], c0[3]);    
    (*_colors)[1].set(c1[0], c1[1], c1[2], c1[3]);    
    (*_colors)[2].set(c2[0], c2[1], c2[2], c2[3]);

    if( (c0[3] != 1.0) || (c1[3] != 1.0) || (c2[3] != 1.0))
        getOrCreateStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    else
        getOrCreateStateSet()->setRenderingHint(osg::StateSet::DEFAULT_BIN);

}

void TriangleShape::update(std::string command)
{
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    _dirty = true;

    // check for changed values
    addParameter(command, "x1");
    addParameter(command, "y1");
    addParameter(command, "z1");
    addParameter(command, "r1");
    addParameter(command, "g1");
    addParameter(command, "b1");
    addParameter(command, "a1");

	addParameter(command, "x2");
    addParameter(command, "y2");
    addParameter(command, "z2");
    addParameter(command, "r2");
    addParameter(command, "g2");
    addParameter(command, "b2");
    addParameter(command, "a2");

	addParameter(command, "x3");
    addParameter(command, "y3");
    addParameter(command, "z3");
    addParameter(command, "r3");
    addParameter(command, "g3");
    addParameter(command, "b3");
    addParameter(command, "a3");
}

void TriangleShape::update()
{
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    if( !_dirty )
        return;

    osg::Vec3 p1((*_vertices)[0]);
    osg::Vec4 c1((*_colors)[0]);

    setParameter("x1", p1.x()); 
    setParameter("y1", p1.y()); 
    setParameter("z1", p1.z()); 
    setParameter("r1", c1.r()); 
    setParameter("g1", c1.g()); 
    setParameter("b1", c1.b()); 
    setParameter("a1", c1.a()); 

	osg::Vec3 p2((*_vertices)[1]);
    osg::Vec4 c2((*_colors)[1]);

    setParameter("x2", p2.x()); 
    setParameter("y2", p2.y()); 
    setParameter("z2", p2.z()); 
    setParameter("r2", c2.r()); 
    setParameter("g2", c2.g()); 
    setParameter("b2", c2.b()); 
    setParameter("a2", c2.a()); 

	osg::Vec3 p3((*_vertices)[2]);
    osg::Vec4 c3((*_colors)[2]);

    setParameter("x3", p3.x()); 
    setParameter("y3", p3.y()); 
    setParameter("z3", p3.z()); 
    setParameter("r3", c3.r()); 
    setParameter("g3", c3.g()); 
    setParameter("b3", c3.b()); 
    setParameter("a3", c3.a()); 

    setPosition(p1, p2, p3);
    setColor(c1, c2 ,c3);

	_colors->dirty();
	_vertices->dirty();
    dirtyBound();
    
	// reset flag
    _dirty = false;
}

