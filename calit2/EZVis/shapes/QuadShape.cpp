#include "QuadShape.h"

#include <osg/Geometry>
#include <osg/Material>

#include <string>
#include <vector>

QuadShape::QuadShape(std::string command, std::string name) 
{
    // check for changed values
    createParameter("pos1", new Vec3Type());
    createParameter("pos2", new Vec3Type());
    createParameter("pos3", new Vec3Type());
    createParameter("pos4", new Vec3Type());
    createParameter("color1", new Vec4Type());
    createParameter("color2", new Vec4Type());
    createParameter("color3", new Vec4Type());

    _type = SimpleShape::QUAD; 
    
    BasicShape::setName(name);

    _vertices = new osg::Vec3Array(4);
    _colors = new osg::Vec4Array(4);
   
    setPosition(osg::Vec3(0.0, 0.0, 0.0), osg::Vec3(1.0, 0.0, 0.0), osg::Vec3(1.0, 0.0, 1.0), osg::Vec3(0.0, 0.0, 1.0));
    setColor(osg::Vec4(1.0, 0.0, 0.0, 1.0),osg::Vec4(0.0, 1.0, 0.0, 1.0),osg::Vec4(0.0, 0.0, 1.0, 1.0), osg::Vec4(0.0, 1.0, 1.0, 1.0));
    update(command);
    
    setVertexArray(_vertices); 
    setColorArray(_colors); 
    setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,4));

    osg::StateSet* state = getOrCreateStateSet();
    osg::Material* mat = new osg::Material();
    state->setMode(GL_BLEND, osg::StateAttribute::ON);
    mat->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    state->setAttributeAndModes(mat, osg::StateAttribute::ON);
}

QuadShape::~QuadShape()
{
}

void QuadShape::setPosition(osg::Vec3 p0, osg::Vec3 p1, osg::Vec3 p2, osg::Vec3 p3)
{
    (*_vertices)[0].set(p0[0], p0[1], p0[2]);    
    (*_vertices)[1].set(p1[0], p1[1], p1[2]);    
    (*_vertices)[2].set(p2[0], p2[1], p2[2]);    
    (*_vertices)[3].set(p3[0], p3[1], p3[2]);    
}

void QuadShape::setColor(osg::Vec4 c0, osg::Vec4 c1, osg::Vec4 c2, osg::Vec4 c3)
{
    (*_colors)[0].set(c0[0], c0[1], c0[2], c0[3]);    
    (*_colors)[1].set(c1[0], c1[1], c1[2], c1[3]);    
    (*_colors)[2].set(c2[0], c2[1], c2[2], c2[3]);    
    (*_colors)[3].set(c3[0], c3[1], c3[2], c3[3]);

    if( (c0[3] != 1.0) || (c1[3] != 1.0) || (c2[3] != 1.0) || (c3[3] != 1.0))
        getOrCreateStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    else
        getOrCreateStateSet()->setRenderingHint(osg::StateSet::DEFAULT_BIN);

}

void QuadShape::update(std::string command)
{
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    _dirty = true;

    // check for changed values
    setParameter(command, "pos1");
    setParameter(command, "pos2");
    setParameter(command, "pos3");
    setParameter(command, "pos4");
    setParameter(command, "color1");
    setParameter(command, "color2");
    setParameter(command, "color3");
    setParameter(command, "color4");
}

void QuadShape::update()
{
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    if( !_dirty )
        return;

    osg::Vec3 p1((*_vertices)[0]);
    osg::Vec4 c1((*_colors)[0]);

    p1 = getParameter("pos1")->asVec3Type()->getValue();
    c1 = getParameter("color1")->asVec4Type()->getValue();

	osg::Vec3 p2((*_vertices)[1]);
    osg::Vec4 c2((*_colors)[1]);
    
    p2 = getParameter("pos2")->asVec3Type()->getValue();
    c2 = getParameter("color2")->asVec4Type()->getValue();

	osg::Vec3 p3((*_vertices)[2]);
    osg::Vec4 c3((*_colors)[2]);
    
    p3 = getParameter("pos3")->asVec3Type()->getValue();
    c3 = getParameter("color3")->asVec4Type()->getValue();

	osg::Vec3 p4((*_vertices)[3]);
    osg::Vec4 c4((*_colors)[3]);
    
    p4 = getParameter("pos4")->asVec3Type()->getValue();
    c4 = getParameter("color4")->asVec4Type()->getValue();

    setPosition(p1, p2, p3, p4);
    setColor(c1, c2 ,c3, c4);
    
	_colors->dirty();
	_vertices->dirty();
    dirtyBound();

	// reset flag
    _dirty = false;
}
