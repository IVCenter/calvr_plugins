#include "ScalableLineShape.h"

#include <osg/Geometry>
#include <osg/Material>

#include <string>
#include <vector>
#include <iostream>

ScalableLineShape::ScalableLineShape(std::string command, std::string name) 
{
    // check for changed values
    createParameter("pos1", new Vec3Type());
    createParameter("pos2", new Vec3Type());
    createParameter("color", new Vec4Type());
    createParameter("width", new FloatType());

    _type = SimpleShape::SCALABLELINE;

    BasicShape::setName(name);
    
    _vertices = new osg::Vec3Array(4);
    _colors = new osg::Vec4Array(1);
    
    setPosition(osg::Vec3(0.0, 0.0, 0.0), osg::Vec3(1.0, 0.0, 0.0), 1.0);
    setColor(osg::Vec4(1.0, 0.0, 0.0, 1.0));
    update(command);
    
    setVertexArray(_vertices); 
    setColorArray(_colors); 
    setColorBinding(osg::Geometry::BIND_OVERALL);
    addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_STRIP,0,4));

    //osg::StateSet* state = getOrCreateStateSet();
    //state->setMode(GL_BLEND, osg::StateAttribute::ON);
    //osg::Material* mat = new osg::Material();
    //mat->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    //state->setAttributeAndModes(mat, osg::StateAttribute::ON);
}

ScalableLineShape::~ScalableLineShape()
{
}

void ScalableLineShape::setPosition(osg::Vec3 p1, osg::Vec3 p2, float width)
{
    // make sure width is not set to zero (will make it positive even though a negative width would work below)
    if( width <= 0 )
        width = 0.1; 

    // just set default screen normal
    osg::Vec3 screenNormal(0, -1, 0);
    osg::Vec3 tangent(0, 0, 1);

    // check which scene system is being used
    if( Globals::G_ROTAXIS )
    {
        screenNormal.set(0,0,1);
        tangent.set(0,1,0);
    }

    // compute line (direction doesnt matter)
    osg::Vec3 line = p2 - p1;

    // need normal for line for parallel test
    osg::Vec3 lineNormal = line;
    lineNormal.normalize();
    
    //check if parallel with screen normal or line length is not zero
    if ( line.length2() != 0 || (lineNormal * screenNormal) != 1 || (lineNormal * screenNormal) != -1 )
    {
        // can compute tangent 
        tangent = line ^ screenNormal;
        tangent.normalize();
    }

    // set the 4 points of the triangle strip
    (*_vertices)[0] = p1 + (tangent * width);
	(*_vertices)[1] = p2 + (tangent * width);
	(*_vertices)[2] = p1 + (tangent * -width);
	(*_vertices)[3] = p2 + (tangent * -width);
}

void ScalableLineShape::setColor(osg::Vec4 c0)
{
    (*_colors)[0].set(c0[0], c0[1], c0[2], c0[3]);
    
    if(c0[3] != 1.0)
        getOrCreateStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    else
        getOrCreateStateSet()->setRenderingHint(osg::StateSet::DEFAULT_BIN);

}

void ScalableLineShape::update(std::string command)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    _dirty = true;

    // check for changed values
    setParameter(command, "pos1");
    setParameter(command, "pos2");
    setParameter(command, "color");
    setParameter(command, "width");
}

void ScalableLineShape::update()
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    if( !_dirty )
        return;

    //float width = (*_vertices)[1].x() - (*_vertices)[0].x();
    //float height = (*_vertices)[2].z() - (*_vertices)[1].z();
    osg::Vec4 c = (*_colors)[0];

    osg::Vec3 p1 = getParameter("pos1")->asVec3Type()->getValue();
    osg::Vec3 p2 = getParameter("pos2")->asVec3Type()->getValue();
    c = getParameter("color")->asVec4Type()->getValue();
    float width = getParameter("width")->asFloatType()->getValue();

    setPosition(p1, p2, width);
    setColor(c);
    _vertices->dirty();
    _colors->dirty();
    dirtyBound();

    // reset flag
    _dirty = false;
}

