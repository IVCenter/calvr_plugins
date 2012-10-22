#include "TexturedSphere.h"

#include <osg/Geometry>
#include <osg/Texture2D>
#include <osgDB/ReadFile>

#include <iostream>
#include <cmath>

osg::Geode * TexturedSphere::makeSphere(std::string file, float radius, float tfactor)
{
    osg::Geometry * geom = new osg::Geometry();
    osg::Vec3Array * verts = new osg::Vec3Array();
    osg::Vec4Array * colors = new osg::Vec4Array(1);
    osg::Vec2Array * tex = new osg::Vec2Array();
    colors->at(0) = osg::Vec4(1.0,1.0,1.0,1.0);
    geom->setUseDisplayList(false);
    geom->setVertexArray(verts);
    geom->setColorArray(colors);
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);
    geom->setTexCoordArray(0,tex);

    int segments = (int)(20.0 * tfactor);
    if(segments < 3)
    {
	segments = 3;
    }

    float rotPerSegmentH = 360.0 / (float)segments;
    float rotPerSegmentV = 180.0 / (float)segments;
    rotPerSegmentH *= M_PI / 180.0;
    rotPerSegmentV *= M_PI / 180.0;

    float currentRotH = 0.0;
    float currentRotV = 0.0;
    for(int i = 0; i < segments; ++i)
    {
	currentRotH = 0.0;
	for(int j = 0; j < segments; ++j)
	{
	    float tcoordL,tcoordR,tcoordT,tcoordB;
	    tcoordL = ((float)j) / ((float)segments);
	    tcoordR = ((float)(j+1)) / ((float)segments);
	    tcoordT = ((float)i) / ((float)segments);
	    tcoordB = ((float)(i+1)) / ((float)segments);

	    osg::Vec3 point;
	    // top left
	    point.x() = radius * sin(currentRotV) * cos(currentRotH);
	    point.y() = radius * sin(currentRotV) * sin(currentRotH);
	    point.z() = -radius * cos(currentRotV);
	    verts->push_back(point);
	    tex->push_back(osg::Vec2(tcoordL,tcoordT));

	    // top right;
	    point.x() = radius * sin(currentRotV) * cos(currentRotH+rotPerSegmentH);
	    point.y() = radius * sin(currentRotV) * sin(currentRotH+rotPerSegmentH);
	    point.z() = -radius * cos(currentRotV);
	    verts->push_back(point);
	    tex->push_back(osg::Vec2(tcoordR,tcoordT));

	    // bottom right;
	    point.x() = radius * sin(currentRotV+rotPerSegmentV) * cos(currentRotH+rotPerSegmentH);
	    point.y() = radius * sin(currentRotV+rotPerSegmentV) * sin(currentRotH+rotPerSegmentH);
	    point.z() = -radius * cos(currentRotV+rotPerSegmentV);
	    verts->push_back(point);
	    tex->push_back(osg::Vec2(tcoordR,tcoordB));

	    // bottom left;
	    point.x() = radius * sin(currentRotV+rotPerSegmentV) * cos(currentRotH);
	    point.y() = radius * sin(currentRotV+rotPerSegmentV) * sin(currentRotH);
	    point.z() = -radius * cos(currentRotV+rotPerSegmentV);
	    verts->push_back(point);
	    tex->push_back(osg::Vec2(tcoordL,tcoordB));
	    
	    currentRotH += rotPerSegmentH;
	}
	currentRotV += rotPerSegmentV;
    }   

    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,verts->size()));
    osg::Geode * geode = new osg::Geode();
    geode->addDrawable(geom);

    osg::StateSet * stateset = geode->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

    osg::Image * image = osgDB::readImageFile(file);
    if(image)
    {
	osg::Texture2D * texture = new osg::Texture2D();
	texture->setImage(image);
	stateset->setTextureAttributeAndModes(0,texture,osg::StateAttribute::ON);
    }
    else
    {
	std::cerr << "Unable to read image file: " << file << std::endl;
    }

    return geode;
}
