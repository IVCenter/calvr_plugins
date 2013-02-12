#ifndef _QUADSHAPE_
#define _QUADSHAPE_

#include "GeometryShape.h"

#include <osg/Geometry>
#include <osg/Point>

class QuadShape : public GeometryShape
{
public:        

    QuadShape(std::string command, std::string name);
    virtual ~QuadShape();
    void update(std::string);

protected:
    QuadShape();
    void setPosition(osg::Vec3, osg::Vec3, osg::Vec3, osg::Vec3);
    void setColor(osg::Vec4, osg::Vec4, osg::Vec4, osg::Vec4);
    void setTextureCoords(osg::Vec2, osg::Vec2, osg::Vec2, osg::Vec2);
    void setTextureImage(std::string);
    void update();
};

#endif
