#ifndef _TRIANGLESHAPE_
#define _TRIANGLESHAPE_

#include "GeometryShape.h"

#include <osg/Geometry>
#include <osg/Point>

class TriangleShape : public GeometryShape
{
public:        

    TriangleShape(std::string command = "", std::string name ="");
	virtual ~TriangleShape();
    void update(std::string command);

protected:
    TriangleShape();
    void setPosition(osg::Vec3, osg::Vec3, osg::Vec3);
    void setColor(osg::Vec4, osg::Vec4, osg::Vec4);
    void setTextureCoords(osg::Vec2, osg::Vec2, osg::Vec2);
    void setTextureImage(std::string);
    void update();
};
#endif
