#ifndef _POINTSHAPE_
#define _POINTSHAPE_

#include "BasicShape.h"

#include <osg/Geometry>
#include <osg/Point>

class PointShape : public BasicShape
{
public:        

    PointShape(std::string command = "", std::string name = "");
	virtual ~PointShape();
    void update(std::string);

protected:
    PointShape();
    void setPosition(osg::Vec3);
    void setColor(osg::Vec4);
    void setSize(float);
    void update();
    osg::Point* _point;
};

#endif
