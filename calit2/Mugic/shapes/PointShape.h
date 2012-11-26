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
    virtual void update(std::string);
    virtual void update(osg::NodeVisitor*, osg::Drawable*);

protected:
    PointShape();
    void setPosition(osg::Vec3);
    void setColor(osg::Vec4);
    void setSize(float);
    osg::Point* _point;
};

#endif
