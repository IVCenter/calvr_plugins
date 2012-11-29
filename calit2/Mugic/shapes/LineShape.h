#ifndef _LINESHAPE_
#define _LINESHAPE_

#include "BasicShape.h"

#include <osg/Geometry>
#include <osg/LineWidth>

class LineShape : public BasicShape
{
public:        

    LineShape(std::string command = "", std::string name ="");
	virtual ~LineShape();
    void update(std::string);

protected:
    LineShape();
    void setPosition(osg::Vec3, osg::Vec3);
    void setColor(osg::Vec4, osg::Vec4);
    void setWidth(float);
    void update();
    osg::LineWidth* _width;
};
#endif
