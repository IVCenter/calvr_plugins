#ifndef _SCALEABLELINESHAPE_
#define _SCALEABLELINESHAPE_

#include "GeometryShape.h"
#include "../Globals.h"

#include <osg/Geometry>

class ScalableLineShape : public GeometryShape
{
public:        

    ScalableLineShape(std::string command = "", std::string name = "");
	virtual ~ScalableLineShape();
    void update(std::string);

protected:
    ScalableLineShape();
    void setPosition(osg::Vec3, osg::Vec3, float width);
    void setColor(osg::Vec4);
    void update();
};

#endif
