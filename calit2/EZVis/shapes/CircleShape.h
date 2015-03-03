#ifndef _CIRCLESHAPE_
#define _CIRCLESHAPE_

#include "GeometryShape.h"
#include "../Globals.h"

#include <osg/Geometry>

class CircleShape : public GeometryShape
{
public:        

    CircleShape(std::string command = "", std::string name = "");
	virtual ~CircleShape();
    void update(std::string);

protected:
    CircleShape();
    void setPosition(osg::Vec3, float);
    void setColor(osg::Vec4);
    void update();
    int _numFaces;
};

#endif
