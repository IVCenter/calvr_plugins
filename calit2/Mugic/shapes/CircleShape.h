#ifndef _CIRCLESHAPE_
#define _CIRCLESHAPE_

#include "BasicShape.h"

#include <osg/Geometry>

class CircleShape : public BasicShape
{
public:        

    CircleShape(std::string command = "", std::string name = "");
	virtual ~CircleShape();
    void update(osg::NodeVisitor*, osg::Drawable*);
    void update(std::string);

protected:
    CircleShape();
    void setPosition(osg::Vec3, float);
    void setColor(osg::Vec4, osg::Vec4);
    int _numFaces;
};

#endif
