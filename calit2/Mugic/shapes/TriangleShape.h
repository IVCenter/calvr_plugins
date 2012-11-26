#ifndef _TRIANGLESHAPE_
#define _TRIANGLESHAPE_

#include "BasicShape.h"

#include <osg/Geometry>
#include <osg/Point>

class TriangleShape : public BasicShape
{
public:        

    TriangleShape(std::string command = "", std::string name ="");
	virtual ~TriangleShape();
    void update(osg::NodeVisitor*, osg::Drawable*);
    void update(std::string);

protected:
    TriangleShape();
    void setPosition(osg::Vec3, osg::Vec3, osg::Vec3);
    void setColor(osg::Vec4, osg::Vec4, osg::Vec4);
};
#endif
