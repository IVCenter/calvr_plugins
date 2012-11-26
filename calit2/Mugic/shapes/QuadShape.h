#ifndef _QUADSHAPE_
#define _QUADSHAPE_

#include "BasicShape.h"

#include <osg/Geometry>
#include <osg/Point>

class QuadShape : public BasicShape
{
public:        

    QuadShape(std::string command, std::string name);
    virtual ~QuadShape();
    void update(std::string);
    void update(osg::NodeVisitor*, osg::Drawable*);

protected:
    QuadShape();
    void setPosition(osg::Vec3, osg::Vec3, osg::Vec3, osg::Vec3);
    void setColor(osg::Vec4, osg::Vec4, osg::Vec4, osg::Vec4);
};

#endif
