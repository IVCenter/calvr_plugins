#ifndef _TEXTSHAPE_
#define _TEXTSHAPE_

#include "BasicShape.h"

#include <osg/Geometry>
#include <osgText/Text>

class TextShape : public BasicShape, public osgText::Text
{
public:        

    TextShape(std::string command = "", std::string name = "");
	virtual ~TextShape();
    void update(std::string);

protected:
    TextShape();
    virtual void drawImplementation(osg::RenderInfo &renderInfo) const;
    void update();
};

#endif
