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
    osg::Geode* getParent();
    osg::MatrixTransform* getMatrixParent();
    osg::Drawable* asDrawable();

    struct TextUpdateCallback : public osg::Drawable::UpdateCallback
    {
        virtual void update(osg::NodeVisitor*, osg::Drawable* drawable)
        {
            TextShape* shape = dynamic_cast<TextShape*> (drawable);
            if( shape )
                shape->update();
        }
    };


protected:
    TextShape();
    void update();
};

#endif
