#ifndef _GEOMETRYSHAPE_
#define _GEOMETRYSHAPE_

#include "BasicShape.h"
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Drawable>

class GeometryShape : public BasicShape, public osg::Geometry
{
    public:        

        virtual void update(std::string command) = 0;
        osg::Geode* getParent();
        osg::Drawable* asDrawable();

        struct GeometryUpdateCallback : public osg::Drawable::UpdateCallback
        {
            virtual void update(osg::NodeVisitor*, osg::Drawable* drawable)
            {
                GeometryShape* shape = dynamic_cast<GeometryShape*> (drawable);
                if( shape )
                    shape->update();
            }
        };


    protected:
        GeometryShape();
	    virtual ~GeometryShape();
        osg::Vec3Array* _vertices;
        osg::Vec4Array* _colors;

        virtual void update() = 0;
};

#endif
