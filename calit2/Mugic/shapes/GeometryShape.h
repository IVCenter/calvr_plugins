#ifndef _GEOMETRYSHAPE_
#define _GEOMETRYSHAPE_

#include "BasicShape.h"
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Drawable>
#include <osg/MatrixTransform>

class GeometryShape : public BasicShape, public osg::Geometry
{
    public:        

        virtual void update(std::string command) = 0;
	osg::MatrixTransform* getMatrixParent();
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
	osg::Vec2Array* _textures; //for texture coordinates
	std::string _texture_name; //path to texture

        virtual void update() = 0;
};

#endif
