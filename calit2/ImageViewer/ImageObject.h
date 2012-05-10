#ifndef IMAGE_OBJECT_H
#define IMAGE_OBJECT_H

#include <cvrKernel/SceneObject.h>
#include <cvrMenu/MenuRangeValueCompact.h>

#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/Texture2D>
#include <osg/Image>

class ImageObject : public cvr::SceneObject
{
    public:
        ImageObject(std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~ImageObject();

        void loadImages(std::string fileLeft, std::string fileRight = "");

        void setWidth(float width);
        float getWidth();
        void setAspectRatio(float ratio);
        float getAspectRatio();
        void setScale(float scale);
        float getScale();

        virtual void menuCallback(cvr::MenuItem * item);

    protected:
        void setScaleMatrix();

        osg::ref_ptr<osg::MatrixTransform> _scaleMT;
        osg::ref_ptr<osg::Geode> _imageGeodeLeft;
        osg::ref_ptr<osg::Geode> _imageGeodeRight;
        osg::ref_ptr<osg::Texture2D> _imageTextureLeft;
        osg::ref_ptr<osg::Texture2D> _imageTextureRight;
        osg::ref_ptr<osg::Image> _imageImageLeft;
        osg::ref_ptr<osg::Image> _imageImageRight;

        float _imageWidth;
        float _imageHeight;

        float _width;
        float _scale;
        float _aspectRatio;

        cvr::MenuRangeValueCompact * _scaleRV;
};

#endif
