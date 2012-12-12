#ifndef COLOR_SELECTOR_H
#define COLOR_SELECTOR_H

#include <osg/Vec4>
#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/ShapeDrawable>

class ColorSelector
{
    public:
        ColorSelector(osg::Vec4 color, osg::Vec3 pos = osg::Vec3(0,0,0), float scale = 1.0);
        virtual ~ColorSelector();

        bool buttonEvent(int type, const osg::Matrix & mat);

        void setColor(osg::Vec4 color);
        osg::Vec4 getColor();

        void setVisible(bool v);
        bool isVisible();
        void setPosition(osg::Vec3 pos);
        osg::Vec3 getPosition();
        void setScale(float scale);
        float getScale();

    protected:
        void setMatrix();
        osg::Vec3 hsl2rgb(osg::Vec3 hsl, bool undefh = false);
        osg::Vec3 hcl2rgb(osg::Vec3 hcl, bool undefh = false);
        osg::Vec3 rgb2hcl(osg::Vec4 color);
        osg::Vec3 hcl2xyz(osg::Vec3 hcl);
        osg::Vec3 xyz2hcl(osg::Vec3 xyz, bool & undefh);
        void createGeometry();

        osg::ref_ptr<osg::MatrixTransform> _root;
        osg::Geode * _mainGeode;
        osg::MatrixTransform * _sphereTransform;
        osg::Geode * _sphereGeode;
        osg::ShapeDrawable * _sphereDrawable;

        bool _visible;
        float _scale;
        float _sphereRad;
        osg::Vec3 _position;
        osg::Vec4 _color;

        bool _moving;
        //osg::Matrix _lastPointerInv;
        osg::Vec3 _pointerSpaceCenter;
};

#endif
