

#ifndef LAYOUT_H
#define LAYOUT_H

#include <osg/PositionAttitudeTransform>

#include "SketchShape.h"

#include <vector>

class Layout
{
    public:
        std::vector<osg::PositionAttitudeTransform *> children;
        float R, r;
        int count;
        osg::Vec3 center;
        SketchShape * shape;
        enum LayoutType
        {
            NONE = -1,
            TORUS,
            CYLINDER
        };

        LayoutType _type;
        Layout();

        Layout(LayoutType type, float majorRadius, float minorRadius);
        virtual ~Layout();

        void setCenter(osg::Vec3 vec);
        osg::Vec3 addChild(osg::PositionAttitudeTransform * p);
        bool removeChild(osg::PositionAttitudeTransform * p);
        void setRadii(float majorRadius, float minorRadius);
        void setPat(osg::PositionAttitudeTransform * p) { _pat = p; }
        void setShape(SketchShape * sketchshape) { shape = sketchshape; }
        osg::PositionAttitudeTransform * getPat() { return _pat; }
        void hide();
        void show();
        void scale(osg::Vec3 scale);
        void scaleMajorRadius(float scale);
        void scaleMinorRadius(float scale);
        bool containsPoint(osg::Vec3 point);

   protected:
        float scaleMaj, scaleMin;

        osg::Vec3 positionChildren();
        osg::PositionAttitudeTransform * _pat;
};

#endif
