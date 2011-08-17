#ifndef SKETCH_RIBBON_H
#define SKETCH_RIBBON_H

#include "SketchObject.h"

#include <osg/Geometry>
#include <osg/LineWidth>
#include <osg/ShapeDrawable>
#include <osg/Geode>

#include <sys/time.h>

class SketchRibbon : public SketchObject
{
    public:
        SketchRibbon(osg::Vec4 color, float size);
        virtual ~SketchRibbon();

        virtual bool buttonEvent(int type, const osg::Matrix & mat);
        virtual void addBrush(osg::MatrixTransform * mt);
        virtual void removeBrush(osg::MatrixTransform * mt);
        virtual void updateBrush(osg::MatrixTransform * mt);
        virtual void finish();
        virtual osg::Drawable * getDrawable();
        virtual void setColor(osg::Vec4 color);

    protected:
        bool _drawing;
        unsigned int _count;
        osg::ref_ptr<osg::Vec3Array> _verts;
        osg::ref_ptr<osg::Vec4Array> _colors;
        osg::ref_ptr<osg::Vec3Array> _normals;
        osg::ref_ptr<osg::DrawArrays> _primitive;
        osg::ref_ptr<osg::Geometry> _geometry;

        osg::ref_ptr<MyComputeBounds> _mcb;

        osg::ref_ptr<osg::ShapeDrawable> _brushDrawable;
        osg::ref_ptr<osg::Geode> _brushGeode;

        struct timeval _lastPointTime;
        osg::Vec3 _lastPoint1, _lastPoint2;
};

#endif
