#ifndef SKETCH_LINE_H
#define SKETCH_LINE_H

#include "SketchObject.h"

#include <osg/Geometry>
#include <osg/LineWidth>
#include <osg/ShapeDrawable>
#include <osg/Geode>

#ifndef WIN32
#include <sys/time.h>
#else
#include <windows.h>
#endif

class SketchLine : public SketchObject
{
    public:
        enum LineType
        {
            NONE = -1,
            SEGMENT,
            MULTI_SEGMENT,
            FREEHAND
        };

        SketchLine(LineType type, bool tube, bool snap, osg::Vec4 color, float size);
        virtual ~SketchLine();

        virtual bool buttonEvent(int type, const osg::Matrix & mat);
        virtual void addBrush(osg::MatrixTransform * mt);
        virtual void removeBrush(osg::MatrixTransform * mt);
        virtual void updateBrush(osg::MatrixTransform * mt);
        virtual void finish();
        virtual osg::Drawable * getDrawable();
        virtual void setColor(osg::Vec4 color);
        virtual void setSize(float size);

        void setTube(bool b);
        bool getTube() { return _tube; }
        void setSnap(bool b);
        bool getSnap() { return _snap; }

    protected:
        LineType _type;
        bool _tube;
        bool _snap;
        bool _drawing;

        unsigned int _count;
        osg::ref_ptr<osg::Vec3Array> _verts;
        osg::ref_ptr<osg::Vec4Array> _colors;
        osg::ref_ptr<osg::DrawArrays> _primitive;
        osg::ref_ptr<osg::Geometry> _geometry;

        osg::ref_ptr<MyComputeBounds> _mcb;
        osg::ref_ptr<osg::LineWidth> _lineWidth;

        osg::ref_ptr<osg::ShapeDrawable> _brushDrawable;
        osg::ref_ptr<osg::Geode> _brushGeode;

        struct timeval _lastPointTime;
        osg::Vec3 _lastPoint;
};

#endif
