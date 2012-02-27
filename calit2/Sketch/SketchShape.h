#ifndef SKETCH_SHAPE_H
#define SKETCH_SHAPE_H

#include "SketchObject.h"

#include <osg/Geometry>
#include <osg/LineWidth>
#include <osg/ShapeDrawable>
#include <osg/Geode>
#include <osg/PositionAttitudeTransform>

#ifndef WIN32
#include <sys/time.h>
#else
#include <windows.h>
#endif

#define HL_ON_MASK  0xFFFFFF
#define HL_OFF_MASK 0x0

class SketchShape: public SketchObject
{
    public:
        enum ShapeType 
        {
            NONE = -1,
            SPHERE,
            BOX,
            CYLINDER,
            CONE,
            TORUS
        };

        SketchShape(ShapeType type, bool wireframe, osg::Vec4 color,
                    int tessellations, float size);

        virtual ~SketchShape();

        virtual void setPat(osg::PositionAttitudeTransform **pat);

        virtual bool buttonEvent(int type, const osg::Matrix & mat);
        virtual void addBrush(osg::MatrixTransform * mt);
        virtual void removeBrush(osg::MatrixTransform * mt);
        virtual void updateBrush(osg::MatrixTransform * mt);
        virtual void finish();
        virtual osg::Drawable * getDrawable();

        virtual void setColor(osg::Vec4 color);
        virtual void setSize(float size);
        virtual void setWireframe(bool b);
        virtual void setTessellations(int t);

        virtual void highlight();
        virtual void unhighlight();
        virtual bool containsPoint(const osg::Vec3 point);
        virtual osg::PositionAttitudeTransform* getPat();
        virtual void scale(osg::Vec3 scale);

        virtual void resizeTorus(float majorRad, float minorRad);

        bool getWireframe() { return _wireframe; }

    protected:
        ShapeType _type;
        bool _wireframe;
        bool _drawing;
        int _tessellations;

        osg::PositionAttitudeTransform * _pat, * _highlightPat, * _modelPat;
        osg::ref_ptr<osg::Geode> _shapeGeode;
        osg::ref_ptr<osg::Geode> _highlightGeode;
        osg::ref_ptr<osg::ShapeDrawable> _highlightDrawable;

        SketchShape * _torusHLShape;

        unsigned int _count;
        osg::ref_ptr<osg::Vec3Array> _verts;
        osg::ref_ptr<osg::Vec4Array> _colors;
        osg::ref_ptr<osg::Vec3Array> _normals;
        osg::ref_ptr<osg::DrawArrays> _primitive;
        osg::ref_ptr<osg::Geometry> _geometry;

        osg::ref_ptr<osg::ShapeDrawable> _shapeDrawable;
        osg::ref_ptr<osg::Sphere> _sphere;
        osg::ref_ptr<osg::Cone> _cone;
        osg::ref_ptr<osg::Box> _box;
        osg::ref_ptr<osg::Cylinder> _cylinder;

        osg::ref_ptr<MyComputeBounds> _mcb;

        osg::ref_ptr<osg::ShapeDrawable> _brushDrawable;
        osg::ref_ptr<osg::Geode> _brushGeode;

        struct timeval _lastPointTime;
        osg::Vec3 _lastPoint;

        void drawBox();
        void drawCylinder();
        void drawCone();
        void drawSphere();
        void drawTorus(float majorRad, float minorRad);
};

#endif
