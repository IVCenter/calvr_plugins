#ifndef SKETCH_LAYOUT_H
#define SKETCH_LAYOUT_H

#include "SketchLayout.h"

#include <osg/Geometry>
#include <osg/ShapeDrawable>
#include <osg/Geode>

#ifndef WIN32
#include <sys/time.h>
#else
#include <windows.h>
#endif

class SketchLayout: public SketchObject
{
    public:
        enum LayoutType 
        {
            NONE = -1,
            SPHERE,
            BOX,
            CYLINDER,
            CONE,
            TORUS
        };

        SketchLayout(ShapeType type, bool wireframe, osg::Vec4 color,
                    int tessellations, float size);
        virtual ~SketchLayout();

        virtual bool buttonEvent(int type, const osg::Matrix & mat);
        virtual void addBrush(osg::MatrixTransform * mt);
        virtual void removeBrush(osg::MatrixTransform * mt);
        virtual void updateBrush(osg::MatrixTransform * mt);
        virtual void finish();
        virtual osg::Drawable * getDrawable();
        virtual void setColor(osg::Vec4 color);
        virtual void setSize(float size);

    protected:
        LayoutType _type;
        bool _drawing;

        void drawBox();
        void drawCylinder();
        void drawCone();
        void drawSphere();
        void drawTorus();

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

};

#endif
