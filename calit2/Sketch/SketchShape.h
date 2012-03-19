#ifndef SKETCH_SHAPE_H
#define SKETCH_SHAPE_H

#include "SketchObject.h"

#include <osg/Geometry>
#include <osg/LineWidth>
#include <osg/ShapeDrawable>
#include <osg/Geode>
#include <osg/PositionAttitudeTransform>
#include <osgText/Text>
#include <osgText/Text3D>

#include <string.h>

#ifndef WIN32
#include <sys/time.h>
#else
#include <windows.h>
#endif

#define HL_ON_MASK  0xFFFFFF /* 0x0 off, 0xFFFFFF on */
#define HL_OFF_MASK 0x0

#define TEXT_ON_MASK  0xFFFFFF
#define TEXT_OFF_MASK 0x0

#define HL_BOLD     2   /* 1 off, >1 on */
#define HL_UNBOLD   1

#define HL_STEP     0.003 /* 0.0 for off, >0.0 on (.003 works well) */
#define HL_MAX_DIFF 0.03  /* max scale to expand/contract to */

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

        void setPat(osg::PositionAttitudeTransform **pat);
        bool buttonEvent(int type, const osg::Matrix & mat);
        void addBrush(osg::MatrixTransform * mt);
        void removeBrush(osg::MatrixTransform * mt);
        void updateBrush(osg::MatrixTransform * mt);
        void finish();
        osg::Drawable * getDrawable();

        void setColor(osg::Vec4 color);
        void setSize(float size);
        void setWireframe(bool b);
        void setTessellations(int t);

        void highlight();
        void unhighlight();
        bool containsPoint(const osg::Vec3 point);
        osg::PositionAttitudeTransform* getPat();
        void scale(osg::Vec3 scale);
        bool getWireframe() { return _wireframe; }
        void resizeTorus(float majorRad, float minorRad);
        void setFont(std::string font);

        static void updateHighlight();

    protected:
        ShapeType _type;
        bool _wireframe;
        bool _drawing;
        int _tessellations;

        static bool _growing;
        static float _scale;

        osg::PositionAttitudeTransform * _pat, * _highlightPat, * _modelPat, * _shapePat;
        osg::ref_ptr<osg::Geode> _shapeGeode;
        osg::ref_ptr<osg::Geode> _highlightGeode;
        osg::ref_ptr<osg::ShapeDrawable> _highlightDrawable;

        osg::PositionAttitudeTransform * _textPat; 
        osgText::Text3D * text;

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

        void pulsate();
        void drawBox();
        void drawCylinder();
        void drawCone();
        void drawSphere();
        void drawTorus(float majorRad, float minorRad);
};
#endif
