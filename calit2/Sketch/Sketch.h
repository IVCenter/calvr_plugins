#ifndef SKETCH_PLUGIN_H
#define SKETCH_PLUGIN_H

#include "SketchObject.h"
#include "SketchLine.h"
#include "ColorSelector.h"

#include <kernel/CVRPlugin.h>

#include <menu/SubMenu.h>
#include <menu/MenuTextButtonSet.h>
#include <menu/MenuRangeValue.h>
#include <menu/MenuCheckbox.h>
#include <menu/MenuButton.h>

#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osg/Geometry>

#include <string>
#include <vector>
#include <map>

class Sketch : public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:
        Sketch();
        virtual ~Sketch();
        
        static Sketch * instance();

        bool init();
        void menuCallback(cvr::MenuItem * item);
        void preFrame();
        bool buttonEvent(int type, int button, int hand, const osg::Matrix & mat);
        bool mouseButtonEvent(int type, int button, int x, int y, const osg::Matrix & mat);

        float getPointerDistance() { return _pointerDistance; }

    protected:
        static Sketch * _myPtr;

        enum DrawMode
        {
            NONE = -1,
            RIBBON,
            LINE,
            SHAPE
        };

        /*enum LineType
        {
            LINE_NONE = -1,
            SEGMENT,
            MULTI_SEGMENT,
            FREEHAND
        };*/

        enum ShapeType
        {
            SHAPE_NONE = -1,
            BOX,
            CYLINDER,
            CONE,
            SPHERE
        };

        struct MyComputeBounds : public osg::Drawable::ComputeBoundingBoxCallback
        {
            MyComputeBounds() {}
            MyComputeBounds(const MyComputeBounds & mcb, const osg::CopyOp &) {}
            virtual osg::BoundingBox computeBound(const osg::Drawable &) const;

            osg::BoundingBox _bound;
        };

        void removeMenuItems(DrawMode dm);
        void addMenuItems(DrawMode dm);
        void finishGeometry();
        void createGeometry();

        cvr::SubMenu * _sketchMenu;
        cvr::MenuTextButtonSet * _modeButtons;
        cvr::MenuRangeValue * _sizeRV;
        cvr::MenuCheckbox * _csCB;

        cvr::MenuTextButtonSet * _lineType;
        cvr::MenuCheckbox * _lineTube;
        cvr::MenuCheckbox * _lineSnap;

        cvr::MenuTextButtonSet * _shapeType;
        cvr::MenuCheckbox * _shapeWireframe;

        cvr::MenuButton * _saveButton;

        DrawMode _mode;
        SketchLine::LineType _lt;
        ShapeType _st;

        std::string _dir;
        bool _drawing;
        bool _updateLastPoint;

        float _brushScale;
        float _pointerDistance;

        osg::Matrix _lastTransform;

        osg::ref_ptr<osg::MatrixTransform> _brushRoot;
        std::vector<osg::ref_ptr<osg::Geode> > _brushes;

        osg::ref_ptr<osg::MatrixTransform> _sketchRoot;
        osg::ref_ptr<osg::Geode> _sketchGeode;

        osg::BoundingBox * _currentBound;

        osg::Vec4 _color;
        unsigned int _count;
        osg::Vec3Array * _verts;
        osg::Vec4Array * _colors;
        osg::Vec3Array * _normals;
        osg::DrawArrays * _primitive;
        osg::Geometry * _currentGeometry;

        std::map<double, osg::ref_ptr<osg::Geometry> > _geometryMap;

        SketchObject * _activeObject;
        std::vector<SketchObject*> _objectList;

        ColorSelector * _colorSelector;
};

#endif
