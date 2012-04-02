#ifndef SKETCH_PLUGIN_H
#define SKETCH_PLUGIN_H

#include "SketchObject.h"
#include "SketchLine.h"
#include "ColorSelector.h"

#include "SketchShape.h"
#include "Layout.h"

#include <kernel/CVRPlugin.h>
#include <kernel/FileHandler.h>

#include <menu/SubMenu.h>
#include <menu/MenuTextButtonSet.h>
#include <menu/MenuRangeValue.h>
#include <menu/MenuCheckbox.h>
#include <menu/MenuButton.h>
#include <menu/ScrollingDialogPanel.h>
#include <menu/MenuText.h>

#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osg/Geometry>
#include <osg/PositionAttitudeTransform>

#include <string.h>
#include <vector>
#include <map>

class Sketch : public cvr::CVRPlugin, public cvr::MenuCallback, public cvr::FileLoadCallback
{
    public:
        Sketch();
        virtual ~Sketch();
        static Sketch * instance();
        bool init();
        void menuCallback(cvr::MenuItem * item);
        void preFrame();
        bool processEvent(cvr::InteractionEvent * event);
        float getPointerDistance() { return _pointerDistance; }
        virtual bool loadFile(std::string file);

    protected:
        static Sketch * _myPtr;

        enum DrawMode
        {
            NONE = -1,
            SHAPE,
            LAYOUT,
            RIBBON,
            LINE
        };

        enum Mode
        {
            DRAW,
            SELECT,
            MOVE,
            OPTIONS
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

        void removeMenuItems(Mode dm);
        void addMenuItems(Mode dm);

        osg::PositionAttitudeTransform * getNextModel();

        osg::Vec3 getCurrentPoint();

        cvr::SubMenu * _sketchMenu;
        cvr::MenuTextButtonSet * _modeButtons;
        cvr::MenuRangeValue * _sizeRV;
        cvr::MenuCheckbox * _csCB;

        cvr::MenuTextButtonSet * _drawModeButtons;
        
        cvr::MenuRangeValue * _sizeAllRV;
        cvr::MenuRangeValue * _tessellationsRV;

        cvr::MenuCheckbox * _freezeCB;
        cvr::MenuCheckbox * _modelCB;

        cvr::MenuTextButtonSet * _lineType;
        cvr::MenuCheckbox * _lineTube;
        cvr::MenuCheckbox * _lineSnap;

        cvr::MenuTextButtonSet * _shapeType;
        cvr::MenuCheckbox * _shapeWireframe;

        cvr::MenuTextButtonSet * _layoutType;
        cvr::MenuCheckbox * _showLayoutCB;
        cvr::MenuRangeValue * _layoutSizeRV;

        cvr::MenuButton * _saveButton;
        cvr::MenuButton * _clearButton;

        cvr::MenuButton * _selectAllButton;
        cvr::MenuButton * _clearSelectButton;

        cvr::MenuText * _highlightLabel;
        cvr::MenuCheckbox * _transparentHLCB;
        cvr::MenuCheckbox * _textHLCB;
        cvr::MenuCheckbox * _boldHLCB;
        cvr::MenuCheckbox * _pulsatingHLCB;

        cvr::ScrollingDialogPanel * _dialogPanel;

        cvr::SubMenu * _loadMenu;
        std::vector<cvr::MenuButton*> _loadFileButtons;
        std::vector<std::string> _loadFileList;
        std::string _dataDir;
        std::string _modelDir;
        
        int _gridSize;
        cvr::MenuCheckbox * _snapToGridCB;
        
        Mode _mode;
        DrawMode _drawMode;

        SketchLine::LineType _lt;
        SketchShape::ShapeType _st;
        Layout::LayoutType _lot;

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
        std::vector<SketchShape *> _shapeList;
        std::vector<Layout *> _layoutList;
        std::vector<osg::PositionAttitudeTransform *> _movingList;
        std::vector<osg::PositionAttitudeTransform *> _patList;

        ColorSelector * _colorSelector;

        osg::PositionAttitudeTransform * _pat;
        osg::PositionAttitudeTransform * _modelpat;
        osg::PositionAttitudeTransform * _modelpatScale;
        osg::Node * _model;

        osg::Geode * _shapeGeode;
        osg::Vec3 _lastPoint;
        osg::ShapeDrawable * _highlightDrawable;
        osg::Geode * _highlightGeode;
        osg::PositionAttitudeTransform * _highlightPat;

        osg::Shape * _moveBrushShape;
        osg::ShapeDrawable * _moveBrushDrawable;
        osg::Geode * _moveBrushGeode;

        osg::PositionAttitudeTransform * _layoutPat;
        osg::Geode * _layoutGeode;

        bool _isObjectRoot;
        bool _movingLayout;
        bool _isIntersecting;
        bool _orientToViewer;

        int _sizeScale;
        float _modelScale;

        int _modelCounter;

};


#endif
