#include "Sketch.h"
#include "SketchLine.h"
#include "SketchRibbon.h"
#include "SketchShape.h"
#include "Layout.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/InteractionManager.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/SceneObject.h>

#include <osg/ShapeDrawable>
#include <osg/Material>
#include <osg/LightModel>
#include <osg/LineWidth>
#include <osgDB/WriteFile>
#include <osgDB/ReadFile>

#include <osg/Version>
#include <osgText/Text3D>
#include <osgText/Text>
#include <cvrUtil/Intersection.h>

#include <iostream>
#include <dirent.h>
#include <math.h>

#define MAX_GRID_SIZE 200

Sketch * Sketch::_myPtr = NULL;

CVRPLUGIN(Sketch)

using namespace cvr;
using namespace osg;
using namespace std;

Sketch::Sketch() : FileLoadCallback("obj")
{
    _myPtr = this;
}

Sketch::~Sketch()
{
}

Sketch * Sketch::instance()
{
    return _myPtr;
}

bool Sketch::init()
{
    _sketchMenu = new SubMenu("Sketch");

    PluginHelper::addRootMenuItem(_sketchMenu);

    _modeButtons = new MenuTextButtonSet(true, 450, 30, 4);
    _modeButtons->setCallback(this);
    _modeButtons->addButton("Draw");
    _modeButtons->addButton("Select");
    _modeButtons->addButton("Move");
    _modeButtons->addButton("Options");
    _sketchMenu->addItem(_modeButtons);

    _drawModeButtons = new MenuTextButtonSet(true, 450, 30, 4);
    _drawModeButtons->setCallback(this);
    _drawModeButtons->addButton("Shape");
    _drawModeButtons->addButton("Layout");
    _drawModeButtons->addButton("Ribbon");
    _drawModeButtons->addButton("Line");

    _sizeRV = new MenuRangeValue("Size",0.1,10.0,1.0);
    _sizeRV->setCallback(this);
    _sketchMenu->addItem(_sizeRV);

    _tessellationsRV = new MenuRangeValue("Tessellations", 6, 30, 12, 2);
    _tessellationsRV->setCallback(this);
    _sketchMenu->addItem(_tessellationsRV);

    _csCB = new MenuCheckbox("Color Selector",false);
    _csCB->setCallback(this);
    _sketchMenu->addItem(_csCB);

    _saveButton = new MenuButton("Save");
    _saveButton->setCallback(this);
    _sketchMenu->addItem(_saveButton);

    _loadMenu = new SubMenu("Load", "Load");
    _loadMenu->setCallback(this);
    _sketchMenu->addItem(_loadMenu);

    _clearButton = new MenuButton("Clear");
    _clearButton->setCallback(this);
    _sketchMenu->addItem(_clearButton);

    _selectAllButton = new MenuButton("Select All Shapes");
    _selectAllButton->setCallback(this);

    _clearSelectButton = new MenuButton("Clear Selection");
    _clearSelectButton->setCallback(this);

    _freezeCB = new MenuCheckbox("Freeze", false);
    _freezeCB->setCallback(this);
    _sketchMenu->addItem(_freezeCB);

    _snapToGridCB = new MenuCheckbox("Snap To Grid", false);
    _snapToGridCB->setCallback(this);
    _sketchMenu->addItem(_snapToGridCB);

    _showLayoutCB = new MenuCheckbox("Show Layouts", true);
    _showLayoutCB->setCallback(this);
    _sketchMenu->addItem(_showLayoutCB);

    _modelCB = new MenuCheckbox("Place Models", true);
    _modelCB->setCallback(this);
    _sketchMenu->addItem(_modelCB);

    _orientToViewerCB = new MenuCheckbox("Orient To Viewer", false);
    _orientToViewerCB->setCallback(this);
    _sketchMenu->addItem(_orientToViewerCB);

    _lineType = new MenuTextButtonSet(true, 400, 30, 3);
    _lineType->setCallback(this);
    _lineType->addButton("Segment");
    _lineType->addButton("Mult-Segment");
    _lineType->addButton("Freehand");

    _lineTube = new MenuCheckbox("Tube", false);
    _lineTube->setCallback(this);

    _lineSnap = new MenuCheckbox("Snap", false);
    _lineSnap->setCallback(this);

    _shapeType = new MenuTextButtonSet(true, 400, 30, 4);
    _shapeType->setCallback(this);
    _shapeType->addButton("Box");
    _shapeType->addButton("Cylinder");
    _shapeType->addButton("Cone");
    _shapeType->addButton("Sphere");

    _layoutType = new MenuTextButtonSet(true, 400, 30, 3);
    _layoutType->setCallback(this);
    _layoutType->addButton("Torus");
    _layoutType->addButton("Horizontal");
    _layoutType->addButton("Vertical");

    _layoutSizeRV = new MenuRangeValue("Layout Size", 0.1,10.0,1.0);
    _layoutSizeRV->setCallback(this);

    _shapeWireframe = new MenuCheckbox("Wireframe", true);
    _shapeWireframe->setCallback(this);
   
    _highlightLabel = new MenuText("Highlight Options");
    _transparentHLCB = new MenuCheckbox("Transparent", true);
    _transparentHLCB->setCallback(this);

    _textHLCB = new MenuCheckbox("Text Labels", true);
    _textHLCB->setCallback(this);

    _boldHLCB = new MenuCheckbox("Bolded", true);
    _boldHLCB->setCallback(this);

    _pulsatingHLCB = new MenuCheckbox("Pulsating", true);
    _pulsatingHLCB->setCallback(this);

    //_dialogPanel = new ScrollingDialogPanel(100, 20, 1, false, "Dialog Panel");
    //_dialogPanel->addText("This is text.");

    //_panel = new DialogPanel(100, "Panel");

    _mode = SELECT;
    _drawMode = NONE;
    _lt = SketchLine::NONE;
    _st = SketchShape::NONE;
    _drawing = false;

    _dataDir = ConfigManager::getEntry("Plugin.Sketch.DataDir");
    _dataDir = _dataDir + "/";

    _modelDir = ConfigManager::getEntry("Plugin.Sketch.ModelDir");
    _modelDir = _modelDir + "/";

    std::string gridSize = ConfigManager::getEntry("Plugin.Sketch.GridSize");
    _gridSize = atoi(gridSize.c_str());

    std::string orient = ConfigManager::getEntry("Plugin.Sketch.OrientToViewer");
    //_orientToViewer = orient == "on";

    if (!_gridSize || _gridSize > MAX_GRID_SIZE)
        _gridSize = 1;

    cvr::MenuButton * button;
    std::string filename;

    _pointerDistance = 1000.0;
    _sizeScale = 100;
    _modelScale = 12;
    _modelCounter = 0;

    _color = osg::Vec4(0.0,1.0,0.0,1.0);

    _sketchRoot = new osg::MatrixTransform();
    _sketchGeode = new osg::Geode();
    _sketchRoot->addChild(_sketchGeode);

    PluginHelper::getObjectsRoot()->addChild(_sketchRoot);
    _isObjectRoot = true;

    osg::StateSet * stateset = _sketchGeode->getOrCreateStateSet();
    osg::Material * mat = new osg::Material();
    stateset->setAttributeAndModes(mat,osg::StateAttribute::ON);

    _brushRoot = new osg::MatrixTransform();
    PluginHelper::getScene()->addChild(_brushRoot);

    _activeObject = NULL;

    _pat = new osg::PositionAttitudeTransform();

    _moveBrushShape = new osg::Sphere(osg::Vec3(0,0,0),10);
    _moveBrushDrawable = new osg::ShapeDrawable(_moveBrushShape);
    _moveBrushDrawable->setColor(osg::Vec4(.5,.5,.5,1));
    _moveBrushGeode = new osg::Geode();
    _moveBrushGeode->addDrawable(_moveBrushDrawable);
    
    _colorSelector = new ColorSelector(_color);
    osg::Vec3 pos = ConfigManager::getVec3("Plugin.Sketch.ColorSelector");
    _colorSelector->setPosition(pos);


    return true;
}

void Sketch::menuCallback(MenuItem * item)
{
    if(item == _modeButtons)
    {
        finishGeometry();
        removeMenuItems(_mode);
        preFrame();
        _mode = (Mode)_modeButtons->firstNumOn();

        _sizeRV->setValue(1.0);
        addMenuItems(_mode);

        if (_mode == MOVE || _mode == SELECT)
        {
            _moveBrushGeode = new osg::Geode();
            _moveBrushGeode->addDrawable(_moveBrushDrawable);
            _brushRoot->addChild(_moveBrushGeode);
        }
        else
        {
            _brushRoot->removeChild(_moveBrushGeode);
        }
    } 

    else if (item == _drawModeButtons)
    {
        finishGeometry();
        removeMenuItems(_drawMode);
        _drawMode = (DrawMode)_drawModeButtons->firstNumOn(); 
        addMenuItems(_drawMode);

        // Only draw shapes and layouts when type is picked
        if (_drawMode != SHAPE && _drawMode != LAYOUT)
        {
            createGeometry();
        }
    }

    else if (item == _shapeType)
    {
        finishGeometry();
        _st = (SketchShape::ShapeType)_shapeType->firstNumOn();
        createGeometry();
    }

    else if (item == _layoutType)
    {
        finishGeometry();
        _st = (SketchShape::ShapeType)(_layoutType->firstNumOn() + 4);
        _lot = (Layout::LayoutType)(_layoutType->firstNumOn());
        createGeometry();
    }

    else if (item == _csCB)
    {
        _colorSelector->setVisible(_csCB->getValue());
    }

    else if (item == _freezeCB)
    {
        if(_freezeCB->getValue()) // Object -> Scene
        {
            _sketchRoot->postMult(PluginHelper::getObjectToWorldTransform());
            _lastPoint = PluginHelper::getObjectToWorldTransform() * _lastPoint;

            PluginHelper::getObjectsRoot()->removeChild(_sketchRoot);
            PluginHelper::getScene()->addChild(_sketchRoot);
            _isObjectRoot = false;
        }
        else // Scene -> Object
        {
            _sketchRoot->postMult(PluginHelper::getWorldToObjectTransform());
            _lastPoint = PluginHelper::getWorldToObjectTransform() * _lastPoint;

            PluginHelper::getScene()->removeChild(_sketchRoot);
            PluginHelper::getObjectsRoot()->addChild(_sketchRoot);
            _isObjectRoot = true;
        }
    }

    else if (item == _snapToGridCB)
    {
    }

    else if (item == _orientToViewerCB)
    {
    }

    else if (item == _sizeRV)
    {
        if (_mode == DRAW)
        {
            if(_activeObject)
            {
                _activeObject->setSize(_sizeRV->getValue());
            }
            if (_highlightPat)
            {
                _highlightPat->setScale(osg::Vec3(_sizeRV->getValue(),
                                                  _sizeRV->getValue(),
                                                  _sizeRV->getValue()));
            }
        }
        else if (_mode == SELECT)
        {
            float s = _sizeRV->getValue();
            vector<PositionAttitudeTransform *>::iterator it;
            bool isLayout;

            for (it = _movingList.begin(); it != _movingList.end(); ++it)
            {
                isLayout = false;

                // layouts do not get uniformly scaled
                for (int i = 0; i < _layoutList.size(); ++i)
                {
                    if ((*it) == _layoutList[i]->getPat())
                    {
                        _layoutList[i]->scaleMajorRadius(s);
                        isLayout = true;
                    }
                }
                if (!isLayout)
                {
                    (*it)->setScale(osg::Vec3(s,s,s));
                }
            }
        }
    }

    else if (item == _selectAllButton)
    {
        _movingList.clear();
        for (int i = 0; i < _shapeList.size(); ++i)
        {
            _movingList.push_back(_shapeList[i]->getPat());
            _shapeList[i]->highlight();
        }
    }
    
    else if (item == _clearSelectButton)
    {
        _movingList.clear();
    }

    else if (item == _tessellationsRV)
    {
        float t = _tessellationsRV->getValue();
        int tes = (int)floor(t);
        SketchShape * shape;
        for (int i = 0; i < _objectList.size(); ++i)
        {
            shape = dynamic_cast<SketchShape *>(_objectList[i]);
            if (shape)
            {
                shape->setTessellations(tes);
            }
        }
    }

    else if (item == _lineType)
    {
        finishGeometry();
        _lt = (SketchLine::LineType)_lineType->firstNumOn();
        createGeometry();
    }

    else if (item == _lineTube)
    {
        SketchLine * line = dynamic_cast<SketchLine*>(_activeObject);
        if(line)
        {
            line->setTube(_lineTube->getValue());
        }
    }

    else if (item == _lineSnap)
    {
        SketchLine * line = dynamic_cast<SketchLine*>(_activeObject);
        if(line)
        {
            line->setSnap(_lineSnap->getValue());
        }
    }

    else if (item == _showLayoutCB)
    {
        for (int i = 0; i < _layoutList.size(); ++i)
        {
            if (_showLayoutCB->getValue())
            {
                _layoutList[i]->show();
            }
            else
            {
                _layoutList[i]->hide();
            }
        }
    }

    else if (item == _layoutSizeRV)
    {
        for (int i = 0; i < _layoutList.size(); ++i)
        {
            _layoutList[i]->scaleMajorRadius(_layoutSizeRV->getValue());
        }
    }

    else if (item == _shapeWireframe)
    {
        SketchShape * shape = dynamic_cast<SketchShape*>(_activeObject);
        if(shape)
        {
            shape->setWireframe(_shapeWireframe->getValue());
        }
    }


    else if (item == _transparentHLCB)
    {
        SketchShape::setTransparentHighlight(_transparentHLCB->getValue());    
    }

    else if (item == _textHLCB)
    {
         SketchShape::setTextHighlight(_textHLCB->getValue());       
    }

    else if (item == _boldHLCB)
    {
        SketchShape::setBoldHighlight(_boldHLCB->getValue());
    }

    else if (item == _pulsatingHLCB)
    {
        SketchShape::setPulsatingHighlight(_pulsatingHLCB->getValue());
    }


    else if (item == _saveButton)
    {
        std::string filename;
        size_t pos;
        string sub;
        int i, max = 0;
        
        // traverses directory searching for files like sketch-005.osg
        if (DIR *dir = opendir(_dataDir.c_str()))
        {
            while (struct dirent *entry = readdir(dir))
            {
                filename = entry->d_name;
                // sketch-005.osg
                pos = filename.rfind("-");

                // 005.osg
                sub = filename.substr(pos + 1, filename.size() - pos);

                pos = filename.rfind(".");
                // 005
                sub = sub.substr(0, filename.size() - pos + 1);

                i = atoi(sub.c_str());

                if (i != 0 && i > max)
                {
                    max = i; 
                }
            }

            char buf [10];
            sprintf(buf, "%03d", ++max);
            filename = _dataDir + "sketch-" + buf + ".osg";
            std::cout << "Saving " << filename << std::endl;
            osgDB::writeNodeFile(*_sketchRoot.get(), filename);
            closedir(dir);
        }
    }

    else if (item == _clearButton)
    {
        _sketchRoot->removeChildren(0, _sketchRoot->getNumChildren());
        for (int i = 0; i < _objectList.size(); ++i)
        {
            delete _objectList[i];
        }
        _objectList.clear();
        _shapeList.clear();
        _layoutList.clear();

        _patList.clear();
        _movingList.clear();

        _sketchGeode = new osg::Geode();
        _sketchRoot->addChild(_sketchGeode);

        finishGeometry();
        createGeometry();
    }

    else if (item == _loadMenu)
    {
        cvr::MenuButton * button;
        std::string filename;

        size_t pos;
        string sub;
        
        vector<std::string> addList;

        if (DIR *dir = opendir(_dataDir.c_str()))
        {
            while (struct dirent *entry = readdir(dir))
            {
                filename = entry->d_name; 
                if (strcmp(entry->d_name, ".") && strcmp(entry->d_name, ".."))
                {
                    pos = filename.rfind(".");
                    sub = filename.substr(pos + 1, filename.size() - pos);

                    if (!strcmp(sub.c_str(), "osg"))
                    {
                        bool found = false;
                        for (vector<std::string>::iterator it = _loadFileList.begin();
                            it != _loadFileList.end(); ++it)
                        {
                            if ((*it).compare(filename))
                            {
                                found = true;
                                break;
                            }
                        }

                        if (!found)
                        {
                            addList.push_back(filename);
                        }
                    }
                }
            }
            closedir(dir);
        }

        sort(_loadFileList.begin(), _loadFileList.end());
        sort(addList.begin(), addList.end());       

        for (int i = 0; i < addList.size(); ++i)
        {
            button = new MenuButton(addList[i]);
            button->setCallback(this);
            _loadFileButtons.push_back(button);
            _loadMenu->addItem(button);
            _loadFileList.push_back(_dataDir + addList[i]);
        }
    }

    for (int i = 0; i < _loadFileButtons.size(); ++i)
    {
        if (item == _loadMenu->getChild(i))
        {
            std::cout << "Loading " << _loadFileList[i] << std::endl;
            osg::Node * node = osgDB::readNodeFile(_loadFileList[i]);

            if (node == NULL)
            {
                std::cout << "Error loading file." << std::endl;
                return;
            }
            _sketchRoot->removeChildren(0, _sketchRoot->getNumChildren());
            _sketchRoot->addChild(node);
            
            // get all children of root that are PATs, for shape objects
            osg::MatrixTransform * mat = dynamic_cast<MatrixTransform *>(node);
            if (mat)
            {
                for (int i = 0; i < mat->getNumChildren(); ++i)
                {
                    osg::PositionAttitudeTransform * pat = 
                        dynamic_cast<PositionAttitudeTransform*>(mat->getChild(i));
                    if (pat)
                    {
                        _patList.push_back(pat);
                    }
                } 
            }
        }
    }
}

void Sketch::preFrame()
{
    if (_orientToViewerCB->getValue())
    {
        osg::Quat rot = (TrackingManager::instance()->getHeadMat(0)
            * PluginHelper::getWorldToObjectTransform()).getRotate();

        for (int i = 0; i < _shapeList.size(); ++i)
        {
            _shapeList[i]->getPat()->setAttitude(rot);
        }
    }

    if(_activeObject)
    {
        _activeObject->updateBrush(_brushRoot.get());
    } 

    osg::Matrix m;
    osg::Quat rot = TrackingManager::instance()->getHandMat(0).getRotate();
    osg::Vec3 point(0,Sketch::instance()->getPointerDistance(),0);
    point = point * TrackingManager::instance()->getHandMat(0);

    m.makeRotate(rot);
    m = m * osg::Matrix::translate(point);
    (_brushRoot.get())->setMatrix(m);

    if (_isObjectRoot)
        point = point * PluginHelper::getWorldToObjectTransform();

    SketchShape::updateHighlight();
    
    // approximate position of hand for testing
    osg::Vec3 hpoint(0,0,0);
    hpoint = hpoint * TrackingManager::instance()->getHandMat(0);
    if (_isObjectRoot)
        hpoint = hpoint * PluginHelper::getWorldToObjectTransform();
    
    // highlight layouts when placing shapes
    if (_mode == DRAW && _drawMode == SHAPE)
    {
        for (int i = 0; i < _layoutList.size(); ++i)
        {
            _layoutList[i]->getPat()->dirtyBound();
            if (_layoutList[i]->shape->containsPoint(point))
            {
                if (_showLayoutCB->getValue())
                {
                    _layoutList[i]->shape->highlight();
                }
            }
            else
            {
                _layoutList[i]->shape->unhighlight();
            }
        }
    }

    else if (_mode == MOVE)
    {
        bool isShapeHighlight = false;

        for (int i = 0; i < _shapeList.size(); ++i)
        {
            if (_shapeList[i]->containsPoint(point))
            {
                _shapeList[i]->highlight();
                isShapeHighlight = true;
            }
            else
            {
                _shapeList[i]->unhighlight();
            }
        }
        
        // do not highlight layout when point inside shape
        for (int i = 0; i < _layoutList.size(); ++i)
        {
            if (isShapeHighlight)
            {
                _layoutList[i]->shape->unhighlight();
            }
            else if (_layoutList[i]->shape->containsPoint(point))
            {
                if (_showLayoutCB->getValue())
                {
                    _layoutList[i]->shape->highlight();
                }
            }
            else
            {
                _layoutList[i]->shape->unhighlight();
            }
        }
    }
    
    if (_mode == SELECT || _mode == OPTIONS || _mode == MOVE)
    {
        bool isShapeHighlight = false;
        bool isMoving;
        for (int i = 0; i < _shapeList.size(); ++i)
        {
            isMoving = false;
            for (int j = 0; j < _movingList.size(); ++j)
            {
                if (_movingList[j] == _shapeList[i]->getPat())
                {
                    _shapeList[i]->highlight();
                    isMoving = true;
                    isShapeHighlight = true;
                }
            }
            // don't double-highlight selected things when point inside them
            if (!isMoving)
            {
                if (_shapeList[i]->containsPoint(point) || _shapeList[i]->containsPoint(hpoint))
                {
                    _shapeList[i]->highlight();
                }
                else
                {
                    _shapeList[i]->unhighlight();
                }
            }
        }

        for (int i = 0; i < _layoutList.size(); ++i)
        {
            isMoving = false;
            for (int j = 0; j < _movingList.size(); ++j)
            {
                if (_layoutList[i]->getPat() == _movingList[j])
                {
                    _layoutList[i]->shape->highlight();
                   isMoving = true;
                }
            }
            
            if (!isMoving)
            {
                //if (_layoutList[i]->shape->containsPoint(point) ||
                //    _layoutList[i]->shape->containsPoint(hpoint))
                if (_layoutList[i]->containsPoint(point) ||
                    _layoutList[i]->containsPoint(hpoint))

                {
                    if (!isShapeHighlight && _showLayoutCB->getValue())
                    {
                        _layoutList[i]->shape->highlight();
                    }
                }
                else
                {
                    _layoutList[i]->shape->unhighlight();
                }
            }
        } // for layoutList
    } // if mode == SELECT
}

bool Sketch::processEvent(InteractionEvent * event)
{
    TrackedButtonInteractionEvent * tie = event->asTrackedButtonEvent();
    if(!tie)
    {
        return false;
    }

    if(tie->getHand() == 0 && tie->getButton() == 0)
    {
        if(_csCB->getValue())
        {
            if(_colorSelector->buttonEvent(tie->getInteraction(), 
                                           tie->getTransform()))
            {
                _color = _colorSelector->getColor();
                if(_activeObject)
                {
                    _activeObject->setColor(_color);
                }
                return true;
            }
        }

        osg::Vec3 point(0,Sketch::instance()->getPointerDistance(),0);
        point = point * TrackingManager::instance()->getHandMat(0);

        osg::Vec3 distance;

        if (!_freezeCB->getValue())
        {
            point = point * PluginHelper::getWorldToObjectTransform();
        }
        else
        {

        }
        if (_snapToGridCB->getValue()) 
        {
            for (int i = 0; i < 3; ++i)
            {
                int diff = (int)floor(point[i]) % _gridSize;     
                point[i] -= diff;
            }

            distance = point - _lastPoint;

            distance[0] = (int)distance[0];
            distance[1] = (int)distance[1];
            distance[2] = (int)distance[2];
        }
        else
        {
            distance = point - _lastPoint;
        }
        _lastPoint = point;


        if (_mode == MOVE)
        {
            if (tie->getInteraction() == BUTTON_DOWN)
            {
                bool inSphere = false;
                
                for (int i = 0; i < _shapeList.size(); ++i)
                {
                    if (_shapeList[i]->containsPoint(_lastPoint))
                    {
                        _movingList.push_back(_shapeList[i]->getPat());

                        _movingLayout = false;
                        inSphere = true;
                    }
                }
                
                for (int i = 0; i < _layoutList.size(); ++i)
                {
                    if (_layoutList[i]->shape->containsPoint(_lastPoint))
                    {
                        if (!inSphere)
                        {
                            _movingLayout = true;
                            _movingList.push_back(_layoutList[i]->getPat());

                            for (int j = 0; j < _layoutList[i]->children.size(); ++j)
                            {
                                _movingList.push_back(_layoutList[i]->children[j]);
                            }
                        }
                    }
                }
                return !_movingList.empty();
            }
            else if (tie->getInteraction() == BUTTON_DRAG)
            {
                for (int i = 0; i < _movingList.size(); ++i)
                {
                    // remove child shapes that are dragged out of layouts
                    if (!_movingLayout)
                    {
                        for (int j = 0; j < _layoutList.size(); ++j)
                        {
                            _layoutList[j]->removeChild(_movingList[i]);
                             
                            if (_layoutList[j]->shape->containsPoint(_lastPoint))
                            {
                                _layoutList[j]->shape->highlight();
                                _layoutList[j]->addChild(_movingList[i]);
                            }
                        }
                    }

                    _movingList[i]->setPosition(
                        _movingList[i]->getPosition() + distance);
                    _movingList[i]->dirtyBound();                   
                }
                return !_movingList.empty();
            }
            else if (tie->getInteraction() == BUTTON_UP)
            {
                _movingList.clear();
                return false;
            }

            return true;
        }

        else if (_mode == DRAW && !_activeObject)
        {
            return false;
        }

        else if (_mode == DRAW && _activeObject)
        {
            bool ret = _activeObject->buttonEvent(tie->getInteraction(),
                                                  tie->getTransform());
            
            if (tie->getInteraction() == BUTTON_DOWN)
            {
                if (_drawMode == SHAPE)
                {
                    SketchShape * shape = dynamic_cast<SketchShape*>(_activeObject);
                    if (shape)
                    {
                        _sketchRoot->addChild(_pat);
                        _shapeList.push_back((SketchShape*)_activeObject);
                        shape->setSize(_sizeRV->getValue() * _sizeScale);
                    }

                    // add shape as child of layout if point in layout
                    for (int i = 0; i < _layoutList.size(); ++i)
                    {
                        if (_layoutList[i]->shape->containsPoint(point))
                        {
                            point = _layoutList[i]->addChild(_pat);
                            _lastPoint = point;
                            break;
                        }
                    }

                    _pat->setPosition(_lastPoint);

                    if (_modelCB->getValue())
                    {
                        _pat->addChild(getNextModel());
                    }
                    return true;
                }
            
                else if (_drawMode == LAYOUT)
                 {
                    SketchShape * shape = dynamic_cast<SketchShape*>(_activeObject);
                    if (shape)
                    {
                        Layout * lo;

                        switch (_st)
                        {
                            case 4:
                                lo = new Layout(_lot,
                                _layoutSizeRV->getValue() * _sizeScale * 1.5,
                                _layoutSizeRV->getValue() * _sizeScale * 0.5);
                                break;
                            
                            case 5:
                                lo = new Layout(_lot,
                                _layoutSizeRV->getValue() * _sizeScale * 0.5, 
                                _layoutSizeRV->getValue() * 1.5 * _sizeScale * 4);
                                break;
                           
                            case 6:
                                lo = new Layout(_lot,
                                _layoutSizeRV->getValue() * _sizeScale * 0.5, 
                                _layoutSizeRV->getValue() * 1.5 * _sizeScale * 4);
                            break;
                        }

                        lo->setPat(_layoutPat);
                        lo->setCenter(_lastPoint);
                        lo->setShape(shape);

                        _layoutList.push_back(lo);

                        _sketchRoot->addChild(_layoutPat);
                        _layoutPat->setPosition(_lastPoint);
                    }
                    return true;
                 }
                 
                if(_activeObject->isDone())
                {
                    finishGeometry();
                    createGeometry();
                }
            }
            else if (tie->getInteraction() == BUTTON_DRAG)
            {
                if(_drawMode == SHAPE)
                {
                    bool inLayout = false;

                    for (int i = 0; i < _layoutList.size(); ++i)
                    {
                        if (_layoutList[i]->shape->containsPoint(point))
                        {
                            inLayout = true;
                         //  point = _layoutList[i]->addChild(_pat);
                            break;
                        }
                        else
                        {
                            _layoutList[i]->removeChild(_pat);
                        }
                    }
                    if (!inLayout)
                    {
                        _pat->setPosition(_pat->getPosition() + distance);
                        _modelpat->setPosition(_modelpat->getPosition() + distance);
                    }
                }

                if (_drawMode == LAYOUT)
                {
                    _layoutPat->setPosition(_layoutPat->getPosition() + distance);

                    _layoutList[_layoutList.size() - 1]->setCenter(
                        _layoutPat->getPosition() + distance);
                }
            }
            else if (tie->getInteraction() == BUTTON_UP)
            {
                finishGeometry();
                createGeometry();
            }

            return ret;
        }

        else if (_mode == SELECT || _mode == OPTIONS)
        {
            if (tie->getInteraction() == BUTTON_DOWN)
            {
                bool inNone = true;
                bool alreadyIn = false; 
                bool inSphere = false;

                for (int i = 0; i < _shapeList.size(); ++i)
                {
                    if (_shapeList[i]->containsPoint(_lastPoint))
                    {
                        vector<osg::PositionAttitudeTransform*>::iterator it;
                        for (it = _movingList.begin(); it != _movingList.end(); ++it)
                        {
                            if (*it == _shapeList[i]->getPat())
                            {
                                _movingList.erase(it);
                                alreadyIn = true;
                                break;
                            }
                        }
                        if (!alreadyIn)
                        {
                            inSphere = true;
                            _movingList.push_back(_shapeList[i]->getPat());
                        }
                        inNone = false;
                    }
                }
                
                for (int i = 0; i < _layoutList.size(); ++i)
                {
                    if (_layoutList[i]->shape->containsPoint(_lastPoint))
                    {
                        if (!inSphere)
                        {
                            inNone = false;
                            _movingList.push_back(_layoutList[i]->getPat());
                        }
                    }
                }
                if (inNone)
                {
                    _movingList.clear();
                    return false;
                }
            }
            else if (tie->getInteraction() == BUTTON_DRAG)
            {
                return false;
            }
            else if (tie->getInteraction() == BUTTON_UP)
            {
                return false;
            }
        }

        return true;
    } 

    return false;
}

osg::BoundingBox Sketch::MyComputeBounds::computeBound(const osg::Drawable &) const
{
    return _bound;
}

void Sketch::removeMenuItems(DrawMode dm)
{
    switch(dm)
    {
	case RIBBON:
	    break;
	case LINE:
	    _sketchMenu->removeItem(_lineType);
	    _sketchMenu->removeItem(_lineTube);
	    _sketchMenu->removeItem(_lineSnap);
	    break;
	case SHAPE:
	    _sketchMenu->removeItem(_shapeType);
	    _sketchMenu->removeItem(_shapeWireframe);
	    break;
    case LAYOUT:
        _sketchMenu->removeItem(_layoutType);
        _sketchMenu->removeItem(_layoutSizeRV);
	default:
	    break;
    }
}

void Sketch::addMenuItems(DrawMode dm)
{
    switch(dm)
    {
	case RIBBON:
	    break;
	case LINE:
	    _lt = SketchLine::NONE;
	    if(_lineType->firstNumOn() >= 0)
	    {
            _lineType->setValue(_lineType->firstNumOn(), false);
	    }
	    _lineTube->setValue(false);
	    _lineSnap->setValue(false);
	    _updateLastPoint = false;
	    _sketchMenu->addItem(_lineType);
	    _sketchMenu->addItem(_lineTube);
	    _sketchMenu->addItem(_lineSnap);
	    break;
	case SHAPE:
	    _st = SketchShape::NONE;
	    if(_shapeType->firstNumOn() >= 0)
	    {
            _shapeType->setValue(_shapeType->firstNumOn(), false);
	    }
	    _shapeWireframe->setValue(true);
	    _sketchMenu->addItem(_shapeType);
	    _sketchMenu->addItem(_shapeWireframe);
	    break;
     case LAYOUT:
        _st = SketchShape::NONE;
        if(_layoutType->firstNumOn() >= 0)
        {
            _layoutType->setValue(_layoutType->firstNumOn(), false);
        }
        _sketchMenu->addItem(_layoutType);
        _sketchMenu->addItem(_layoutSizeRV);
	default:
	    break;
    }
}

void Sketch::removeMenuItems(Mode dm)
{
    switch(dm)
    {
	case DRAW:
	    _sketchMenu->removeItem(_drawModeButtons);
        removeMenuItems(_drawMode);
	    break;
	case SELECT:
        _sketchMenu->removeItem(_selectAllButton);
        _sketchMenu->removeItem(_clearSelectButton);
	    break;
    case MOVE:
        break;
    case OPTIONS:
        _sketchMenu->removeItem(_highlightLabel);
        _sketchMenu->removeItem(_transparentHLCB);
        _sketchMenu->removeItem(_textHLCB);
        _sketchMenu->removeItem(_boldHLCB);
        _sketchMenu->removeItem(_pulsatingHLCB);       
        break;
	default:
	    break;
    }
}

void Sketch::addMenuItems(Mode dm)
{
    switch(dm)
    {
	case DRAW:
	    _sketchMenu->addItem(_drawModeButtons);
	    if(_drawModeButtons->firstNumOn() >= 0)
	    {
            _drawModeButtons->setValue(_drawModeButtons->firstNumOn(), false);
	    }

        for (int i = 0; i < _shapeList.size(); ++i)
        {
            for (int j = 0; j < _movingList.size(); ++j)
            {
                if (_movingList[j] = _shapeList[i]->getPat())
                {
                    _shapeList[i]->unhighlight();
                }
            }
        }

        for (int i = 0; i < _layoutList.size(); ++i)
        {
            for (int j = 0; j < _movingList.size(); ++j)
            {
                if (_movingList[j] = _layoutList[i]->shape->getPat())
                {
                    _layoutList[i]->shape->unhighlight();
                }
            }
        } 

        _movingList.clear();
	    break;
	case MOVE:
	    break;
    case SELECT:
        _sketchMenu->addItem(_selectAllButton);
        _sketchMenu->addItem(_clearSelectButton);
        break;
    case OPTIONS:
        _sketchMenu->addItem(_highlightLabel);
        _sketchMenu->addItem(_transparentHLCB);
        _sketchMenu->addItem(_textHLCB);
        _sketchMenu->addItem(_boldHLCB);
        _sketchMenu->addItem(_pulsatingHLCB);
        break;
	default:
	    break;
    }
}

void Sketch::finishGeometry()
{
    if(!_activeObject)
    {
        return;
    }
    if(!_activeObject->isDone())
    {
        _activeObject->finish();
    }

    _activeObject->removeBrush(_brushRoot.get());

    if(_activeObject->isValid())
    {
        _objectList.push_back(_activeObject);
    }
    else
    {
        _sketchGeode->removeDrawable(_activeObject->getDrawable());
        delete _activeObject;
    }
    _activeObject = NULL;
}

void Sketch::createGeometry()
{
    if(_activeObject)
    {
        finishGeometry();
    }

    float size = _sizeScale * _sizeRV->getValue(),
        layoutSize = _sizeScale * _layoutSizeRV->getValue();

    float highlightScale = .95;

    osg::Vec3 cylinderScaleVec(1,1,4);
    
    SketchShape * p;

    switch(_drawMode)
    {
	case RIBBON:
	    _activeObject = new SketchRibbon(_color, size/_sizeScale);
	    break;

	case LINE:
	    _activeObject = new SketchLine(_lt, _lineTube->getValue(), 
            _lineSnap->getValue(), _color, _sizeRV->getValue());
	    break;

	case SHAPE:
        // do not create object if no shape type selected
        if (_st == (SketchShape::ShapeType)NONE)
        {
            _activeObject = NULL;
            break;
        }

        _pat = new osg::PositionAttitudeTransform();
        
	    p = new SketchShape(_st, _shapeWireframe->getValue(), _color, 
                            (int) _tessellationsRV->getValue(), size);
        p->setPat(&_pat);

//        p->setFont(_dataDir + "arial.ttf");

        _activeObject = p;

        _modelpat = new osg::PositionAttitudeTransform();
        _modelpatScale = new osg::PositionAttitudeTransform();

	    break;

   case LAYOUT:
        if (_st < 4 || _st > 6 )
        {
            _activeObject = NULL;
            break;
        }

        _layoutPat = new osg::PositionAttitudeTransform();
        
	    p = new SketchShape(_st, _shapeWireframe->getValue(), osg::Vec4(1,1,1,1), 
                            (int) _tessellationsRV->getValue(), layoutSize);
        p->setPat(&_layoutPat);
        _activeObject = p;


        switch (_st)
        {
            case 5: // HORIZONTAL
                _layoutPat->setScale(cylinderScaleVec);
                _layoutPat->setAttitude(osg::Quat(M_PI/2, osg::Vec3(0,1,0)));
                break;

            case 6: // VERTICAL
                _layoutPat->setScale(cylinderScaleVec);
                break;

            default:
                break;
        }
        break;

	default:
	    break;
    }

    if(_activeObject)
    {
        if (_drawMode == SHAPE || _drawMode == LAYOUT)    
        {
        }
        else 
        {
            _sketchGeode->addDrawable(_activeObject->getDrawable());
        }
        _activeObject->addBrush(_brushRoot.get());
    }
}

bool Sketch::loadFile(std::string file)
{
    osg::Node * node = osgDB::readNodeFile(file);
    if (node)
    {
        _sketchRoot->addChild(node);
        return true;
    }
    else
    {
        return false;
    }
}

osg::Vec3 Sketch::getCurrentPoint()
{
    osg::Vec3 pos(0,Sketch::instance()->getPointerDistance(),0);
    pos = pos * TrackingManager::instance()->getHandMat(0);

    if (_isObjectRoot)
    {
        pos = pos * PluginHelper::getWorldToObjectTransform();
    }
    
    return pos;
}

osg::PositionAttitudeTransform * Sketch::getNextModel()
{
    _modelpat->setPosition(_lastPoint);
    _modelpatScale->setPosition(osg::Vec3(0,0, - _sizeRV->getValue() * 10));
    
    // position adjusted forward in cylinder due to the way cylinders are tessellated
    if (_st == 1)
    {
        _modelpatScale->setPosition(_modelpatScale->getPosition() +
            osg::Vec3(0, -_sizeRV->getValue() * _sizeScale / 2, 0));
    }

    
    int numIcons = 10;
    osgText::Text3D * text;
    osg::Geode * geode;
    osg::StateSet * stateset;
    osg::Material * material;

    switch (_modelCounter % numIcons)
    {
    case 0:
        _model = osgDB::readNodeFile(_modelDir + "fileIcon.obj");

        _modelpatScale->setScale(osg::Vec3(_sizeRV->getValue() * _modelScale,
                                           _sizeRV->getValue() * _modelScale, 
                                           _sizeRV->getValue() * _modelScale));
       break;
    
    case 1:
        _model = osgDB::readNodeFile(_modelDir + "bicycleIcon.obj");

        _modelpatScale->setScale(osg::Vec3(_sizeRV->getValue() * 11,
                                           _sizeRV->getValue() * 11, 
                                           _sizeRV->getValue() * 11));

        _modelpatScale->setAttitude(osg::Quat(M_PI/2, Vec3(0,0,1)));
        break;
    case 2:
        _model = osgDB::readNodeFile(_modelDir + "handIcon.obj");

        _modelpatScale->setScale(osg::Vec3(_sizeRV->getValue() * 16,
                                           _sizeRV->getValue() * 16, 
                                           _sizeRV->getValue() * 16));

        _modelpatScale->setPosition(_modelpatScale->getPosition() -
             osg::Vec3(-_sizeRV->getValue() * 6, 0, _sizeRV->getValue() * 20));
        break;
    case 3:
        _model = osgDB::readNodeFile(_modelDir + "magnifyingIcon.obj");

        _modelpatScale->setScale(osg::Vec3(_sizeRV->getValue() * 15,
                                           _sizeRV->getValue() * 15,
                                           _sizeRV->getValue() * 15));

        _modelpatScale->setPosition(_modelpatScale->getPosition() +
             osg::Vec3(-_sizeRV->getValue() * 10, 0, _sizeRV->getValue() * 25));

        _modelpatScale->setAttitude(osg::Quat(-M_PI/5, Vec3(0,1,0)));
        break;
    case 4: 
        _model = osgDB::readNodeFile(_modelDir + "birdIcon.obj");

        _modelpatScale->setScale(osg::Vec3(_sizeRV->getValue() * 8,
                                           _sizeRV->getValue() * 8, 
                                           _sizeRV->getValue() * 8));

        _modelpatScale->setPosition(_modelpatScale->getPosition() -
             osg::Vec3(-_sizeRV->getValue() * 5, 0, _sizeRV->getValue() * 15));

        _modelpatScale->setAttitude(osg::Quat(M_PI/3, Vec3(0,0,1)));
        break;
    case 5:
        _model = osgDB::readNodeFile(_modelDir + "carIcon.obj");

        _modelpatScale->setScale(osg::Vec3(_sizeRV->getValue() * 12,
                                           _sizeRV->getValue() * 12, 
                                           _sizeRV->getValue() * 12));

        _modelpatScale->setPosition(_modelpatScale->getPosition() -
             osg::Vec3(-_sizeRV->getValue() * 5, 0, _sizeRV->getValue() * 0));

        break;
    case 6:
        _model = osgDB::readNodeFile(_modelDir + "planeIcon.obj");

        _modelpatScale->setScale(osg::Vec3(_sizeRV->getValue() * 9,
                                           _sizeRV->getValue() * 9, 
                                           _sizeRV->getValue() * 9));

        _modelpatScale->setPosition(_modelpatScale->getPosition() -
             osg::Vec3(_sizeRV->getValue() * 5, _sizeRV->getValue() * 5, _sizeRV->getValue() * 5));

        _modelpatScale->setAttitude(osg::Quat(M_PI/3,  Vec3(0,0,1), 
                                              M_PI/8,  Vec3(1,0,0), 
                                              0,       Vec3(0,1,0)));
        break;
    case 7:
        text = new osgText::Text3D();
        geode = new osg::Geode();
        text->setFont(_modelDir + "arial.ttf");
        text->setText("Drive");
        text->setCharacterSize(35);
        text->setCharacterDepth(5);
        text->setDrawMode(osgText::Text3D::TEXT);
        text->setAxisAlignment(osgText::Text3D::XZ_PLANE);
#if OPENSCENEGRAPH_MAJOR_VERSION >= 3
        text->setColor(osg::Vec4(1,1,1,1));
#endif
        geode->addDrawable(text);

        stateset = text->getOrCreateStateSet();
        stateset->setRenderingHint(osg::StateSet::OPAQUE_BIN);

        material = new osg::Material();
        material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(1,0,0,1));
        material->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(1,0,0,1));
        //stateset->setAttributeAndModes(material, osg::StateAttribute::ON);

        stateset->setMode(GL_LIGHTING, osg::StateAttribute::ON);

        _modelpatScale->setPosition(_modelpatScale->getPosition() -
             osg::Vec3(40, 10, 0));

        _model = geode;
        break;
    case 8:
        text = new osgText::Text3D();
        geode = new osg::Geode();
        text->setFont(_modelDir + "arial.ttf");
        text->setText("Fly");
        text->setCharacterSize(35);
        text->setCharacterDepth(15);
        text->setDrawMode(osgText::Text3D::TEXT);
        text->setAxisAlignment(osgText::Text3D::XZ_PLANE);
#if OPENSCENEGRAPH_MAJOR_VERSION >= 3
        text->setColor(osg::Vec4(1,1,1,1));
#endif
        geode->addDrawable(text);

        stateset = text->getOrCreateStateSet();
        stateset->setRenderingHint(osg::StateSet::OPAQUE_BIN);

        _modelpatScale->setPosition(_modelpatScale->getPosition() - 
            osg::Vec3(25, 10,0));

        _model = geode;
        break;
    case 9:
        text = new osgText::Text3D();
        geode = new osg::Geode();
        text->setFont(_modelDir + "arial.ttf");
        text->setText("Scale");
        text->setCharacterSize(35);
        text->setCharacterDepth(20);
        text->setDrawMode(osgText::Text3D::TEXT);
        text->setAxisAlignment(osgText::Text3D::XZ_PLANE);
#if OPENSCENEGRAPH_MAJOR_VERSION >= 3
        text->setColor(osg::Vec4(1,1,1,1));
#endif
        geode->addDrawable(text);

        stateset = text->getOrCreateStateSet();
        stateset->setRenderingHint(osg::StateSet::OPAQUE_BIN);

        material = new osg::Material();
        material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(1,1,0,1));
        material->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0,0,0,1));
        //stateset->setAttributeAndModes(material, osg::StateAttribute::ON);
        
        stateset->setMode(GL_LIGHTING, osg::StateAttribute::ON);

        _modelpatScale->setPosition(_modelpatScale->getPosition() -
             osg::Vec3(43, 10,0));//_sizeRV->getValue() * 5, _sizeRV->getValue() * 5));

        _model = geode;
        break;
    }

    _modelCounter++;
    _modelpatScale->addChild(_model);
    return _modelpatScale; 
}
