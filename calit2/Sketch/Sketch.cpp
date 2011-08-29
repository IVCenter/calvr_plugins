#include "Sketch.h"
#include "SketchLine.h"
#include "SketchRibbon.h"

#include <config/ConfigManager.h>
#include <kernel/PluginHelper.h>
#include <kernel/InteractionManager.h>

#include <osg/ShapeDrawable>
#include <osg/Material>
#include <osg/LightModel>
#include <osg/LineWidth>
#include <osgDB/WriteFile>

#include <iostream>

Sketch * Sketch::_myPtr = NULL;

CVRPLUGIN(Sketch)

using namespace cvr;
using namespace osg;

Sketch::Sketch()
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

    _modeButtons = new MenuTextButtonSet(true, 400, 35, 3);
    _modeButtons->setCallback(this);
    _modeButtons->addButton("Ribbon");
    _modeButtons->addButton("Line");
    _modeButtons->addButton("Shape");
    _sketchMenu->addItem(_modeButtons);

    _csCB = new MenuCheckbox("Color Selector",false);
    _csCB->setCallback(this);
    _sketchMenu->addItem(_csCB);

    _sizeRV = new MenuRangeValue("Size",0.1,10.0,1.0);
    _sizeRV->setCallback(this);
    _sketchMenu->addItem(_sizeRV);

    _saveButton = new MenuButton("Save");
    _saveButton->setCallback(this);
    _sketchMenu->addItem(_saveButton);

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

    _shapeWireframe = new MenuCheckbox("Wireframe", false);
    _shapeWireframe->setCallback(this);

    _mode = NONE;
    _lt = SketchLine::NONE;
    _st = SHAPE_NONE;
    _drawing = false;

    _pointerDistance = 1000.0;

    _color = osg::Vec4(1.0,1.0,1.0,1.0);

    _sketchRoot = new osg::MatrixTransform();
    _sketchGeode = new osg::Geode();
    _sketchRoot->addChild(_sketchGeode);

    PluginHelper::getObjectsRoot()->addChild(_sketchRoot);

    osg::StateSet * stateset = _sketchGeode->getOrCreateStateSet();
    //stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
    //osg::LightModel * lm = new osg::LightModel();
    //lm->setTwoSided(true);
    //stateset->setAttributeAndModes(lm,osg::StateAttribute::ON);
    osg::Material * mat = new osg::Material();
    stateset->setAttributeAndModes(mat,osg::StateAttribute::ON);

    _brushRoot = new osg::MatrixTransform();
    PluginHelper::getScene()->addChild(_brushRoot);

    _activeObject = NULL;

    /*osg::Shape * shape;
    osg::ShapeDrawable * sd;
    osg::Geode * geode;

    osg::Quat rot;

    shape = new osg::Cylinder(osg::Vec3(0,0,0),5,100);
    rot.makeRotate(M_PI/ 2.0, osg::Vec3(0,1.0,0));
    ((osg::Cylinder*)shape)->setRotation(rot);

    sd = new osg::ShapeDrawable(shape);
    geode = new osg::Geode();
    geode->addDrawable(sd);
    _brushes.push_back(geode);

    shape = new osg::Sphere(osg::Vec3(0,0,0),5);
    sd = new osg::ShapeDrawable(shape);
    geode = new osg::Geode();
    geode->addDrawable(sd);
    _brushes.push_back(geode);

    _brushes.push_back(geode);*/

    _colorSelector = new ColorSelector(_color);
    //_colorSelector->setVisible(true);
    osg::Vec3 pos = ConfigManager::getVec3("Plugin.Sketch.ColorSelector");
    _colorSelector->setPosition(pos);


    return true;
}

void Sketch::menuCallback(MenuItem * item)
{
    /*if(item == _modeButtons)
    {
	finishGeometry();
	removeMenuItems(_mode);
	_mode = (DrawMode)_modeButtons->firstNumOn();
	_brushRoot->removeChildren(0,_brushRoot->getNumChildren());
	if(_mode >= 0)
	{
	    _brushRoot->addChild(_brushes[_mode]);
	}

	_sizeRV->setValue(1.0);
	addMenuItems(_mode);
    }
    else if(item == _sizeRV)
    {
    }
    else if(item == _lineType)
    {
	finishGeometry();
	_lt = (LineType)_lineType->firstNumOn();
    }
    else if(item == _shapeType)
    {
	finishGeometry();
	_st = (ShapeType)_shapeType->firstNumOn();
    }*/

    if(item == _modeButtons)
    {
	finishGeometry();
	removeMenuItems(_mode);
	_mode = (DrawMode)_modeButtons->firstNumOn();
	//_brushRoot->removeChildren(0,_brushRoot->getNumChildren());
	//if(_mode >= 0)
	//{
	//    _brushRoot->addChild(_brushes[_mode]);
	//}

	_sizeRV->setValue(1.0);
	addMenuItems(_mode);
	createGeometry();
    }
    else if(item == _sizeRV)
    {
	if(_activeObject)
	{
	    _activeObject->setSize(_sizeRV->getValue());
	}
    }
    else if(item == _lineType)
    {
	finishGeometry();
	_lt = (SketchLine::LineType)_lineType->firstNumOn();
	createGeometry();
    }
    else if(item == _lineTube)
    {
	SketchLine * line = dynamic_cast<SketchLine*>(_activeObject);
	if(line)
	{
	    line->setTube(_lineTube->getValue());
	}
    }
    else if(item == _lineSnap)
    {
	SketchLine * line = dynamic_cast<SketchLine*>(_activeObject);
	if(line)
	{
	    line->setSnap(_lineSnap->getValue());
	}
    }
    else if(item == _shapeType)
    {
	finishGeometry();
	_st = (ShapeType)_shapeType->firstNumOn();
	createGeometry();
    }
    else if(item == _saveButton)
    {
	osgDB::writeNodeFile(*_sketchRoot.get(), "/home/aprudhom/ribbontest.obj");
    }
    else if(item == _csCB)
    {
	_colorSelector->setVisible(_csCB->getValue());
    }
}

void Sketch::preFrame()
{
    if(_activeObject)
    {
	_activeObject->updateBrush(_brushRoot.get());
    }
    /*if(_mode >= 0)
    {
	osg::Matrix m;
	osg::Quat rot = TrackingManager::instance()->getHandMat(0).getRotate();
	osg::Vec3 pos(0,_pointerDistance,0);
	pos = pos * TrackingManager::instance()->getHandMat(0);
	m.makeRotate(rot);

	osg::Matrix scale;
	if(_mode == RIBBON)
	{
	    scale.makeScale(osg::Vec3(_sizeRV->getValue(),1.0,1.0));
	}

	m = scale *  m * osg::Matrix::translate(pos);
	_brushRoot->setMatrix(m);
    }*/
}

bool Sketch::buttonEvent(int type, int button, int hand, const osg::Matrix & mat)
{
    if(hand == 0 && button == 0)
    {
	if(_csCB->getValue())
	{
	    if(_colorSelector->buttonEvent(type, mat))
	    {
		_color = _colorSelector->getColor();
		if(_activeObject)
		{
		    _activeObject->setColor(_color);
		}
		return true;
	    }
	}

	if(_activeObject)
	{
	    bool ret = _activeObject->buttonEvent(type, mat);
	    if(_activeObject->isDone())
	    {
		finishGeometry();
		createGeometry();
	    }
	    return ret;
	}
    }
    /*if(hand == 0 && button == 0 && type == BUTTON_DOWN && _mode >= 0)
    {
	//std::cerr << "Start drawing." << std::endl;

	if(_mode == RIBBON)
	{
	    _verts = new Vec3Array(0);
	    _colors = new Vec4Array(1);
	    _normals = new Vec3Array(0);
	    _primitive = new DrawArrays(PrimitiveSet::TRIANGLE_STRIP, 0, 0);
	    _currentGeometry = new osg::Geometry();
	    
	    (*_colors)[0] = _color;

	    _currentGeometry->setVertexArray(_verts);
	    _currentGeometry->setColorArray(_colors);
	    _currentGeometry->setNormalArray(_normals);
	    _currentGeometry->setNormalBinding(Geometry::BIND_PER_VERTEX);
	    _currentGeometry->setColorBinding(Geometry::BIND_OVERALL);
	    _currentGeometry->setUseDisplayList(false);
	    //_currentGeometry->setUseVertexBufferObjects(true);
	    _currentGeometry->addPrimitiveSet(_primitive);

	    _sketchGeode->addDrawable(_currentGeometry);

	    MyComputeBounds * mcb = new MyComputeBounds();
	    _currentBound = &mcb->_bound;
	    _currentGeometry->setComputeBoundingBoxCallback(mcb);

	    osg::Vec3 point(-50.0 * _sizeRV->getValue(), _pointerDistance, 0);
	    point = point * mat * PluginHelper::getWorldToObjectTransform();

	    _verts->push_back(point);
	    _normals->push_back(osg::Vec3(0,0,1));
	    _currentBound->expandBy(point);

	    point = osg::Vec3(50.0 * _sizeRV->getValue(), _pointerDistance, 0);
	    point = point * mat * PluginHelper::getWorldToObjectTransform();

	    _verts->push_back(point);
	    _normals->push_back(osg::Vec3(0,0,1));
	    _currentBound->expandBy(point);

	    _lastTransform = mat * PluginHelper::getWorldToObjectTransform();

	    _count = 2;

	    _primitive->setCount(_count);
	    _currentGeometry->dirtyBound();

	    _drawing = true;
	}
	else if(_mode == LINE)
	{
	    if(_lt == LINE_NONE)
	    {
		//std::cerr << "LINE_NONE" << std::endl;
		return false;
	    }

	    if(!_drawing)
	    {
		//std::cerr << "Starting Line." << std::endl;
		_verts = new Vec3Array(0);
		_colors = new Vec4Array(1);
		_primitive = new DrawArrays(PrimitiveSet::LINE_STRIP, 0, 0);
		_currentGeometry = new osg::Geometry();

		(*_colors)[0] = _color;

		_currentGeometry->setVertexArray(_verts);
		_currentGeometry->setColorArray(_colors);
		_currentGeometry->setColorBinding(Geometry::BIND_OVERALL);
		_currentGeometry->setUseDisplayList(false);
		//_currentGeometry->setUseVertexBufferObjects(true);
		_currentGeometry->addPrimitiveSet(_primitive);

		_sketchGeode->addDrawable(_currentGeometry);

		MyComputeBounds * mcb = new MyComputeBounds();
		_currentBound = &mcb->_bound;
		_currentGeometry->setComputeBoundingBoxCallback(mcb);

		osg::Vec3 point;
		if(_lineSnap->getValue())
		{
		    point = _brushRoot->getMatrix().getTrans();
		}
		else 
		{
		    point = osg::Vec3(0, _pointerDistance, 0);
		    point = point * mat;
		}

		point = point * PluginHelper::getWorldToObjectTransform();

		_verts->push_back(point);
		_currentBound->expandBy(point);

		_count = 1;

		if(_lt == SEGMENT || _lt == MULTI_SEGMENT)
		{
		    _verts->push_back(point);
		    _count = 2;
		    _updateLastPoint = true;
		}

		_primitive->setCount(_count);
		_currentGeometry->dirtyBound();

		osg::StateSet * stateset = _currentGeometry->getOrCreateStateSet();
		stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
		osg::LineWidth * lw = new osg::LineWidth();
		lw->setWidth(_sizeRV->getValue());
		stateset->setAttributeAndModes(lw,osg::StateAttribute::ON);
		//TODO: tube shader?

		_drawing = true;
	    }
	    else if(_lt == MULTI_SEGMENT)
	    {
		osg::Vec3 point;
		if(_lineSnap->getValue())
		{
		    point = _brushRoot->getMatrix().getTrans();
		}
		else 
		{
		    point = osg::Vec3(0, _pointerDistance, 0);
		    point = point * mat;
		}

		point = point * PluginHelper::getWorldToObjectTransform();

		_verts->push_back(point);
		_currentBound->expandBy(point);

		_count++;

		_primitive->setCount(_count);
		_currentGeometry->dirtyBound();
		_updateLastPoint = true;
	    }
	}

	return true;
    }
    else if(hand == 0 && button == 0 && type == BUTTON_UP && _drawing)
    {
	//std::cerr << "Stop drawing." << std::endl;

	if(_mode == RIBBON || (_mode == LINE && (_lt == SEGMENT || _lt == FREEHAND)))
	{
	    //std::cerr << "Finish geometry" << std::endl;
	    finishGeometry();
	}
	if(_mode == LINE && (_lt == SEGMENT || _lt == MULTI_SEGMENT))
	{
	    _updateLastPoint = false;
	}

	return true;
    }
    else if(hand == 0 && button == 0 && type == BUTTON_DRAG && _drawing)
    {
	if(_mode == RIBBON)
	{
	    osg::Vec3 lastpoint1(-50.0 * _sizeRV->getValue(), _pointerDistance, 0), lastpoint2(50.0 * _sizeRV->getValue(), _pointerDistance, 0), newpoint1(-50.0 * _sizeRV->getValue(), _pointerDistance, 0), newpoint2(50.0 * _sizeRV->getValue(), _pointerDistance, 0);

	    lastpoint1 = lastpoint1 * _lastTransform;
	    lastpoint2 = lastpoint2 * _lastTransform;

	    newpoint1 = newpoint1 * mat * PluginHelper::getWorldToObjectTransform();
	    newpoint2 = newpoint2 * mat * PluginHelper::getWorldToObjectTransform();

	    if((newpoint1 - lastpoint1).length2() > 10.0 || (newpoint2 - lastpoint2).length2() > 10.0)
	    {
		_verts->push_back(newpoint1);
		_verts->push_back(newpoint2);


		osg::Vec3 v1, v2, normal1, normal2, normala;
		v1 = newpoint1 - (*_verts)[_count-2];
		v1.normalize();
		v2 = (*_verts)[_count-1] - (*_verts)[_count-2];
		v2.normalize();

		normal1 = v1 ^ v2;
		normal1.normalize();

		v1 = (*_verts)[_count-1] - newpoint2;
		v1.normalize();
		v2 = newpoint1 - newpoint2;
		v2.normalize();

		normal2 = v1 ^ v2;
		normal2.normalize();

		//normal1 = osg::Vec3(normal1.x(),-normal1.z(),normal1.y());
		//normal2 = osg::Vec3(normal2.x(),-normal2.z(),normal2.y());

		normala = (normal1 + normal2) / 2.0;
		normala.normalize();

		//_normals->push_back(osg::Vec3(0,0,1));
		//_normals->push_back(osg::Vec3(0,0,1));

		_normals->push_back(normala);
		_normals->push_back(normal2);

		if(_count == 2)
		{
		    (*_normals)[0] = normal1;
		    (*_normals)[1] = normala;
		}
		else
		{
		    (*_normals)[_count-2] = ((*_normals)[_count-2] + normal1) / 2.0;
		    (*_normals)[_count-2].normalize();

		    (*_normals)[_count-1] = ((*_normals)[_count-1] + normala) / 2.0;
		    (*_normals)[_count-1].normalize();
		}

		_count += 2;
		_lastTransform = mat * PluginHelper::getWorldToObjectTransform();

		_primitive->setCount(_count);

		_currentBound->expandBy(newpoint1);
		_currentBound->expandBy(newpoint2);

		_currentGeometry->dirtyBound();

		//std::cerr << "Count: " << _count << std::endl;
	    }
	}
	if(_mode == LINE)
	{
	    //std::cerr << "Line update" << std::endl;
	    osg::Vec3 point;
	    if(_lineSnap->getValue())
	    {
		point = _brushRoot->getMatrix().getTrans();
	    }
	    else 
	    {
		point = osg::Vec3(0, _pointerDistance, 0);
		point = point * mat;
	    }

	    point = point * PluginHelper::getWorldToObjectTransform();

	    if(_lt == FREEHAND)
	    {
		_verts->push_back(point);
		_count++;
		_primitive->setCount(_count);
	    }
	    else if(_updateLastPoint)
	    {
		//std::cerr << "Changing last point" << std::endl;
		//std::cerr << "Point x: " << point.x() << " y: " << point.y() << " z: " << point.z() << std::endl;
		(*_verts)[_count-1] = point;
	    }

	    _currentBound->expandBy(point);
	    _currentGeometry->dirtyBound();

	    //std::cerr << "Count: " << _count << std::endl;
	}
	return true;
    }*/

    return false;
}

bool Sketch::mouseButtonEvent(int type, int button, int x, int y, const osg::Matrix & mat)
{
    if(type == MOUSE_BUTTON_DOWN)
    {
	buttonEvent(BUTTON_DOWN, button, -1, mat);
    }
    else if(type == MOUSE_BUTTON_UP)
    {
	buttonEvent(BUTTON_UP, button, -1, mat);
    }
    else if(type == MOUSE_DOUBLE_CLICK)
    {
	buttonEvent(BUTTON_DOUBLE_CLICK, button, -1, mat);
    }
    else if(type == MOUSE_DRAG)
    {
	buttonEvent(BUTTON_DRAG, button, -1, mat);
    }
    return false;
}

osg::BoundingBox Sketch::MyComputeBounds::computeBound(const osg::Drawable &) const
{
    //std::cerr << "Returning bound." << std::endl;
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
	    _st = SHAPE_NONE;
	    if(_shapeType->firstNumOn() >= 0)
	    {
		_shapeType->setValue(_shapeType->firstNumOn(), false);
	    }
	    _shapeWireframe->setValue(false);
	    _sketchMenu->addItem(_shapeType);
	    _sketchMenu->addItem(_shapeWireframe);
	    break;
	default:
	    break;
    }
}

void Sketch::finishGeometry()
{
    /*if(!_drawing)
    {
	return;
    }

    if(_mode == RIBBON || _mode == LINE)
    {
	_geometryMap[PluginHelper::getProgramDuration()] = _currentGeometry;
    }

    _drawing = false;*/

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

    switch(_mode)
    {
	case RIBBON:
	    _activeObject = new SketchRibbon(_color, _sizeRV->getValue());
	    break;
	case LINE:
	    _activeObject = new SketchLine(_lt, _lineTube->getValue(), _lineSnap->getValue(), _color, _sizeRV->getValue());
	    break;
	case SHAPE:
	    break;
	default:
	    break;
    }

    if(_activeObject)
    {
	_sketchGeode->addDrawable(_activeObject->getDrawable());
	_activeObject->addBrush(_brushRoot.get());
    }
}
