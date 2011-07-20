#include "Sketch.h"

#include <kernel/PluginHelper.h>
#include <kernel/InteractionManager.h>

#include <osg/ShapeDrawable>
#include <osg/Material>
#include <osg/LightModel>

#include <iostream>

CVRPLUGIN(Sketch)

using namespace cvr;
using namespace osg;

Sketch::Sketch()
{
}

Sketch::~Sketch()
{
}

bool Sketch::init()
{
    _sketchMenu = new SubMenu("Sketch");

    PluginHelper::addRootMenuItem(_sketchMenu);

    _modeButtons = new MenuTextButtonSet(true, 400, 35, 3);
    _modeButtons->setCallback(this);
    _modeButtons->addButton("Ribbon");
    _modeButtons->addButton("Tube");
    _modeButtons->addButton("Sphere");

    _sketchMenu->addItem(_modeButtons);

    _mode = NONE;
    _drawing = false;

    _brushScale = 1.0;
    _pointerDistance = 1000.0;

    _color = osg::Vec4(1.0,1.0,1.0,1.0);

    _sketchRoot = new osg::MatrixTransform();
    _sketchGeode = new osg::Geode();
    _sketchRoot->addChild(_sketchGeode);

    PluginHelper::getObjectsRoot()->addChild(_sketchRoot);

    osg::StateSet * stateset = _sketchGeode->getOrCreateStateSet();
    //stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
    osg::LightModel * lm = new osg::LightModel();
    lm->setTwoSided(true);
    stateset->setAttributeAndModes(lm,osg::StateAttribute::ON);
    osg::Material * mat = new osg::Material();
    stateset->setAttributeAndModes(mat,osg::StateAttribute::ON);

    _brushRoot = new osg::MatrixTransform();
    PluginHelper::getScene()->addChild(_brushRoot);

    osg::Shape * shape;
    osg::ShapeDrawable * sd;
    osg::Geode * geode;

    osg::Quat rot;

    shape = new osg::Cylinder(osg::Vec3(0,0,0),10,100);
    rot.makeRotate(M_PI/ 2.0, osg::Vec3(0,1.0,0));
    ((osg::Cylinder*)shape)->setRotation(rot);

    sd = new osg::ShapeDrawable(shape);
    geode = new osg::Geode();
    geode->addDrawable(sd);
    _brushes.push_back(geode);

    shape = new osg::Sphere(osg::Vec3(0,0,0),10);
    sd = new osg::ShapeDrawable(shape);
    geode = new osg::Geode();
    geode->addDrawable(sd);
    _brushes.push_back(geode);

    _brushes.push_back(geode);


    return true;
}

void Sketch::menuCallback(MenuItem * item)
{
    if(item == _modeButtons)
    {
	_mode = (DrawMode)_modeButtons->firstNumOn();
	_brushRoot->removeChildren(0,_brushRoot->getNumChildren());
	if(_mode >= 0)
	{
	    _brushRoot->addChild(_brushes[_mode]);
	}
    }
}

void Sketch::preFrame()
{
    if(_mode >= 0)
    {
	osg::Matrix m;
	osg::Quat rot = TrackingManager::instance()->getHandMat(0).getRotate();
	osg::Vec3 pos(0,_pointerDistance,0);
	pos = pos * TrackingManager::instance()->getHandMat(0);
	m.makeRotate(rot);
	m = m * osg::Matrix::translate(pos);
	_brushRoot->setMatrix(m);
    }
}

bool Sketch::buttonEvent(int type, int button, int hand, const osg::Matrix & mat)
{
    if(hand == 0 && button == 0 && type == BUTTON_DOWN && !_drawing && _mode >= 0)
    {
	std::cerr << "Start drawing." << std::endl;

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

	    osg::Vec3 point(-50.0 * _brushScale, _pointerDistance, 0);
	    point = point * mat * PluginHelper::getWorldToObjectTransform();

	    _verts->push_back(point);
	    _normals->push_back(osg::Vec3(0,0,1));
	    _currentBound->expandBy(point);

	    point = osg::Vec3(50.0 * _brushScale, _pointerDistance, 0);
	    point = point * mat * PluginHelper::getWorldToObjectTransform();

	    _verts->push_back(point);
	    _normals->push_back(osg::Vec3(0,0,1));
	    _currentBound->expandBy(point);

	    _lastTransform = mat * PluginHelper::getWorldToObjectTransform();

	    _count = 2;

	    _primitive->setCount(_count);
	    _currentGeometry->dirtyBound();
	    
	}

	_drawing = true;
	return true;
    }
    else if(hand == 0 && button == 0 && type == BUTTON_UP && _drawing)
    {
	std::cerr << "Stop drawing." << std::endl;

	_drawing = false;
	return true;
    }
    else if(hand == 0 && button == 0 && type == BUTTON_DRAG && _drawing)
    {
	if(_mode == RIBBON)
	{
	    osg::Vec3 lastpoint1(-50.0 * _brushScale, _pointerDistance, 0), lastpoint2(50.0 * _brushScale, _pointerDistance, 0), newpoint1(-50.0 * _brushScale, _pointerDistance, 0), newpoint2(50.0 * _brushScale, _pointerDistance, 0);

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

		normala = (normal1 + normal2) / 2.0;
		normala.normalize();

		_normals->push_back(osg::Vec3(0,0,1));
		_normals->push_back(osg::Vec3(0,0,1));

		//_normals->push_back(normala);
		//_normals->push_back(normal2);

		/*if(_count == 2)
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
		}*/

		_count += 2;
		_lastTransform = mat * PluginHelper::getWorldToObjectTransform();

		_primitive->setCount(_count);

		_currentBound->expandBy(newpoint1);
		_currentBound->expandBy(newpoint2);

		_currentGeometry->dirtyBound();

		//std::cerr << "Count: " << _count << std::endl;
	    }
	}
	return true;
    }

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
