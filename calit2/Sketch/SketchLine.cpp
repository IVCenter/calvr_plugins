#include "SketchLine.h"
#include "Sketch.h"

#include <cvrInput/TrackingManager.h>
#include <cvrKernel/InteractionManager.h>
#include <cvrKernel/PluginHelper.h>
#ifdef WIN32
#include <cvrUtil/TimeOfDay.h>
#endif

#include <iostream>
#include <osg/Material>

using namespace cvr;

SketchLine::SketchLine(LineType type, bool tube, bool snap, osg::Vec4 color, float size) : SketchObject(color,size)
{
    _tube = tube;
    _snap = snap;
    _type = type;

    _verts = new osg::Vec3Array(0);
    _colors = new osg::Vec4Array(1);
    _primitive = new osg::DrawArrays(osg::PrimitiveSet::LINE_STRIP, 0, 0);
    _geometry = new osg::Geometry();

    (*_colors)[0] = _color;

    _geometry->setVertexArray(_verts.get());
    _geometry->setColorArray(_colors.get());
    _geometry->setColorBinding(osg::Geometry::BIND_OVERALL);
    _geometry->setUseDisplayList(false);
    _geometry->addPrimitiveSet(_primitive.get());

    _mcb = new MyComputeBounds();
    _geometry->setComputeBoundingBoxCallback(_mcb.get());

    osg::StateSet * stateset = _geometry->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    _lineWidth = new osg::LineWidth();
    _lineWidth->setWidth(_size);
    stateset->setAttributeAndModes(_lineWidth,osg::StateAttribute::ON);

    osg::Shape * shape = new osg::Sphere(osg::Vec3(0,0,0),10);
    _brushDrawable = new osg::ShapeDrawable(shape);
    _brushDrawable->setColor(_color);
    _brushGeode = new osg::Geode();
    _brushGeode->addDrawable(_brushDrawable.get());

    stateset = _brushDrawable->getOrCreateStateSet();
    osg::Material * mat = new osg::Material();
    //stateset->setAttributeAndModes(mat,osg::StateAttribute::ON);

    _drawing = false;

    _count = 0;
}

SketchLine::~SketchLine()
{
    
}

bool SketchLine::buttonEvent(int type, const osg::Matrix & mat)
{
    if(_done || _type == NONE)
    {
	return false;
    }

    if(type == BUTTON_DOWN)
    {
	if(!_drawing)
	{
	    osg::Vec3 point;
	    if(!_snap)
	    {
		point = osg::Vec3(0, Sketch::instance()->getPointerDistance(), 0);
		point = point * mat * PluginHelper::getWorldToObjectTransform();
	    }
	    else
	    {
		//TODO: find this point
	    }

	    gettimeofday(&_lastPointTime, NULL);

	    // set object timestamp
	    _timeStamp = _lastPointTime;

	    if(_type == FREEHAND)
	    {
		_lastPoint = point;
	    }

	    _verts->push_back(point);
	    _mcb->_bound.expandBy(point);

	    if(_type == SEGMENT || _type == MULTI_SEGMENT)
	    {
		_verts->push_back(point);
		_count = 2;
	    }
	    else
	    {
		_count = 1;
	    }

	    _primitive->setCount(_count);
	    _geometry->dirtyBound();
	    _drawing = true;
	}
	else if(_type == MULTI_SEGMENT)
	{
	    _valid = true;
	    osg::Vec3 point;
	    if(!_snap)
	    {
		point = osg::Vec3(0, Sketch::instance()->getPointerDistance(), 0);
		point = point * mat * PluginHelper::getWorldToObjectTransform();
	    }
	    else
	    {
		//TODO: find this point
	    }

	    if(_type == FREEHAND)
	    {
		gettimeofday(&_lastPointTime, NULL);
		_lastPoint = point;
	    }

	    _verts->push_back(point);
	    _mcb->_bound.expandBy(point);

	    _count++;

	    _primitive->setCount(_count);
	    _geometry->dirtyBound();
	}

	return true;
    }
    else if(type == BUTTON_DRAG)
    {
	if(!_drawing)
	{
	    return false;
	}

	osg::Vec3 point;
	if(!_snap)
	{
	    point = osg::Vec3(0, Sketch::instance()->getPointerDistance(), 0);
	    point = point * mat * PluginHelper::getWorldToObjectTransform();
	}
	else
	{
	    //TODO: find this point
	}

	if(_type == FREEHAND)
	{
	    struct timeval currentTime;
	    gettimeofday(&currentTime,NULL);

	    double timediff = (currentTime.tv_sec - _lastPointTime.tv_sec) + ((currentTime.tv_usec - _lastPointTime.tv_usec) / 1000000.0);
	    float distance2 = (point - _lastPoint).length2();

	    if(timediff > 0.75 || distance2 > 10.0)
	    {
		_valid = true;

		_verts->push_back(point);
		_count++;
		_primitive->setCount(_count);

		_mcb->_bound.expandBy(point);
		_geometry->dirtyBound();

		_lastPoint = point;
		_lastPointTime = currentTime;
	    }
	}
	else if(_type == SEGMENT || _type == MULTI_SEGMENT)
	{
	    (*_verts)[_count-1] = point;
	    _mcb->_bound.expandBy(point);
	    _geometry->dirtyBound();
	}
	return true;
    }
    else if(type == BUTTON_UP)
    {
	if(_type == SEGMENT || _type == FREEHAND)
	{
	    _done = true;
	}
	if(_type == SEGMENT)
	{
	    _valid = true;
	}
	return true;
    }
	return false;
}

void SketchLine::addBrush(osg::MatrixTransform * mt)
{
    mt->addChild(_brushGeode.get());
}

void SketchLine::removeBrush(osg::MatrixTransform * mt)
{
    mt->removeChild(_brushGeode.get());
}

void SketchLine::updateBrush(osg::MatrixTransform * mt)
{
    osg::Matrix m;
    osg::Quat rot = TrackingManager::instance()->getHandMat(0).getRotate();
    osg::Vec3 pos(0,Sketch::instance()->getPointerDistance(),0);
    pos = pos * TrackingManager::instance()->getHandMat(0);
    m.makeRotate(rot);

    m = m * osg::Matrix::translate(pos);
    mt->setMatrix(m);
}

void SketchLine::finish()
{
    _done = true;
}

osg::Drawable * SketchLine::getDrawable()
{
    return _geometry;
}

void SketchLine::setColor(osg::Vec4 color)
{
    _color = color;
    (*_colors)[0] = _color;
    _brushDrawable->setColor(_color);
}

void SketchLine::setSize(float size)
{
    _size = size;
    _lineWidth->setWidth(_size);
}

void SketchLine::setTube(bool b)
{
    if(b == _tube)
    {
	return;
    }

    //TODO: implement

    _tube = b;
}

void SketchLine::setSnap(bool b)
{
    _snap = b;
}
