#include "SketchLine.h"
#include "Sketch.h"

#include <input/TrackingManager.h>
#include <kernel/InteractionManager.h>
#include <kernel/PluginHelper.h>

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
    stateset->setAttributeAndModes(mat,osg::StateAttribute::ON);

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
	    point = osg::Vec3(0, Sketch::instance()->getPointerDistance(), 0);
	    point = point * mat;

	    
	}
    }
    else if(type == BUTTON_DRAG)
    {
    }
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
