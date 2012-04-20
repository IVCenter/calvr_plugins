#include "SketchRibbon.h"
#include "Sketch.h"

#include <cvrInput/TrackingManager.h>
#include <cvrKernel/InteractionManager.h>
#include <cvrKernel/PluginHelper.h>
#ifdef WIN32
#include <cvrUtil/TimeOfDay.h>
#endif

#ifdef WIN32
#define M_PI 3.141592653589793238462643
#endif

#include <iostream>
#include <osg/Material>

using namespace cvr;

SketchRibbon::SketchRibbon(osg::Vec4 color, float size) : SketchObject(color,size)
{
    _verts = new osg::Vec3Array(0);
    _colors = new osg::Vec4Array(1);
    _normals = new osg::Vec3Array(0);
    _primitive = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_STRIP, 0, 0);
    _geometry = new osg::Geometry();

    _color = color;
    (*_colors)[0] = _color;

    _geometry->setVertexArray(_verts.get());
    _geometry->setColorArray(_colors.get());
    _geometry->setColorBinding(osg::Geometry::BIND_OVERALL);
    _geometry->setNormalArray(_normals.get());
    _geometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    _geometry->setUseDisplayList(false);
    _geometry->addPrimitiveSet(_primitive.get());

    _mcb = new MyComputeBounds();
    _geometry->setComputeBoundingBoxCallback(_mcb.get());

    osg::Quat rot;

    osg::Shape * shape = new osg::Cylinder(osg::Vec3(0,0,0),5,100);
    rot.makeRotate(M_PI/ 2.0, osg::Vec3(0,1.0,0));
    ((osg::Cylinder*)shape)->setRotation(rot);

    _brushDrawable = new osg::ShapeDrawable(shape);
    _brushDrawable->setColor(_color);
    _brushGeode = new osg::Geode();
    _brushGeode->addDrawable(_brushDrawable.get());

    osg::StateSet * stateset = _brushDrawable->getOrCreateStateSet();
    osg::Material * mat = new osg::Material();
    //stateset->setAttributeAndModes(mat,osg::StateAttribute::ON);

    stateset = _geometry->getOrCreateStateSet();
    //stateset->setAttributeAndModes(mat,osg::StateAttribute::ON);
    stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    _drawing = false;

    _count = 0;
}

SketchRibbon::~SketchRibbon()
{
}

bool SketchRibbon::buttonEvent(int type, const osg::Matrix & mat)
{
    if(_done)
    {
	return false;
    }

    if(type == BUTTON_DOWN)
    {
	if(!_drawing)
	{
	    osg::Vec3 point(-50.0 * _size, Sketch::instance()->getPointerDistance(), 0);
	    point = point * mat * PluginHelper::getWorldToObjectTransform();

	    _verts->push_back(point);
	    _normals->push_back(osg::Vec3(0,0,1));
	    _mcb->_bound.expandBy(point);
	    _lastPoint1 = point;

	    point = osg::Vec3(50.0 * _size, Sketch::instance()->getPointerDistance(), 0);
	    point = point * mat * PluginHelper::getWorldToObjectTransform();
	    _verts->push_back(point);
	    _normals->push_back(osg::Vec3(0,0,1));
	    _mcb->_bound.expandBy(point);
	    _lastPoint2 = point;

	    _count = 2;

	    gettimeofday(&_lastPointTime, NULL);

	    // set object timestamp
	    _timeStamp = _lastPointTime;

	    _primitive->setCount(_count);
	    _geometry->dirtyBound();

	    _drawing = true;
	}

	return true;
    }
    else if(type == BUTTON_DRAG)
    {
	if(!_drawing)
	{
	    return false;
	}

	osg::Vec3 newpoint1(-50.0 * _size, Sketch::instance()->getPointerDistance(), 0), 
              newpoint2(50.0 * _size, Sketch::instance()->getPointerDistance(), 0);

	newpoint1 = newpoint1 * mat * PluginHelper::getWorldToObjectTransform();
	newpoint2 = newpoint2 * mat * PluginHelper::getWorldToObjectTransform();

	struct timeval currentTime;
	gettimeofday(&currentTime,NULL);

	double timediff = (currentTime.tv_sec - _lastPointTime.tv_sec) + 
        ((currentTime.tv_usec - _lastPointTime.tv_usec) / 1000000.0);

	if(timediff > 0.75 || (newpoint1 - _lastPoint1).length2() > 10.0 || (newpoint2 - _lastPoint2).length2() > 10.0)
	{
	    _valid = true;

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
		//std::cerr << "Normal x: " << (*_normals)[_count-2].x() << " y: " << (*_normals)[_count-2].y() << " z: " << (*_normals)[_count-2].z() << std::endl;

		(*_normals)[_count-1] = ((*_normals)[_count-1] + normala) / 2.0;
		(*_normals)[_count-1].normalize();
		//std::cerr << "Normal x: " << (*_normals)[_count-1].x() << " y: " << (*_normals)[_count-1].y() << " z: " << (*_normals)[_count-1].z() << std::endl;
	    }

	    _count += 2;

	    _primitive->setCount(_count);

	    _mcb->_bound.expandBy(newpoint1);
	    _mcb->_bound.expandBy(newpoint2);

	    _geometry->dirtyBound();

	    _lastPoint1 = newpoint1;
	    _lastPoint2 = newpoint2;

	    _lastPointTime = currentTime;
	}

	return true;
    }
    else if(type == BUTTON_UP)
    {
	if(!_drawing)
	{
	    return false;
	}

	_done = true;

	return true;
    }
}

void SketchRibbon::addBrush(osg::MatrixTransform * mt)
{
    mt->addChild(_brushGeode.get());
}

void SketchRibbon::removeBrush(osg::MatrixTransform * mt)
{
    mt->removeChild(_brushGeode.get());
}

void SketchRibbon::updateBrush(osg::MatrixTransform * mt)
{
    osg::Matrix m;
    osg::Quat rot = TrackingManager::instance()->getHandMat(0).getRotate();
    osg::Vec3 pos(0,Sketch::instance()->getPointerDistance(),0);
    pos = pos * TrackingManager::instance()->getHandMat(0);
    m.makeRotate(rot);

    osg::Matrix scale;
    scale.makeScale(_size,1.0,1.0);

    m = scale * m * osg::Matrix::translate(pos);
    mt->setMatrix(m);
}

void SketchRibbon::finish()
{
    _done = true;
}

osg::Drawable * SketchRibbon::getDrawable()
{
    return _geometry;
}

void SketchRibbon::setColor(osg::Vec4 color)
{
    _color = color;
    (*_colors)[0] = _color;
    _brushDrawable->setColor(_color);
}
