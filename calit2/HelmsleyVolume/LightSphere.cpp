#include <iostream>

#include "LightSphere.h"
#include "UIExtensions.h"
#include "HelmsleyVolume.h"

LightSphere::LightSphere(float radius, osg::Vec3 position)
	:SceneObject("light_sphere", false, true, false, false, true), position(position), radius(radius) 
{
	sphere = new osg::Sphere(position, radius);
	//sphere = new osg::Cone(position, radius, 100.0f);
	sd = new osg::ShapeDrawable(sphere);
	transform = new osg::MatrixTransform();
	geode = new osg::Geode();

	osg::Vec4Array* colors = new osg::Vec4Array;
	colors->push_back(osg::Vec4(1, 1, 1, 1));

	sd->setColorArray(colors);
	sd->setColorBinding(osg::Geometry::BIND_OVERALL);

	sd->setDataVariance(osg::Object::DYNAMIC);
	sd->setUseDisplayList(false);

	geode->addChild(sd);

	//transform->addChild(geode);
	
	this->addChild(geode);

	this->getRoot()->addUpdateCallback(new LightUpdate(this));

	createGeometry();
}

void LightSphere::createGeometry()
{

	updateGeometry();
}

void LightSphere::updateGeometry()
{
}

osg::Vec3 LightSphere::getWorldPosition() {
	osg::MatrixTransform* _obj2World = new osg::MatrixTransform(HelmsleyVolume::instance()->getSceneObjects()[0]->getObjectToWorldMatrix());

	return (this->getPosition() * _obj2World->getMatrix());
}

bool LightSphere::processEvent(cvr::InteractionEvent* ie)
{
	//osg::Vec3 tempPos = transform->getMatrix().getTrans();
	//osg::Vec3 tempPos = this->getPosition();
	//static_cast<MarchingCubesRender*>(_mcr)->setPointLightPos(tempPos);
	
	return SceneObject::processEvent(ie);
}

void LightSphere::setSceneObject(SceneObject* so)
{
	_so = so;
}

osg::MatrixTransform* LightSphere::getTransform()
{
	return transform;
}

void LightUpdate::operator()(osg::Node* node, osg::NodeVisitor* nv)
{
	if (_ls->getWorldPosition() != _lastPos) {
		std::cout << "spherePos: " << HelmsleyVolume::instance()->printVec3OSG(_lastPos) << std::endl;
		static_cast<MarchingCubesRender*>(_ls->_mcr)->setPointLightPos(_ls->getWorldPosition());
		_lastPos = _ls->getWorldPosition();
	}

	traverse(node, nv);
}
