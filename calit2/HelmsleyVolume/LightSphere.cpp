#include <iostream>

#include "LightSphere.h"



LightSphere::LightSphere(float radius, osg::Vec3 position)
	:SceneObject("light_sphere", false, true, false, false, true), position(position), radius(radius) 
{
	sphere = new osg::Sphere(position, radius);
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

	std::cout << "========================Constructor=======================" << std::endl;
	createGeometry();
}

void LightSphere::createGeometry()
{
	//_intersect = new osg::Geode();

	//_group->addChild(transform);
	//transform->addChild(_intersect);

	//_intersect->setNodeMask(cvr::INTERSECT_MASK);

	//_intersect->addDrawable(sd);

	std::cout << "======================Creating Geometrey for Light Sphere============================" << std::endl;
	updateGeometry();
}

void LightSphere::updateGeometry()
{
	std::cout << "=========================Geometry Updated================================" << std::endl;
}

bool LightSphere::processEvent(cvr::InteractionEvent* ie)
{
	std::cout << "======================processing event===================================" << std::endl;

	return SceneObject::processEvent(ie);
}

/*
void LightSphere::processHover(bool event)
{
	std::cout << "======================processing event===================================" << std::endl;
	/*cvr::TrackedButtonInteractionEvent* tie = event->asTrackedButtonEvent();
	osg::Matrix mat = cvr::TrackingManager::instance()->getHandMat(tie->getHand());
	osg::Quat q = osg::Quat();
	osg::Quat q2 = osg::Quat();
	osg::Vec3 v2 = osg::Vec3();
	osg::Vec3 pos = osg::Vec3();
	mat.decompose(pos, q, v2, q2);

	//if correct button is pressed
	if (true) {
	}
	
	moveLightPos(osg::Vec3(0, 0, 0));
}
*/

void LightSphere::setSceneObject(SceneObject* so)
{
	_so = so;
}

void LightSphere::moveLightPos(osg::Vec3 position)
{
	osg::Matrix sm = transform->getMatrix();
	sm = sm.translate(position);
	transform->setMatrix(sm);
	std::cout << "moving light sphere: " << position.x() << ", " << position.y() << ", " << position.z() << std::endl;

}

osg::MatrixTransform* LightSphere::getTransform()
{
	return transform;
}
