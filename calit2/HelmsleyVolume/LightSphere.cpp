#include <iostream>

#include "LightSphere.h"



LightSphere::LightSphere(float radius, osg::Vec3 position)
	:UIElement(), position(position), radius(radius)
{
	color = osg::Vec4(1, 0, 0, 1);
	geode = new osg::Geode();
	transform = new osg::MatrixTransform();

	sphere = new osg::Sphere(position, radius);
	sd = new osg::ShapeDrawable(sphere);

	std::cout << "========================Constructor=======================" << std::endl;
	createGeometry();
}

void LightSphere::createGeometry()
{

	geode->addDrawable(sd);
	transform->addChild(geode);

	std::cout << "======================Creating Geometrey for Light Sphere============================" << std::endl;
	updateGeometry();
}

void LightSphere::updateGeometry()
{
	osg::Vec4Array* colors = new osg::Vec4Array;
	colors->push_back(color);
	((osg::Geometry*)geode->getDrawable(0))->setColorArray(colors, osg::Array::BIND_OVERALL);

	osg::Matrix mat = osg::Matrix();
	mat.makeScale(_actualSize);
	mat.postMultTranslate(_actualPos);
	transform->setMatrix(mat);
	std::cout << "=========================Geometry Updated================================" << std::endl;
}
