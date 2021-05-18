#pragma once
#include <cvrMenu/NewUI/UIElement.h>
#include <osg/ShapeDrawable>


class LightSphere : public cvr::UIElement {
public:
	LightSphere(float radius = 5.0f, osg::Vec3 position = osg::Vec3(0, 0, 0));

	osg::Shape* sphere;
	osg::ShapeDrawable* sd;
	osg::ref_ptr<osg::MatrixTransform> transform;
	osg::ref_ptr<osg::Geode> geode;
	osg::Geometry* polyGeom;

	osg::Vec4 color;
	osg::Vec3 position;
	float radius;

	virtual void createGeometry();
	virtual void updateGeometry();

};
