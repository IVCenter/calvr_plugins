#pragma once

#include <cvrMenu/NewUI/UIElement.h>
#include <cvrKernel/InteractionEvent.h>
#include <cvrKernel/SceneObject.h>
#include <cvrInput/TrackingManager.h>
#include <osg/ShapeDrawable>
#include "cvrKernel/NodeMask.h"


class LightSphere : public cvr::SceneObject {
public:
	LightSphere(float radius = 25.0f, osg::Vec3 position = osg::Vec3(0, 0, 0));

	osg::ref_ptr<osg::Sphere> sphere;
	osg::ref_ptr<osg::ShapeDrawable> sd;
	osg::ref_ptr<osg::MatrixTransform> transform;
	osg::ref_ptr<osg::Geode> geode;
	osg::Geometry* polyGeom;
	SceneObject* _so;

	osg::Vec4 color;
	osg::Vec3 position;
	float radius;

	virtual void createGeometry();
	virtual void updateGeometry();
	virtual bool processEvent(cvr::InteractionEvent* event) override;
	//virtual void processHover(bool event) override;

	void setSceneObject(SceneObject* so);

	void moveLightPos(osg::Vec3 position);
	osg::MatrixTransform* getTransform();

};
