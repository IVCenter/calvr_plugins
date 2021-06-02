#pragma once

#include <cvrMenu/NewUI/UIElement.h>
#include <cvrKernel/InteractionEvent.h>
#include <cvrKernel/SceneObject.h>
#include <cvrInput/TrackingManager.h>
#include <osg/ShapeDrawable>
#include "cvrKernel/NodeMask.h"

class LightSphere;

class LightUpdate : public osg::NodeCallback {
public:
	LightUpdate(LightSphere* ls)
	:_ls(ls){};

	virtual void operator()(osg::Node* node, osg::NodeVisitor* nv);

private:
	LightSphere* _ls;
	osg::Vec3 _lastPos;
};

class LightSphere : public cvr::SceneObject {
public:
	LightSphere(float radius = 25.0f, osg::Vec3 position = osg::Vec3(100, 100, 100));

	osg::ref_ptr<osg::Sphere> sphere;
	osg::ref_ptr<osg::ShapeDrawable> sd;
	osg::ref_ptr<osg::MatrixTransform> transform;
	osg::ref_ptr<osg::Geode> geode;
	osg::Geometry* polyGeom;
	SceneObject* _so;
	void* _mcr;

	osg::Vec4 color;
	osg::Vec3 position;
	float radius;

	virtual void createGeometry();
	virtual void updateGeometry();
	osg::Vec3 getWorldPosition();
	virtual bool processEvent(cvr::InteractionEvent* event) override;

	void setSceneObject(SceneObject* so);
	//void setMCRenderer(MarchingCubesRender* mcr);

	osg::MatrixTransform* getTransform();

};
