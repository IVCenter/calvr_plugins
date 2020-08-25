#ifndef CENTERLINE_TOOL_H
#define CENTERLINE_TOOL_H

#include <osg/Group>
#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <osg/PositionAttitudeTransform>
#include <osg/MatrixTransform>
#include <osgText/Text>
#include <osg/CullFace>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osg/Texture2D>
#include <OpenThreads/Thread>

#include <cvrKernel/InteractionEvent.h>
#include <cvrKernel/SceneObject.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/PopupMenu.h>

#include <iostream>

#include "Interactable.h"

class UpdateCenterlineCam : public osg::NodeCallback
{
public:
	UpdateCenterlineCam(osg::Camera* cam, cvr::SceneObject* object = nullptr) : _camera(cam), _object(object)  {}

	virtual void operator()(osg::Node* node, osg::NodeVisitor* nv)
	{
		if (_index >= 0 && _index < _coords->size() - 1) {
			
			osg::Vec3d pos = _coords->at(_index) * _mTransform->getMatrix();
			osg::Vec3d next = _coords->at((_index + 1)) * _mTransform->getMatrix();
			osg::Matrix mat = _object->getTransform();

			osg::Vec3d posSwitch = osg::Vec3d(pos.x(), pos.z(), pos.y());
			osg::Vec3d nextSwitch = osg::Vec3d(next.x(), next.z(), next.y());
			osg::Vec3 up = osg::Vec3(0, 0, 1);
			mat.makeLookAt(posSwitch, nextSwitch, up);
			mat.setTrans(pos);
		
			
			_camera->setViewMatrix(mat);
			

			_index++;

			//osg::Matrix localToWorld = *_mTransform;
			//osg::Vec4 eye = osg::Vec4(0, 0, 0, 1) * localToWorld;
			//osg::Vec4 center = osg::Vec4(0, 10, 0, 1) * localToWorld;
			//osg::Vec4 up = osg::Vec4(0, 0, 1, 0) * localToWorld;
			////up.normalize();
			//_camera->setViewMatrixAsLookAt(osg::Vec3(eye.x(), eye.y(), eye.z()),
			//	osg::Vec3(center.x(), center.y(), center.z()),
			//	osg::Vec3(up.x(), up.y(), up.z()));

			//_index++;
		}
	}

	void setCoords(osg::Vec3dArray* coords, osg::MatrixTransform* transform) {
		_coords = coords;
		_mTransform = transform;
		_index = 0;
	}

private:
	cvr::SceneObject* _object;
	osg::Vec3dArray* _coords = nullptr;
	clock_t this_time = clock();
	clock_t last_time = this_time;
	double time_counter = 0;
	double testSeconds = .01;
	osg::MatrixTransform* _mTransform;
	osg::Camera* _camera;
	int _index = -1;
};



class CenterlineTool : public cvr::SceneObject
{
public:
	CenterlineTool(std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds = false)
		: SceneObject(name, navigation, movable, clip, contextMenu, showBounds)
	{
		init();
		_updateCallback = new UpdateCenterlineCam(_camera, this);
		this->getRoot()->addUpdateCallback(_updateCallback);
	}

	//virtual void updateCallback(int handID, const osg::Matrix& mat);
	//virtual void menuCallback(cvr::MenuItem* menuItem);
	//virtual void enterCallback(int handID, const osg::Matrix& mat);
	//virtual void leaveCallback(int handID, const osg::Matrix& mat);
	void setCoords(osg::Vec3dArray* coords, osg::MatrixTransform* trans) {
		_updateCallback->setCoords(coords, trans);
	}


	void setParams(double fov, double aspect);
	void activate();
	void deactivate();
	const bool getIsActive() { return _cameraActive; }

protected:
	void init();

	osg::ref_ptr<osg::Image> _image;
	osg::ref_ptr<osg::Texture2D> _texture;
	osg::ref_ptr<osg::Camera> _camera;
	osg::ref_ptr<osg::Node> _tablet;
	osg::ref_ptr<osg::Geometry> _display;

	cvr::PopupMenu* _cameraMenu;
	
	osg::ref_ptr<osg::Camera::DrawCallback> _pdc;

	UpdateCenterlineCam* _updateCallback = nullptr;
	

	bool _cameraActive;
};

#endif
#pragma once
