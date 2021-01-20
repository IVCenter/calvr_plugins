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
#include <cvrMenu/NewUI/UIPopup.h>
#include <cvrMenu/NewUI/UIToggle.h>

#include <iostream>
#include "CuttingPlane.h"
#include "Interactable.h"
#include "UIExtensions.h"
 
class CenterlineTool : public cvr::SceneObject, public UICallback, public UICallbackCaller
{
public:
	CenterlineTool(std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds = false)
		: SceneObject(name, navigation, movable, clip, contextMenu, showBounds)
	{
		init();
	
		_updateCallback = new CenterlineUpdate(this, _camera);
		this->getRoot()->addUpdateCallback(_updateCallback);
	}
	virtual void uiCallback(UICallbackCaller* ui);
	void setCoords(osg::Vec3dArray* coords, osg::MatrixTransform* trans) {
		_updateCallback->setCoords(coords, trans);
	}
	void setCoords(osg::Vec3dArray* coords) {
		_updateCallback->setCoords(coords);
	}


	void setParams(double fov, double aspect);
	void activate();
	void deactivate();
	const bool getIsActive() { return _cameraActive; }
	
	void setCP(CuttingPlane* cp) { _cp = cp; }
protected:
	void init();

	osg::ref_ptr<osg::Image> _image;
	osg::ref_ptr<osg::Texture2D> _texture;
	osg::ref_ptr<osg::Camera> _camera;
	osg::ref_ptr<osg::Node> _tablet;
	osg::ref_ptr<osg::Geometry> _display;

	cvr::PopupMenu* _cameraMenu;
	cvr::UIPopup* _cameraPop;
	ToolToggle* _playButton;
	CallbackButton* _colonButton;
	CallbackButton* _illeumButton;
	CuttingPlane* _cp;
	
	osg::ref_ptr<osg::Camera::DrawCallback> _pdc;

	//UpdateCenterlineCam* _updateCallback = nullptr;
	CenterlineUpdate* _updateCallback = nullptr;
	std::string _imgDir;

	bool _cameraActive;
};

#endif
#pragma once
