#ifndef SCREENSHOT_TOOL_H
#define SCREENSHOT_TOOL_H

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
//#include "UIExtensions.h"
#include "VolumeMenu.h"

class ScreenshotSaverThread : public OpenThreads::Thread
{
public:
	ScreenshotSaverThread(osg::Image* i, std::string filename) : _image(i), _filename(filename), _done(false) {}
	virtual ~ScreenshotSaverThread() {}

	virtual void run();

protected:
	osg::ref_ptr<osg::Image> _image;
	std::string _filename;
	bool _done;
};

class ScreenshotTool : public cvr::SceneObject, public UICallback
{
public:
	ScreenshotTool(std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds = false)
		: SceneObject(name, navigation, movable, clip, contextMenu, showBounds)
	{
		init();
	}
	~ScreenshotTool();

	virtual void updateCallback(int handID, const osg::Matrix& mat);
	virtual void menuCallback(cvr::MenuItem* menuItem);
	//virtual void uiCallback(cvr::MenuItem* menuItem);
	virtual void uiCallback(UICallbackCaller* ui);
	//virtual void enterCallback(int handID, const osg::Matrix& mat);
	//virtual void leaveCallback(int handID, const osg::Matrix& mat);

	void setParams(double fov, double aspect);
	void activate();
	void deactivate();
	const bool getIsActive() { return _cameraActive; }
	void takePhoto(std::string filename);
	void setPhoto(std::string imgLocation);

	void setWorldMenu(NewVolumeMenu* vm) { _vm = vm; }
	void saveAsYaml();
	void takingPhoto(bool takingPhoto) { takingPhoto = _takingPhoto; }
	osg::ref_ptr<osg::Camera> getCam(){ return _camera; }
protected:
	void init();

	osg::ref_ptr<osg::Image> _image;
	osg::ref_ptr<osg::Texture2D> _texture;
	osg::ref_ptr<osg::Camera> _camera;
	osg::ref_ptr<osg::Node> _tablet;
	osg::ref_ptr<osg::Geometry> _display;

	cvr::PopupMenu* _cameraMenu;
	cvr::MenuButton* _pictureButton;
	osg::ref_ptr<osg::Camera::DrawCallback> _pdc;

	CallbackButton* _takePicture;
	//CINE
#ifdef CINE
	CallbackButton* _renderCinematic;
	std::string _imgCineFP;
#endif

	NewVolumeMenu* _vm;

	std::vector<ScreenshotSaverThread*> _saveThreads;

	bool _cameraActive;
	bool _showingPhoto = false;
	bool _takingPhoto = false;
	bool _pictureSaved = false;
};

class ScreenshotCallback : public osg::Camera::DrawCallback
{
public:
	ScreenshotCallback(ScreenshotTool* tool, std::string path) : _tool(tool), _path(path) {}

	virtual void operator()(osg::RenderInfo& renderInfo) const
	{
		_tool->takePhoto(_path);
	}

protected:
	ScreenshotTool* _tool;
	std::string _path;
	
};

#endif
