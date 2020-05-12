#ifndef HELMSLEY_VOLUME_MENU_H
#define HELMSLEY_VOLUME_MENU_H

#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/SceneObject.h>

#include <cvrMenu/MenuItem.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/MenuRangeValueCompact.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuList.h>

#include "UIExtensions.h"
#include <cvrMenu/NewUI/UIPopup.h>

#include "VolumeGroup.h"

class VolumeMenu : public cvr::MenuCallback {
public:
	VolumeMenu(cvr::SceneObject* scene, VolumeGroup* volume) : _scene(scene), _volume(volume) {}

	void init();

	virtual void menuCallback(cvr::MenuItem * item);

	enum ColorFunction {
		DEFAULT = 0,
		RAINBOW = 1
	};
	ColorFunction transferFunction;
protected:
	cvr::SceneObject* _scene;
	VolumeGroup* _volume;

	cvr::MenuRangeValueCompact* scale;
	cvr::MenuRangeValueCompact* sampleDistance;

	cvr::MenuRangeValueCompact* contrastBottom;
	cvr::MenuRangeValueCompact* contrastTop;

	cvr::MenuRangeValueCompact* opacityMult;
	cvr::MenuRangeValueCompact* opacityCenter;
	cvr::MenuRangeValueCompact* opacityWidth;
	cvr::MenuRangeValueCompact* opacityTop;

	cvr::MenuCheckbox* adaptiveQuality;
	cvr::MenuCheckbox* highlightColon;
	cvr::MenuCheckbox* organsOnly;

	cvr::MenuList* colorFunction;
};


class ToolMenu : public UICallback {
public:
	ToolMenu(int index = 0, bool movable = true, cvr::SceneObject* parent = nullptr);
	~ToolMenu();

	cvr::SceneObject* getContainer()
	{
		return _container;
	}


	virtual void uiCallback(UICallbackCaller* item);

protected:

	cvr::UIPopup* _menu = nullptr;

	//CallbackRadial* _tool;

	ToolToggle* _cuttingPlane;
	ToolToggle* _measuringTool;
	ToolToggle* _screenshotTool;
	//ToolRadialButton* _prevButton = nullptr;

private:
	bool _movable;
	int _index;
	cvr::SceneObject* _container = nullptr;
};

class NewVolumeMenu : public UICallback {
public:
	NewVolumeMenu(cvr::SceneObject* scene, VolumeGroup* volume, bool movable = true) : _scene(scene), _volume(volume), _movable(movable) {}
	~NewVolumeMenu();

	void init();

	virtual void uiCallback(UICallbackCaller * item);

	cvr::SceneObject* getSceneObject() { return _so; }
	std::string transferFunction;

	inline void setOrganColPicker(std::string organName)
	{
		_maskContainer->addChild(_colorMenu->getRoot());
		_cpHLabel->setText(organName);
	}

	void colorButtonPress(cvr::UIQuadElement* button, organRGB organRGB, std::string organName, VisibilityToggle* organEye);

protected:
	cvr::SceneObject* _scene;
	VolumeGroup* _volume;
	TentWindow* _tentWindow;

	cvr::UIPopup* _menu = nullptr;
	cvr::UIPopup* _maskMenu = nullptr;
	cvr::UIPopup* _colorMenu = nullptr;

	cvr::UIQuadElement* _maskBknd;

	CallbackButton* _horizontalflip;
	CallbackButton* _verticalflip;
	CallbackButton* _depthflip;
	CallbackButton* _addTriangle;
	
	cvr::UIList* _triangleList;
	cvr::UIList* _mainMaskList;

	CallbackSlider* _density;
	CallbackSlider* _contrastBottom;
	CallbackSlider* _contrastTop;
	CallbackSlider* _brightness;

	cvr::UIText* _contrastValueLabel;
	cvr::UIText* _brightValueLabel;

	VisibilityToggle* _organs;
	VisibilityToggle* _colon;
	VisibilityToggle* _kidney;
	VisibilityToggle* _bladder;
	VisibilityToggle* _spleen;

	cvr::UIQuadElement* _bodyColorButton;
	cvr::UIQuadElement* _colonColorButton;
	cvr::UIQuadElement* _kidneyColorButton;
	cvr::UIQuadElement* _spleenColorButton;
	cvr::UIQuadElement* _bladderColorButton;

	CallbackButton* _bodyColCallback;
	CallbackButton* _colonColCallback;
	CallbackButton* _kidneyColCallback;
	CallbackButton* _bladderColCallback;
	CallbackButton* _spleenColCallback;
	CallbackButton* _exitCPCallback;

	ColorPicker* _cp;
	cvr::UIQuadElement* _cpHeader;
	cvr::UIText* _cpHLabel;
	CallbackRadial* _transferFunction;
	ShaderQuad* _colorDisplay;
	ShaderQuad* _opacityDisplay;
	ShaderQuad* _opacityColorDisplay;
	cvr::UIRadialButton* _blacknwhite;
	cvr::UIRadialButton* _rainbow;


	osg::Vec3 _bodyCol;
	osg::Vec3 _colonCol;
	osg::Vec3 _kidneyCol;
	osg::Vec3 _bladderCol;
	osg::Vec3 _spleenCol;


	int _triangleCount;
	osg::Vec3 triangleColors[6] = { UI_BLUE_COLOR, UI_GREEN_COLOR, UI_PURPLE_COLOR, UI_RED_COLOR, UI_PINK_COLOR, UI_YELLOW_COLOR};
private:
	ToolMenu* _toolMenu = nullptr;
	bool _movable;
	cvr::SceneObject* _container = nullptr;
	cvr::SceneObject* _maskContainer = nullptr;
	cvr::SceneObject* _so = nullptr;
};

#endif
