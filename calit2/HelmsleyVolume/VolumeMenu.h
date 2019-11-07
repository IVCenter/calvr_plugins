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

	cvr::MenuCheckbox* adaptiveQuality;
	cvr::MenuCheckbox* highlightColon;
	cvr::MenuCheckbox* organsOnly;

	cvr::MenuList* colorFunction;
};


class NewVolumeMenu : public UICallback {
public:
	NewVolumeMenu(cvr::SceneObject* scene, VolumeGroup* volume) : _scene(scene), _volume(volume) {}
	~NewVolumeMenu();

	void init();

	virtual void uiCallback(UICallbackCaller * item);

	std::string transferFunction;

protected:
	cvr::SceneObject* _scene;
	VolumeGroup* _volume;

	cvr::UIPopup* _menu = nullptr;
	cvr::UIPopup* _maskMenu = nullptr;

	CallbackButton* _horizontalflip;
	CallbackButton* _verticalflip;
	CallbackButton* _depthflip;

	CallbackSlider* _density;
	CallbackSlider* _contrastBottom;
	CallbackSlider* _contrastTop;


	VisibilityToggle* _organs;
	VisibilityToggle* _colon;
	VisibilityToggle* _kidney;
	VisibilityToggle* _bladder;
	VisibilityToggle* _spleen;

	CallbackRadial* _transferFunction;
	ShaderQuad* _colorDisplay;
	cvr::UIRadialButton* _blacknwhite;
	cvr::UIRadialButton* _rainbow;
};

class ToolMenu : public UICallback {
public:
	ToolMenu();
	~ToolMenu();


	virtual void uiCallback(UICallbackCaller * item);

protected:

	cvr::UIPopup* _menu = nullptr;

	CallbackRadial* _tool;

	ToolRadialButton* _cuttingPlane;
	ToolRadialButton* _measuringTool;
	ToolToggle* _screenshotTool;
	ToolRadialButton* _prevButton = nullptr;
};

#endif
