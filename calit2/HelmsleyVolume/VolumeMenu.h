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

#include "YamlParser.h"

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
	void setVolume(VolumeGroup* volume) { _volume = volume; }
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

	YamlParser yp;

	
};


class ToolMenu : public UICallback {
public:
	ToolMenu(int index = 0, bool movable = true, cvr::SceneObject* parent = nullptr);
	~ToolMenu();

	cvr::SceneObject* getContainer()
	{
		return _container;
	}

	ToolToggle* getCenterLineTool() { return  _centerLIneTool; }
	ToolToggle* getCuttingPlaneTool() { return  _cuttingPlane; }

	virtual void uiCallback(UICallbackCaller* item);

protected:

	cvr::UIPopup* _menu = nullptr;

	//CallbackRadial* _tool;

	ToolToggle* _cuttingPlane;
	ToolToggle* _measuringTool;
	ToolToggle* _screenshotTool;
	ToolToggle* _centerLIneTool;
	
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
	void setNewVolume(VolumeGroup* volume) { _volume = volume; }

	std::string _transferFunction;

	inline void setOrganColPicker(std::string organName)
	{
		_maskContainer->addChild(_colorMenu->getRoot());
		_cpHLabel->setText(organName);
	}

	void colorButtonPress(cvr::UIQuadElement* button, organRGB organRGB, std::string organName, VisibilityToggle* organEye);
	cvr::UIList* addPresets(cvr::UIQuadElement* bknd);

protected:
	cvr::SceneObject* _scene;
	VolumeGroup* _volume;
	TentWindow* _tentWindow;
	TentWindowOnly* _tentWindowOnly;

	cvr::UIPopup* _menu = nullptr;
	cvr::UIPopup* _maskMenu = nullptr;
	cvr::UIPopup* _colorMenu = nullptr;
	cvr::UIPopup* _presetPopup = nullptr;
	cvr::UIPopup* _tentMenu = nullptr;
	cvr::UIQuadElement* _maskBknd;
	cvr::UIQuadElement* _presetBknd;

	CallbackButton* _horizontalflip;
	CallbackButton* _verticalflip;
	CallbackButton* _depthflip;
	CallbackButton* _addTriangle;
	CallbackButton* _addPreset;
	CallbackButton* _loadPreset;
	cvr::UIList* _mainMaskList;
	cvr::UIList* _presetUIList;

	CallbackSlider* _contrastBottom;
	CallbackSlider* _contrastTop;
	CallbackSlider* _brightness;

	ColorSlider* _colorSliderLeft;
	ColorSlider* _colorSliderRight;
	std::vector<ColorSlider*> _colSliderList = { _colorSliderLeft, _colorSliderRight };
	
	cvr::UIText* _contrastValueLabel;
	cvr::UIText* _brightValueLabel;

	VisibilityToggle* _organs;
	VisibilityToggle* _colon;
	VisibilityToggle* _kidney;
	VisibilityToggle* _bladder;
	VisibilityToggle* _illeum;
	VisibilityToggle* _spleen;
	VisibilityToggle* _aorta;

	cvr::UIQuadElement* _bodyColorButton;
	cvr::UIQuadElement* _colonColorButton;
	cvr::UIQuadElement* _kidneyColorButton;
	cvr::UIQuadElement* _spleenColorButton;
	cvr::UIQuadElement* _bladderColorButton;
	cvr::UIQuadElement* _aortaColorButton;
	cvr::UIQuadElement* _illeumColorButton;

	CallbackButton* _bodyColCallback;
	CallbackButton* _colonColCallback;
	CallbackButton* _kidneyColCallback;
	CallbackButton* _bladderColCallback;
	CallbackButton* _spleenColCallback;
	CallbackButton* _aortaColCallback;
	CallbackButton* _illeumColCallback;
	CallbackButton* _exitCPCallback;

	ColorPicker* _cp;
	cvr::UIQuadElement* _cpHeader;
	cvr::UIText* _cpHLabel;
	CallbackRadial* _transferFunctionRadial;
	ShaderQuad* _colorDisplay;
	ShaderQuad* _opacityDisplay;
	ShaderQuad* _opacityColorDisplay;
	cvr::UIRadialButton* _blacknwhite;
	cvr::UIRadialButton* _rainbow;
	cvr::UIRadialButton* _custom;
	void useTransferFunction(int tfID);
	void setContrastValues(float contrastLow, float contrastHigh, float brightness);

	osg::Vec3 _bodyCol;
	osg::Vec3 _colonCol;
	osg::Vec3 _kidneyCol;
	osg::Vec3 _bladderCol;
	osg::Vec3 _spleenCol;
	osg::Vec3 _aortaCol;
	osg::Vec3 _illeumCol;

	CallbackButton* _Triangle0;
	CallbackButton* _Triangle1;
	CallbackButton* _Triangle2;
	CallbackButton* _Triangle3;
	CallbackButton* _Triangle4;
	CallbackButton* _Triangle5;

	std::vector<CallbackButton*> _triangleCallbacks = {_Triangle0, _Triangle1, _Triangle2, _Triangle3, _Triangle4, _Triangle5};
	std::vector<CallbackButton*> _presetCallbacks;

	cvr::UIList* _triangleList;
	int _triangleIndex;
	int _colorSliderIndex;
	osg::Vec3 triangleColors[6] = { UI_BLUE_COLOR, UI_RED_COLOR, UI_YELLOW_COLOR, UI_GREEN_COLOR, UI_PURPLE_COLOR, UI_PINK_COLOR};
	bool checkTriangleCallbacks(UICallbackCaller* item);
	bool checkTriangleVisCallbacks(UICallbackCaller* item);
	bool checkPresetCallbacks(UICallbackCaller* item);
	bool checkColorSliderCallbacks(UICallbackCaller* item);
	void usePreset(std::string filename);
	void clearRegionPreviews();
	Tent* addRegion();
	void savePreset();
	void NewVolumeMenu::upDatePreviewDefines(std::string tf);
private:
	ToolMenu* _toolMenu = nullptr;
	bool _movable;
	cvr::SceneObject* _container = nullptr;
	cvr::SceneObject* _maskContainer = nullptr;
	cvr::SceneObject* _tentWindowContainer = nullptr;
	cvr::SceneObject* _so = nullptr;
};

#endif
