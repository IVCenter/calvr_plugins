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
	~VolumeMenu();
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
	cvr::MenuRangeValueCompact* testScale;
	cvr::MenuRangeValueCompact* maxSteps;
	cvr::MenuRangeValueCompact* sampleDistance;
	 
	cvr::MenuRangeValueCompact* contrastBottom;
	cvr::MenuRangeValueCompact* contrastTop;

	cvr::MenuRangeValueCompact* opacityMult;
	cvr::MenuRangeValueCompact* opacityCenter;
	cvr::MenuRangeValueCompact* opacityWidth;
	cvr::MenuRangeValueCompact* opacityTop;

	cvr::MenuCheckbox* adaptiveQuality;
	cvr::MenuCheckbox* test;
	cvr::MenuCheckbox* edgeDetectBox;
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
	ToolToggle* getMeasuringTool() { return  _measuringTool; }
	ToolToggle* getScreenShotTool() { return  _screenshotTool; }

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

class VolumeMenuUpdate : public osg::NodeCallback
{
public:
	VolumeMenuUpdate(){}

	virtual void operator()(osg::Node* node, osg::NodeVisitor* nv)
	{
		if (_Linked) {
			Link();
		}

	}

	void setLinkOn() {
		_Linked = true;
	}
	void setLinkOff() {
		_Linked = false;
	}

	void setCuttingPlanes(cvr::SceneObject* cuttingPlane, cvr::SceneObject* cuttingPlane2) { _cP1 = cuttingPlane; _cP2 = cuttingPlane2; }

	void Link();


private:

	cvr::SceneObject* _cP1;
	cvr::SceneObject* _cP2;
	bool _Linked = false;
	osg::Vec3 _prevPos = osg::Vec3(0,0,0);
};

class NewVolumeMenu : public UICallback {
public:
	NewVolumeMenu(cvr::SceneObject* scene, VolumeGroup* volume, bool movable = true) : _scene(scene), _volume(volume), _movable(movable) {
		
	}
	~NewVolumeMenu();

	void init();
	

	virtual void uiCallback(UICallbackCaller * item);

	cvr::SceneObject* getSceneObject() { return _so; }
	void setNewVolume(VolumeGroup* volume, int index = -1);
	

	void setSecondVolume(VolumeGroup* v2) {
		_volume2 = v2;
		_volume1 = _volume;
		switchVolumes(1);
		

	}
	void clearVolumes();
	void resetValues();
	std::string _transferFunction;

	inline void setOrganColPicker(std::string organName)
	{
		_maskContainer->addChild(_colorMenu->getRoot());
		_cpHLabel->setText(organName);
	}

	cvr::UIList* addPresets(cvr::UIQuadElement* bknd);
	void setLinkOff() { _updateCallback->setLinkOff(); }
	void toggleSwapOpacity();
	void toggleLinkOpacity(bool turnOn);
protected:
	cvr::SceneObject* _scene;
	VolumeGroup* _volume;
	VolumeGroup* _volume1 = nullptr;
	VolumeGroup* _volume2 = nullptr;
	bool _prevMask = false;

	TentWindow* _tentWindow;
	TentWindowOnly* _tentWindowOnly;

	cvr::UIPopup* _menu = nullptr;
	cvr::UIPopup* _maskMenu = nullptr;
	cvr::UIPopup* _contrastMenu = nullptr;
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
	CallbackButton* _volume1Button;
	CallbackButton* _volume2Button;
	CallbackButton* _linkButton;
	CallbackButton* _swapButton;
	bool _swapOpacity = false;

	cvr::UIList* _mainMaskList;
	cvr::UIList* _presetUIList;

	CallbackSlider* _contrastBottom;
	CallbackSlider* _contrastTop;
	CallbackSlider* _brightness;

	CallbackSlider* _trueContrast;
	CallbackSlider* _contrastCenter;

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
	VisibilityToggle* _vein;

	cvr::UIQuadElement* _bodyColorButton;
	cvr::UIQuadElement* _colonColorButton;
	cvr::UIQuadElement* _kidneyColorButton;
	cvr::UIQuadElement* _spleenColorButton;
	cvr::UIQuadElement* _bladderColorButton;
	cvr::UIQuadElement* _aortaColorButton;
	cvr::UIQuadElement* _illeumColorButton;

	
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
	void switchVolumes(int index = -1);
	void linkVolumes();
	void saveValues(VolumeGroup* vg);
	void useTransferFunction(int tfID);
	void setContrastValues(float contrastLow, float contrastHigh, float brightness);
	void fillFromVolume(VolumeGroup* vg);


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
	int _volumeIndex = 0;
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
	cvr::SceneObject* _contrastContainer = nullptr;
	cvr::SceneObject* _tentWindowContainer = nullptr;
	cvr::SceneObject* _so = nullptr;

	VolumeMenuUpdate* _updateCallback;
};

#endif
