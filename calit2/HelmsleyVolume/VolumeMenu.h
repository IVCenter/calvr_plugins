#ifndef HELMSLEY_VOLUME_MENU_H
#define HELMSLEY_VOLUME_MENU_H

#include <future>
#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/SceneObject.h>

#include <cvrMenu/MenuItem.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/MenuRangeValueCompact.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuList.h>

#include "CurvedMenu.h"
#include "UIExtensions.h"

#include <cvrMenu/NewUI/UIPopup.h>

#include "VolumeGroup.h"

#include <yaml-cpp/yaml.h>




enum ToolIndex {
	UNDEFINED0,
	CUTTINGPLANE,
	UNDEFINED1
};

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


	
};



class ToolMenu : public UICallback {
	


public:
	enum class TOOLID {
		CUTTINGPLANE,
		CLAHE,
		MARCHINGCUBES,
		CENTERLINE,
		SCREENSHOT,
		HISTOGRAM,
		RULER,
		MASKMENU,
		TFMENU
	};

	ToolMenu(int index = 0, bool movable = true, cvr::SceneObject* parent = nullptr);
	~ToolMenu();

	cvr::SceneObject* getContainer()
	{
		return _container;
	}

	ToolToggle* getCenterLineTool() { return  _centerLIneTool; }
	CurvedQuad* getCuttingPlaneTool() {
		auto curvedMenuItems = _curvedMenu->getCurvedMenuItems();
		int toolIndex = 0;
		for (std::vector<CurvedQuad*>::iterator it = curvedMenuItems.begin(); it != curvedMenuItems.end(); ++it) {
			if (toolIndex == CUTTINGPLANE)
				return (*it);
		toolIndex++;
		}
		return nullptr;
	}
	ToolToggle* getMeasuringTool() { return  _measuringTool; }
	ToolToggle* getScreenShotTool() { return  _screenshotTool; }

	void toggleOtherMenus(TOOLID currentActiveTool);

	virtual void uiCallback(UICallbackCaller* item);
	std::vector<CurvedQuad*> getCurvedMenuItems();
protected:

	cvr::UIPopup* _menu = nullptr;
	CurvedMenu* _curvedMenu = nullptr;
	//CallbackRadial* _tool;

	ToolToggle* _cuttingPlane;
	ToolToggle* _measuringTool;
	ToolToggle* _screenshotTool;
	ToolToggle* _centerLIneTool;
	ToolToggle* _histogramTool;
	ToolToggle* _claheTool;
	
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
		traverse(node, nv);
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

class UIMenuUpdate;
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
	void toggleHistogram(bool on);
	void toggleClaheTools(bool on);
	void toggleMaskMenu(bool on);
	void toggleTFUI(bool on);
	void toggleMCRender(bool on);
	void removeAllToolMenus();

	void updateMCUI(bool on);
 	void saveYamlForCinematic();
protected:
	cvr::SceneObject* _scene;
	VolumeGroup* _volume;
	VolumeGroup* _volume1 = nullptr;
	VolumeGroup* _volume2 = nullptr;
	bool _prevMask = false;

	TentWindow* _tentWindow;
	TentWindowOnly* _tentWindowOnly;
	HistQuad* _histQuad;

	cvr::UIPopup* _menu = nullptr;
	cvr::UIPopup* _maskMenu = nullptr;
	cvr::UIPopup* _contrastMenu = nullptr;
	cvr::UIPopup* _colorMenu = nullptr;
	cvr::UIPopup* _presetPopup = nullptr;
	cvr::UIPopup* _tentMenu = nullptr;

	cvr::UIPopup* _claheMenu = nullptr;
	cvr::UIPopup* _marchingCubesMenu = nullptr;
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
	cvr::UIList* _volumeList;

	CallbackSlider* _contrastBottom;
	CallbackSlider* _contrastTop;
	CallbackSlider* _brightness;

	CallbackSlider* _numBinsSlider;
	CallbackSlider* _clipLimitSlider;
	CallbackSlider* _claheResSlider;
	CallbackButton* _genClaheButton;
	CallbackButton* _useClaheButton;

	//Marchinc Cubes UI
	CallbackButton* _UseMarchingCubesButton;
	CallbackButton* _GenMarchCubesButton;
	CallbackButton* _PrintStlCallbackButton;
	FullButton* _printStlButton;

	//Contrast UI
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
	cvr::UIText* _numBinsLabel;
	cvr::UIText* _clipLimitLabel;
	cvr::UIText* _claheResLabel;
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
	osg::Vec3 triangleColors[6] = { UI_BLUE_COLOR, UI_RED_HOVER_COLOR, UI_YELLOW_COLOR, UI_GREEN_COLOR, UI_PURPLE_COLOR, UI_PINK_COLOR};
	bool checkTriangleCallbacks(UICallbackCaller* item);
	bool checkTriangleVisCallbacks(UICallbackCaller* item);
	bool checkPresetCallbacks(UICallbackCaller* item);
	bool checkColorSliderCallbacks(UICallbackCaller* item);
	void usePreset(std::string filename);
	void clearRegionPreviews();
	Tent* addRegion();
	void savePreset();
	void NewVolumeMenu::upDatePreviewDefines(std::string tf);

	static void runCinematicThread(std::string datasetPath, std::string configPath);
	std::vector<std::future<void>> _futures;

	
private:
	ToolMenu* _toolMenu = nullptr;
	bool _movable;
	cvr::SceneObject* _container = nullptr;
	cvr::SceneObject* _maskContainer = nullptr;
	cvr::SceneObject* _contrastContainer = nullptr;
	cvr::SceneObject* _tentWindowContainer = nullptr;
	cvr::SceneObject* _toolContainer = nullptr;
	cvr::SceneObject* _so = nullptr;

	VolumeMenuUpdate* _updateCallback;
	UIMenuUpdate* _uiMenuUpdate;


};


class UIMenuUpdate : public osg::NodeCallback
{
public:
	UIMenuUpdate(VolumeGroup* vg, NewVolumeMenu* vm) { _vg = vg; _vm = vm; }

	virtual void operator()(osg::Node* node, osg::NodeVisitor* nv)
	{
		if (_vg->_UIDirty) {
			if (_vg->_mcIsReady) {
				_vm->updateMCUI(false);
				_vg->_UIDirty = false;
			}
		}
		traverse(node, nv);
	}




private:
	VolumeGroup* _vg = nullptr;
	NewVolumeMenu* _vm = nullptr;
};

#endif
