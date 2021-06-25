#include "VolumeMenu.h"
#include "HelmsleyVolume.h"
#include "Utils.h"
#include "cvrMenu/MenuManager.h"
#include "cvrConfig/ConfigManager.h"
#include <windows.h>
#include <tchar.h>


using namespace cvr;

void VolumeMenu::init()
{
	scale = new MenuRangeValueCompact("Scale", 0.1, 100.0, 1.0, true);
	scale->setCallback(this);
	_scene->addMenuItem(scale);
	
	maxSteps = new MenuRangeValueCompact("Max Steps", 0.01, 1.0, 0.98, true);
	maxSteps->setCallback(this);
	_scene->addMenuItem(maxSteps);

	sampleDistance = new MenuRangeValueCompact("SampleDistance", .0001, 0.01, .00150f, true);
	sampleDistance->setCallback(this);
	_scene->addMenuItem(sampleDistance);
	
	adaptiveQuality = new MenuCheckbox("Adaptive Quality", false);
	adaptiveQuality->setCallback(this);
	_scene->addMenuItem(adaptiveQuality);

	edgeDetectBox = new MenuCheckbox("Edges", false);
	edgeDetectBox->setCallback(this);
	_scene->addMenuItem(edgeDetectBox);

	attnMapBox = new MenuCheckbox("Attention Masks Menu", false);
	attnMapBox->setCallback(this);
	_scene->addMenuItem(attnMapBox);

	fireColorMap = new MenuCheckbox("Fire Color Map", false);
	fireColorMap->setCallback(this);
	_scene->addMenuItem(fireColorMap);

	cetColorMap = new MenuCheckbox("CET Color Map", false);
	cetColorMap->setCallback(this);
	_scene->addMenuItem(cetColorMap);

	
}

void VolumeMenu::menuCallback(cvr::MenuItem * item)
{
	if (item == sampleDistance)
	{
		_volume->_StepSize->set(sampleDistance->getValue());
	}
	else if (item == scale)
	{
		_scene->setScale(scale->getValue());
	}
	else if (item == maxSteps)
	{
		_volume->_maxSteps->set(maxSteps->getValue());
	}
	else if (item == adaptiveQuality)
	{
		_volume->getDrawable()->getOrCreateStateSet()->setDefine("VR_ADAPTIVE_QUALITY", adaptiveQuality->getValue());
	}
	else if (item == edgeDetectBox)
	{
		_volume->getCompute()->getOrCreateStateSet()->setDefine("EDGE_DETECTION", edgeDetectBox->getValue());
		_volume->setDirtyAll();
	}

	else if (item == attnMapBox)
	{
		/*_volume->getCompute()->getOrCreateStateSet()->setDefine("ATTN_MAPS", attnMapBox->getValue());
		_volume->setDirtyAll();*/
		HelmsleyVolume::instance()->toggleAttnMapsTools(attnMapBox->getValue());
	}
	else if (item == fireColorMap)
	{ 
		std::string transferFunction = "useLut(ra.r,0);";
		_volume->getCompute()->getOrCreateStateSet()->setDefine("COLOR_FUNCTION", transferFunction, fireColorMap->getValue());
		cetColorMap->setValue(false);
		_volume->setDirtyAll();
	}
	else if (item == cetColorMap)
	{ 
		fireColorMap->setValue(false);
		std::string transferFunction = "useLut(ra.r,1);";
		_volume->getCompute()->getOrCreateStateSet()->setDefine("COLOR_FUNCTION", transferFunction, cetColorMap->getValue());
		_volume->setDirtyAll();
	}

	
}

VolumeMenu::~VolumeMenu() {
	if (_scene)
	{
		delete _scene;
	}
}

NewVolumeMenu::~NewVolumeMenu()
{
	delete _toolMenu;

	_menu->setActive(false, false);
	MenuManager::instance()->removeMenuSystem(_menu);
	delete _menu;

	if (_maskMenu)
	{
		_maskMenu->setActive(false, false);
		MenuManager::instance()->removeMenuSystem(_maskMenu);
		delete _maskMenu;
	}
	if (_contrastMenu)
	{
		_contrastMenu->setActive(false, false);
		MenuManager::instance()->removeMenuSystem(_contrastMenu);
		delete _contrastMenu;
	}
	if (_tentMenu)
	{
		_tentMenu->setActive(false, false);
		MenuManager::instance()->removeMenuSystem(_tentMenu);
		delete _tentMenu;
	}
	if (_claheMenu)
	{
		_claheMenu->setActive(false, false);
		MenuManager::instance()->removeMenuSystem(_claheMenu);
		delete _claheMenu;
	}
	if (_attnMapsMenu)
	{
		_attnMapsMenu->setActive(false, false);
		MenuManager::instance()->removeMenuSystem(_attnMapsMenu);
		delete _attnMapsMenu;
	}
	if (_marchingCubesMenu)
	{
		_marchingCubesMenu->setActive(false, false);
		MenuManager::instance()->removeMenuSystem(_marchingCubesMenu);
		delete _marchingCubesMenu;
	}
	
	if (_selectionMenu)
	{
		_selectionMenu->setActive(false, false);
		MenuManager::instance()->removeMenuSystem(_selectionMenu);
		delete _selectionMenu;
	}

	if (_container)
	{
		delete _container;
	}
	if (_maskContainer)
	{
		delete _maskContainer;
	}
	if (_contrastContainer)
	{
		delete _contrastContainer;
	}
	if (_tentWindowContainer)
	{
		delete _tentWindowContainer;
	}
	if (_toolContainer)
	{
		delete _toolContainer;
	}

	if(_cp->_parent == nullptr){
		delete _cp;
	}
	
	

	delete _so;

	delete _volume;

	

}

void NewVolumeMenu::init()
{
#pragma region VolumeOptions
	_so = new SceneObject("VolumeMenu", false, false, false, false, false);
	PluginHelper::registerSceneObject(_so, "HelmsleyVolume");
	_so->attachToScene();
#ifdef WITH_OPENVR
	_so->getRoot()->addUpdateCallback(new FollowSceneObjectLerp(_scene, 0.2f, _so));
	_so->getRoot()->addUpdateCallback(new PointAtHeadLerp(0.2f, _so));
#endif
	_updateCallback = new VolumeMenuUpdate();
	_uiMenuUpdate = new UIMenuUpdate(_volume, this, _so);
	
	_so->getRoot()->addUpdateCallback(_updateCallback);
	_so->getRoot()->addUpdateCallback(_uiMenuUpdate);

	osg::Vec3 volumePos = ConfigManager::getVec3("Plugin.HelmsleyVolume.Orientation.Volume.Position", osg::Vec3(-413, 1052, 885));
	_so->setPosition(volumePos);

	_menu = new UIPopup();
	
	UIQuadElement* bknd = new UIQuadElement(UI_BACKGROUND_COLOR);
	_menu->addChild(bknd);
	_menu->setPosition(osg::Vec3(1000, 450, 1300));

	_menu->getRootElement()->setAbsoluteSize(ConfigManager::getVec3("Plugin.HelmsleyVolume.Orientation.OptionsMenu.Scale", osg::Vec3(1000, 1, 600)));
	


	
	_cp = new ColorPicker();
	_tentWindowOnly = new TentWindowOnly();
	//_histQuad = new HistQuad();
	 _tentWindow = new TentWindow(_tentWindowOnly);
	_menu->addChild(_tentWindow);
	_tentWindow->setPercentSize(osg::Vec3(1, 1, .75));
	_tentWindow->setPercentPos(osg::Vec3(0, 0, -1));
	_tentWindow->setVolume(_volume);
	//_histQuad->setVolume(_volume);
	//_histQuad->setMax(_volume->getHistMax());
	//_histQuad->setBB(_volume->getHistBB());
	//


	UIList* list = new UIList(UIList::TOP_TO_BOTTOM, UIList::CONTINUE);
	list->setPercentPos(osg::Vec3(0, 0, -0.2));
	list->setPercentSize(osg::Vec3(1, 1, 0.8));
	list->setAbsoluteSpacing(10);
	bknd->addChild(list);

	

	UIText* label = new UIText("Volume Options", 50.0f, osgText::TextBase::CENTER_CENTER);
	label->setPercentSize(osg::Vec3(1, 1, 0.2));
	bknd->addChild(label);


	UIList* fliplist = new UIList(UIList::LEFT_TO_RIGHT, UIList::CUT);

 	label = new UIText("Flip Sagittal", 30.0f, osgText::TextBase::CENTER_CENTER);
	label->setColor(UI_WHITE_COLOR);
	label->setPercentPos(osg::Vec3(0.0, -1.0, 0.0));
	UIQuadElement* labelbknd = new UIQuadElement(UI_RED_HOVER_COLOR_VEC4);
	labelbknd->setPercentPos(osg::Vec3(.1, 1, -.15));
	labelbknd->setPercentSize(osg::Vec3(.8, 1, .65));
 	labelbknd->setRounding(0, .2);
	labelbknd->setTransparent(true);
	label->addChild(labelbknd);
	_horiFlipButton = std::make_unique<HoverButton>(labelbknd, UI_RED_HOVER_COLOR_VEC4, UI_RED_ACTIVE_COLOR);
	_horiFlipButton->setCallback(this);
	_horiFlipButton->addChild(label);

  	label = new UIText("Flip Axial", 30.0f, osgText::TextBase::CENTER_CENTER);
	label->setColor(UI_WHITE_COLOR);
	label->setPercentPos(osg::Vec3(0.0, -1.0, 0.0));
	labelbknd = new UIQuadElement(UI_RED_HOVER_COLOR_VEC4);
	labelbknd->setPercentPos(osg::Vec3(.1, 1, -.15));
	labelbknd->setPercentSize(osg::Vec3(.8, 1, .65));
	labelbknd->setRounding(0, .2);
	labelbknd->setTransparent(true);
	label->addChild(labelbknd);
 	_vertiFlipButton = std::make_unique<HoverButton>(labelbknd, UI_RED_HOVER_COLOR_VEC4, UI_RED_ACTIVE_COLOR);
	_vertiFlipButton->setCallback(this);
	_vertiFlipButton->addChild(label);


  	label = new UIText("Flip Coronal", 30.0f, osgText::TextBase::CENTER_CENTER);
	label->setColor(UI_WHITE_COLOR);
	label->setPercentPos(osg::Vec3(0.0, -1.0, 0.0));
	labelbknd = new UIQuadElement(UI_RED_HOVER_COLOR_VEC4);
	labelbknd->setPercentPos(osg::Vec3(.1, 1, -.15));
	labelbknd->setPercentSize(osg::Vec3(.8, 1, .65));
	labelbknd->setRounding(0, .2);
	labelbknd->setTransparent(true);
	label->addChild(labelbknd);
 	_depthFlipButton = std::make_unique<HoverButton>(labelbknd, UI_RED_HOVER_COLOR_VEC4, UI_RED_ACTIVE_COLOR);
	_depthFlipButton->setCallback(this);
	_depthFlipButton->addChild(label);

	fliplist->addChild(_horiFlipButton.get());
	fliplist->addChild(_vertiFlipButton.get());
	fliplist->addChild(_depthFlipButton.get());
	list->addChild(fliplist);

	label = new UIText("Density", 30.0f, osgText::TextBase::LEFT_CENTER);
	label->setPercentPos(osg::Vec3(0.1, 0, 0));

	UIList* valueList = new UIList(UIList::LEFT_TO_RIGHT, UIList::CUT);
	label = new UIText("Low Range", 40.0f, osgText::TextBase::LEFT_CENTER);
	label->setPercentPos(osg::Vec3(0.1, 0, 0));
	valueList->addChild(label);
	_contrastValueLabel = new UIText("Low 0.00 / High 1.00", 35.0f, osgText::TextBase::RIGHT_CENTER);
	_contrastValueLabel->setPercentPos(osg::Vec3(-0.1, 0, 0));
	valueList->addChild(_contrastValueLabel);
	list->addChild(valueList);

	_contrastBottom = new CallbackSlider();
	_contrastBottom->setPercentPos(osg::Vec3(0.025, 0, 0.05));
	_contrastBottom->setPercentSize(osg::Vec3(0.95, 1, 0.9));
	_contrastBottom->handle->setAbsoluteSize(osg::Vec3(20, 0, 0));
	_contrastBottom->handle->setAbsolutePos(osg::Vec3(-10, -0.2f, 0));
	_contrastBottom->handle->setPercentSize(osg::Vec3(0, 1, 1));
	_contrastBottom->setCallback(this);
	_contrastBottom->setPercent(0);

	label = new UIText("High Range", 40.0f, osgText::TextBase::LEFT_CENTER);
	label->setPercentPos(osg::Vec3(0.1, 0, 0));

	_contrastTop = new CallbackSlider();
	_contrastTop->setPercentPos(osg::Vec3(0.025, 0, 0.05));
	_contrastTop->setPercentSize(osg::Vec3(0.95, 1, 0.9));
	_contrastTop->handle->setAbsoluteSize(osg::Vec3(20, 0, 0));
	_contrastTop->handle->setAbsolutePos(osg::Vec3(-10, -0.2f, 0));
	_contrastTop->handle->setPercentSize(osg::Vec3(0, 1, 1));
	_contrastTop->setCallback(this);
	_contrastTop->setPercent(1);

	

	list->addChild(_contrastBottom);
	list->addChild(label);
	list->addChild(_contrastTop);
	


	

	UIList* list2 = new UIList(UIList::LEFT_TO_RIGHT, UIList::CUT);
	
	
	_transferFunctionRadial = new CallbackRadial();
	_transferFunctionRadial->allowNoneSelected(false);
	_blacknwhite = new UIRadialButton(_transferFunctionRadial);
	_rainbow = new UIRadialButton(_transferFunctionRadial);
	_custom = new UIRadialButton(_transferFunctionRadial);


	_transferFunctionRadial->setCurrent(0);
	_transferFunctionRadial->setCallback(this);

	_colorDisplay = new ShaderQuad();
	
	std::string vert = HelmsleyVolume::loadShaderFile("transferFunction.vert");
	std::string frag = HelmsleyVolume::loadShaderFile("transferFunction.frag");
	osg::Program* p = new osg::Program;
	p->setName("TransferFunction");
	p->addShader(new osg::Shader(osg::Shader::VERTEX, vert));
	p->addShader(new osg::Shader(osg::Shader::FRAGMENT, frag));
	_colorDisplay->setProgram(p);
	_colorDisplay->addUniform(_volume->_computeUniforms["ContrastBottom"]);
	_colorDisplay->addUniform(_volume->_computeUniforms["ContrastTop"]);
	_colorDisplay->addUniform(_volume->_computeUniforms["leftColor"]);
	_colorDisplay->addUniform(_volume->_computeUniforms["rightColor"]);
	_colorDisplay->setPercentSize(osg::Vec3(1.0, 1.0, 0.5));
	_colorDisplay->getGeode()->getOrCreateStateSet()->setRenderBinDetails(3, "RenderBin");

	_opacityDisplay = new ShaderQuad();
	_opacityDisplay->getGeode()->getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::ON);
	frag = HelmsleyVolume::loadShaderFile("transferFunctionOpacity.frag");
	p = new osg::Program;
	p->setName("TransferFunctionOpacity");
	p->addShader(new osg::Shader(osg::Shader::VERTEX, vert));
	p->addShader(new osg::Shader(osg::Shader::FRAGMENT, frag));
	_opacityDisplay->setProgram(p);
	_opacityDisplay->addUniform(_volume->_computeUniforms["OpacityCenter"]);
	_opacityDisplay->addUniform(_volume->_computeUniforms["OpacityWidth"]);
	_opacityDisplay->addUniform(_volume->_computeUniforms["OpacityTopWidth"]);
	_opacityDisplay->addUniform(_volume->_computeUniforms["OpacityMult"]);
	_opacityDisplay->addUniform(_volume->_computeUniforms["Lowest"]);
	_opacityDisplay->addUniform(_volume->_computeUniforms["ContrastBottom"]);
	_opacityDisplay->addUniform(_volume->_computeUniforms["ContrastTop"]);
	_opacityDisplay->addUniform(_volume->_computeUniforms["TriangleCount"]);
	_opacityDisplay->addUniform(_volume->_computeUniforms["leftColor"]);
	_opacityDisplay->addUniform(_volume->_computeUniforms["rightColor"]);
	_opacityDisplay->setPercentSize(osg::Vec3(1.0, 1.0, 0.5));
	_opacityDisplay->setPercentPos(osg::Vec3(0.0, 0.0, 1.0));
	_opacityDisplay->getGeode()->getOrCreateStateSet()->setRenderBinDetails(3, "RenderBin");

	_opacityColorDisplay = new ShaderQuad();
	_opacityColorDisplay->getGeode()->getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::ON);
	frag = HelmsleyVolume::loadShaderFile("transferFunctionColorOpacity.frag");
	p = new osg::Program;
	p->setName("TransferFunctionColorOpacity");
	p->addShader(new osg::Shader(osg::Shader::VERTEX, vert));
	p->addShader(new osg::Shader(osg::Shader::FRAGMENT, frag));
	_opacityColorDisplay->setProgram(p);
	_opacityColorDisplay->addUniform(_volume->_computeUniforms["OpacityCenter"]);
	_opacityColorDisplay->addUniform(_volume->_computeUniforms["OpacityWidth"]);
	_opacityColorDisplay->addUniform(_volume->_computeUniforms["OpacityTopWidth"]);
	_opacityColorDisplay->addUniform(_volume->_computeUniforms["OpacityMult"]);
	_opacityColorDisplay->addUniform(_volume->_computeUniforms["Lowest"]);
	_opacityColorDisplay->addUniform(_volume->_computeUniforms["ContrastBottom"]);
	_opacityColorDisplay->addUniform(_volume->_computeUniforms["ContrastTop"]);
	_opacityColorDisplay->addUniform(_volume->_computeUniforms["TriangleCount"]);
	_opacityColorDisplay->addUniform(_volume->_computeUniforms["leftColor"]);
	_opacityColorDisplay->addUniform(_volume->_computeUniforms["rightColor"]);
	_opacityColorDisplay->setPercentSize(osg::Vec3(1.0, 1.0, 0.5));

	_opacityColorDisplay->setPercentPos(osg::Vec3(0.0, 0.0, 0.5));
	_opacityColorDisplay->getGeode()->getOrCreateStateSet()->setRenderBinDetails(3, "RenderBin");
	
	
	
	UIText* bnw = new UIText("Grayscale", 40.0f, osgText::TextBase::CENTER_CENTER);
	bnw->setColor(UI_WHITE_COLOR);
	bnw->setPercentPos(osg::Vec3(0.0, -1.0, 0.0));
	labelbknd = new UIQuadElement(UI_BLUE_COLOR2);
	labelbknd->setPercentPos(osg::Vec3(.1, 1, -.15));
	labelbknd->setPercentSize(osg::Vec3(.8, 1, .65));
	labelbknd->setRounding(0, .2);
	labelbknd->setTransparent(true);
	labelbknd->setBorderColor(UI_BLUE_COLOR2);
	labelbknd->setBorderSize(.1f);
	bnw->addChild(labelbknd);

	UIText* rnbw = new UIText("Rainbow", 40.0f, osgText::TextBase::CENTER_CENTER);
	rnbw->setColor(UI_WHITE_COLOR);
	rnbw->setPercentPos(osg::Vec3(0.0, -1.0, 0.0));
	labelbknd = new UIQuadElement(UI_BACKGROUND_COLOR);
	labelbknd->setPercentPos(osg::Vec3(.1, 1, -.15));
	labelbknd->setPercentSize(osg::Vec3(.8, 1, .65));
	labelbknd->setRounding(0, .2);
	labelbknd->setTransparent(true);
	labelbknd->setBorderColor(UI_BLUE_COLOR2);
	labelbknd->setBorderSize(.1f);
	rnbw->addChild(labelbknd);

	UIText* cstm = new UIText("Custom", 40.0f, osgText::TextBase::CENTER_CENTER);
	cstm->setColor(UI_WHITE_COLOR);
	cstm->setPercentPos(osg::Vec3(0.0, -1.0, 0.0));
	labelbknd = new UIQuadElement(UI_BACKGROUND_COLOR);
	labelbknd->setPercentPos(osg::Vec3(.1, 1, -.15));
	labelbknd->setPercentSize(osg::Vec3(.8, 1, .65));
	labelbknd->setRounding(0, .2);
	labelbknd->setTransparent(true);
	labelbknd->setBorderColor(UI_BLUE_COLOR2);
	labelbknd->setBorderSize(.1f);
	cstm->addChild(labelbknd);


	

	_blacknwhite->addChild(bnw);
	_rainbow->addChild(rnbw);
	_custom->addChild(cstm);


	list2->addChild(_blacknwhite);
	list2->addChild(_rainbow);
	list2->addChild(_custom);

	list->addChild(list2);
	list->setAbsoluteSpacing(0.0);
	_tentMenu = new UIPopup();
	_claheMenu = new UIPopup();
	popupMenus.push_back(_claheMenu);
	_attnMapsMenu = new UIPopup();
	popupMenus.push_back(_attnMapsMenu);



	_tentWindowOnly->setPercentSize(osg::Vec3(1, 1, 3.0));
	_tentWindowOnly->setPercentPos(osg::Vec3(0, 0, 1.50));


	UIList* gradientList = new UIList(UIList::TOP_TO_BOTTOM, UIList::CONTINUE);
	gradientList->addChild(_colorDisplay);
	gradientList->addChild(_opacityColorDisplay);
	gradientList->addChild(_opacityDisplay);
	gradientList->addChild(_tentWindowOnly);


	
	UIQuadElement* gradientBknd = new UIQuadElement(UI_BACKGROUND_COLOR);
	gradientBknd->setPercentSize(osg::Vec3(1, 1, .4));

	
	_colorSliderIndex = 0;
	_colSliderList[_colorSliderIndex] = new ColorSlider(_cp, osg::Vec4(1.0, 0.0, 0.0, 1.0));
	_colSliderList[_colorSliderIndex]->setPercentPos(osg::Vec3(0.0, -1.0, 0.0));
	_colSliderList[_colorSliderIndex]->setAbsoluteSize(osg::Vec3(50.0, 50.0, 50.0));
	_colSliderList[_colorSliderIndex]->setPercentSize(osg::Vec3(0.0, 0.0, 0.0));
	_colSliderList[_colorSliderIndex]->setCallback(this);

	_colorSliderIndex = 1;
	_colSliderList[_colorSliderIndex] = new ColorSlider(_cp, osg::Vec4(1.0, 1.0, 1.0, 1.0));
	_colSliderList[_colorSliderIndex]->setPercentPos(osg::Vec3(1.0, -1.0, 0.0));
	_colSliderList[_colorSliderIndex]->setAbsoluteSize(osg::Vec3(50.0, 50.0, 50.0));
	_colSliderList[_colorSliderIndex]->setPercentSize(osg::Vec3(0.0, 0.0, 0.0));
	_colSliderList[_colorSliderIndex]->setCallback(this);
	_colorSliderIndex = 0;

	_colSliderList[0]->getGeode()->setNodeMask(0);
	_colSliderList[1]->getGeode()->setNodeMask(0);

	_cp->setPercentPos(osg::Vec3(-0.35, 0.0, 0.0));
	_cp->setPercentSize(osg::Vec3(0.3, 0.0, 1.0));
	_cp->setCallback(this);
	


	_tentMenu->addChild(_colSliderList[0]);
	_tentMenu->addChild(_colSliderList[1]);
	_tentMenu->addChild(gradientBknd);
	_tentMenu->addChild(gradientList);


	UIQuadElement* claheBknd = new UIQuadElement(UI_BACKGROUND_COLOR);
	claheBknd->setRounding(0, .2);
	claheBknd->setTransparent(true);
	claheBknd->setBorderColor(UI_BLACK_COLOR);
	claheBknd->setBorderSize(.02);

	_claheMenu->addChild(claheBknd);
	
	
	std::cout << "moveable" << _movable << std::endl;
	if (!_movable)
	{
		_menu->setActive(true, true);
	}
	else {
		_menu->setActive(true, false);
		_container = new SceneObject("VolumeMenu", false, true, false, false, false);
		_so->addChild(_container);
		_container->setShowBounds(true);
		//_container->addChild(_menu->getRoot());
		_menu->getRootElement()->updateElement(osg::Vec3(0, 0, 0), osg::Vec3(0, 0, 0));
		_container->dirtyBounds();

		_toolContainer = new SceneObject("toolmenu", false, true, false, false, false);
		_so->addChild(_toolContainer);
		_toolContainer->setShowBounds(false);
 	}

#pragma endregion
	//>===============================MASKS AND REGIONS==============================<//
	_toolMenu = new ToolMenu(0, true, _so);
	_toolMenu->disableUnavailableButtons(_volume);
	
	_maskMenu = new UIPopup();
	_contrastMenu = new UIPopup();
	UIQuadElement* regionHeaderBknd = new UIQuadElement(UI_BACKGROUND_COLOR);
	_presetBknd = new UIQuadElement(UI_BACKGROUND_COLOR);
	_presetBknd->setBorderSize(.01);
	_presetPopup = new UIPopup;

	UIText* regionLabel = new UIText("Regions", 50.0f, osgText::TextBase::CENTER_CENTER);
	regionLabel->setPercentSize(osg::Vec3(1, 1, 0.2));

	


	label = new UIText(" Add Region ", 35.0f, osgText::TextBase::CENTER_CENTER);
	label->setColor(UI_WHITE_COLOR);
	label->setPercentPos(osg::Vec3(0.0, -1.0, 0.0));
	labelbknd = new UIQuadElement(UI_RED_HOVER_COLOR_VEC4);
	labelbknd->setPercentPos(osg::Vec3(.1, -1, -.15));
	labelbknd->setPercentSize(osg::Vec3(.8, 1, .65));
  	labelbknd->setRounding(0, .2);
	labelbknd->setTransparent(true);
	labelbknd->addChild(label);
	_addTriangleButton = std::make_shared<HoverButton>(labelbknd, UI_RED_HOVER_COLOR_VEC4, UI_RED_ACTIVE_COLOR);
	_addTriangleButton->setCallback(this);
	_addTriangleButton->addChild(labelbknd);

	label = new UIText(" Save Preset ", 35.0f, osgText::TextBase::CENTER_CENTER);
	label->setColor(UI_WHITE_COLOR);
	label->setPercentPos(osg::Vec3(0.0, -1.0, 0.0));
	labelbknd = new UIQuadElement(UI_RED_HOVER_COLOR_VEC4);
	labelbknd->setPercentPos(osg::Vec3(.1, -1, -.15));
	labelbknd->setPercentSize(osg::Vec3(.8, 1, .65));
	labelbknd->setRounding(0, .2);
	labelbknd->setTransparent(true);
	labelbknd->addChild(label);
	_addPresetButton = std::make_shared<HoverButton>(labelbknd, UI_RED_HOVER_COLOR_VEC4, UI_RED_ACTIVE_COLOR);
	_addPresetButton->setCallback(this);
	_addPresetButton->addChild(labelbknd);

	label = new UIText(" Load Preset ", 35.0f, osgText::TextBase::CENTER_CENTER);
	label->setColor(UI_WHITE_COLOR);
	label->setPercentPos(osg::Vec3(0.0, -1.0, 0.0));
	labelbknd = new UIQuadElement(UI_RED_HOVER_COLOR_VEC4);
	labelbknd->setPercentPos(osg::Vec3(.1, -1, -.15));
	labelbknd->setPercentSize(osg::Vec3(.8, 1, .65));
	labelbknd->setRounding(0, .2);
	labelbknd->setTransparent(true);
	labelbknd->addChild(label);
	_loadPresetButton = std::make_shared<HoverButton>(labelbknd, UI_RED_HOVER_COLOR_VEC4, UI_RED_ACTIVE_COLOR);
	_loadPresetButton->setCallback(this);
	_loadPresetButton->addChild(labelbknd);


	regionHeaderBknd->addChild(regionLabel);
	regionHeaderBknd->setPercentSize(osg::Vec3(1.62,1.0,0.7));
	regionHeaderBknd->setBorderSize(.01);
	
	UIText* presetsLabel = new UIText("Presets", 50.0f, osgText::TextBase::CENTER_CENTER);
	presetsLabel->setPercentSize(osg::Vec3(1, 0, 0.2));
	presetsLabel->setPercentPos(osg::Vec3(0, -.1, -.05));

	
	_presetBknd->setPercentSize(osg::Vec3(.3, 0.0, .3));
	_presetBknd->setBorderSize(.01);
	_presetBknd->addChild(presetsLabel);
	_presetPopup->addChild(_presetBknd);

	_presetUIList = addPresets(_presetBknd);
	if (_presetUIList == nullptr) {
		_presetUIList = new UIList(UIList::TOP_TO_BOTTOM, UIList::CONTINUE);
		_presetUIList->setPercentPos(osg::Vec3(0, 0, -.25));
		_presetUIList->setPercentSize(osg::Vec3(1.0, 1.0, .75));
		_presetBknd->addChild(_presetUIList);
	}

	_triangleList = new UIList(UIList::TOP_TO_BOTTOM, UIList::CONTINUE);
	_triangleList->setPercentPos(osg::Vec3(0.0, 0.0, -.25));
	_triangleList->setPercentSize(osg::Vec3(1.0, 1.0, .75));
	
	_triangleIndex = 0;
	_triangleCallbacks[_triangleIndex] = new CallbackButton();
	_triangleCallbacks[_triangleIndex]->setCallback(this);
	_triangleCallbacks[_triangleIndex]->setPercentPos(osg::Vec3(0.20, 0.0, 0.0));

	label = new UIText("Triangle 0", 50.0f, osgText::TextBase::LEFT_TOP); 
	label->setColor(osg::Vec4(triangleColors[_triangleIndex], 1.0));
	label->setPercentPos(osg::Vec3(-.05, -10.0, 0.0));
	label->getTextObject()->setPosition(osg::Vec3(0.0, -5, 0.0));
	_triangleCallbacks[_triangleIndex]->addChild(label);

	label->setPercentSize(osg::Vec3(.25, 0.0, .5));
	VisibilityToggle* vT = new VisibilityToggle("");
	vT->getChild(0)->setPercentSize(osg::Vec3(1, 1, 1));
	vT->getChild(0)->setPercentPos(osg::Vec3(0, 0, 0));
	vT->setCallback(this);
	vT->setPercentPos(osg::Vec3(-0.20, 0.0, 0.1));
	vT->setPercentSize(osg::Vec3(0.185, 0.0, 0.5));
	vT->toggle();


	ShaderQuad* sq = new ShaderQuad();
	frag = HelmsleyVolume::loadShaderFile("shaderQuadPreview.frag");
	vert = HelmsleyVolume::loadShaderFile("transferFunction.vert");
	p = new osg::Program;
	p->setName("trianglePreview");
	p->addShader(new osg::Shader(osg::Shader::VERTEX, vert));
	p->addShader(new osg::Shader(osg::Shader::FRAGMENT, frag));
	sq->setProgram(p);
	sq->setPercentSize(osg::Vec3(0.5, 1.0, 0.4));
	sq->setPercentPos(osg::Vec3(.27, 0.0, 0.1));
	sq->addUniform(_tentWindow->_tWOnly->_tents.at(0)->_centerUniform);
	sq->addUniform(_tentWindow->_tWOnly->_tents.at(0)->_widthUniform);
	sq->addUniform(_volume->_computeUniforms["ContrastBottom"]);
	sq->addUniform(_volume->_computeUniforms["ContrastTop"]);
	sq->addUniform(_volume->_computeUniforms["leftColor"]);
	sq->addUniform(_volume->_computeUniforms["rightColor"]);
	_transferFunction = "vec3(ra.r);";
	sq->setShaderDefine("COLOR_FUNCTION", _transferFunction, osg::StateAttribute::ON);



	_triangleCallbacks[_triangleIndex]->addChild(vT);
	_triangleCallbacks[_triangleIndex]->addChild(sq);


	label->setAbsoluteSize(osg::Vec3(0.0f, 0.0f, 50.0));
	UIList* regionTopList = new UIList(UIList::LEFT_TO_RIGHT, UIList::CONTINUE);
	regionTopList->setPercentPos(osg::Vec3(0.0, 0.0, .25));
	regionTopList->addChild(_addTriangleButton.get());
	regionTopList->addChild(_addPresetButton.get());
	regionTopList->addChild(_loadPresetButton.get());
	_triangleList->addChild(regionTopList);
	_triangleList->addChild(_triangleCallbacks[_triangleIndex]);
	_triangleList->setMaxSize(label->getAbsoluteSize().z()*3);

	regionHeaderBknd->addChild(_triangleList);
	_maskMenu->addChild(regionHeaderBknd);
	
	osg::Quat rot;
	rot.makeRotate(0.707, 0, 0, 1);
	_maskMenu->setRotation(rot);
	_maskMenu->setPosition(osg::Vec3(-1000, -100, 1300));
	_maskMenu->getRootElement()->setAbsoluteSize(osg::Vec3(500, 1, 800));


	UIQuadElement* contrastBknd = new UIQuadElement(UI_BACKGROUND_COLOR);
	_maskMenu->addChild(contrastBknd);
	contrastBknd->setPercentPos(osg::Vec3(0.0, 0.0, -.7));
	contrastBknd->setPercentSize(osg::Vec3(1.62, 1.0, 0.33));
	contrastBknd->setBorderSize(0.02);
 
	UIList* contrastList = new UIList(UIList::TOP_TO_BOTTOM, UIList::CONTINUE);


	label = new UIText("Contrast", 40.0f, osgText::TextBase::LEFT_CENTER);
	label->setPercentPos(osg::Vec3(0.1, 0, 0));
	contrastList->addChild(label);

	_trueContrast = new CallbackSlider();
	_trueContrast->setPercentPos(osg::Vec3(0.025, 0, 0.05));
	_trueContrast->setPercentSize(osg::Vec3(0.95, 1, 0.9));
	_trueContrast->handle->setAbsoluteSize(osg::Vec3(20, 0, 0));
	_trueContrast->handle->setAbsolutePos(osg::Vec3(-10, -0.2f, 0));
	_trueContrast->handle->setPercentSize(osg::Vec3(0, 1, 1));
	_trueContrast->setCallback(this);
	_trueContrast->setPercent(0);
	_trueContrast->setMax(10.0);
	_trueContrast->setMin(1.0);
	contrastList->addChild(_trueContrast);

	label = new UIText("Brightness", 40.0f, osgText::TextBase::LEFT_CENTER);
	label->setPercentPos(osg::Vec3(0.1, 0, 0));
	contrastList->addChild(label);

	_brightness = new CallbackSlider();
	_brightness->setPercentPos(osg::Vec3(0.025, 0, 0.05));
	_brightness->setPercentSize(osg::Vec3(0.95, 1, 0.9));
	_brightness->handle->setAbsoluteSize(osg::Vec3(20, 0, 0));
	_brightness->handle->setAbsolutePos(osg::Vec3(-10, -0.2f, 0));
	_brightness->handle->setPercentSize(osg::Vec3(0, 1, 1));
	_brightness->setCallback(this);
	_brightness->setPercent(.5f);
	contrastList->addChild(_brightness);

	label = new UIText("Center", 40.0f, osgText::TextBase::LEFT_CENTER);
	label->setPercentPos(osg::Vec3(0.1, 0, 0));
	contrastList->addChild(label);

	_contrastCenter = new CallbackSlider();
	_contrastCenter->setPercentPos(osg::Vec3(0.025, 0, 0.05));
	_contrastCenter->setPercentSize(osg::Vec3(0.95, 1, 0.9));
	_contrastCenter->handle->setAbsoluteSize(osg::Vec3(20, 0, 0));
	_contrastCenter->handle->setAbsolutePos(osg::Vec3(-10, -0.2f, 0));
	_contrastCenter->handle->setPercentSize(osg::Vec3(0, 1, 1));
	_contrastCenter->setCallback(this);
	_contrastCenter->setPercent(.5);
	contrastList->addChild(_contrastCenter);

	contrastBknd->addChild(contrastList);
	
	_tentMenu->setPosition(osg::Vec3(-1200, 675, 1880));
	_tentMenu->getRootElement()->setAbsoluteSize(osg::Vec3(1500, 1, 600));

	////////////Clahe UI////////////////////
	label = new UIText("CLAHE Options", 32.0f, osgText::TextBase::CENTER_TOP);
	UIList* claheUI = new UIList(UIList::TOP_TO_BOTTOM, UIList::CONTINUE);
 	label->setPercentPos(osg::Vec3(0.0, 0.0, -.5));
	claheUI->addChild(label);

	UIList* numBinTexts = new UIList(UIList::LEFT_TO_RIGHT, UIList::CONTINUE);
	label = new UIText("Number of Bins: ", 24.0f, osgText::TextBase::LEFT_CENTER);
	label->setPercentPos(osg::Vec3(0.05, 0.0, 0.0));
	numBinTexts->addChild(label);
	_numBinsLabel = new UIText("255", 24.0f, osgText::TextBase::RIGHT_CENTER);
	_numBinsLabel->setPercentPos(osg::Vec3(-0.05, 0.0, 0.0));
	numBinTexts->addChild(_numBinsLabel);
	claheUI->addChild(numBinTexts);
	

	_numBinsSlider = new CallbackSlider();
	_numBinsSlider->setCallback(this);
	_numBinsSlider->setPercent(.5f);
	_numBinsSlider->setPercentPos(osg::Vec3(0.025, 0, 0.05));
	_numBinsSlider->setPercentSize(osg::Vec3(0.95, 1, 0.5));
	_numBinsSlider->handle->setAbsoluteSize(osg::Vec3(20, 0, 0));
	_numBinsSlider->handle->setAbsolutePos(osg::Vec3(-10, -0.2f, 0));
	_numBinsSlider->handle->setPercentSize(osg::Vec3(0, 1, 1));
	claheUI->addChild(_numBinsSlider);

	UIList* clipLimitTexts = new UIList(UIList::LEFT_TO_RIGHT, UIList::CONTINUE);
	label = new UIText("Clip Limit: ", 24.0f, osgText::TextBase::LEFT_CENTER);
	label->setPercentPos(osg::Vec3(0.05, 0.0, 0.0));
	clipLimitTexts->addChild(label);
	_clipLimitLabel = new UIText(".85", 24.0f, osgText::TextBase::RIGHT_CENTER);
	_clipLimitLabel->setPercentPos(osg::Vec3(-0.05, 0.0, 0.0));
	clipLimitTexts->addChild(_clipLimitLabel);
	claheUI->addChild(clipLimitTexts);

	_clipLimitSlider = new CallbackSlider();
	_clipLimitSlider->setCallback(this);
	_clipLimitSlider->setPercent(.85f);
	_clipLimitSlider->setPercentPos(osg::Vec3(0.025, 0, 0.05));
	_clipLimitSlider->setPercentSize(osg::Vec3(0.95, 1, 0.5));
	_clipLimitSlider->handle->setAbsoluteSize(osg::Vec3(20, 0, 0));
	_clipLimitSlider->handle->setAbsolutePos(osg::Vec3(-10, -0.2f, 0));
	_clipLimitSlider->handle->setPercentSize(osg::Vec3(0, 1, 1));
	claheUI->addChild(_clipLimitSlider);

	UIList* claheResTexts = new UIList(UIList::LEFT_TO_RIGHT, UIList::CONTINUE);
	label = new UIText("Resolution: ", 24.0f, osgText::TextBase::LEFT_CENTER);
	label->setPercentPos(osg::Vec3(0.05, 0.0, 0.0));
	claheResTexts->addChild(label);
	_claheResLabel = new UIText("4", 24.0f, osgText::TextBase::RIGHT_CENTER);
	_claheResLabel->setPercentPos(osg::Vec3(-0.05, 0.0, 0.0));
	claheResTexts->addChild(_claheResLabel);
	claheUI->addChild(claheResTexts);

	_claheResSlider = new CallbackSlider();
	_claheResSlider->setCallback(this);
	_claheResSlider->setPercent(.4f);
	_claheResSlider->setMin(0.1);
	_claheResSlider->setPercentPos(osg::Vec3(0.025, 0, 0.05));
	_claheResSlider->setPercentSize(osg::Vec3(0.95, 1, 0.5));
	_claheResSlider->handle->setAbsoluteSize(osg::Vec3(20, 0, 0));
	_claheResSlider->handle->setAbsolutePos(osg::Vec3(-10, -0.2f, 0));
	_claheResSlider->handle->setPercentSize(osg::Vec3(0, 1, 1));
	claheUI->addChild(_claheResSlider);


	UIList* buttonList = new UIList(UIList::LEFT_TO_RIGHT, UIList::CONTINUE);

	_genClaheButton = new CallbackButton();
	_genClaheButton->setCallback(this);

	_genClaheButton->setPercentSize(osg::Vec3(.7, 1.0, 1));
 	UIQuadElement* buttonBknd = new UIQuadElement(UI_RED_ACTIVE_COLOR);
	label = new UIText("Generate CLAHE", 24.0f, osgText::TextBase::CENTER_CENTER);
	label->setPercentPos(osg::Vec3(0.0, -1.0, 0.0));

	_genClaheButton->addChild(buttonBknd);
	_genClaheButton->addChild(label);
	claheUI->addChild(_genClaheButton);

	_useClaheButton = new CallbackButton();
	_useClaheButton->setCallback(this);
	_useClaheButton->setPercentSize(osg::Vec3(.7, 1.0, 1));
 	buttonBknd = new UIQuadElement(UI_INACTIVE_RED_COLOR);
	label = new UIText("Enable CLAHE", 24.0f, osgText::TextBase::CENTER_CENTER);
	label->setColor(UI_INACTIVE_WHITE_COLOR);
	label->setPercentPos(osg::Vec3(0.0, -1.0, 0.0));

	_useClaheButton->addChild(buttonBknd);
	_useClaheButton->addChild(label);

	_useClaheSelection = new CallbackButton();
	_useClaheSelection->setCallback(this);
	_useClaheSelection->setPercentSize(osg::Vec3(.7, 1.0, 1));
 	buttonBknd = new UIQuadElement(UI_RED_ACTIVE_COLOR);
	label = new UIText("Enable Selection", 24.0f, osgText::TextBase::CENTER_CENTER);
	label->setColor(UI_INACTIVE_WHITE_COLOR);
	label->setPercentPos(osg::Vec3(0.0, -1.0, 0.0));

	_useClaheSelection->addChild(buttonBknd);
	_useClaheSelection->addChild(label);

	buttonList->setPercentPos(osg::Vec3(.05, 0.0, 1.0));
	buttonList->addChild(_genClaheButton);
	buttonList->addChild(_useClaheButton);
	buttonList->addChild(_useClaheSelection);
	claheUI->addChild(buttonList);

	

	_claheMenu->addChild(claheUI);
	
	_claheMenu->setPosition(POPUP_POS);
	_claheMenu->getRootElement()->setAbsoluteSize(osg::Vec3(POPUP_WIDTH, 1, 450));
	
	////////////Attention Maps UI/////////////////

	UIQuadElement* attnMapsBknd = new UIQuadElement(UI_BACKGROUND_COLOR);
	attnMapsBknd->setRounding(0, .2);
	attnMapsBknd->setTransparent(true);
	attnMapsBknd->setBorderColor(UI_BLACK_COLOR);
	attnMapsBknd->setBorderSize(.02);

	_attnMapsMenu->addChild(attnMapsBknd);

	label = new UIText("Attention Maps Options", 32.0f, osgText::TextBase::CENTER_TOP);
	UIList* attnMapsUI = new UIList(UIList::TOP_TO_BOTTOM, UIList::CONTINUE);
	label->setPercentPos(osg::Vec3(0.0, 0.0, -.5));
	attnMapsUI->addChild(label);

	UIList* minAttentionTexts = new UIList(UIList::LEFT_TO_RIGHT, UIList::CONTINUE);
	label = new UIText("Minimum Attention Value: ", 24.0f, osgText::TextBase::LEFT_CENTER);
	label->setPercentPos(osg::Vec3(0.05, 0.0, 0.0));
	minAttentionTexts->addChild(label);
	_minAttentionLabel = new UIText("0.00", 24.0f, osgText::TextBase::RIGHT_CENTER);
	_minAttentionLabel->setPercentPos(osg::Vec3(-0.05, 0.0, 0.0));
	minAttentionTexts->addChild(_minAttentionLabel);
	attnMapsUI->addChild(minAttentionTexts);
	_volume->getCompute()->getOrCreateStateSet()->setDefine("MIN_ATTN_VALUE", "0.0", osg::StateAttribute::ON);

	_minAttnSlider = new CallbackSlider();
	_minAttnSlider->setCallback(this);
	_minAttnSlider->setPercent(0.0f);
	_minAttnSlider->setPercentPos(osg::Vec3(0.025, 0, 0.05));
	_minAttnSlider->setPercentSize(osg::Vec3(0.95, 1, 0.5));
	_minAttnSlider->handle->setAbsoluteSize(osg::Vec3(20, 0, 0));
	_minAttnSlider->handle->setAbsolutePos(osg::Vec3(-10, -0.2f, 0));
	_minAttnSlider->handle->setPercentSize(osg::Vec3(0, 1, 1));
	attnMapsUI->addChild(_minAttnSlider);

	buttonList = new UIList(UIList::LEFT_TO_RIGHT, UIList::CONTINUE);

	_attnMapOnlyButton = new CallbackButton();
	_attnMapOnlyButton->setCallback(this);
	_attnMapOnlyButton->setPercentSize(osg::Vec3(.5, 1.0, .7));
	buttonBknd = new UIQuadElement(UI_INACTIVE_RED_COLOR);
	label = new UIText("Enable Maps Only", 24.0f, osgText::TextBase::CENTER_CENTER);
	label->setPercentPos(osg::Vec3(0.0, -1.0, 0.0));

	_attnMapOnlyButton->addChild(buttonBknd);
	_attnMapOnlyButton->addChild(label);

	_useAttnMapsButton = new CallbackButton();
	_useAttnMapsButton->setCallback(this);
	_useAttnMapsButton->setPercentSize(osg::Vec3(.5, 1.0, .7));
	buttonBknd = new UIQuadElement(UI_INACTIVE_RED_COLOR);
	label = new UIText("Show Maps", 24.0f, osgText::TextBase::CENTER_CENTER);
	label->setColor(UI_WHITE_COLOR);
	label->setPercentPos(osg::Vec3(0.0, -1.0, 0.0));

	_useAttnMapsButton->addChild(buttonBknd);
	_useAttnMapsButton->addChild(label);

	buttonList->setPercentPos(osg::Vec3(.125, 0.0, 0.35));
	buttonList->addChild(_useAttnMapsButton);
	buttonList->addChild(_attnMapOnlyButton);
	attnMapsUI->addChild(buttonList);



	_attnMapsMenu->addChild(attnMapsUI);
	_attnMapsMenu->setPosition(POPUP_POS);
	_attnMapsMenu->getRootElement()->setAbsoluteSize(osg::Vec3(POPUP_WIDTH, 1, 450));
	
	////////////Marching Cubes UI/////////////////
	_marchingCubesMenu = new UIPopup();
	popupMenus.push_back(_marchingCubesMenu);
	UIQuadElement* mcBknd = new UIQuadElement(UI_BACKGROUND_COLOR);
	mcBknd->setRounding(0, .2);
	mcBknd->setTransparent(true);
	mcBknd->setBorderColor(UI_BLACK_COLOR);
	mcBknd->setBorderSize(.02);

	_marchingCubesMenu->addChild(mcBknd);

	label = new UIText("Marching Cubes Options", 32.0f, osgText::TextBase::CENTER_TOP);
	UIList* mcUI = new UIList(UIList::TOP_TO_BOTTOM, UIList::CONTINUE);
	mcUI->setPercentSize(osg::Vec3(1, 1, .9));
	mcUI->setPercentPos(osg::Vec3(0, 0, -.025));
	label->setPercentPos(osg::Vec3(0.0, 0.0, -.5));
	mcUI->addChild(label);


	UIList* mcResolutionTexts = new UIList(UIList::LEFT_TO_RIGHT, UIList::CONTINUE);
	label = new UIText("Resolution ", 24.0f, osgText::TextBase::LEFT_CENTER);
	label->setPercentPos(osg::Vec3(0.05, 0.0, 0.0));
	mcResolutionTexts->addChild(label);
	_mcResolutionLabel = new UIText("Lowest", 24.0f, osgText::TextBase::RIGHT_CENTER);
	_mcResolutionLabel->setPercentPos(osg::Vec3(-0.05, 0.0, 0.0));
	mcResolutionTexts->addChild(_mcResolutionLabel);
	mcUI->addChild(mcResolutionTexts);

	_mcResSlider = new CallbackSlider();
	_mcResSlider->setCallback(this);
	_mcResSlider->setPercent(0);
 	_mcResSlider->setPercentPos(osg::Vec3(0.025, 0, 0.05));
	_mcResSlider->setPercentSize(osg::Vec3(0.95, 1, 0.5));
	_mcResSlider->handle->setAbsoluteSize(osg::Vec3(20, 0, 0));
	_mcResSlider->handle->setAbsolutePos(osg::Vec3(-10, -0.2f, 0));
	_mcResSlider->handle->setPercentSize(osg::Vec3(0, 1, 1));
	mcUI->addChild(_mcResSlider);

	UIList* mcOrganPickTexts = new UIList(UIList::LEFT_TO_RIGHT, UIList::CONTINUE);
	label = new UIText("Organ ", 24.0f, osgText::TextBase::LEFT_CENTER);
	label->setPercentPos(osg::Vec3(0.05, 0.0, 0.0));
	mcOrganPickTexts->addChild(label);
	_mcOrganPickLabel = new UIText("Colon", 24.0f, osgText::TextBase::RIGHT_CENTER);
	_mcOrganPickLabel->setPercentPos(osg::Vec3(-0.05, 0.0, 0.0));
	mcOrganPickTexts->addChild(_mcOrganPickLabel);
	mcUI->addChild(mcOrganPickTexts);

	_mcOrganSlider = new CallbackSlider();
	_mcOrganSlider->setCallback(this);
	_mcOrganSlider->setPercent(0.0f);
	_mcOrganSlider->setPercentPos(osg::Vec3(0.025, 0, 0.05));
	_mcOrganSlider->setPercentSize(osg::Vec3(0.95, 1, 0.5));
	_mcOrganSlider->handle->setAbsoluteSize(osg::Vec3(20, 0, 0));
	_mcOrganSlider->handle->setAbsolutePos(osg::Vec3(-10, -0.2f, 0));
	_mcOrganSlider->handle->setPercentSize(osg::Vec3(0, 1, 1));
	mcUI->addChild(_mcOrganSlider);

	buttonList = new UIList(UIList::LEFT_TO_RIGHT, UIList::CONTINUE);

	_UseMarchingCubesButton = new CallbackButton();
	_UseMarchingCubesButton->setCallback(this);
	_UseMarchingCubesButton->setPercentSize(osg::Vec3(.7, 1.0, .7));

	buttonBknd = new UIQuadElement(UI_INACTIVE_RED_COLOR);
	label = new UIText("Use Polygon", 24.0f, osgText::TextBase::CENTER_CENTER);
	label->setColor(UI_INACTIVE_WHITE_COLOR);
	label->setPercentPos(osg::Vec3(0.0, -1.0, 0.0));

	_UseMarchingCubesButton->addChild(buttonBknd);
	_UseMarchingCubesButton->addChild(label);

	_GenMarchCubesButton = new CallbackButton();
	_GenMarchCubesButton->setCallback(this);
	_GenMarchCubesButton->setPercentSize(osg::Vec3(.7, 1.0, .7));

	buttonBknd = new UIQuadElement(osg::Vec4(1.0, 0.0, 0.0, 1.0));
	label = new UIText("Gen Polygon", 24.0f, osgText::TextBase::CENTER_CENTER);
	label->setPercentPos(osg::Vec3(0.0, -1.0, 0.0));

	_GenMarchCubesButton->addChild(buttonBknd);
	_GenMarchCubesButton->addChild(label);

	_printStlButton = new FullButton("Print STL File", 24.0f, UI_RED_ACTIVE_COLOR, UI_RED_ACTIVE_COLOR);
	_printStlButton->getButton()->setDisabledColor(UI_INACTIVE_RED_COLOR);
	_printStlButton->getButton()->setCallback(this);
	_printStlButton->getText()->setColor(UI_INACTIVE_WHITE_COLOR);
	_printStlButton->setPercentSize(osg::Vec3(.7, 1.0, .7));
	_PrintStlCallbackButton = _printStlButton->getButton();

	buttonList->setPercentPos(osg::Vec3(.05, 0.0, 0.0));
	buttonList->addChild(_GenMarchCubesButton);
	buttonList->addChild(_UseMarchingCubesButton);
	buttonList->addChild(_printStlButton);

	mcUI->addChild(buttonList);
	_marchingCubesMenu->addChild(mcUI);
	_marchingCubesMenu->setPosition(POPUP_POS);
	_marchingCubesMenu->getRootElement()->setAbsoluteSize(osg::Vec3(POPUP_WIDTH, 1, 450));
	
	////////////Selection UI/////////////////
	_selectionMenu = new UIPopup();
	popupMenus.push_back(_selectionMenu);
	UIQuadElement* selectionBknd = new UIQuadElement(UI_BACKGROUND_COLOR);
	selectionBknd->setRounding(0, .2);
	selectionBknd->setTransparent(true);
	selectionBknd->setBorderColor(UI_BLACK_COLOR);
	selectionBknd->setBorderSize(.02);

	_selectionMenu->addChild(selectionBknd);

	label = new UIText("3D Selection", 32.0f, osgText::TextBase::CENTER_TOP);
	UIList* selectUI = new UIList(UIList::TOP_TO_BOTTOM, UIList::CONTINUE);
	selectUI->setPercentSize(osg::Vec3(1, 1, .9));
	selectUI->setPercentPos(osg::Vec3(0, 0, -.025));
	label->setPercentPos(osg::Vec3(0.0, 0.0, -.5));
	selectUI->addChild(label);

	buttonList = new UIList(UIList::LEFT_TO_RIGHT, UIList::CONTINUE);

	_lockSelectionButton = new CallbackButton();
	_lockSelectionButton->setCallback(this);
	_lockSelectionButton->setPercentSize(osg::Vec3(.7, 1.0, .7));

	buttonBknd = new UIQuadElement(UI_RED_ACTIVE_COLOR);
	label = new UIText("Lock Size", 24.0f, osgText::TextBase::CENTER_CENTER);
	label->setColor(UI_WHITE_COLOR);
	label->setPercentPos(osg::Vec3(0.0, -1.0, 0.0));

	_lockSelectionButton->addChild(buttonBknd);
	_lockSelectionButton->addChild(label);

	_resetSelectionButton = new CallbackButton();
	_resetSelectionButton->setCallback(this);
	_resetSelectionButton->setPercentSize(osg::Vec3(.7, 1.0, .7));

	buttonBknd = new UIQuadElement(UI_RED_ACTIVE_COLOR);
	label = new UIText("Remove Selection", 24.0f, osgText::TextBase::CENTER_CENTER);
	label->setColor(UI_WHITE_COLOR);
	label->setPercentPos(osg::Vec3(0.0, -1.0, 0.0));

	_resetSelectionButton->addChild(buttonBknd);
	_resetSelectionButton->addChild(label);

	_toggleSelection = new CallbackButton();
	_toggleSelection->setCallback(this);
	_toggleSelection->setPercentSize(osg::Vec3(.7, 1.0, .7));

	buttonBknd = new UIQuadElement(UI_RED_ACTIVE_COLOR);
	label = new UIText("Disable Selection", 24.0f, osgText::TextBase::CENTER_CENTER);
	label->setColor(UI_WHITE_COLOR);
	label->setPercentPos(osg::Vec3(0.0, -1.0, 0.0));

	_toggleSelection->addChild(buttonBknd);
	_toggleSelection->addChild(label);


	buttonList->setPercentPos(osg::Vec3(.05, 0.0, 0.0));
	buttonList->addChild(_lockSelectionButton);
	buttonList->addChild(_toggleSelection);
	buttonList->addChild(_resetSelectionButton);


	selectUI->addChild(buttonList);
	_selectionMenu->addChild(selectUI);
	_selectionMenu->setPosition(POPUP_POS);
	_selectionMenu->getRootElement()->setAbsoluteSize(osg::Vec3(POPUP_WIDTH, 1, 450));


	///////////Masks UI///////////////
		_maskBknd = new UIQuadElement(UI_BACKGROUND_COLOR);
		_maskBknd->setPercentPos((osg::Vec3(0, 0, -1.025)));
		_maskBknd->setPercentSize((osg::Vec3(1.62, 1, 1)));
		_maskBknd->setBorderSize(.01);
		_maskMenu->addChild(_maskBknd);

		label = new UIText("Organ Masks", 50.0f, osgText::TextBase::CENTER_CENTER);
		label->setPercentSize(osg::Vec3(1, 1, 0.2));
		_maskBknd->addChild(label);

		_mainMaskList = new UIList(UIList::TOP_TO_BOTTOM, UIList::CONTINUE);
		_mainMaskList->setPercentPos(osg::Vec3(0, 0, -.2));
		_mainMaskList->setPercentSize(osg::Vec3(1, 1, .8));
		_maskBknd->addChild(_mainMaskList);
		
		
		
		UIQuadElement* bodyColorbutton = new UIQuadElement(osg::Vec4(1, 0, 0, 0));
		bodyColorbutton->setBorderSize(.05);
		bodyColorbutton->setBorderColor(UI_BACKGROUND_COLOR);
		bodyColorbutton->setPercentPos(osg::Vec3(0.6, 0.0, 0.0));
		bodyColorbutton->setPercentSize(osg::Vec3(.25, 1.0, 0.8));
		bodyColorbutton->setTransparent(true);


		_colonColorButton = new UIQuadElement(osg::Vec4(0.752, 0.635, 0.996, 1));
		_colonColorButton->setBorderSize(.05);
  		_colonColorButton->setPercentPos(osg::Vec3(0.6, 0.0, 0.0));
		_colonColorButton->setPercentSize(osg::Vec3(.25, 1.0, 0.8));

		_kidneyColorButton = new UIQuadElement(osg::Vec4(0, 0.278, 1, 1));
		_kidneyColorButton->setBorderSize(.05);
  		_kidneyColorButton->setPercentPos(osg::Vec3(0.6, 0.0, 0.0));
		_kidneyColorButton->setPercentSize(osg::Vec3(.25, 1.0, 0.8));
		
	
		_bladderColorButton = new UIQuadElement(osg::Vec4(0.992, 0.968, 0.843, 1));
		_bladderColorButton->setBorderSize(.05);
  		_bladderColorButton->setPercentPos(osg::Vec3(0.6, 0.0, 0.0));
		_bladderColorButton->setPercentSize(osg::Vec3(.25, 1.0, 0.8));
		
		_spleenColorButton = new UIQuadElement(osg::Vec4(1, 0.874, 0.109, 1));
		_spleenColorButton->setBorderSize(.05);
  		_spleenColorButton->setPercentPos(osg::Vec3(0.6, 0.0, 0.0));
		_spleenColorButton->setPercentSize(osg::Vec3(.25, 1.0, 0.8));

		_aortaColorButton = new UIQuadElement(osg::Vec4(1, 0, 0, 1));
		_aortaColorButton->setBorderSize(.05);
  		_aortaColorButton->setPercentPos(osg::Vec3(0.6, 0.0, 0.0));
		_aortaColorButton->setPercentSize(osg::Vec3(.25, 1.0, 0.8));

		_illeumColorButton = new UIQuadElement(osg::Vec4(0.968, 0.780, 1, 1));
		_illeumColorButton->setBorderSize(.05);
  		_illeumColorButton->setPercentPos(osg::Vec3(0.6, 0.0, 0.0));
		_illeumColorButton->setPercentSize(osg::Vec3(.25, 1.0, 0.8));

		UIQuadElement* veinColorButton = new UIQuadElement(osg::Vec4(0, .8, 1, 0));
		veinColorButton->setBorderSize(.05);
		veinColorButton->setPercentPos(osg::Vec3(0.6, 0.0, 0.0));
		veinColorButton->setPercentSize(osg::Vec3(.25, 1.0, 0.8));
	


		_organs = new VisibilityToggle("Body");
		_organs->toggle();
		_organs->setCallback(this);
		_volume->getCompute()->getOrCreateStateSet()->setDefine("ORGANS_ONLY", _organs->isOn());

		_colon = new VisibilityToggle("Colon");
		_colon->setCallback(this);

		_kidney = new VisibilityToggle("Kidney");
		_kidney->setCallback(this);

		_bladder = new VisibilityToggle("Bladder");
		_bladder->setCallback(this);

		_spleen = new VisibilityToggle("Spleen");
		_spleen->setCallback(this);

		_illeum = new VisibilityToggle("Illeum");
		_illeum->setCallback(this);

		_aorta = new VisibilityToggle("Aorta");
		_aorta->setCallback(this);

		_vein = new VisibilityToggle("Vena Cava");
		_vein->setCallback(this);

		UIList* bodyList = new UIList(UIList::LEFT_TO_RIGHT, UIList::CONTINUE);
		bodyList->addChild(_organs);
		bodyList->addChild(bodyColorbutton);
		bodyList->setPercentPos(osg::Vec3(.02, 0.0, 0.0));
		UIList* colonList = new UIList(UIList::LEFT_TO_RIGHT, UIList::CONTINUE);
		colonList->addChild(_colon);
		colonList->addChild(_colonColorButton);
		colonList->setPercentPos(osg::Vec3(.02, 0.0, 0.0));
		UIList* kidneyList = new UIList(UIList::LEFT_TO_RIGHT, UIList::CONTINUE);
		kidneyList->addChild(_kidney);
		kidneyList->addChild(_kidneyColorButton);
		kidneyList->setPercentPos(osg::Vec3(.02, 0.0, 0.0));
		UIList* bladderList = new UIList(UIList::LEFT_TO_RIGHT, UIList::CONTINUE);
		bladderList->addChild(_bladder);
		bladderList->addChild(_bladderColorButton);
		bladderList->setPercentPos(osg::Vec3(.02, 0.0, 0.0));
		UIList* spleenList = new UIList(UIList::LEFT_TO_RIGHT, UIList::CONTINUE);
		spleenList->addChild(_spleen);
		spleenList->addChild(_spleenColorButton);
		spleenList->setPercentPos(osg::Vec3(.02, 0.0, 0.0));
		UIList* aortaList = new UIList(UIList::LEFT_TO_RIGHT, UIList::CONTINUE);
		aortaList->addChild(_aorta);
		aortaList->addChild(_aortaColorButton);
		aortaList->setPercentPos(osg::Vec3(.02, 0.0, 0.0));
		UIList* illeumList = new UIList(UIList::LEFT_TO_RIGHT, UIList::CONTINUE);
		illeumList->addChild(_illeum);
		illeumList->addChild(_illeumColorButton);
		illeumList->setPercentPos(osg::Vec3(.02, 0.0, 0.0));
		UIList* veinList = new UIList(UIList::LEFT_TO_RIGHT, UIList::CONTINUE);
		veinList->addChild(_vein);
		veinList->addChild(veinColorButton);
		veinList->setPercentPos(osg::Vec3(.02, 0.0, 0.0));
	
		_mainMaskList->addChild(bodyList);
		_mainMaskList->addChild(colonList);
		_mainMaskList->addChild(kidneyList);
		_mainMaskList->addChild(bladderList);
		_mainMaskList->addChild(spleenList);
		_mainMaskList->addChild(aortaList);
		_mainMaskList->addChild(illeumList);
		_mainMaskList->addChild(veinList);

		UIQuadElement* volumeSwitchBknd = new UIQuadElement(UI_BACKGROUND_COLOR);
		volumeSwitchBknd->setPercentPos((osg::Vec3(0, 0, -2.025)));
		volumeSwitchBknd->setPercentSize((osg::Vec3(1.62, 1, .3)));
		volumeSwitchBknd->setBorderSize(.04);
		_maskMenu->addChild(volumeSwitchBknd);

		_volumeList = new UIList(UIList::LEFT_TO_RIGHT , UIList::CONTINUE);
		//_volume1Button = new CallbackButton();
		//_volume1Button->setCallback(this);
		UIText* v1Text = new UIText("Volume 1", 40.0f, osgText::TextBase::CENTER_CENTER);
		v1Text->setColor(UI_ACTIVE_COLOR);
		//_volume1Button->addChild(v1Text);
		//_volume2Button = new CallbackButton();
		//_volume2Button->setCallback(this);
		UIText* v2Text = new UIText("Volume 2", 40.0f, osgText::TextBase::CENTER_CENTER);
		//_volume2Button->addChild(v2Text);
		_volumeList->addChild(v1Text);
		_volumeList->addChild(v2Text);


		UIList* volumeInteraction = new UIList(UIList::TOP_TO_BOTTOM, UIList::CONTINUE);
		_swapButton = new CallbackButton();
		_swapButton->setCallback(this);
		v1Text = new UIText("Swap", 40.0f, osgText::TextBase::CENTER_CENTER);
		v1Text->setPercentPos(osg::Vec3(0.0, -1.0, 0.0));
		v1Text->setColor(osg::Vec4(1.0, 1.0, 1.0, 0.4));
		_swapButton->addChild(v1Text);
		UIQuadElement* swpbknd = new UIQuadElement(osg::Vec4(.8, .1, .1, 0.4));
		swpbknd->setPercentSize(osg::Vec3(.8, 1, .8));
		swpbknd->setPercentPos(osg::Vec3(.1, 0, -.1));
		_swapButton->addChild(swpbknd);
		swpbknd->setRounding(0, .2);
		//swpbknd->setBorderSize(.1);
		swpbknd->setTransparent(true);


		_linkButton = new CallbackButton();
		_linkButton->setCallback(this);
		v2Text = new UIText("Link", 40.0f, osgText::TextBase::CENTER_CENTER);
		v2Text->setColor(osg::Vec4(1.0, 1.0, 1.0, 0.4));
		v2Text->setPercentPos(osg::Vec3(0.0, -1.0, 0.0));
		_linkButton->addChild(v2Text);
		UIQuadElement* linkbknd = new UIQuadElement(osg::Vec4(.8, .1, .1, 0.4));
		linkbknd->setPercentSize(osg::Vec3(.8, 1, .8));
		linkbknd->setPercentPos(osg::Vec3(.1, 0, -.1));
		_linkButton->addChild(linkbknd);
		linkbknd->setRounding(0, .2);
		//swpbknd->setBorderSize(.1);
		linkbknd->setTransparent(true);

		volumeInteraction->addChild(_swapButton);
		volumeInteraction->addChild(_linkButton);

		UIList* lists = new UIList(UIList::LEFT_TO_RIGHT, UIList::CONTINUE);
		lists->addChild(volumeInteraction);
		lists->addChild(_volumeList);
		volumeSwitchBknd->addChild(lists);
		
		
		
	//}
		std::cout << "moveable" << _movable << std::endl;

	if (!_movable)
	{
		_maskMenu->setActive(true, true);
		_contrastMenu->setActive(true, true);
		_tentMenu->setActive(true, true);
	}
	else {
		_maskMenu->setActive(true, false);
		_contrastMenu->setActive(true, false);
		_tentMenu->setActive(true, false);
		_claheMenu->setActive(false, false);
		_marchingCubesMenu->setActive(false, false);
		_attnMapsMenu->setActive(false, false);
		_selectionMenu->setActive(false, false);
		
		_maskContainer = new SceneObject("MaskMenu", false, true, false, false, false);
		_contrastContainer = new SceneObject("ContrastMenu", false, true, false, false, false);
		_tentWindowContainer = new SceneObject("TentWindow", false, true, false, false, false);
		_so->addChild(_maskContainer);
		_so->addChild(_tentWindowContainer);
		_so->addChild(_contrastContainer);

		_maskContainer->setShowBounds(false);
		//_maskContainer->addChild(_maskMenu->getRoot());

		_tentWindowContainer->setShowBounds(false);
		//_tentWindowContainer->addChild(_tentMenu->getRoot());

		_contrastContainer->setShowBounds(false);
		//_contrastContainer->addChild(_contrastMenu->getRoot());
		
		_maskMenu->getRootElement()->updateElement(osg::Vec3(0, 0, 0), osg::Vec3(0, 0, 0));
		_tentMenu->getRootElement()->updateElement(osg::Vec3(0, 0, 0), osg::Vec3(0, 0, 0));
		_claheMenu->getRootElement()->updateElement(osg::Vec3(0, 0, 0), osg::Vec3(0, 0, 0));
		_marchingCubesMenu->getRootElement()->updateElement(osg::Vec3(0, 0, 0), osg::Vec3(0, 0, 0));
		_selectionMenu->getRootElement()->updateElement(osg::Vec3(0, 0, 0), osg::Vec3(0, 0, 0));
		_contrastMenu->getRootElement()->updateElement(osg::Vec3(0, 0, 0), osg::Vec3(0, 0, 0));
		_maskContainer->dirtyBounds();
	}

	if (!_volume->hasMask()) {
		_maskMenu->getRootElement()->getGroup()->removeChild(_maskBknd->getGroup());
		_maskBknd->_parent = nullptr;
		_maskBknd->setActive(false);
	}
}

void NewVolumeMenu::toggleSwapOpacity() {
	_swapOpacity ? _swapOpacity = false : _swapOpacity = true;
	if (_swapOpacity) {
		((UIText*)_swapButton->getChild(0))->setColor(osg::Vec4(1.0, 1.0, 1.0, 1.0));
		((UIQuadElement*)_swapButton->getChild(1))->setColor(osg::Vec4(0.8, 0.1, 0.1, 1.0));
 	}
	else {
		((UIText*)_swapButton->getChild(0))->setColor(osg::Vec4(1.0, 1.0, 1.0, 0.4));
		((UIQuadElement*)_swapButton->getChild(1))->setColor(osg::Vec4(0.8, 0.1, 0.1, 0.4));
	}
}
void NewVolumeMenu::toggleLinkOpacity(bool turnOn) {
	if (turnOn) {
		((UIText*)_linkButton->getChild(0))->setColor(osg::Vec4(1.0, 1.0, 1.0, 1.0));
		((UIQuadElement*)_linkButton->getChild(1))->setColor(osg::Vec4(0.8, 0.1, 0.1, 1.0));
	}
	else {
		((UIText*)_linkButton->getChild(0))->setColor(osg::Vec4(1.0, 1.0, 1.0, 0.4));
		((UIQuadElement*)_linkButton->getChild(1))->setColor(osg::Vec4(0.8, 0.1, 0.1, 0.4));
	}
}

UIList* NewVolumeMenu::addPresets(UIQuadElement* bknd) {
	std::vector<std::string> presetFilePaths = FileSelector::getPresets();
	UIList* presetUIList;
	if (!presetFilePaths.empty()) {
		presetUIList = new UIList(UIList::TOP_TO_BOTTOM, UIList::CONTINUE);
		presetUIList->setPercentPos(osg::Vec3(0, 0, -.25));
		presetUIList->setPercentSize(osg::Vec3(1.0, 1.0, .75));
	}
	else {
		return nullptr;
	}
	for (int i = 0; i < presetFilePaths.size(); i++) {
		CallbackButton* presetbutton = new CallbackButton();
		presetbutton->setCallback(this);
		int presetNameStart = strrchr(presetFilePaths[i].c_str(), '\\') - presetFilePaths[i].c_str() + 1;
		int presetNameLength = strrchr(presetFilePaths[i].c_str(), '.') - presetFilePaths[i].c_str() - presetNameStart;
		UIText* presetText = new UIText(presetFilePaths[i].substr(presetNameStart, presetNameLength), 40.0f, osgText::TextBase::CENTER_CENTER);
		presetbutton->addChild(presetText);
		presetUIList->addChild(presetbutton);
		_presetCallbacks.push_back(presetbutton);
	}
	bknd->addChild(presetUIList);
	return presetUIList;
}

void NewVolumeMenu::uiCallback(UICallbackCaller * item)
{	
	///////////////Rotations/////////////////////
	if (item == _horiFlipButton.get())
	{
		osg::Matrix m;
		m.makeScale(osg::Vec3(-1, 1, 1));
		_volume->_transform->postMult(m);
		_volume->flipCull();
	}
	else if (item == _vertiFlipButton.get())
	{
		osg::Matrix m;
		m.makeScale(osg::Vec3(1, 1, -1));
		_volume->_transform->postMult(m);
		_volume->flipCull();
	}
	else if (item == _depthFlipButton.get())
	{
		osg::Matrix m;
		m.makeScale(osg::Vec3(1, -1, 1));
		_volume->_transform->postMult(m);
		_volume->flipCull();
	}

	/////////////////////////Masks/////////////////
	else if (item == _organs)
	{
		_volume->getCompute()->getOrCreateStateSet()->setDefine("ORGANS_ONLY", _organs->isOn());
		_volume->setDirtyAll();
	}
	else if (item == _colon)
	{
		_volume->getCompute()->getOrCreateStateSet()->setDefine("COLON", _colon->isOn());
		_volume->setDirtyAll();
	}
	else if (item == _kidney)
	{
		_volume->getCompute()->getOrCreateStateSet()->setDefine("KIDNEY", _kidney->isOn());
		_volume->setDirtyAll();
	}
	else if (item == _bladder)
	{
		_volume->getCompute()->getOrCreateStateSet()->setDefine("BLADDER", _bladder->isOn());
		_volume->setDirtyAll();
	}
	else if (item == _spleen)
	{
		_volume->getCompute()->getOrCreateStateSet()->setDefine("SPLEEN", _spleen->isOn());
		_volume->setDirtyAll();
	}
	else if (item == _illeum)
	{
		_volume->getCompute()->getOrCreateStateSet()->setDefine("ILLEUM", _illeum->isOn());
		_volume->setDirtyAll();
	}
	else if (item == _aorta)
	{
		_volume->getCompute()->getOrCreateStateSet()->setDefine("AORTA", _aorta->isOn());
		_volume->setDirtyAll();
	}
	else if (item == _vein)
	{
		_volume->getCompute()->getOrCreateStateSet()->setDefine("VEIN", _vein->isOn());
		_volume->setDirtyAll();
	}



	else if (item == _addTriangleButton.get())
	{
		if (_triangleIndex < 5) {
			_triangleIndex++;
			addRegion();
		}
	}
	
	else if (item == _contrastBottom)
	{
		if (_contrastBottom->getPercent() >= _contrastTop->getPercent())
		{
			_contrastBottom->setPercent(_contrastTop->getPercent() - 0.001f);
		}
		_volume->_computeUniforms["ContrastBottom"]->set(_contrastBottom->getAdjustedValue());
		_colSliderList[0]->setPercentPos(osg::Vec3(_contrastBottom->getAdjustedValue(), -1.0, 0.0));
		
		
		std::string low = std::to_string(_contrastBottom->getAdjustedValue()).substr(0, 4);
		std::string high = std::to_string(_contrastTop->getAdjustedValue()).substr(0, 4);
		_contrastValueLabel->setText("Low " + low + " / " + "High " + high);
		_volume->setDirtyAll();
	}
	else if (item == _contrastTop)
	{
		if (_contrastBottom->getPercent() >= _contrastTop->getPercent())
		{
			_contrastTop->setPercent(_contrastBottom->getPercent() + 0.001f);
		}
		_volume->_computeUniforms["ContrastTop"]->set(_contrastTop->getAdjustedValue());
		_colSliderList[1]->setPercentPos(osg::Vec3(_contrastTop->getAdjustedValue(), -1.0, 0.0));

		std::string low = std::to_string(_contrastBottom->getAdjustedValue()).substr(0, 4);
		std::string high = std::to_string(_contrastTop->getAdjustedValue()).substr(0, 4);
		_contrastValueLabel->setText("Low " + low + " / " + "High " + high);
		_volume->setDirtyAll();
	}
	else if (item == _trueContrast)
	{
		
		_volume->_computeUniforms["TrueContrast"]->set(_trueContrast->getAdjustedValue());
		
		_volume->setDirtyAll();
	}

	else if (item == _contrastCenter) {
		_volume->_computeUniforms["ContrastCenter"]->set(_contrastCenter->getAdjustedValue());
		_volume->setDirtyAll();
	}
	else if (item == _brightness)
	{
		_volume->_computeUniforms["Brightness"]->set(_brightness->getAdjustedValue());
		_volume->setDirtyAll();
	}
	else if (item == _transferFunctionRadial)
	{
		useTransferFunction(_transferFunctionRadial->getCurrent());
	}
	else if (item == _addPresetButton.get()) {
		savePreset();
	}
	else if (item == _loadPresetButton.get()) {
		_maskMenu->addChild(_presetPopup->getRootElement());
		_presetPopup->getRootElement()->setPercentPos(osg::Vec3(1.62, 0.0, 0.0));
	}
	else if (checkTriangleCallbacks(item)) {
		return;
	}
	else if (checkTriangleVisCallbacks(item)) {
		return;
	}
	else if (checkPresetCallbacks(item)) {
		return;
	}
	else if (item == _cp) {
		osg::Vec3 col = _cp->returnColor();

		
		_colSliderList[_colorSliderIndex]->setColor(osg::Vec4(col, 1.0));
		if(_colorSliderIndex == 0)
			_volume->_computeUniforms["leftColor"]->set(col);
		else
			_volume->_computeUniforms["rightColor"]->set(col);
		_volume->setDirtyAll();
	}
	else if (checkColorSliderCallbacks(item)) {

 
		osg::Vec4 col4 = _colSliderList[_colorSliderIndex]->getColor();
		osg::Vec3 col; col.x() = col4.x(); col.y() = col4.y(); col.z() = col4.z();
		_cp->setCPColor(col);
		return;
	}
	
	else if (item == _swapButton) {
		switchVolumes();
		

	}
	else if (item == _linkButton) {
		linkVolumes();
	}

	//CLAHE Callbacks
	else if (item == _numBinsSlider) {
		if (_numBinsSlider->getAdjustedValue() > .8) {
			_numBinsLabel->setText("65536");
			_volume->setNumBins(65536);
			
		}
		if (_numBinsSlider->getAdjustedValue() > .5 && _numBinsSlider->getAdjustedValue() < .8) {
			_numBinsLabel->setText("255");
			_volume->setNumBins(255);
		}
		if (_numBinsSlider->getAdjustedValue() < .5) {
			_numBinsLabel->setText("16");
			_volume->setNumBins(16);
		}

	}

	else if (item == _clipLimitSlider) {
		_clipLimitLabel->setText(std::to_string(_clipLimitSlider->getAdjustedValue()).substr(0, 4));
		_volume->setClipLimit(_clipLimitSlider->getAdjustedValue());
	}
	
	else if (item == _claheResSlider) {
		if(_claheResSlider->getAdjustedValue() * 10 < 10.0)
			_claheResLabel->setText(std::to_string(_claheResSlider->getAdjustedValue()*10).substr(0, 1));
		else
			_claheResLabel->setText(std::to_string(_claheResSlider->getAdjustedValue()*10).substr(0, 2));
		
		float rawFloat = _claheResSlider->getAdjustedValue();
		if (rawFloat <= .25) {
			_volume->setClaheRes(2);
		}
		else if (rawFloat <= .50) {
			_volume->setClaheRes(4);
		}
		else if (rawFloat <= .75) {
			_volume->setClaheRes(8);
		}
		else if (rawFloat <= 1) {
			_volume->setClaheRes(16);
		}

		
	}

	else if (item == _genClaheButton) {
		_volume->genClahe();
		((UIQuadElement*)_useClaheButton->getChild(0))->setColor(UI_RED_ACTIVE_COLOR);
		((UIText*)_useClaheButton->getChild(1))->setColor(UI_WHITE_COLOR);
		((UIText*)_useClaheButton->getChild(1))->setText("Enable CLAHE");
		 
		_volume->getCompute()->getOrCreateStateSet()->setDefine("CLAHE", true);
		_volume->setDirtyAll();
		
		((UIText*)_useClaheButton->getChild(1))->setText("Disable CLAHE");
	}
	else if (item == _useClaheButton) {
		bool on = ((UIText*)_useClaheButton->getChild(int(UI_ID::TEXT)))->getText() == "Enable CLAHE";
		_volume->getCompute()->getOrCreateStateSet()->setDefine("CLAHE", on);
		_volume->setDirtyAll();
		if(on)
			((UIText*)_useClaheButton->getChild(int(UI_ID::TEXT)))->setText("Disable CLAHE");
		else
			((UIText*)_useClaheButton->getChild(int(UI_ID::TEXT)))->setText("Enable CLAHE");
	}

	else if (item == _useClaheSelection) {
		bool on = ((UIText*)_useClaheSelection->getChild(int(UI_ID::TEXT)))->getText() == "Enable Selection";
		_volume->setCLAHEUseSelection(on);
		_volume->setDirtyAll();
		if(on)
			((UIText*)_useClaheSelection->getChild(int(UI_ID::TEXT)))->setText("Disable Selection");
		else
			((UIText*)_useClaheSelection->getChild(int(UI_ID::TEXT)))->setText("Enable Selection");
	}
	

	//AttnMaps
	else if (item == _useAttnMapsButton) {
		_useAttnMapsToggle = _useAttnMapsToggle == true ? false : true;
		std::cout << "use maps bool: " << _useAttnMapsToggle << std::endl;
		_volume->getCompute()->getOrCreateStateSet()->setDefine("ATTN_MAPS", _useAttnMapsToggle);
		_volume->setDirtyAll();
		if (_useAttnMapsToggle) {
			((UIText*)_useAttnMapsButton->getChild(int(UI_ID::TEXT)))->setText("Hide Maps");
			((UIQuadElement*)_useAttnMapsButton->getChild(int(UI_ID::BACKGROUND)))->setColor(UI_RED_ACTIVE_COLOR);
		}
		else {
			((UIText*)_useAttnMapsButton->getChild(int(UI_ID::TEXT)))->setText("Show Maps");
			((UIQuadElement*)_useAttnMapsButton->getChild(int(UI_ID::BACKGROUND)))->setColor(UI_INACTIVE_RED_COLOR);
		}
	}
	else if (item == _attnMapOnlyButton) {
 		_attnMapOnlyToggle = _attnMapOnlyToggle == true ? false : true;
		
 		_volume->getCompute()->getOrCreateStateSet()->setDefine("ATTN_MAP_ONLY", _attnMapOnlyToggle);
		_volume->setDirtyAll();
		if (_attnMapOnlyToggle) {
			((UIText*)_attnMapOnlyButton->getChild(int(UI_ID::TEXT)))->setText("Disable Maps Only");
			((UIQuadElement*)_attnMapOnlyButton->getChild(int(UI_ID::BACKGROUND)))->setColor(UI_RED_ACTIVE_COLOR);
		}
		else {
			((UIText*)_attnMapOnlyButton->getChild(int(UI_ID::TEXT)))->setText("Enable Maps Only");
			((UIQuadElement*)_attnMapOnlyButton->getChild(int(UI_ID::BACKGROUND)))->setColor(UI_INACTIVE_RED_COLOR);

		}
	}


	else if (item == _minAttnSlider) {
	_minAttentionLabel->setText(std::to_string(_minAttnSlider->getAdjustedValue()).substr(0, 4));
	_volume->getCompute()->getOrCreateStateSet()->setDefine("MIN_ATTN_VALUE", std::to_string(_minAttnSlider->getAdjustedValue()).substr(0, 4), osg::StateAttribute::ON);
	}

	//Marching Cubes
	else if (item == _GenMarchCubesButton) {
		if (!_volume->isMCInitialized()) {
 			_volume->intializeMC();
			updateMCUI(true);
 		}
		else {
			_volume->genMCs();
		}
	}
	

	else if (item == _UseMarchingCubesButton) {
			updateMCUI(_volume->toggleMC());
			
	}
	
	else if (item == _mcResSlider) {
		float rawFloat = _mcResSlider->getAdjustedValue();
		if (rawFloat <= .25) {
			_volume->_mcRes = 16;
			_mcResolutionLabel->setText("Lowest");
		}
		else if (rawFloat <= .50) {
			_volume->_mcRes = 8;
			_mcResolutionLabel->setText("Medium");
		}
		else if (rawFloat <= .75) {
			_volume->_mcRes = 4;
			_mcResolutionLabel->setText("High");
		}
		else if (rawFloat <= 1) {
			_volume->_mcRes = 2;
			_mcResolutionLabel->setText("Max");
		}
	
	}

	else if (item == _mcOrganSlider) {
		float rawFloat = _mcOrganSlider->getAdjustedValue();
		if (rawFloat <= .25) {
			_volume->_mcOrgan = ORGANID::COLON;
			_mcOrganPickLabel->setText("Colon");
		}
		else if (rawFloat <= .50) {
			_volume->_mcOrgan = ORGANID::ILLEUM;
			_mcOrganPickLabel->setText("Illeum");
		}
		else if (rawFloat <= .75) {
			_volume->_mcOrgan = ORGANID::VEIN;
			_mcOrganPickLabel->setText("Vena Cava");
		}
		else if (rawFloat <= 1) {
			_volume->_mcOrgan = ORGANID::AORTA;
			_mcOrganPickLabel->setText("Aorta");
		}
 	}
	//Print stl file if mcs are gen-ed
	else if (item == _PrintStlCallbackButton) {
		_volume->printSTLFile();
	}

	//Selection UI
	else if (item == _lockSelectionButton) {
		if (((cvr::UIText*)_lockSelectionButton->getChild(int(UI_ID::TEXT)))->getText() == "Lock Size") {
			_volume->lockSelection(true);
			((cvr::UIText*)_lockSelectionButton->getChild(int(UI_ID::TEXT)))->setText("Unlock Size");
		}
		else {
			_volume->lockSelection(false);
			((cvr::UIText*)_lockSelectionButton->getChild(int(UI_ID::TEXT)))->setText("Lock Size");
		}
	}
	else if (item == _resetSelectionButton) {
		if (((cvr::UIText*)_resetSelectionButton->getChild(int(UI_ID::TEXT)))->getText() == "Remove Selection") {
			_volume->removeSelection(true);
			((cvr::UIText*)_resetSelectionButton->getChild(int(UI_ID::TEXT)))->setText("Add Selection");
		}
		else {
			_volume->removeSelection(false);
			((cvr::UIText*)_resetSelectionButton->getChild(int(UI_ID::TEXT)))->setText("Remove Selection");
		}
	}
	else if (item == _toggleSelection) {
		if (((cvr::UIText*)_toggleSelection->getChild(int(UI_ID::TEXT)))->getText() == "Disable Selection") {
			_volume->disableSelection(true);
			((cvr::UIText*)_toggleSelection->getChild(int(UI_ID::TEXT)))->setText("Enable Selection");
		}
		else {
			_volume->disableSelection(false);
			((cvr::UIText*)_toggleSelection->getChild(int(UI_ID::TEXT)))->setText("Disable Selection");
		}
	}
	
	
}

void NewVolumeMenu::updateMCUI(bool on) {
	if (_volume->_mcIsReady == true) {
		((UIQuadElement*)_UseMarchingCubesButton->getChild(0))->setColor(UI_RED_ACTIVE_COLOR);
		((UIText*)_UseMarchingCubesButton->getChild(1))->setColor(UI_WHITE_COLOR);
	}
	if (!on) {		
		((UIText*)_UseMarchingCubesButton->getChild(1))->setText("Show Polygon");
	}
	else {
		((UIText*)_UseMarchingCubesButton->getChild(1))->setText("Hide Polygon");
		_printStlButton->getButton()->setDisabledColor(UI_NULL_COLOR_VEC4);
		_printStlButton->getText()->setColor(UI_WHITE_COLOR);
	}
}



void NewVolumeMenu::switchVolumes(int index) {

	if (_volume2 == nullptr)
		return;

	saveValues(_volume);
	_volume1 = _volume2;
	_volume2 = _volume;
	setNewVolume(_volume1, index);

	if (HelmsleyVolume::instance()->getVolumeIndex() == 0) {
		((UIText*)_volumeList->getChild(0))->setColor(osg::Vec4(UI_ACTIVE_COLOR));
		((UIText*)_volumeList->getChild(1))->setColor(osg::Vec4(UI_INACTIVE_COLOR));
	}
	else {
		((UIText*)_volumeList->getChild(1))->setColor(osg::Vec4(UI_ACTIVE_COLOR));
		((UIText*)_volumeList->getChild(0))->setColor(osg::Vec4(UI_INACTIVE_COLOR));
	}	
}

void VolumeMenuUpdate::Link() {
	if(_cP1 && _cP2)
	{
		osg::Vec3 pos = ((CuttingPlane*)_cP1)->getPosition();
		if (_prevPos != pos) {
			osg::Quat rot = ((CuttingPlane*)_cP1)->getRotation();
			((CuttingPlane*)_cP2)->setPosition(_prevPos);
			((CuttingPlane*)_cP2)->setRotation(rot);
			((CuttingPlane*)_cP2)->changePlane();
			_prevPos = pos;
		}
	}
	else {
		setLinkOff();
	}
}

void NewVolumeMenu::linkVolumes() {
	auto cps = HelmsleyVolume::instance()->getCuttingPlanes();
	/*if (cps.size() == 2) {
		osg::Vec3 pos = cps[0]->getPosition();
		osg::Quat rot = cps[0]->getRotation();
		cps[1]->setPosition(pos);
		cps[1]->setRotation(rot);
	}*/
	if (cps.size() == 2) {
		_updateCallback->setCuttingPlanes(cps[0], cps[1]);
		_updateCallback->setLinkOn();
	}
}

void NewVolumeMenu::clearVolumes() {
	bool _prevMask = _volume->hasMask();
	_updateCallback->setLinkOff();
	if (_volume1) {
		delete (_volume1);
		_volume1 = nullptr;
	}
	if (_volume2) {
		delete(_volume2);
		_volume2 = nullptr;
	}
	else if (_volume) {
		delete(_volume);
	}
	
}

void NewVolumeMenu::setNewVolume(VolumeGroup* volume , int index) {
	
	
	_volume = volume;

	if (_volume2 != nullptr) {
		HelmsleyVolume::instance()->getVolumeIndex() == 0 ? HelmsleyVolume::instance()->setVolumeIndex(1, true) :
			HelmsleyVolume::instance()->setVolumeIndex(0, true);
	}
	if (index > -1) {
		HelmsleyVolume::instance()->setVolumeIndex(index, false);
	}
	

	if (!_volume->hasMask()) {
		_maskMenu->getRootElement()->getGroup()->removeChild(_maskBknd->getGroup());
		_maskBknd->_parent = nullptr;
		_maskBknd->setActive(false);
	}
	else if (_volume->hasMask() && !_prevMask) {
		_maskMenu->addChild(_maskBknd);
		_maskBknd->setActive(true);
	}

	_tentWindow->setVolume(_volume);
	_colorDisplay->addUniform(_volume->_computeUniforms["ContrastBottom"]);
	_colorDisplay->addUniform(_volume->_computeUniforms["ContrastTop"]);
	_colorDisplay->addUniform(_volume->_computeUniforms["leftColor"]);
	_colorDisplay->addUniform(_volume->_computeUniforms["rightColor"]);
	_opacityDisplay->addUniform(_volume->_computeUniforms["OpacityCenter"]);
	_opacityDisplay->addUniform(_volume->_computeUniforms["OpacityWidth"]);
	_opacityDisplay->addUniform(_volume->_computeUniforms["OpacityTopWidth"]);
	_opacityDisplay->addUniform(_volume->_computeUniforms["OpacityMult"]);
	_opacityDisplay->addUniform(_volume->_computeUniforms["Lowest"]);
	_opacityDisplay->addUniform(_volume->_computeUniforms["ContrastBottom"]);
	_opacityDisplay->addUniform(_volume->_computeUniforms["ContrastTop"]);
	_opacityDisplay->addUniform(_volume->_computeUniforms["TriangleCount"]);
	_opacityDisplay->addUniform(_volume->_computeUniforms["leftColor"]);
	_opacityDisplay->addUniform(_volume->_computeUniforms["rightColor"]);
	_opacityColorDisplay->addUniform(_volume->_computeUniforms["OpacityCenter"]);
	_opacityColorDisplay->addUniform(_volume->_computeUniforms["OpacityWidth"]);
	_opacityColorDisplay->addUniform(_volume->_computeUniforms["OpacityTopWidth"]);
	_opacityColorDisplay->addUniform(_volume->_computeUniforms["OpacityMult"]);
	_opacityColorDisplay->addUniform(_volume->_computeUniforms["Lowest"]);
	_opacityColorDisplay->addUniform(_volume->_computeUniforms["ContrastBottom"]);
	_opacityColorDisplay->addUniform(_volume->_computeUniforms["ContrastTop"]);
	_opacityColorDisplay->addUniform(_volume->_computeUniforms["TriangleCount"]);
	_opacityColorDisplay->addUniform(_volume->_computeUniforms["leftColor"]);
	_opacityColorDisplay->addUniform(_volume->_computeUniforms["rightColor"]);

	if (!volume->values.saved) {
		resetValues();
	}
	else {
		fillFromVolume(volume);
	}

}

void NewVolumeMenu::saveValues(VolumeGroup* vg) {
	vg->values.contrastData = { _contrastBottom->getAdjustedValue(), _contrastTop->getAdjustedValue(), _brightness->getAdjustedValue() };
	if (_transferFunctionRadial->getCurrent() == 0) {
		vg->values.tf = 0;
	}
	else {
		vg->values.tf = 1;
	}
	
	vg->values.tentCount = _triangleIndex+1;
	vg->values.opacityData.clear();
	for (int i = 0; i < vg->values.tentCount; i++) {
		vg->values.opacityData.push_back(_tentWindow->getPresetData(i));
	}
	
	///////////////////Cutting Plane///////////////////////
	std::vector<CuttingPlane*> cPs = HelmsleyVolume::instance()->getCuttingPlanes();
	std::vector<CuttingPlane*>::iterator it = cPs.begin();

	while (it != cPs.end()) {
		if ((*it)->getVolume() == vg)
		{
			vg->values.cpPos = (*it)->getPosition();
			vg->values.cpRot = (*it)->getRotation();
			vg->values.cpToggle = true;
			break;
		}
		else
		{
			vg->values.cpToggle = false;
		}
		++it;
	}

	std::vector<CurvedQuad*> curvedMenuItems = _toolMenu->getCurvedMenuItems();

	int index = 0;
	for (std::vector<CurvedQuad*>::iterator it = curvedMenuItems.begin(); it != curvedMenuItems.end(); ++it) {
		vg->values.toolToggles[index] = (*it)->isOn();
		index++;
	}

	///////////////////Masks////////////////////////
	std::vector<VisibilityToggle*> organToggles = {_organs, _colon, _kidney, _bladder,
													_spleen, _illeum, _aorta, _vein};

	vg->values.masks.clear();
	for (int i = 0; i < organToggles.size(); i++) {
		organToggles[i]->isOn() ? vg->values.masks.push_back(true) : vg->values.masks.push_back(false);
	}

	vg->values.saved = true;
}

void NewVolumeMenu::fillFromVolume(VolumeGroup* vg) {
	_tentWindow->clearTents();
	clearRegionPreviews();
	int tentCount = vg->values.tentCount;
	_tentWindow->setVolume(vg);
	_tentWindowOnly->setVolume(vg);
	_triangleIndex = -1;
	//////////////////opacities//////////////////
	for (int i = 0; i < tentCount; i++) {
		_triangleIndex++;
		char presetIndex = '0' + i;
		std::string presetName = "tent ";
		presetName += presetIndex;
		addRegion();
		float center = vg->values.opacityData[i][0];
		float bottomWidth = vg->values.opacityData[i][1];
		float topWidth = vg->values.opacityData[i][2];
		float height = vg->values.opacityData[i][3];
		float lowest = vg->values.opacityData[i][4];
		_tentWindow->fillTentDetails(_triangleIndex, center, bottomWidth, topWidth, height, lowest);
	}

	/////////////////tf & contrast////////////


	int tfid = vg->values.tf;

	_transferFunctionRadial->setCurrent(tfid);
	useTransferFunction(tfid);

	//contrast Data 0 = bottom, 1 = top, 2 = brightness
	float contrastLow = vg->values.contrastData[0];
	float contrastHigh = vg->values.contrastData[1];
	float brightness = vg->values.contrastData[2];
	setContrastValues(contrastLow, contrastHigh, brightness);

	///////////////Masks////////////////////
	if (vg->hasMask()) {
		_maskMenu->addChild(_maskBknd);
		_maskBknd->setActive(true);

		std::vector<bool> volumeToggles = vg->values.masks ;
		

		std::vector<VisibilityToggle*> organToggles = { _organs, _colon, _kidney, _bladder,
													_spleen, _illeum, _aorta, _vein };

		std::vector<std::string> organNames = { "ORGANS_ONLY", "COLON","KIDNEY", "BLADDER","SPLEEN","ILLEUM", "AORTA", "VEIN" };
		for (int i = 0; i < organToggles.size(); i++) {
			if (volumeToggles[i] ^ organToggles[i]->isOn()) {
				organToggles[i]->toggle();

				
				vg->getCompute()->getOrCreateStateSet()->setDefine(organNames[i], volumeToggles[i]);
			}
		}
		_volume->setDirtyAll();
	}
	else {
		_maskMenu->getRootElement()->getGroup()->removeChild(_maskBknd->getGroup());
		_maskBknd->_parent = nullptr;
		_maskBknd->setActive(false);
	}

	//Tools
#define NUMBER_OF_VOLUME_DEPENDENT_TOOLS 4
	////////////Cutting Plane/////////////
	std::vector<CurvedQuad*> curvedMenuItems = _toolMenu->getCurvedMenuItems();
	int index = 0;
	for (std::vector<CurvedQuad*>::iterator it = curvedMenuItems.begin(); it != curvedMenuItems.end(); ++it) {
		if (index == int(TOOLID::CUTTINGPLANE)) {
			if (!(*it)->isOn() && _volume->values.toolToggles[index]) {
				(*it)->turnOn();
				(*it)->setColor(UI_RED_ACTIVE_COLOR);
			}
			if ((*it)->isOn() && !_volume->values.toolToggles[index]) {
				(*it)->turnOff();
				(*it)->setColor(UI_BACKGROUND_COLOR);
			}
			index++;
			continue;
		}
		if (index < NUMBER_OF_VOLUME_DEPENDENT_TOOLS) {
			if (!(*it)->isOn() && _volume->values.toolToggles[index]) {
				(*it)->toggle();
				(*it)->setColor(UI_RED_ACTIVE_COLOR);
			}
			if ((*it)->isOn() && !_volume->values.toolToggles[index]) {
				(*it)->toggle();
				(*it)->setColor(UI_BACKGROUND_COLOR);
			}
			index++;
		}
		else
			break;
	}

}

void NewVolumeMenu::useTransferFunction(int tfID) {
	if (tfID == 0)
	{
		_transferFunction = "vec3(ra.r);";
		((UIQuadElement*)_blacknwhite->getChild(0)->getChild(0))->setColor(UI_BLUE_COLOR2);
		((UIQuadElement*)_rainbow->getChild(0)->getChild(0))->setColor(UI_BACKGROUND_COLOR);
		((UIQuadElement*)_custom->getChild(0)->getChild(0))->setColor(UI_BACKGROUND_COLOR);
 


		_colSliderList[0]->getGeode()->setNodeMask(0);
		_colSliderList[1]->getGeode()->setNodeMask(0);
		_tentMenu->getRootElement()->getGroup()->removeChild(_cp->getGroup());
		_cp->_parent = nullptr;
	}
	else if (tfID == 1)
	{
		_transferFunction = "hsv2rgb(vec3(ra.r * 0.8, 1, 1));";
		((UIQuadElement*)_blacknwhite->getChild(0)->getChild(0))->setColor(UI_BACKGROUND_COLOR);
		((UIQuadElement*)_rainbow->getChild(0)->getChild(0))->setColor(UI_BLUE_COLOR2);
		((UIQuadElement*)_custom->getChild(0)->getChild(0))->setColor(UI_BACKGROUND_COLOR);
		_colSliderList[0]->getGeode()->setNodeMask(0);
		_colSliderList[1]->getGeode()->setNodeMask(0);
		_tentMenu->getRootElement()->getGroup()->removeChild(_cp->getGroup());
		_cp->_parent = nullptr;
	}
	else if (tfID == 2)
	{
		_transferFunction = "custom(vec3(ra.r));";;
		((UIQuadElement*)_blacknwhite->getChild(0)->getChild(0))->setColor(UI_BACKGROUND_COLOR);
		((UIQuadElement*)_rainbow->getChild(0)->getChild(0))->setColor(UI_BACKGROUND_COLOR);
		((UIQuadElement*)_custom->getChild(0)->getChild(0))->setColor(UI_BLUE_COLOR2);
		_colSliderList[0]->getGeode()->setNodeMask(0xffffffff);
		_colSliderList[1]->getGeode()->setNodeMask(0xffffffff);
		_tentMenu->addChild(_cp);
	}
	_volume->getCompute()->getOrCreateStateSet()->setDefine("COLOR_FUNCTION", _transferFunction, osg::StateAttribute::ON);
	_colorDisplay->setShaderDefine("COLOR_FUNCTION", _transferFunction, osg::StateAttribute::ON);
	_opacityDisplay->setShaderDefine("COLOR_FUNCTION", _transferFunction, osg::StateAttribute::ON);
	_opacityColorDisplay->setShaderDefine("COLOR_FUNCTION", _transferFunction, osg::StateAttribute::ON);
	//upDatePreviewDefines(_transferFunction);
	_volume->setDirtyAll();

	osg::Vec3 pos = _container->getPosition();
}

void NewVolumeMenu::upDatePreviewDefines(std::string tf) {
	for (int i = 0; i < _triangleIndex+1; i++) {
		ShaderQuad* sq = (ShaderQuad*)_triangleCallbacks[i]->getChild(2);
		sq->setShaderDefine("COLOR_FUNCTION", tf, osg::StateAttribute::ON);
	}
}

bool NewVolumeMenu::checkTriangleCallbacks(UICallbackCaller* item) {
	bool found = false;
	int i = 0;
	while (i <	_triangleIndex+1) {
		if (item == _triangleCallbacks[i]) {
			found = true;
			_tentWindow->setTent(i);
			break;
		}
		i++;
	}
	return found;
}

bool NewVolumeMenu::checkTriangleVisCallbacks(UICallbackCaller* item) {
	bool found = false;
	int i = 0;
	while (i < _triangleIndex + 1) {
		if (item == (VisibilityToggle*)_triangleCallbacks[i]->getChild(1)) {
			found = true;
			_tentWindow->toggleTent(i);
			break;
		}
		i++;
	}
	return found;
}

bool NewVolumeMenu::checkPresetCallbacks(UICallbackCaller* item) {
	bool found = false;
	for (int i = 0; i < _presetCallbacks.size(); i++) {
		if (item == _presetCallbacks[i]) {
			usePreset(((UIText*)_presetCallbacks[i]->getChild(0))->getText());
			_maskMenu->getRootElement()->getGroup()->removeChild(_presetPopup->getRootElement()->getGroup());
			_presetPopup->getRootElement()->_parent = nullptr;
			found = true;
			break;
		}
	}
	return found;
}

bool NewVolumeMenu::checkColorSliderCallbacks(UICallbackCaller* item) {
	bool found = false;
	for (int i = 0; i < _colSliderList.size(); i++) {
		if (item == _colSliderList[i]) {
			_colorSliderIndex = i;
			found = true;
			break;
		}
	}
	return found;
}

//#include <sys/types.h>
//#include <sys/stat.h>

void NewVolumeMenu::usePreset(std::string filename) {
	std::string currPath = cvr::ConfigManager::getEntry("Plugin.HelmsleyVolume.PresetsFolder", "C:/", false);
	std::string presetPath = currPath + "\\" + filename + ".yml";
	YAML::Node config = YAML::LoadFile(presetPath);

	std::string datasetPath = config["series name"].as<std::string>();
	FileSelector* fs = HelmsleyVolume::instance()->getFileSelector();
	fs->loadVolumeOnly(true, datasetPath);
	_tentWindow->clearTents();
	clearRegionPreviews();
	_triangleIndex = -1;
	//_tentWindow->setVolume(_volume);
	_tentWindowOnly->setVolume(_volume);
	//////////////////opacities//////////////////
	for (int i = 0; i < config["tent count"].as<int>(); i++) {
		_triangleIndex++;
		char presetIndex = '0' + i;
		std::string presetName = "tent ";
		presetName+=presetIndex;
		addRegion();
		float center = config[presetName]["opacity"][0].as<float>();
		float bottomWidth = config[presetName]["opacity"][1].as<float>();
		float topWidth = config[presetName]["opacity"][2].as<float>();
		float height = config[presetName]["opacity"][3].as<float>();
		float lowest = config[presetName]["opacity"][4].as<float>();
		_tentWindow->fillTentDetails(_triangleIndex, center, bottomWidth, topWidth, height, lowest);	
	}

	/////////////////tf & contrast////////////
	int tfid = -1;
	if (config["color scheme"].as<std::string>() == "grayscale") {
		tfid = 0;
	}
	else if (config["color scheme"].as<std::string>() == "color") {
		tfid = 1;
	}
	_transferFunctionRadial->setCurrent(tfid);
	useTransferFunction(tfid);
	float contrastLow = config["contrast"][0].as<float>();
	float contrastHigh = config["contrast"][1].as<float>();
	float brightness = config["contrast"][2].as<float>();
	setContrastValues(contrastLow, contrastHigh, brightness);

	///////////////Masks////////////////////
	if (_volume->hasMask()) {


		bool val = (bool)(config["mask"][0].as<int>());

		if (_organs->isOn() != val)
			_organs->toggle();
		_volume->getCompute()->getOrCreateStateSet()->setDefine("ORGANS_ONLY", val);
		val = config["mask"][1].as<int>();
		if (_colon->isOn() != val)
			_colon->toggle();
		_volume->getCompute()->getOrCreateStateSet()->setDefine("COLON", val);
		val = config["mask"][2].as<int>();
		if (_kidney->isOn() != val)
			_kidney->toggle();
		_volume->getCompute()->getOrCreateStateSet()->setDefine("KIDNEY", val);
		val = config["mask"][3].as<int>();
		if (_bladder->isOn() != val)
			_bladder->toggle();
		_volume->getCompute()->getOrCreateStateSet()->setDefine("BLADDER", val);
		val = config["mask"][4].as<int>();
		if (_spleen->isOn() != val)
			_spleen->toggle();
		_volume->getCompute()->getOrCreateStateSet()->setDefine("SPLEEN", val);
		val = config["mask"][5].as<int>();
		if (_illeum->isOn() != val)
			_illeum->toggle();
		_volume->getCompute()->getOrCreateStateSet()->setDefine("ILLEUM", val);
		val = config["mask"][6].as<int>();
		if (_aorta->isOn() != val)
			_aorta->toggle();
		_volume->getCompute()->getOrCreateStateSet()->setDefine("AORTA", val);
		if (_vein->isOn() != val)
			_vein->toggle();
		_volume->getCompute()->getOrCreateStateSet()->setDefine("VEIN", val);

		_volume->setDirtyAll();
	}
	

	//////////Cutting Plane/////////////

	//ToolToggle* cuttingPlaneTool = _toolMenu->getCuttingPlaneTool();
	CurvedQuad* cuttingPlaneTool = _toolMenu->getTool(int(TOOLID::CUTTINGPLANE));

		if (config["cutting plane value"].as<int>() == 0) {
			cuttingPlaneTool->setColor(UI_BACKGROUND_COLOR);
			cuttingPlaneTool->turnOff();
			
		}
		else {
			osg::Vec3 cutPpos;
			cutPpos.x() = config["cutting plane position"][0].as<float>();
			cutPpos.y() = config["cutting plane position"][1].as<float>();
			cutPpos.z() = config["cutting plane position"][2].as<float>();

			osg::Quat cutPQuat;
			cutPQuat.x() = config["cutting plane rotation"][0].as<float>();
			cutPQuat.y() = config["cutting plane rotation"][1].as<float>();
			cutPQuat.z() = config["cutting plane rotation"][2].as<float>();
			cutPQuat.w() = config["cutting plane rotation"][3].as<float>();


			if (!cuttingPlaneTool->isOn()) {
				cuttingPlaneTool->setColor(UI_RED_ACTIVE_COLOR);
				cuttingPlaneTool->turnOn();
			}

			CuttingPlane* cutP = HelmsleyVolume::instance()->HelmsleyVolume::createCuttingPlane();
			cutP->setPosition(cutPpos);
			cutP->setRotation(cutPQuat);

		}
	

	CurvedQuad* mTool = _toolMenu->getTool(int(TOOLID::RULER));
	CurvedQuad* centerTool = _toolMenu->getTool(int(TOOLID::CENTERLINE));
	CurvedQuad* screenTool = _toolMenu->getTool(int(TOOLID::SCREENSHOT));
	if (mTool->isOn()) {
		mTool->toggle();
 		HelmsleyVolume::instance()->deactivateMeasurementTool(0);
	}
	if (centerTool->isOn()) {
		centerTool->toggle();
 		HelmsleyVolume::instance()->toggleCenterlineTool(false);
	}
	if (screenTool->isOn()) {
		screenTool->toggle();
 		HelmsleyVolume::instance()->toggleScreenshotTool(false);
	}
}

void NewVolumeMenu::resetValues() {
	_tentWindow->clearTents();
	clearRegionPreviews();
	

	_triangleIndex = 0;
	_tentWindow->setVolume(_volume);
	_tentWindowOnly->setVolume(_volume);
	//////////////////opacities//////////////////

		
		addRegion();
		float center = .5f;
		float bottomWidth = 1.0f;
		float topWidth = 1.0f;
		float height = 1.0f;
		float lowest = 0.0f;
		_tentWindow->fillTentDetails(_triangleIndex, center, bottomWidth, topWidth, height, lowest);
	

	/////////////////tf & contrast////////////

	
	int tfid = 0;
	
	_transferFunctionRadial->setCurrent(tfid);
	useTransferFunction(tfid);
	float contrastLow = 0.0f;
	float contrastHigh = 1.0f;
	float brightness = 0.5f;
	setContrastValues(contrastLow, contrastHigh, brightness);

	_colSliderList[0]->setColor(osg::Vec4(1.0,0.0,0.0,1.0));
	_colSliderList[1]->setColor(osg::Vec4(1.0,1.0,1.0,1.0));

	///////////////Masks////////////////////
	if (_volume->hasMask()) {
		/*_maskMenu->addChild(_maskBknd);
		_maskBknd->setActive(true);*/

		

		if (!_organs->isOn())
			_organs->toggle();
		_volume->getCompute()->getOrCreateStateSet()->setDefine("ORGANS_ONLY", true);
		
		if (_colon->isOn())
			_colon->toggle();
		_volume->getCompute()->getOrCreateStateSet()->setDefine("COLON", false);
 		if (_kidney->isOn())
			_kidney->toggle();
		_volume->getCompute()->getOrCreateStateSet()->setDefine("KIDNEY", false);
 		if (_bladder->isOn())
			_bladder->toggle();
		_volume->getCompute()->getOrCreateStateSet()->setDefine("BLADDER", false);
 		if (_spleen->isOn())
			_spleen->toggle();
		_volume->getCompute()->getOrCreateStateSet()->setDefine("SPLEEN", false);
 		if (_illeum->isOn())
			_illeum->toggle();
		_volume->getCompute()->getOrCreateStateSet()->setDefine("ILLEUM", false);
 		if (_aorta->isOn())
			_aorta->toggle();
		_volume->getCompute()->getOrCreateStateSet()->setDefine("AORTA", false);
		if (_vein->isOn())
			_vein->toggle();
		_volume->getCompute()->getOrCreateStateSet()->setDefine("VEIN", false);

		_volume->setDirtyAll();
	}
	/*else {
		_maskMenu->getRootElement()->getGroup()->removeChild(_maskBknd->getGroup());
		_maskBknd->_parent = nullptr;
		_maskBknd->setActive(false);
	}*/

	//Tools
#define NUMBER_OF_VOLUME_DEPENDENT_TOOLS 4
	////////////Cutting Plane/////////////
	std::vector<CurvedQuad*> curvedMenuItems = _toolMenu->getCurvedMenuItems();
	int index = 0;
	for (std::vector<CurvedQuad*>::iterator it = curvedMenuItems.begin(); it != curvedMenuItems.end(); ++it) {
		if (index == int(TOOLID::CUTTINGPLANE)) {	
 			(*it)->turnOff();
			(*it)->setColor(UI_BACKGROUND_COLOR);
 			index++;
			continue;
		}
		else {
			if ((*it)->isOn()) {
				(*it)->toggle();
				(*it)->setColor(UI_BACKGROUND_COLOR);
			}
			index++;
		}
		
		
	}

	//TOOLS
	

	/////Etc/////
	((UIText*)_volumeList->getChild(0))->setColor(UI_ACTIVE_COLOR);
	((UIText*)_volumeList->getChild(1))->setColor(UI_INACTIVE_COLOR);

}

void NewVolumeMenu::clearRegionPreviews() {
	for(int i = 0; i < _triangleIndex+1; i++) {
		_triangleList->removeChild(_triangleCallbacks[i]);
	}
}

Tent* NewVolumeMenu::addRegion() {
	_triangleCallbacks[_triangleIndex] = new CallbackButton();
	_triangleCallbacks[_triangleIndex]->setCallback(this);
	_triangleCallbacks[_triangleIndex]->setPercentPos(osg::Vec3(0.20, 0.0, 0.0));

	osg::Vec3 color = triangleColors[_triangleIndex];
	std::string name = "Triangle " + std::to_string(_triangleIndex);
	UIText* label = new UIText(name, 50.0f, osgText::TextBase::LEFT_TOP);
	label->setColor(osg::Vec4(color, 1.0));
	VisibilityToggle* vT = new VisibilityToggle("");
	vT->getChild(0)->setPercentSize(osg::Vec3(1, 1, 1));
	vT->getChild(0)->setPercentPos(osg::Vec3(0, 0, 0));
	vT->setCallback(this);
	vT->setPercentPos(osg::Vec3(-0.20, 0.0, 0.1));
	vT->setPercentSize(osg::Vec3(0.185, 0.0, 0.5));
	vT->toggle();

	Tent* tent = _tentWindow->addTent(_triangleIndex, color);

	ShaderQuad* sq = new ShaderQuad();
	std::string frag = HelmsleyVolume::loadShaderFile("shaderQuadPreview.frag");
	std::string vert = HelmsleyVolume::loadShaderFile("transferFunction.vert");
	osg::Program* p = new osg::Program;
	p->setName("trianglePreview");
	p->addShader(new osg::Shader(osg::Shader::VERTEX, vert));
	p->addShader(new osg::Shader(osg::Shader::FRAGMENT, frag));
	sq->setProgram(p);
	sq->setPercentSize(osg::Vec3(0.5, 1.0, 0.4));
	sq->setPercentPos(osg::Vec3(.27, 0.0, 0.1));
	/*sq->addUniform(tent->_centerUniform);
	sq->addUniform(tent->_widthUniform);*/
	sq->setShaderDefine("COLOR_FUNCTION", _transferFunction, osg::StateAttribute::ON);

	sq->addUniform(_volume->_computeUniforms["ContrastBottom"]);
	sq->addUniform(_volume->_computeUniforms["ContrastTop"]);
	sq->addUniform(_volume->_computeUniforms["leftColor"]);
	sq->addUniform(_volume->_computeUniforms["rightColor"]);



	_triangleList->addChild(_triangleCallbacks[_triangleIndex]);



	_triangleCallbacks[_triangleIndex]->addChild(label);
	_triangleCallbacks[_triangleIndex]->addChild(vT);
	_triangleCallbacks[_triangleIndex]->addChild(sq);

	return nullptr;
	//return tent;
}

void NewVolumeMenu::setContrastValues(float contrastLow, float contrastHigh, float brightness) {
	_volume->_computeUniforms["ContrastBottom"]->set(contrastLow);
	_volume->_computeUniforms["ContrastTop"]->set(contrastHigh);
	_volume->_computeUniforms["Brightness"]->set(brightness);
	_contrastBottom->setPercent(contrastLow);
	_contrastTop->setPercent(contrastHigh);
	_brightness->setPercent(brightness);
	float realValue = brightness - .5;
	/*std::string value;
	if (realValue >= 0.0) {
		value = "+" + std::to_string(realValue).substr(0, 4);
	}
	else {
		value = std::to_string(realValue).substr(0, 5);
	}
	_brightValueLabel->setText(value);*/

	std::string low = std::to_string(contrastLow).substr(0, 4);
	std::string high = std::to_string(contrastHigh).substr(0, 4);
	_contrastValueLabel->setText("Low " + low + " / " + "High " + high);
	_volume->setDirtyAll();
}

void NewVolumeMenu::savePreset(){
	std::vector<float> opacityData;//0=Center, 1=BottomWidth, 2=TopWidth, 3= Height, 4 = Lowest
	std::vector<float> contrastData = { _contrastBottom->getAdjustedValue(), _contrastTop->getAdjustedValue(), _brightness->getAdjustedValue()};
	std::string tf;
	if (_transferFunctionRadial->getCurrent() == 0) {
		tf = "grayscale";
	}
	else {
		tf = "color";
	}
	YAML::Emitter out;
	std::string name = "Preset";
	char nextPresetIndex = '0' + _presetCallbacks.size();
	name += nextPresetIndex;

	out << YAML::BeginMap;
	out << YAML::Key << "Name";
	out << YAML::Value << name;
	out << YAML::Key << "contrast";
	out << YAML::Value << YAML::Flow << contrastData;
	out << YAML::Key << "color scheme";
	out << YAML::Value << tf;
	out << YAML::Key << "tent count";
	out << YAML::Value << _triangleIndex + 1;

	for (int i = 0; i < _triangleIndex + 1; i++) {
		opacityData = _tentWindow->getPresetData(i);
		std::string tentString = "tent ";
		char tentIndex = '0' + i;
		std::string tentStringIndex = tentString += tentIndex;
		out << YAML::Key << tentStringIndex;
		out << YAML::Value << YAML::BeginMap;
		out << YAML::Key << "opacity";
		out << YAML::Value << YAML::Flow << opacityData;
		out << YAML::EndMap;
	}
	///////////////////Cutting Plane///////////////////////
	std::vector<CuttingPlane*> cPs = HelmsleyVolume::instance()->getCuttingPlanes();
	if (!cPs.empty()) {
		out << YAML::Key << "cutting plane value";
		out << YAML::Value << 1;


		osg::Vec3 cPPos = cPs[0]->getPosition();
		std::vector<float> fPos;
		fPos.push_back(cPPos.x());
		fPos.push_back(cPPos.y());
		fPos.push_back(cPPos.z());
		out << YAML::Key << "cutting plane position";
		out << YAML::Value << YAML::Flow << fPos;


		osg::Quat cPRot = cPs[0]->getRotation();
		std::vector<float> fQuat;
		fQuat.push_back(cPRot.x());
		fQuat.push_back(cPRot.y());
		fQuat.push_back(cPRot.z());
		fQuat.push_back(cPRot.w());

		out << YAML::Key << "cutting plane rotation";
		out << YAML::Value << YAML::Flow << fQuat;
	}
	else {
		out << YAML::Key << "cutting plane value";
		out << YAML::Value << 0;
	}
	///////////////////Masks////////////////////////
	if (_volume->hasMask()) {
		out << YAML::Key << "mask";
		std::vector<unsigned char> maskValues;
		_organs->isOn() ? maskValues.push_back('1') : maskValues.push_back('0');
		_colon->isOn() ? maskValues.push_back('1') : maskValues.push_back('0');
		_kidney->isOn() ? maskValues.push_back('1') : maskValues.push_back('0');
		_bladder->isOn() ? maskValues.push_back('1') : maskValues.push_back('0');
		_spleen->isOn() ? maskValues.push_back('1') : maskValues.push_back('0');
		_illeum->isOn() ? maskValues.push_back('1') : maskValues.push_back('0');
		_aorta->isOn() ? maskValues.push_back('1') : maskValues.push_back('0');
		_vein->isOn() ? maskValues.push_back('1') : maskValues.push_back('0');
		out << YAML::Value << YAML::Flow << maskValues;
	}


	///////////////////Dataset//////////////////////
	out << YAML::Key << "series name";
	FileSelector* fs = HelmsleyVolume::instance()->getFileSelector();
	std::string datasetPath = fs->getCurrPath();
	out << YAML::Value << datasetPath;

	out << YAML::EndMap;

	std::string currPath = cvr::ConfigManager::getEntry("Plugin.HelmsleyVolume.PresetsFolder", "C:/", false);
	std::string presetPath = currPath + "\\" + name + ".yml";
	std::ofstream fout(presetPath);
	fout << out.c_str();

	CallbackButton* presetbutton = new CallbackButton();
	presetbutton->setCallback(this);
	UIText* presetText = new UIText(name, 40.0f, osgText::TextBase::CENTER_CENTER);
	presetbutton->addChild(presetText);
	_presetUIList->addChild(presetbutton);
	_presetCallbacks.push_back(presetbutton);
}

inline osg::Matrix ToEulerAngles(osg::Quat q) {
	

	// roll (x-axis rotation)
	double sinr_cosp = 2 * (q.w() * q.x() + q.y() * q.z());
	double cosr_cosp = 1 - 2 * (q.x() * q.x() + q.y() * q.y());
	double rollAngle = std::atan2(sinr_cosp, cosr_cosp);

	// pitch (y-axis rotation)
	double sinp = 2 * (q.w() * q.y() - q.z() * q.x());

	double pitch;
	if (std::abs(sinp) >= 1)
		pitch = std::copysign(osg::PI / 2, sinp); // use 90 degrees if out of range
	else
		pitch = std::asin(sinp);

	// yaw (z-axis rotation)
	double siny_cosp = 2 * (q.w() * q.z() + q.x() * q.y());
	double cosy_cosp = 1 - 2 * (q.y() * q.y() + q.z() * q.z());
	double yaw = std::atan2(siny_cosp, cosy_cosp);

	osg::Matrix rotMat;
	rotMat.osg::Matrix::makeRotate(rollAngle, osg::Vec3(0, 0, 1));
	std::cout << "rollangle: " << rollAngle << std::endl;

	rotMat.osg::Matrix::makeRotate(yaw, osg::Vec3(0, 1, 0));
	std::cout << "yawangle: " << yaw << std::endl;
	rotMat.osg::Matrix::makeRotate(pitch, osg::Vec3(1, 0, 0));

	return rotMat;
}

void NewVolumeMenu::saveYamlForCinematic() {
	//volume properties
	osg::Matrix worldMatrix = _volume->getObjectToWorldMatrix() * _scene->getObjectToWorldMatrix();
	osg::Vec3f wmScale = worldMatrix.getScale();

	osg::Vec3d osgtrans = worldMatrix.getTrans();
	osgtrans.x()/=wmScale.x(); osgtrans.y()/=wmScale.y(); osgtrans.z()/=wmScale.z(); 
	
	//std::vector<float> trans; trans.push_back(osgtrans.x());  trans.push_back(osgtrans.z());  trans.push_back(osgtrans.y());
	std::vector<float> trans; trans.push_back(0);  trans.push_back(0);  trans.push_back(0);//HARD CODED
	
	float total = wmScale.x() + wmScale.y() + wmScale.z();
 	//std::vector<float> scaleVec = { wmScale.x()/total, wmScale.z()/total, wmScale.y()/total };
 	std::vector<float> scaleVec = { 1, 1, 0.454};//HARD CODED
	
	
	//osg::Quat osgrot = worldMatrix.getRotate();
	osg::Quat osgrot(0,0,0,1);// = worldMatrix.getRotate();
	

	//Camera properties
	osg::Vec3 osgpos;
	osg::Vec3 osgcenter;
	osg::Vec3 osgup;

	HelmsleyVolume::instance()->getScreenshotTool()->getCam()->getViewMatrixAsLookAt(osgpos, osgcenter, osgup);

	osgpos.x() /= wmScale.x(); osgpos.y() /= wmScale.y(); osgpos.z() /= wmScale.z();
	//std::vector<float> pos; pos.push_back(osgpos.x());  pos.push_back(osgpos.z());  pos.push_back(osgpos.y()); 
	std::vector<float> pos; pos.push_back(0);  pos.push_back(0);  pos.push_back(1.5); //HARD CODED


	osgcenter.x() /= wmScale.x(); osgcenter.y() /= wmScale.y(); osgcenter.z() /= wmScale.z();
	//std::vector<float> center; center.push_back(osgcenter.x());  center.push_back(osgcenter.z());  center.push_back(osgcenter.y());
	//std::vector<float> center; center.push_back(osgcenter.x());  center.push_back(osgcenter.z());  center.push_back(osgcenter.y());
	//std::vector<float> center = trans;
	std::vector<float> center = { 0.0,0.0,2.0 };//HARD CODED

	//osgup.x() /= wmScale.x(); osgup.y() /= wmScale.y(); osgup.z() /= wmScale.z();
	osgup.x() = 0; osgup.y() = 0; osgup.z() = 1;
	std::vector<float> up; up.push_back(osgup.x());  up.push_back(osgup.z());  up.push_back(osgup.y());
	//up[0]*=-1.f; up[1]*=-1.f; up[2]*=-1.f;
	

	std::string renderMode = "Texture-based";
	 
	std::vector<std::vector<float>> opacities;
	std::vector<float> opacityData;
	std::vector<float> contrastData = { _contrastBottom->getAdjustedValue(), _contrastTop->getAdjustedValue()};
	
	



	YAML::Emitter out;
	std::string name = "cineparams";
	

	out << YAML::BeginMap;
	out << YAML::Key << "name";
	out << YAML::Value << name;

	out << YAML::Key << "volume";
	out << YAML::Value << YAML::BeginMap;
	out << YAML::Key << "pos";
	out << YAML::Value << YAML::Flow << trans;
	out << YAML::Key << "rotation";
	
	osg::Matrix rotMat; rotMat.osg::Matrix::makeRotate(osgrot);
   //osg::Matrix rotMat = ToEulerAngles(osgrot);


	std::vector<double> rotArray;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			rotArray.push_back(rotMat(i, j));
		}
	}

	out << YAML::Value << YAML::Flow << rotArray;
	out << YAML::Key << "scale";
	out << YAML::Value << YAML::Flow << scaleVec;
	out << YAML::EndMap;

	out << YAML::Key << "camera";
	out << YAML::Value << YAML::BeginMap;
	out << YAML::Key << "pos";
	out << YAML::Value << YAML::Flow << pos;
	out << YAML::Key << "up";
	out << YAML::Value << YAML::Flow << up;
	out << YAML::Key << "center";
	out << YAML::Value << YAML::Flow << center;
	out << YAML::EndMap;

	out << YAML::Key << "render mode";
	out << YAML::Value << renderMode;

	out << YAML::Key << "transfer function";
	out << YAML::Value << YAML::BeginMap;
	out << YAML::Key << "contrast";
	out << YAML::Value << YAML::Flow << contrastData;


	opacityData = _tentWindow->getCinePresetData(0);
	
	opacities.push_back(opacityData);


	out << YAML::Key << "opacity";
	out  << YAML::BeginSeq << YAML::Flow << opacityData;
	out << YAML::EndSeq;
	std::string tf;
	std::vector<std::vector<float>> gradient;
	if (_transferFunctionRadial->getCurrent() == 0) {
		tf = "grayscale";
	}
	else if (_transferFunctionRadial->getCurrent() == 1) {
		tf = "color";
	}
	else if (_transferFunctionRadial->getCurrent() == 2) {
		tf = "RGB-gradient";

		

		osg::Vec4 col_1 = _colSliderList[0]->getColor();
		osg::Vec4 col_2 = _colSliderList[1]->getColor();
		std::vector<float> stdcol1; stdcol1.push_back(col_1.x()); stdcol1.push_back(col_1.y()); stdcol1.push_back(col_1.z());
		std::vector<float> stdcol2; stdcol2.push_back(col_2.x()); stdcol2.push_back(col_2.y()); stdcol2.push_back(col_2.z());

		gradient.push_back(stdcol1);
		gradient.push_back(stdcol2);
	}
	
	out << YAML::Key << "color scheme";
	out << YAML::Value << tf;

	

	if (tf == "RGB-gradient") {
		out << YAML::Key << "gradient";
		out << YAML::Value << YAML::Flow << gradient;
	}
	out << YAML::EndMap;

	out << YAML::Key << "cubemap";
	out << YAML::Value << "studio1";
	out << YAML::Key << "itrs";
	out << YAML::Value << 100;
	
		
	///////////////////Cutting Plane///////////////////////

	
	std::vector<CuttingPlane*> cPs = HelmsleyVolume::instance()->getCuttingPlanes();
	std::vector<float> ppoint;
	std::vector<float> pnorm;
	if (!cPs.empty()) {
		out << YAML::Key << "cutting plane";
		out << YAML::Value << YAML::BeginMap;
		
		osg::FloatArray* osgarraypp = _volume->_PlanePoint->getFloatArray();
		osg::FloatArray* osgarraypn = _volume->_PlaneNormal->getFloatArray();
		std::cout << "cp Z" << osgarraypp->at(2) << std::endl;
		float z = trans.at(2) - (osgarraypp->at(2)+.5);
		z = .1 + (.5 - osgarraypp->at(2));
		z =  .5 + osgarraypp->at(2);
		
		//float z = ((1 - osgarraypp->at(1)) - .5)/2.0 + trans.at(2);
		//ppoint.push_back(0.0); ppoint.push_back(0.0); ppoint.push_back(.6);
		ppoint.push_back(0.0); ppoint.push_back(0.0); ppoint.push_back(z);
		pnorm.push_back(0.0); pnorm.push_back(0.0); pnorm.push_back(1.0);
		
		
		//ppoint.push_back(osgarraypp->at(0)); ppoint.push_back(osgarraypp->at(2)); ppoint.push_back(osgarraypp->at(1));
		//pnorm.push_back(osgarraypn->at(0)); pnorm.push_back(osgarraypn->at(1)); pnorm.push_back(osgarraypn->at(2));
		
		out << YAML::Key << "ppoint";
		out << YAML::Value << YAML::Flow << ppoint;
		out << YAML::Key << "pnorm";
		out << YAML::Value << YAML::Flow << pnorm;
		out << YAML::EndMap;

		std::cout << "ppoint " << "x: " << ppoint.at(0) << "y: " << ppoint.at(1) << "z: " << ppoint.at(2) << std::endl;
		std::cout << "pnorm " << "x: " << pnorm.at(0) << "y: " << pnorm.at(1) << "z: " << pnorm.at(2) << std::endl;
		std::cout << "pos " << "x: " << osgpos.x() << "y: " << osgpos.y() << "z: " << osgpos.z() << std::endl;
 	}
	
	///////////////////Masks////////////////////////
	out << YAML::Key << "mask";
	if (_volume->hasMask()) {
		if(!_organs->isOn() && _colon->isOn())
			out << YAML::Value << "isolate";
		else if (_colon->isOn())
			out << YAML::Value << "body";
		else
			out << YAML::Value << "none";
		//out << YAML::Value << YAML::BeginMap;
		/*out << YAML::Key << "value";
		std::vector<bool> maskValues;
		_organs->isOn() ? maskValues.push_back(true) : maskValues.push_back(false);
		_colon->isOn() ? maskValues.push_back(true) : maskValues.push_back(false);
		_kidney->isOn() ? maskValues.push_back(true) : maskValues.push_back(false);
		_bladder->isOn() ? maskValues.push_back(true) : maskValues.push_back(false);
		_spleen->isOn() ? maskValues.push_back(true) : maskValues.push_back(false);
		_illeum->isOn() ? maskValues.push_back(true) : maskValues.push_back(false);
		_aorta->isOn() ? maskValues.push_back(true) : maskValues.push_back(false);
		_vein->isOn() ? maskValues.push_back(true) : maskValues.push_back(false);
		out << YAML::Value << YAML::Flow << maskValues;
		out << YAML::EndMap;*/
	}
	else {
		out << YAML::Value << "none";
	}
	
	


	///////////////////Dataset//////////////////////
	out << YAML::Key << "series name";
	FileSelector* fs = HelmsleyVolume::instance()->getFileSelector();
	std::string datasetPath = fs->getCurrPath();
	out << YAML::Value << datasetPath;

	out << YAML::EndMap;

	std::string currPath = cvr::ConfigManager::getEntry("Plugin.HelmsleyVolume.PresetsFolder", "C:/", false);
 	std::string presetPath = "C:/Users/g3aguirre/Documents/CAL/ivl-cr/ivl-cr/ivl-cr/configs/" + name + ".yaml";
	std::ofstream fout(presetPath);
	fout << out.c_str();
	std::cout << "wrting to... " << presetPath << std::endl;

	CallbackButton* presetbutton = new CallbackButton();
	presetbutton->setCallback(this);
	UIText* presetText = new UIText(name, 40.0f, osgText::TextBase::CENTER_CENTER);
	presetbutton->addChild(presetText);
	_presetUIList->addChild(presetbutton);
	_presetCallbacks.push_back(presetbutton);

	_futures.push_back(std::async(std::launch::async, runCinematicThread, datasetPath, presetPath));
 }

void NewVolumeMenu::runCinematicThread(std::string datasetPath, std::string configPath) {
 	LPCSTR lp = _T("C:/Users/g3aguirre/Documents/CAL/ivl-cr/ivl-cr/ivl-cr/x64/Release/ivl-cr.exe");
	LPCSTR open = _T("open");
	LPCSTR dir = _T("C:/Users/g3aguirre/Documents/CAL/ivl-cr/ivl-cr/ivl-cr");

	std::replace(datasetPath.begin(), datasetPath.end(), '\\', '/');
	std::cout << datasetPath << std::endl;
	std::cout << configPath << std::endl;

	std::string paramsStr = datasetPath + " " + configPath;
	const char* paramsChar = {paramsStr.c_str()};
	LPCSTR params = _T(paramsChar);


	//HelmsleyVolume::instance()->getScreenshotTool()->takingPhoto(true);
	HANDLE ghMutex = ShellExecute(NULL, open, lp, params, dir, SW_SHOWDEFAULT);
	WaitForSingleObject(
		ghMutex,    // handle to mutex
		10000);  // no time-out interval

	HelmsleyVolume::instance()->getScreenshotTool()->setPhoto(datasetPath);
	//std::remove(configPath.c_str());
	//HelmsleyVolume::instance()->getScreenshotTool()->takingPhoto(false);
}



void NewVolumeMenu::toggleHistogram(bool on) {
	if (on) {
		_histQuad->setMax(_volume->getHistMax());
		_tentWindowOnly->addChild(_histQuad);
	}
	else {
		_tentWindowOnly->getGroup()->removeChild(_histQuad->getGroup());
		_histQuad->_parent = nullptr;
	}
}

void NewVolumeMenu::toggleClaheTools(bool on) {
	if (on) {
		//removeAllToolMenus();
		_menuCount++;
		osg::Vec3 newPos = getCorrectMenuPosition();
		_claheMenu->setPosition(newPos);
		_toolContainer->addChild(_claheMenu->getRoot());
		_claheMenu->setActive(true, true);

	}
	else {
		
		
 		_toolContainer->removeChild(_claheMenu->getRoot());
		_claheMenu->getRootElement()->_parent = nullptr;
		_claheMenu->setActive(false, false);

		if (_menuCount--) {
			resetMenuPos();
		};
	}
}

void NewVolumeMenu::toggleAttnMapsTools(bool on) {
	if (on) {
		//removeAllToolMenus();
		_menuCount++;
		osg::Vec3 newPos = getCorrectMenuPosition();
		_attnMapsMenu->setPosition(newPos);
		_toolContainer->addChild(_attnMapsMenu->getRoot());
		_attnMapsMenu->setActive(true, true);
 	}
	else {		
 		_toolContainer->removeChild(_attnMapsMenu->getRoot());
		_attnMapsMenu->getRootElement()->_parent = nullptr;
		_attnMapsMenu->setActive(false, false);

		if (_menuCount--) {
			resetMenuPos();
		};
	}
}


void NewVolumeMenu::toggleMCRender(bool on) {
	if (on) {
		//removeAllToolMenus();
		_menuCount++;
		_marchingCubesMenu->setPosition(getCorrectMenuPosition());
		_toolContainer->addChild(_marchingCubesMenu->getRoot());
		_marchingCubesMenu->setActive(true, true);
	}
	else {
		 
		_toolContainer->removeChild(_marchingCubesMenu->getRoot());
		_marchingCubesMenu->getRootElement()->_parent = nullptr;
		_marchingCubesMenu->setActive(false, false);

		if (_menuCount--) {
			resetMenuPos();
		};
	}
}

void NewVolumeMenu::toggle3DSelection(bool on) {
	if (on) {
		//removeAllToolMenus();
		_menuCount++;
		_selectionMenu->setPosition(getCorrectMenuPosition());
		_toolContainer->addChild(_selectionMenu->getRoot());
		_selectionMenu->setActive(true, true);
	}
	else {
 		_toolContainer->removeChild(_selectionMenu->getRoot());
		_selectionMenu->getRootElement()->_parent = nullptr;
		_selectionMenu->setActive(false, false);

		if (_menuCount--) {
			resetMenuPos();
		};
	}
}

void NewVolumeMenu::toggleTFUI(bool on) {
	if (on) {
		_tentWindowContainer->addChild(_tentMenu->getRoot());
		_contrastContainer->addChild(_contrastMenu->getRoot());
		_container->addChild(_menu->getRoot());
	}
	else {
		_tentWindowContainer->removeChild(_tentMenu->getRoot());
		_tentMenu->getRootElement()->_parent = nullptr;
		_contrastContainer->removeChild(_contrastMenu->getRoot());
		_contrastMenu->getRootElement()->_parent = nullptr;
		_container->removeChild(_menu->getRoot());
		_menu->getRootElement()->_parent = nullptr;
	}
}

void NewVolumeMenu::removeAllToolMenus() {
	_toolContainer->removeChild(_claheMenu->getRoot());
	_claheMenu->getRootElement()->_parent = nullptr;


	_toolContainer->removeChild(_marchingCubesMenu->getRoot());
	_marchingCubesMenu->getRootElement()->_parent = nullptr;
}

osg::Vec3 NewVolumeMenu::getCorrectMenuPosition(){
	osg::Vec3 newPos = POPUP_POS;
	int leftOrRight = _menuCount % 2 == 0 ? 1 : -1;	//If count is even put it on right side
	int multiplier = (int)(_menuCount / 2) * leftOrRight; 
	newPos.x() += POPUP_WIDTH * multiplier;
	return newPos;
}

void NewVolumeMenu::resetMenuPos() {
	_menuCount = 0;
	for (cvr::UIPopup* currMenu : popupMenus) {
		if (currMenu->isActive()) {
			_menuCount++;
			currMenu->setPosition(getCorrectMenuPosition());
			
		}
	}
}

void NewVolumeMenu::toggleMaskMenu(bool on) {
	if (on) {
		_maskContainer->addChild(_maskMenu->getRoot());
	}
	else {
		_maskContainer->removeChild(_maskMenu->getRoot());
		_maskMenu->getRootElement()->_parent = nullptr;
	}
}

void NewVolumeMenu::togglePopupMenu(bool on, cvr::UIPopup* menu) {
	if (on) {
		osg::Vec3 newPos = getCorrectMenuPosition();
		menu->setPosition(newPos);
		_toolContainer->addChild(menu->getRoot());
		_menuCount++;
	}
	else {
		_toolContainer->removeChild(menu->getRoot());
		menu->getRootElement()->_parent = nullptr;
		_menuCount--;
	}
}

bool NewVolumeMenu::hasCenterLineCoords() {
	return _volume->getColonCoords() == nullptr ? false : true;
}

ToolMenu::ToolMenu(int index, bool movable, cvr::SceneObject* parent)
{
	_movable = movable;
	_index = index;

	_menu = new UIPopup();
	if (parent)
	{
		osg::Vec3 volPos = VOLUME_POS;
		_menu->setPosition(MENU_POS);
 	}
	else
	{
		_menu->setPosition(ConfigManager::getVec3("Plugin.HelmsleyVolume.Orientation.ToolMenu.Position", MENU_POS));
	}
	_menu->getRootElement()->setAbsoluteSize(ConfigManager::getVec3("Plugin.HelmsleyVolume.Orientation.ToolMenu.Scale", osg::Vec3(600, 1, 100)));


	UIList* list = new UIList(UIList::LEFT_TO_RIGHT, UIList::CUT);
	list->setAbsoluteSpacing(5);
	//_menu->addChild(list);	
	std::string dir = ConfigManager::getEntry("Plugin.HelmsleyVolume.ImageDir");
	
	_curvedMenu = new CurvedMenu(this, int(TOOLID::COUNT));
	_curvedMenu->setImage(int(TOOLID::SCREENSHOT), dir + "browser.png");
	_curvedMenu->setImage(int(TOOLID::CUTTINGPLANE), dir + "slice.png");
	//_curvedMenu->setImage(int(TOOLID::HISTOGRAM), dir + "histogram.png");
	_curvedMenu->setImage(int(TOOLID::CLAHE), dir + "clahe.png");
	_curvedMenu->setImage(int(TOOLID::RULER), dir + "ruler.png");
	_curvedMenu->setImage(int(TOOLID::CENTERLINE), dir + "centerline.png");
	
	_curvedMenu->setImage(int(TOOLID::MASKMENU), dir + "maskIcon.png");
	_curvedMenu->setImage(int(TOOLID::TFMENU), dir + "opacityAndGradient.png");
	_curvedMenu->setImage(int(TOOLID::MARCHINGCUBES), dir + "polygon.png");
	_curvedMenu->setImage(int(TOOLID::SELECTION3D), dir + "ruler.png");
	_menu->addChild(_curvedMenu);

	std::cout << "moveable" << _movable << std::endl;

	if (!_movable && !parent)
	{
		_menu->setActive(true, true);
	}
	else {
		_menu->setActive(true, false);
		_container = new SceneObject("VolumeMenu", false, _movable, false, false, false);
		if (parent)
		{
			parent->addChild(_container);
		}
		else
		{
			PluginHelper::registerSceneObject(_container, "VolumeMenu");
			_container->attachToScene();
		}
		_container->addChild(_menu->getRoot());
		_menu->getRootElement()->updateElement(osg::Vec3(0, 0, 0), osg::Vec3(0, 0, 0));
		_container->dirtyBounds();
	}
}

std::vector<CurvedQuad*> ToolMenu::getCurvedMenuItems() {
	std::vector<CurvedQuad*> toReturn = _curvedMenu->getCurvedMenuItems();
	return toReturn;
}

void ToolMenu::disableUnavailableButtons(VolumeGroup* volume) {
	if (volume->getColonCoords()->empty()) {
		_curvedMenu->disableButton(int(TOOLID::CENTERLINE));
	}
}


ToolMenu::~ToolMenu()
{
	_menu->setActive(false, false);
	MenuManager::instance()->removeMenuSystem(_menu);
	delete _menu;

	if (_container)
	{
		_container->detachFromScene();
		delete _container;
	}
}

void ToolMenu::uiCallback(UICallbackCaller* item)
{
	std::pair<int, CurvedQuad*> index = _curvedMenu->getCallbackIndex(item);
	
 	if (index.first == int(TOOLID::SCREENSHOT))
	{
		osg::Matrix mat = PluginHelper::getHandMat(index.second->getLastHand());
		osg::Vec4d position = osg::Vec4(0, 300, 0, 1) * mat;
		osg::Vec3f pos = osg::Vec3(position.x(), position.y(), position.z());

		osg::Quat q = osg::Quat();
		osg::Quat q2 = osg::Quat();
		osg::Vec3 v = osg::Vec3();
		osg::Vec3 v2 = osg::Vec3();
		mat.decompose(v, q, v2, q2);

		HelmsleyVolume::instance()->toggleScreenshotTool(index.second->isOn());
		HelmsleyVolume::instance()->getScreenshotTool()->setRotation(q);
		HelmsleyVolume::instance()->getScreenshotTool()->setPosition(pos);


		if (index.second->isOn())
		{
			index.second->setColor(UI_RED_ACTIVE_COLOR);
 		}
		else 
		{
			index.second->setColor(UI_BACKGROUND_COLOR);
		}
	}
	
	else if (index.first == int(TOOLID::CUTTINGPLANE))
	{
		if (index.second->isOn())
		{
			index.second->setColor(UI_RED_ACTIVE_COLOR);
			HelmsleyVolume::instance()->createCuttingPlane();
		}
		else
		{
			index.second->setColor(UI_BACKGROUND_COLOR);
			HelmsleyVolume::instance()->removeCuttingPlane();
		}
	}

	/*else if (index.first == int(TOOLID::HISTOGRAM))
	{
		if (index.second->isOn())
		{
			index.second->setColor(UI_RED_ACTIVE_COLOR);
			HelmsleyVolume::instance()->toggleHistogram(true);
		}
		else
		{
			index.second->setColor(UI_BACKGROUND_COLOR);
			HelmsleyVolume::instance()->toggleHistogram(false);
		}
	}*/

	else if (index.first == int(TOOLID::CLAHE))
	{
		if (index.second->isOn())
		{
			index.second->setColor(UI_RED_ACTIVE_COLOR);
			HelmsleyVolume::instance()->toggleClaheTools(true);
			//toggleOtherMenus((int)TOOLID::CLAHE);
		}
		else
		{
			index.second->setColor(UI_BACKGROUND_COLOR);
			HelmsleyVolume::instance()->toggleClaheTools(false);
		}
	}
	else if (index.first == int(TOOLID::RULER))
	{
		if (index.second->isOn())
		{
			index.second->setColor(UI_RED_ACTIVE_COLOR);
			HelmsleyVolume::instance()->setTool(HelmsleyVolume::MEASUREMENT_TOOL);
			HelmsleyVolume::instance()->activateMeasurementTool(_index);
		}
		else
		{
			index.second->setColor(UI_BACKGROUND_COLOR);
			HelmsleyVolume::instance()->setTool(HelmsleyVolume::NONE);
			HelmsleyVolume::instance()->deactivateMeasurementTool(_index);
		}
	}
	else if (index.first == int(TOOLID::SELECTION3D))
	{
		if (index.second->isOn())
		{
			index.second->setColor(UI_RED_ACTIVE_COLOR);
			HelmsleyVolume::instance()->setTool(HelmsleyVolume::SELECTION3D);
			HelmsleyVolume::instance()->activateSelectionTool(_index);
			HelmsleyVolume::instance()->toggle3DSelection(true);
		}
		else
		{
			index.second->setColor(UI_BACKGROUND_COLOR);
			HelmsleyVolume::instance()->setTool(HelmsleyVolume::NONE);
			HelmsleyVolume::instance()->deactivateSelectionTool(_index);
			//HelmsleyVolume::instance()->toggle3DSelection(false);
			HelmsleyVolume::instance()->toggle3DSelection(false);

		}
	}
	else if (index.first == int(TOOLID::CENTERLINE))
	{
		if (HelmsleyVolume::instance()->hasCenterLineCoords()) {
			std::cout << "true" << std::endl;
			if (!HelmsleyVolume::instance()->getVolumes()[0]->getColonCoords()->empty()) {
				osg::Matrix mat = PluginHelper::getHandMat(index.second->getLastHand());
				osg::Vec4d position = osg::Vec4(0, 300, 0, 1) * mat;
				osg::Vec3f pos = osg::Vec3(position.x(), position.y(), position.z());



				osg::Quat q = osg::Quat();
				osg::Quat q2 = osg::Quat();
				osg::Vec3 v = osg::Vec3();
				osg::Vec3 v2 = osg::Vec3();
				mat.decompose(v, q, v2, q2);

				HelmsleyVolume::instance()->toggleCenterlineTool(index.second->isOn());
				HelmsleyVolume::instance()->getCenterlineTool()->setRotation(q);
				HelmsleyVolume::instance()->getCenterlineTool()->setPosition(pos);
				if (index.second->isOn())
				{
					HelmsleyVolume::instance()->activateClippingPath();
					index.second->setColor(UI_RED_ACTIVE_COLOR);
				}
				else
				{
					index.second->setColor(UI_BACKGROUND_COLOR);
					HelmsleyVolume::instance()->removeCuttingPlane();

				}
			}
		}
		else {
			std::cout << "false" << std::endl;
		}
	}

	else if (index.first == int(TOOLID::MASKMENU)) {
		if(index.second->isOn())
		{
			index.second->setColor(UI_RED_ACTIVE_COLOR);
			HelmsleyVolume::instance()->toggleMaskAndPresets(true);
 		}
	else
		{
			index.second->setColor(UI_BACKGROUND_COLOR);
			HelmsleyVolume::instance()->toggleMaskAndPresets(false);
 		}
	}

	else if (index.first == int(TOOLID::TFMENU)) {
		if(index.second->isOn())
		{
			index.second->setColor(UI_RED_ACTIVE_COLOR);
			HelmsleyVolume::instance()->toggleTFUI(true);
 		}
	else
		{
			index.second->setColor(UI_BACKGROUND_COLOR);
			HelmsleyVolume::instance()->toggleTFUI(false);
 		}
	}

	else if (index.first == int(TOOLID::MARCHINGCUBES)) {

		if(index.second->isOn())
		{
			index.second->setColor(UI_RED_ACTIVE_COLOR);
			HelmsleyVolume::instance()->toggleMCRender(true);
			//toggleOtherMenus((int)TOOLID::MARCHINGCUBES);
 		}
	else
		{
			index.second->setColor(UI_BACKGROUND_COLOR);
			HelmsleyVolume::instance()->toggleMCRender(false);
 		}
	}

	
}

void ToolMenu::toggleOtherMenus(int currentActiveTool) {
	auto menu = _curvedMenu->getCurvedMenuItems();
	menu[int(TOOLID::CLAHE)]->setColor(UI_BACKGROUND_COLOR);
	menu[int(TOOLID::CLAHE)]->turnOff();
	menu[int(TOOLID::MARCHINGCUBES)]->setColor(UI_BACKGROUND_COLOR);
	menu[int(TOOLID::MARCHINGCUBES)]->turnOff();
	
	menu[int(currentActiveTool)]->setColor(UI_RED_ACTIVE_COLOR);
	menu[int(currentActiveTool)]->turnOn();

}