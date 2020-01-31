#include "VolumeMenu.h"
#include "HelmsleyVolume.h"
#include "Utils.h"
#include "cvrMenu/MenuManager.h"
#include "cvrConfig/ConfigManager.h"

using namespace cvr;

void VolumeMenu::init()
{
	scale = new MenuRangeValueCompact("Scale", 0.1, 100.0, 1.0, true);
	scale->setCallback(this);
	_scene->addMenuItem(scale);

	sampleDistance = new MenuRangeValueCompact("SampleDistance", .0001, 0.01, .00150f, true);
	sampleDistance->setCallback(this);
	_scene->addMenuItem(sampleDistance);
	
	adaptiveQuality = new MenuCheckbox("Adaptive Quality", false);
	adaptiveQuality->setCallback(this);
	_scene->addMenuItem(adaptiveQuality);
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
	else if (item == adaptiveQuality)
	{
		_volume->getDrawable()->getOrCreateStateSet()->setDefine("VR_ADAPTIVE_QUALITY", adaptiveQuality->getValue());
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

	if (_container)
	{
		_container->detachFromScene();
		delete _container;
	}
	if (_maskContainer)
	{
		_maskContainer->detachFromScene();
		delete _maskContainer;
	}

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

	osg::Vec3 volumePos = ConfigManager::getVec3("Plugin.HelmsleyVolume.Orientation.Volume.Position", osg::Vec3(0, 750, 500));
	_so->setPosition(volumePos);

	_menu = new UIPopup();
	UIQuadElement* bknd = new UIQuadElement(UI_BACKGROUND_COLOR);
	//_menu->addChild(bknd);
	_menu->setPosition(ConfigManager::getVec3("Plugin.HelmsleyVolume.Orientation.OptionsMenu.Position", osg::Vec3(500, 500, 1450)) - volumePos);
	_menu->getRootElement()->setAbsoluteSize(ConfigManager::getVec3("Plugin.HelmsleyVolume.Orientation.OptionsMenu.Scale", osg::Vec3(600, 1, 600)));

	UIList* list = new UIList(UIList::TOP_TO_BOTTOM, UIList::CONTINUE);
	list->setPercentPos(osg::Vec3(0, 0, -0.2));
	list->setPercentSize(osg::Vec3(1, 1, 0.8));
	list->setAbsoluteSpacing(10);
	bknd->addChild(list);

	UIText* label = new UIText("Volume Options", 50.0f, osgText::TextBase::CENTER_CENTER);
	label->setPercentSize(osg::Vec3(1, 1, 0.2));
	bknd->addChild(label);


	UIList* fliplist = new UIList(UIList::LEFT_TO_RIGHT, UIList::CUT);

	_horizontalflip = new CallbackButton();
	_horizontalflip->setCallback(this);
	label = new UIText("Flip Sagittal", 30.0f, osgText::TextBase::CENTER_CENTER);
	_horizontalflip->addChild(label);

	_verticalflip = new CallbackButton();
	_verticalflip->setCallback(this);
	label = new UIText("Flip Axial", 30.0f, osgText::TextBase::CENTER_CENTER);
	_verticalflip->addChild(label);

	_depthflip = new CallbackButton();
	_depthflip->setCallback(this);
	label = new UIText("Flip Coronal", 30.0f, osgText::TextBase::CENTER_CENTER);
	_depthflip->addChild(label);

	fliplist->addChild(_horizontalflip);
	fliplist->addChild(_verticalflip);
	fliplist->addChild(_depthflip);
	list->addChild(fliplist);

	label = new UIText("Density", 30.0f, osgText::TextBase::LEFT_CENTER);
	label->setPercentPos(osg::Vec3(0.1, 0, 0));
	list->addChild(label);

	_density = new CallbackSlider();
	_density->setPercentPos(osg::Vec3(0.025, 0, 0.05));
	_density->setPercentSize(osg::Vec3(0.95, 1, 0.9));
	_density->handle->setAbsoluteSize(osg::Vec3(20, 0, 0));
	_density->handle->setAbsolutePos(osg::Vec3(-10, -0.2f, 0));
	_density->handle->setPercentSize(osg::Vec3(0, 1, 1));
	_density->setMax(0.5f);
	_density->setMin(0.0001f);
	_density->setCallback(this);
	_density->setPercent(1);

	list->addChild(_density);


	label = new UIText("Threshold", 30.0f, osgText::TextBase::LEFT_CENTER);
	label->setPercentPos(osg::Vec3(0.1, 0, 0));
	list->addChild(label);

	_contrastBottom = new CallbackSlider();
	_contrastBottom->setPercentPos(osg::Vec3(0.025, 0, 0.05));
	_contrastBottom->setPercentSize(osg::Vec3(0.95, 1, 0.9));
	_contrastBottom->handle->setAbsoluteSize(osg::Vec3(20, 0, 0));
	_contrastBottom->handle->setAbsolutePos(osg::Vec3(-10, -0.2f, 0));
	_contrastBottom->handle->setPercentSize(osg::Vec3(0, 1, 1));
	_contrastBottom->setCallback(this);
	_contrastBottom->setPercent(0);

	_contrastTop = new CallbackSlider();
	_contrastTop->setPercentPos(osg::Vec3(0.025, 0, 0.05));
	_contrastTop->setPercentSize(osg::Vec3(0.95, 1, 0.9));
	_contrastTop->handle->setAbsoluteSize(osg::Vec3(20, 0, 0));
	_contrastTop->handle->setAbsolutePos(osg::Vec3(-10, -0.2f, 0));
	_contrastTop->handle->setPercentSize(osg::Vec3(0, 1, 1));
	_contrastTop->setCallback(this);
	_contrastTop->setPercent(1);

	list->addChild(_contrastBottom);
	list->addChild(_contrastTop);


	label = new UIText("Color", 30.0f, osgText::TextBase::LEFT_CENTER);
	label->setPercentPos(osg::Vec3(0.1, 0, 0));
	list->addChild(label);

	UIList* list2 = new UIList(UIList::LEFT_TO_RIGHT, UIList::CUT);
	
	_transferFunction = new CallbackRadial();
	_transferFunction->allowNoneSelected(false);
	_blacknwhite = new UIRadialButton(_transferFunction);
	_rainbow = new UIRadialButton(_transferFunction);
	_transferFunction->setCurrent(0);
	_transferFunction->setCallback(this);

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

	list->addChild(_colorDisplay);

	UIText* bnw = new UIText("Black and White", 30.0f, osgText::TextBase::CENTER_CENTER);
	bnw->setColor(osg::Vec4(0.8, 1, 0.8, 1));
	UIText* rnbw = new UIText("Rainbow", 40.0f, osgText::TextBase::CENTER_CENTER);
	rnbw->setColor(osg::Vec4(1, 0.8, 0.8, 1));

	_blacknwhite->addChild(bnw);
	_rainbow->addChild(rnbw);
	list2->addChild(_blacknwhite);
	list2->addChild(_rainbow);
	list->addChild(list2);

	if (!_movable)
	{
		_menu->setActive(true, true);
	}
	else {
		_menu->setActive(true, false);
		_container = new SceneObject("VolumeMenu", false, true, false, false, true);
		//PluginHelper::registerSceneObject(_container, "VolumeMenu");
		//_container->attachToScene();
		_so->addChild(_container);
		//_container->getRoot()->addUpdateCallback(new PointAtHeadLerp(0.1f));
		_container->setShowBounds(true);
		_container->addChild(_menu->getRoot());
		_menu->getRootElement()->updateElement(osg::Vec3(0, 0, 0), osg::Vec3(0, 0, 0));
		_container->dirtyBounds();
	}

#pragma endregion

	_toolMenu = new ToolMenu(0, true, _so);

	//_menu->getRootElement()->updateElement(osg::Vec3(0, 0, 0), osg::Vec3(0, 0, 0));
	//UIElement* e = list2->getChild(0)->getChild(1);

	if (_volume->hasMask())
	{
		_maskMenu = new UIPopup();
		bknd = new UIQuadElement(UI_BACKGROUND_COLOR);
		_maskMenu->addChild(bknd);


		_maskMenu->setPosition(ConfigManager::getVec3("Plugin.HelmsleyVolume.Orientation.MaskMenu.Position", osg::Vec3(600, 500, 800)) - volumePos);
		_maskMenu->getRootElement()->setAbsoluteSize(ConfigManager::getVec3("Plugin.HelmsleyVolume.Orientation.MaskMenu.Scale", osg::Vec3(400, 1, 800)));

		list = new UIList(UIList::TOP_TO_BOTTOM, UIList::CONTINUE);
		list->setPercentPos(osg::Vec3(0, 0, -0.2));
		list->setPercentSize(osg::Vec3(1, 1, 0.8));
		bknd->addChild(list);

		UIText* label = new UIText("Organs", 50.0f, osgText::TextBase::CENTER_CENTER);
		label->setPercentSize(osg::Vec3(1, 1, 0.2));
		bknd->addChild(label);

		_organs = new VisibilityToggle("Body");
		_organs->toggle();
		_organs->setCallback(this);
		_colon = new VisibilityToggle("Colon");
		_colon->setCallback(this);
		_kidney = new VisibilityToggle("Kidney");
		_kidney->setCallback(this);
		_bladder = new VisibilityToggle("Bladder");
		_bladder->setCallback(this);
		_spleen = new VisibilityToggle("Spleen");
		_spleen->setCallback(this);

		list->addChild(_organs);
		list->addChild(_colon);
		list->addChild(_kidney);
		list->addChild(_bladder);
		list->addChild(_spleen);

		if (!_movable)
		{
			_maskMenu->setActive(true, true);
		}
		else {
			_maskMenu->setActive(true, false);
			_maskContainer = new SceneObject("MaskMenu", false, true, false, false, true);
			//_maskContainer->getRoot()->addUpdateCallback(new PointAtHeadLerp(0.1f));
			//PluginHelper::registerSceneObject(_maskContainer, "MaskMenu");
			//_maskContainer->attachToScene();
			_so->addChild(_maskContainer);
			_maskContainer->setShowBounds(true);
			_maskContainer->addChild(_maskMenu->getRoot());
			_maskMenu->getRootElement()->updateElement(osg::Vec3(0, 0, 0), osg::Vec3(0, 0, 0));
			_maskContainer->dirtyBounds();
		}
	}
}

void NewVolumeMenu::uiCallback(UICallbackCaller * item)
{
	//if (_container)
	//{
	//	_container->dirtyBounds();
	//}
	if (item == _horizontalflip)
	{
		osg::Matrix m;
		m.makeScale(osg::Vec3(-1, 1, 1));
		_volume->_transform->postMult(m);
		_volume->flipCull();

	}
	else if (item == _verticalflip)
	{
		osg::Matrix m;
		m.makeScale(osg::Vec3(1, 1, -1));
		_volume->_transform->postMult(m);
		_volume->flipCull();
	}
	else if (item == _depthflip)
	{
		osg::Matrix m;
		m.makeScale(osg::Vec3(1, -1, 1));
		_volume->_transform->postMult(m);
		_volume->flipCull();
	}
	else if (item == _organs)
	{
		_volume->getCompute()->getOrCreateStateSet()->setDefine("ORGANS_ONLY", !_organs->isOn());
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
	else if (item == _density)
	{
		_volume->_computeUniforms["OpacityMult"]->set(_density->getAdjustedValue());
		_volume->setDirtyAll();
	}
	else if (item == _contrastBottom)
	{
		if (_contrastBottom->getPercent() >= _contrastTop->getPercent())
		{
			_contrastBottom->setPercent(_contrastTop->getPercent() - 0.001f);
		}
		_volume->_computeUniforms["ContrastBottom"]->set(_contrastBottom->getAdjustedValue());
		_volume->setDirtyAll();
	}
	else if (item == _contrastTop)
	{
		if (_contrastBottom->getPercent() >= _contrastTop->getPercent())
		{
			_contrastTop->setPercent(_contrastBottom->getPercent() + 0.001f);
		}
		_volume->_computeUniforms["ContrastTop"]->set(_contrastTop->getAdjustedValue());
		_volume->setDirtyAll();
	}
	else if (item == _transferFunction)
	{
		if (_transferFunction->getCurrent() == 0)
		{
			transferFunction = "vec3(ra.r);";
			_volume->getCompute()->getOrCreateStateSet()->setDefine("COLOR_FUNCTION", transferFunction, osg::StateAttribute::ON);
			_colorDisplay->setShaderDefine("COLOR_FUNCTION", transferFunction, osg::StateAttribute::ON);

			((UIText*)_blacknwhite->getChild(0))->setColor(UI_ACTIVE_COLOR);
			((UIText*)_rainbow->getChild(0))->setColor(UI_INACTIVE_COLOR);
		}
		else if (_transferFunction->getCurrent() == 1)
		{
			transferFunction = "hsv2rgb(vec3(ra.r * 0.8, 1, 1));";
			_volume->getCompute()->getOrCreateStateSet()->setDefine("COLOR_FUNCTION", transferFunction, osg::StateAttribute::ON);
			_colorDisplay->setShaderDefine("COLOR_FUNCTION", transferFunction, osg::StateAttribute::ON);

			((UIText*)_blacknwhite->getChild(0))->setColor(UI_INACTIVE_COLOR);
			((UIText*)_rainbow->getChild(0))->setColor(UI_ACTIVE_COLOR);
		}
		_volume->setDirtyAll();
	}
}

ToolMenu::ToolMenu(int index, bool movable, cvr::SceneObject* parent)
{
	_movable = movable;
	_index = index;

	_menu = new UIPopup();
	//UIQuadElement* bknd = new UIQuadElement(osg::Vec4(0.3, 0.3, 0.3, 1));
	//_menu->addChild(bknd);
	if (parent)
	{
		osg::Vec3 volumePos = ConfigManager::getVec3("Plugin.HelmsleyVolume.Orientation.Volume.Position", osg::Vec3(0, 750, 500));
		_menu->setPosition(ConfigManager::getVec3("Plugin.HelmsleyVolume.Orientation.ToolMenu.Position", osg::Vec3(-150, 500, 600)) - volumePos);
	}
	else
	{
		_menu->setPosition(ConfigManager::getVec3("Plugin.HelmsleyVolume.Orientation.ToolMenu.Position", osg::Vec3(-150, 500, 600)));
	}
	_menu->getRootElement()->setAbsoluteSize(ConfigManager::getVec3("Plugin.HelmsleyVolume.Orientation.ToolMenu.Scale", osg::Vec3(300, 1, 100)));

	UIList* list = new UIList(UIList::LEFT_TO_RIGHT, UIList::CUT);
	list->setAbsoluteSpacing(5);
	//bknd->addChild(list);
	_menu->addChild(list);

	//_tool = new CallbackRadial();
	//_tool->setCallback(this);
	//_tool->allowNoneSelected(true);

	std::string dir = ConfigManager::getEntry("Plugin.HelmsleyVolume.ImageDir");

	_cuttingPlane = new ToolToggle(dir + "slice.png");
	_cuttingPlane->setCallback(this);
	list->addChild(_cuttingPlane);

	_measuringTool = new ToolToggle(dir + "ruler.png");
	_measuringTool->setCallback(this);
	list->addChild(_measuringTool);

	_screenshotTool = new ToolToggle(dir + "browser.png");
	_screenshotTool->setCallback(this);
	list->addChild(_screenshotTool);

	if (!_movable && !parent)
	{
		_menu->setActive(true, true);
	}
	else {
		_menu->setActive(true, false);
		_container = new SceneObject("VolumeMenu", false, _movable, false, false, true);
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
	if (item == _screenshotTool)
	{
		osg::Matrix mat = PluginHelper::getHandMat(_screenshotTool->getLastHand());
		osg::Vec4d position = osg::Vec4(0, 300, 0, 1) * mat;
		osg::Vec3f pos = osg::Vec3(position.x(), position.y(), position.z());

		osg::Quat q = osg::Quat();
		osg::Quat q2 = osg::Quat();
		osg::Vec3 v = osg::Vec3();
		osg::Vec3 v2 = osg::Vec3();
		mat.decompose(v, q, v2, q2);

		HelmsleyVolume::instance()->toggleScreenshotTool(_screenshotTool->isOn());
		HelmsleyVolume::instance()->getScreenshotTool()->setRotation(q);
		HelmsleyVolume::instance()->getScreenshotTool()->setPosition(pos);


		if (_screenshotTool->isOn())
		{
			_screenshotTool->getIcon()->setColor(osg::Vec4(0.1, 0.4, 0.1, 1));
		}
		else 
		{
			_screenshotTool->getIcon()->setColor(osg::Vec4(0, 0, 0, 1));
		}
	}
	else if (item == _cuttingPlane)
	{
		if (_cuttingPlane->isOn())
		{
			HelmsleyVolume::instance()->createCuttingPlane(0);
			_cuttingPlane->getIcon()->setColor(osg::Vec4(0.1, 0.4, 0.1, 1));
		}
		else
		{
			HelmsleyVolume::instance()->removeCuttingPlane(0);
			_cuttingPlane->getIcon()->setColor(osg::Vec4(0, 0, 0, 1));
		}
	}
	else if (item == _measuringTool)
	{
		if (_measuringTool->isOn())
		{
			HelmsleyVolume::instance()->setTool(HelmsleyVolume::MEASUREMENT_TOOL);
			HelmsleyVolume::instance()->activateMeasurementTool(_index);
			_measuringTool->getIcon()->setColor(osg::Vec4(0.1, 0.4, 0.1, 1));
		}
		else
		{
			HelmsleyVolume::instance()->setTool(HelmsleyVolume::NONE);
			HelmsleyVolume::instance()->deactivateMeasurementTool(_index);
			_measuringTool->getIcon()->setColor(osg::Vec4(0, 0, 0, 1));
		}
	}
}