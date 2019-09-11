#include "HelmsleyVolume.h"

#include <ctime>
#include <iostream>
#include <cvrKernel/NodeMask.h>


using namespace cvr;

CVRPLUGIN(HelmsleyVolume)


HelmsleyVolume::HelmsleyVolume()
{
	_buttonMap = std::map<cvr::MenuItem*, std::string>();
	_stepSizeMap = std::map<cvr::MenuItem*, VolumeGroup*>();
	_scaleMap = std::map<cvr::MenuItem*, SceneObject*>();
	_computeShaderMap = std::map<cvr::MenuItem*, std::pair<std::string, VolumeGroup*> >();
	_volumeDefineMap = std::map<cvr::MenuItem*, std::pair<std::string, VolumeGroup*> >();
	_volumes = std::vector<VolumeGroup*>();
	_sceneObjects = std::vector<SceneObject*>();
}

HelmsleyVolume::~HelmsleyVolume()
{
}

bool HelmsleyVolume::init()
{

	std::vector<osg::Camera*> cameras = std::vector<osg::Camera*>();
	cvr::CVRViewer::instance()->getCameras(cameras);
	for (int i = 0; i < cameras.size(); ++i)
	{
		cameras[i]->getGraphicsContext()->getState()->setUseModelViewAndProjectionUniforms(true);

	}


	_interactButton = cvr::ConfigManager::getInt("Plugin.HelmsleyVolume.InteractButton", 0);
	_cuttingPlaneDistance = cvr::ConfigManager::getFloat("Plugin.HelmsleyVolume.CuttingPlaneDistance", 200.0f);
	float size = cvr::ConfigManager::getFloat("Plugin.HelmsleyVolume.CuttingPlaneSize", 500.0f);

	//Cutting plane setup
	osg::Drawable* cpd1 = new osg::ShapeDrawable(new osg::Box(osg::Vec3(size * 0.495, 0, 0), size * 0.01, size * 0.001, size));
	osg::Drawable* cpd2 = new osg::ShapeDrawable(new osg::Box(osg::Vec3(size * -0.495, 0, 0), size * 0.01, size * 0.001, size));
	osg::Drawable* cpd3 = new osg::ShapeDrawable(new osg::Box(osg::Vec3(0, 0, size * 0.495), size, size * 0.001, size * 0.01));
	osg::Drawable* cpd4 = new osg::ShapeDrawable(new osg::Box(osg::Vec3(0, 0, size * -0.495), size, size * 0.001, size * 0.01));

	osg::Geode* cuttingPlaneGeode = new osg::Geode();
	cuttingPlaneGeode->addDrawable(cpd1);
	cuttingPlaneGeode->addDrawable(cpd2);
	cuttingPlaneGeode->addDrawable(cpd3);
	cuttingPlaneGeode->addDrawable(cpd4);
	cuttingPlaneGeode->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
	cuttingPlane = new osg::MatrixTransform();
	cuttingPlane->addChild(cuttingPlaneGeode);

	SceneObject * cpso = new SceneObject("Cutting Plane Indicator", false, false, false, false, false);
	cpso->addChild(cuttingPlane);
	PluginHelper::registerSceneObject(cpso, "HelmsleyVolume");
	cpso->attachToScene();

	//Measurement tool setup
	measurementTool = new MeasurementTool();
	measurementTool->setNodeMask(0);
	SceneObject * mtso = new SceneObject("Measurement Tool", false, false, false, false, false);
	mtso->addChild(measurementTool);
	PluginHelper::registerSceneObject(mtso, "HelmsleyVolume");
	mtso->attachToScene();


	_selectionMatrix = osg::Matrix();
	_selectionMatrix.makeTranslate(osg::Vec3(-300, 500, 300));




	osg::setNotifyLevel(osg::NOTICE);
	std::cerr << "HelmsleyVolume init" << std::endl;

	_vMenu = new SubMenu("HelmsleyVolume", "HelmsleyVolume");
	_vMenu->setCallback(this);

	SubMenu* fileMenu = new SubMenu("Files", "Files");
	fileMenu->setCallback(this);
	_vMenu->addItem(fileMenu);


	std::string modelDir = cvr::ConfigManager::getEntry("Plugin.HelmsleyVolume.ModelDir");
	std::cout << modelDir << std::endl;

	_selectionMenu = new PopupMenu("Interaction options", "", false);
	_selectionMenu->setVisible(false);
	
	_radial = new MenuRadial();
	std::vector<std::string> labels = std::vector<std::string>();
	std::vector<bool> symbols = std::vector<bool>();
	labels.push_back(modelDir + "scissors_ucsd.obj");
	labels.push_back("measure");
	labels.push_back(modelDir + "pen_ucsd.obj");
	labels.push_back(modelDir + "eraser_ucsd.obj");

	symbols.push_back(true);
	symbols.push_back(false);
	symbols.push_back(true);
	symbols.push_back(true);
	_radial->setLabels(labels, symbols);
	_selectionMenu->addMenuItem(_radial);
	
	//_vMenu->addItem(_radial);

	createList(fileMenu, "Plugin.HelmsleyVolume.Files");

	MenuSystem::instance()->addMenuItem(_vMenu);

    return true;
}

void HelmsleyVolume::createList(SubMenu* menu, std::string configbase)
{
	std::vector<std::string> list;
	ConfigManager::getChildren(configbase, list);

	for (int i = 0; i < list.size(); ++i)
	{
		bool found = false;
		std::string path = ConfigManager::getEntry(configbase + "." + list[i], "", &found);
		if (found)
		{
			MenuButton * button = new MenuButton(list[i]);
			button->setCallback(this);
			menu->addItem(button);
			_buttonMap[button] = path;
		}
		else
		{
			SubMenu* nextMenu = new SubMenu(list[i], list[i]);
			nextMenu->setCallback(this);
			menu->addItem(nextMenu);
			createList(nextMenu, configbase + "." + list[i]);
		}
	}
}

void HelmsleyVolume::preFrame()
{
}

void HelmsleyVolume::postFrame()
{

}

bool HelmsleyVolume::processEvent(InteractionEvent * e)
{
	if (e->getInteraction() == BUTTON_DOWN || e->getInteraction() == BUTTON_DRAG)
	{
		if (e->asTrackedButtonEvent() && e->asTrackedButtonEvent()->getButton() == _interactButton)
		{
			if (_radial->getValue() == 0)
			{
				//Cutting plane
				osg::Matrix mat = PluginHelper::getHandMat(e->asHandEvent()->getHand());
				

				for (int i = 0; i < _volumes.size(); ++i)
				{
					osg::Matrix objhand = mat * _sceneObjects[i]->getWorldToObjectMatrix() * _volumes[i]->getWorldToObjectMatrix();

					osg::Matrix w2o = _volumes[i]->getWorldToObjectMatrix();
					osg::Matrix w2o2 = _sceneObjects[i]->getWorldToObjectMatrix();

					osg::Quat q = osg::Quat();
					osg::Quat q2 = osg::Quat();
					osg::Vec3 v = osg::Vec3();
					osg::Vec3 v2 = osg::Vec3();

					mat.decompose(v, q, v2, q2);
					osg::Matrix m = osg::Matrix();
					m.makeRotate(q);
					_sceneObjects[i]->getWorldToObjectMatrix().decompose(v, q, v2, q2);
					m.postMultRotate(q);
					_volumes[i]->getWorldToObjectMatrix().decompose(v, q, v2, q2);
					m.postMultRotate(q);
					m.postMultScale(osg::Vec3(1.0 / v2.x(), 1.0 / v2.y(), 1.0/v2.z()));

					osg::Vec4d normal = osg::Vec4(0, 1, 0, 0) * m;
					osg::Vec3 norm = osg::Vec3(normal.x(), normal.y(), normal.z());

					osg::Vec4f position = osg::Vec4(0, _cuttingPlaneDistance, 0, 1) * objhand;
					osg::Vec3f pos = osg::Vec3(position.x(), position.y(), position.z());



					_volumes[i]->_PlanePoint->set(pos);
					_volumes[i]->_PlaneNormal->set(norm);

				}

				osg::Vec4d position = osg::Vec4(0, _cuttingPlaneDistance, 0, 1) * mat;
				osg::Vec3f pos = osg::Vec3(position.x(), position.y(), position.z());

				osg::Quat q = osg::Quat();
				osg::Quat q2 = osg::Quat();
				osg::Vec3 v = osg::Vec3();
				osg::Vec3 v2 = osg::Vec3();
				mat.decompose(v, q, v2, q2);

				osg::Matrix m = osg::Matrix();
				m.makeRotate(q);
				m.postMultTranslate(pos);
				cuttingPlane->setMatrix(m);
				cuttingPlane->setNodeMask(0xffffffff);
				return true;
			}
			else if (_radial->getValue() == 1)
			{
				//Measurement tool
				osg::Matrix mat = PluginHelper::getHandMat(e->asHandEvent()->getHand());
				osg::Vec4d position = osg::Vec4(0, 0, 0, 1) * mat;
				osg::Vec3f pos = osg::Vec3(position.x(), position.y(), position.z());

				if (e->getInteraction() == BUTTON_DOWN)
				{
					measurementTool->setStart(pos);
				}
				else
				{
					measurementTool->setEnd(pos);
					measurementTool->setNodeMask(0xffffffff);
				}
				return true;
			}
		}
	}
	else if (e->getInteraction() == BUTTON_UP)
	{
		if (e->asTrackedButtonEvent() && e->asTrackedButtonEvent()->getButton() == _interactButton)
		{
			if (_radial->getValue() == 0)
			{
				//Cutting plane
				cuttingPlane->setNodeMask(0);
				return true;
			}
			else if (_radial->getValue() == 1)
			{
				//Measurement tool
				if (measurementTool->getLength() < 5.0)
				{
					measurementTool->setNodeMask(0);
				}
			}
		}

	}
	
	else if (e->asValuatorEvent() && e->asValuatorEvent()->getValuator() == _radialXVal)
	{
		_radialX = e->asValuatorEvent()->getValue();
	}
	else if (e->asValuatorEvent() && e->asValuatorEvent()->getValuator() == _radialYVal)
	{
		_radialY = e->asValuatorEvent()->getValue();
	}
	if (abs(_radialX) > 0.01 ||abs(_radialY) > 0.01)
	{
		if (!_radialShown)
		{
			_selectionMenu->setVisible(true);
			_radialShown = true;
		}
		std::cout << "x: " << _radialX << ", y: " << _radialY << std::endl;
		if (e->asHandEvent())
		{
			_selectionMenu->setTransform(_selectionMatrix * PluginHelper::getHandMat(e->asHandEvent()->getHand()));
		}
	}
	else
	{
		_radialShown = false;
		_selectionMenu->setVisible(false);
	}
    
	return false;
}

void HelmsleyVolume::menuCallback(MenuItem* menuItem)
{

	if (_buttonMap.find(menuItem) != _buttonMap.end())
	{
		SceneObject * so;
		so = new SceneObject("volume", false, true, true, true, false);
		VolumeGroup * g = new VolumeGroup();
		g->loadVolume(_buttonMap.at(menuItem));

		//std::vector<osg::Camera*> cameras = std::vector<osg::Camera*>();
		//cvr::CVRViewer::instance()->getCameras(cameras);
		//osg::Texture* t = (cameras[0]->getBufferAttachmentMap())[osg::Camera::DEPTH_BUFFER]._texture;

		//osg::Drawable* cube = g->getDrawable();
		//g->getDrawable()->getOrCreateStateSet()->setTextureAttributeAndModes(1, t, osg::StateAttribute::ON);


		_volumes.push_back(g);
		_sceneObjects.push_back(so);
		so->addChild(g);

		PluginHelper::registerSceneObject(so, "HelmsleyVolume");
		so->attachToScene();
		so->setNavigationOn(false);
		so->addMoveMenuItem();
		so->addNavigationMenuItem();
		//so->addScaleMenuItem("Size", 0.1f, 10.0f, 1.0f);
		so->setShowBounds(true);


		MenuRangeValueCompact* scale = new MenuRangeValueCompact("Scale", 0.1, 100.0, 1.0, true);
		scale->setCallback(this);
		so->addMenuItem(scale);
		_scaleMap[scale] = so;


		//Set up uniforms for shaders
		MenuRangeValueCompact* sampleDistance = new MenuRangeValueCompact("SampleDistance", .0001, 0.01, .001, true);
		sampleDistance->setCallback(this);
		so->addMenuItem(sampleDistance);
		_stepSizeMap[sampleDistance] = g;


		SubMenu* contrast = new SubMenu("Contrast");
		so->addMenuItem(contrast);

		MenuRangeValueCompact* contrastbottom = new MenuRangeValueCompact("Contrast Bottom", 0.0, 1.0, 0.0, false);
		contrastbottom->setCallback(this);
		contrast->addItem(contrastbottom);
		_computeShaderMap[contrastbottom] = std::pair<std::string, VolumeGroup*>("ContrastBottom", g);

		MenuRangeValueCompact* contrasttop = new MenuRangeValueCompact("Contrast Top", 0.0, 1.0, 1.0, false);
		contrasttop->setCallback(this);
		contrast->addItem(contrasttop);
		_computeShaderMap[contrasttop] = std::pair<std::string, VolumeGroup*>("ContrastTop", g);


		SubMenu* opacity = new SubMenu("Opacity");
		so->addMenuItem(opacity);

		MenuRangeValueCompact* opacitymult = new MenuRangeValueCompact("Opacity Multiplier", 0.01, 10.0, 1.0, false);
		opacitymult->setCallback(this);
		opacity->addItem(opacitymult);
		_computeShaderMap[opacitymult] = std::pair<std::string, VolumeGroup*>("OpacityMult", g);

		MenuRangeValueCompact* opacitycenter = new MenuRangeValueCompact("Opacity Center", 0.0, 1.0, 1.0, false);
		opacitycenter->setCallback(this);
		opacity->addItem(opacitycenter);
		_computeShaderMap[opacitycenter] = std::pair<std::string, VolumeGroup*>("OpacityCenter", g);

		MenuRangeValueCompact* opacitywidth = new MenuRangeValueCompact("Opacity Width", 0.01, 1.0, 1.0, false);
		opacitywidth->setCallback(this);
		opacity->addItem(opacitywidth);
		_computeShaderMap[opacitywidth] = std::pair<std::string, VolumeGroup*>("OpacityWidth", g);


		//Setu up shader defines
		MenuCheckbox* adaptivequality = new MenuCheckbox("Adaptive Quality", false);
		adaptivequality->setCallback(this);
		so->addMenuItem(adaptivequality);
		_volumeDefineMap[adaptivequality] = std::pair<std::string, VolumeGroup*>("VR_ADAPTIVE_QUALITY", g);

		MenuList* colorfunction = new MenuList();
		std::vector<std::string> colorfunctions = std::vector<std::string>();
		colorfunctions.push_back("Default");
		colorfunctions.push_back("Rainbow");
		colorfunction->setValues(colorfunctions);

		colorfunction->setCallback(this);
		so->addMenuItem(colorfunction);
		_volumeDefineMap[colorfunction] = std::pair<std::string, VolumeGroup*>("COLORFUNCTION", g);
	}
	else if (_stepSizeMap.find(menuItem) != _stepSizeMap.end())
	{
		_stepSizeMap[menuItem]->_StepSize->set(((MenuRangeValueCompact*)menuItem)->getValue());
	}
	else if (_scaleMap.find(menuItem) != _scaleMap.end())
	{
		_scaleMap[menuItem]->setScale(((MenuRangeValueCompact*)menuItem)->getValue());
	}
	else if (_computeShaderMap.find(menuItem) != _computeShaderMap.end())
	{
		_computeShaderMap[menuItem].second->_computeUniforms[_computeShaderMap[menuItem].first]->set(((MenuRangeValueCompact*)menuItem)->getValue());
		_computeShaderMap[menuItem].second->setDirtyAll();
	}
	else if (_volumeDefineMap.find(menuItem) != _volumeDefineMap.end())
	{
		if (_volumeDefineMap[menuItem].first.compare("COLORFUNCTION") == 0)
		{
			MenuList* colorfunction = (MenuList*)menuItem;

			if (colorfunction->getValue().compare("Default") == 0)
			{
				_volumeDefineMap[menuItem].second->getDrawable()->getOrCreateStateSet()->setDefine(_volumeDefineMap[menuItem].first, osg::StateAttribute::OFF);
			}
			else if (colorfunction->getValue().compare("Rainbow") == 0)
			{
				_volumeDefineMap[menuItem].second->getDrawable()->getOrCreateStateSet()->setDefine(_volumeDefineMap[menuItem].first, "hsv2rgb(vec3(ra.r, 1, 1))", osg::StateAttribute::ON);
			}
		}
		else
		{
			MenuCheckbox* checkbox = (MenuCheckbox*)menuItem;
			if (!checkbox)
			{
				return;
			}
			if (checkbox->getValue())
			{
				_volumeDefineMap[menuItem].second->getDrawable()->getOrCreateStateSet()->setDefine(_volumeDefineMap[menuItem].first, osg::StateAttribute::ON);
			}
			else
			{
				_volumeDefineMap[menuItem].second->getDrawable()->getOrCreateStateSet()->setDefine(_volumeDefineMap[menuItem].first, osg::StateAttribute::OFF);
			}
		}
	}
}
