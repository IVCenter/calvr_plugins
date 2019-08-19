#include "HelmsleyVolume.h"

#include <ctime>
#include <iostream>

using namespace cvr;

CVRPLUGIN(HelmsleyVolume)

HelmsleyVolume::HelmsleyVolume()
{
	_buttonMap = std::map<cvr::MenuItem*, std::string>();
	_stepSizeMap = std::map<cvr::MenuItem*, VolumeGroup*>();
	_computeShaderMap = std::map<cvr::MenuItem*, std::pair<std::string, VolumeGroup*>>();
	_volumes = std::vector<VolumeGroup*>();
	_sceneObjects = std::vector<SceneObject*>();
}

HelmsleyVolume::~HelmsleyVolume()
{
}

bool HelmsleyVolume::init()
{
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
	labels.push_back(modelDir + "pen_ucsd.obj");
	labels.push_back(modelDir + "eraser_ucsd.obj");

	symbols.push_back(true);
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
	if (e->getInteraction() == Interaction::BUTTON_DOWN || e->getInteraction() == Interaction::BUTTON_DRAG)
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

					osg::Vec4f position = osg::Vec4(0, 0, 0, 1) * objhand;
					osg::Vec3f pos = osg::Vec3(position.x(), position.y(), position.z());

					osg::Quat q = osg::Quat();
					osg::Quat q2 = osg::Quat();
					osg::Vec3 v = osg::Vec3();
					osg::Vec3 v2 = osg::Vec3();
					objhand.decompose(v, q, v2, q2);
					osg::Matrix m = osg::Matrix();
					m.makeRotate(q);


					osg::Vec4d normal = osg::Vec4(0, 1, 0, 0) * m;
					osg::Vec3 norm = osg::Vec3(normal.x(), normal.y(), normal.z());



					_volumes[i]->_PlanePoint->set(pos);
					_volumes[i]->_PlaneNormal->set(norm);
				}
				return true;
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
		_volumes.push_back(g);
		_sceneObjects.push_back(so);

		//osg::Geode * g = new osg::Geode();
		//g->addDrawable(new VolumeDrawable());
		so->addChild(g);
		PluginHelper::registerSceneObject(so, "HelmsleyVolume");
		so->attachToScene();
		so->setNavigationOn(true);
		so->addMoveMenuItem();
		so->addNavigationMenuItem();
		so->addScaleMenuItem("Size", 0.1f, 10.0f, 1.0f);


		MenuRangeValueCompact* sd = new MenuRangeValueCompact("SampleDistance", .001, 0.1, .00135, true);
		sd->setCallback(this);
		so->addMenuItem(sd);
		_stepSizeMap[sd] = g;

		MenuRangeValueCompact* e = new MenuRangeValueCompact("Exposure", 0.0, 5.0, 1.5, false);
		e->setCallback(this);
		so->addMenuItem(e);
		_computeShaderMap[e] = std::pair<std::string, VolumeGroup*>("Exposure", g);

		MenuRangeValueCompact* d = new MenuRangeValueCompact("Density", 0.01, 1.0, 0.5, true);
		d->setCallback(this);
		so->addMenuItem(d);
		_computeShaderMap[d] = std::pair<std::string, VolumeGroup*>("Density", g);

		MenuRangeValueCompact* t = new MenuRangeValueCompact("Threshold", 0.01, 1.0, 0.2, true);
		t->setCallback(this);
		so->addMenuItem(t);
		_computeShaderMap[t] = std::pair<std::string, VolumeGroup*>("Threshold", g);

	}
	if (_stepSizeMap.find(menuItem) != _stepSizeMap.end())
	{
		_stepSizeMap[menuItem]->_StepSize->set(((MenuRangeValueCompact*)menuItem)->getValue());
	}
	if (_computeShaderMap.find(menuItem) != _computeShaderMap.end())
	{
		_computeShaderMap[menuItem].second->_computeUniforms[_computeShaderMap[menuItem].first]->set(((MenuRangeValueCompact*)menuItem)->getValue());
		_computeShaderMap[menuItem].second->setDirty();
	}
}
