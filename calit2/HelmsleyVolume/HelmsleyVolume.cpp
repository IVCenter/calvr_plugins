#include "HelmsleyVolume.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrMenu/MenuSystem.h>
#include <cvrKernel/ComController.h>

#include <ctime>
#include <iostream>

using namespace cvr;

CVRPLUGIN(HelmsleyVolume)

HelmsleyVolume::HelmsleyVolume()
{
	_buttonMap = std::map<cvr::MenuItem*, std::string>();
}

HelmsleyVolume::~HelmsleyVolume()
{
}

bool HelmsleyVolume::init()
{
	std::cerr << "HelmsleyVolume init" << std::endl;

	_vMenu = new SubMenu("HelmsleyVolume", "HelmsleyVolume");
	_vMenu->setCallback(this);

	SubMenu* fileMenu = new SubMenu("Files", "Files");
	fileMenu->setCallback(this);
	_vMenu->addItem(fileMenu);


	std::string modelDir = cvr::ConfigManager::getEntry("Plugin.HelmsleyVolume.ModelDir");
	std::cout << modelDir << std::endl;
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
	_vMenu->addItem(_radial);

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

void HelmsleyVolume::menuCallback(MenuItem* menuItem)
{

	if (_buttonMap.find(menuItem) != _buttonMap.end())
	{
		SceneObject * so;
		so = new SceneObject("volume", false, true, true, true, false);
		VolumeGroup * g = new VolumeGroup();
		g->loadVolume(_buttonMap.at(menuItem));

		//osg::Geode * g = new osg::Geode();
		//g->addDrawable(new VolumeDrawable());
		so->addChild(g);
		PluginHelper::registerSceneObject(so, "HelmsleyVolume");
		so->attachToScene();
		so->setNavigationOn(true);
		so->addMoveMenuItem();
		so->addNavigationMenuItem();
		so->setShowBounds(true);
		so->addScaleMenuItem("Size", 0.1f, 10.0f, 1.0f);
	}
}
