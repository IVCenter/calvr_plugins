#include "HelmsleyVolume.h"

#include <cvrKernel/NodeMask.h>

#include <ctime>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <fstream>


using namespace cvr;

HelmsleyVolume * HelmsleyVolume::_instance = NULL;


CVRPLUGIN(HelmsleyVolume)

std::string HelmsleyVolume::loadShaderFile(std::string filename)
{
	std::string shaderDir = cvr::ConfigManager::getEntry("Plugin.HelmsleyVolume.ShaderDir");
	std::string fileLoc = shaderDir + filename;

	std::ifstream file(fileLoc.c_str());
	if (!file) return "";

	std::stringstream sstr;
	sstr << file.rdbuf();

	file.close();

	return sstr.str();
}

HelmsleyVolume::HelmsleyVolume()
{
	_buttonMap = std::map<cvr::MenuItem*, std::string>();
	_volumes = std::vector<osg::ref_ptr<VolumeGroup> >();
	_sceneObjects = std::vector<SceneObject*>();
	_volumeMenus = std::vector<VolumeMenu*>();
}

HelmsleyVolume::~HelmsleyVolume()
{
}

bool HelmsleyVolume::init()
{
	_instance = this;

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

	screenshotTool = new ScreenshotTool("Screenshot Tool", false, true, false, false, false);
	screenshotTool->setPosition(osg::Vec3(0, 400, 500));
	PluginHelper::registerSceneObject(screenshotTool, "HelmsleyVolume");
	screenshotTool->attachToScene();


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

	_selectionMatrix = osg::Matrix();
	_selectionMatrix.makeTranslate(osg::Vec3(-300, 500, 300));
	_selectionMenu->setTransform(_selectionMatrix);

	
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
			_buttonMap[button] = configbase + "." + list[i];
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
	if (e->getInteraction() == BUTTON_DOWN || e->getInteraction() == MOVE)
	{
		e->printValues();
	}
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
		if (e->getInteraction() == BUTTON_DOWN && e->asTrackedButtonEvent() && e->asTrackedButtonEvent()->getButton() == _radialButton)
		{
			if (!_radialShown)
			{
				_selectionMenu->setVisible(true);
				_selectionMenu->getRootObject()->getParent(0)->removeChild(_selectionMenu->getRootObject());
				PluginHelper::getHand(e->asHandEvent()->getHand())->addChild(_selectionMenu->getRootObject());
				_radialShown = true;
			}
			else
			{
				_radialShown = false;
				_selectionMenu->setVisible(false);
				PluginHelper::getHand(e->asHandEvent()->getHand())->removeChild(_selectionMenu->getRootObject());
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
		so->addChild(g);

		PluginHelper::registerSceneObject(so, "HelmsleyVolume");
		so->attachToScene();
		so->setNavigationOn(false);
		so->addMoveMenuItem();
		so->addNavigationMenuItem();
		so->setShowBounds(true);

		VolumeMenu* menu = new VolumeMenu(so, g);
		menu->init();

		_sceneObjects.push_back(so);
		_volumes.push_back(g);
		_volumeMenus.push_back(menu);

		MenuButton* removeButton = new MenuButton("Remove Volume");
		so->addMenuItem(removeButton);
		removeButton->setCallback(this);
		_removeButtons.push_back(removeButton);
	}
	else if (std::find(_removeButtons.begin(), _removeButtons.end(), (MenuButton*)menuItem) != _removeButtons.end())
	{
		std::vector<MenuButton*>::iterator it = std::find(_removeButtons.begin(), _removeButtons.end(), (MenuButton*)menuItem);
		int index = std::distance(_removeButtons.begin(), it);

		_sceneObjects[index]->detachFromScene();
		delete _volumeMenus[index];
		delete _removeButtons[index];
		_volumes[index].release();
		delete _sceneObjects[index];
		//delete _sceneObjects[index];
		//delete _volumes[index]; //deleted automatically because no references left once sceneobject is deleted
		_volumeMenus.erase(_volumeMenus.begin() + index);
		_volumes.erase(_volumes.begin() + index);
		_sceneObjects.erase(_sceneObjects.begin() + index);
		_removeButtons.erase(_removeButtons.begin() + index);
	}
	
}
