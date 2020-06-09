#include "HelmsleyVolume.h"
#include "UIExtensions.h"

#include <cvrKernel/NodeMask.h>
#include <cvrMenu/NewUI/UIPopup.h>
#include <cvrMenu/NewUI/UIQuadElement.h>
#include <cvrMenu/NewUI/UIList.h>
#include <cvrMenu/NewUI/UIText.h>
#include <cvrMenu/NewUI/UIButton.h>
#include <cvrMenu/NewUI/UITexture.h>
#include <cvrMenu/NewUI/UISlider.h>
#include <cvrMenu/MenuManager.h>
#include <cvrUtil/ComputeBoundingBoxVisitor.h>


#ifdef WITH_OPENVR
#include <cvrKernel/OpenVRDevice.h>
#endif

#include <osgDB/ReadFile>

#include <ctime>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <fstream>


using namespace cvr;

HelmsleyVolume * HelmsleyVolume::_instance = NULL;


MenuRangeValue* _xpos;
MenuRangeValue* _ypos;
MenuRangeValue* _zpos;
MenuRangeValue* _rotmenu;

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

void HelmsleyVolume::resetOrientation()
{
#ifdef WITH_OPENVR
	if (OpenVRDevice::instance() != nullptr)
	{
		OpenVRDevice* device = OpenVRDevice::instance();
		osg::Matrix curr = device->getUniverseMatrix();

		osg::Vec3 pos = device->position();
		osg::Quat rot = device->orientation();

		osg::Vec3 startPos = cvr::ConfigManager::getVec3("Plugin.HelmsleyVolume.Orientation.Hmd.Position", osg::Vec3(0, 1000, -1000));
		osg::Vec3 diff = (startPos - pos) / 1000.0f;
		double xrot = atan2(2 * (rot.w() * rot.x() + rot.y() * rot.z()), 1 - 2 * (rot.x() * rot.x() + rot.y() * rot.y()));
		osg::Matrix m;
		m.makeTranslate(diff);
		//std::cout << pos.x() << ", " << pos.y() << ", " << pos.z() << std::endl;
		//m.preMultRotate(osg::Quat(xrot, osg::Vec3(0, 1, 0)));
		device->setUniverseMatrix(m * curr);
	}
#endif
}

HelmsleyVolume::HelmsleyVolume()
{
	_buttonMap = std::map<cvr::MenuItem*, std::string>();
	_volumes = std::vector<osg::ref_ptr<VolumeGroup> >();
	_sceneObjects = std::vector<SceneObject*>();
	_contextMenus = std::vector<VolumeMenu*>();
	_worldMenus = std::vector<NewVolumeMenu*>();
	_cuttingPlanes = std::vector<CuttingPlane*>();
	_measurementTools = std::vector<MeasurementTool*>();
	_removeClippingPlaneButtons = std::vector<MenuButton*>();
}

HelmsleyVolume::~HelmsleyVolume()
{
}

bool HelmsleyVolume::init()
{
	_vMenu = new SubMenu("HelmsleyVolume", "HelmsleyVolume");
	_vMenu->setCallback(this);

	
#ifdef WITH_OPENVR
	std::string modelDir = cvr::ConfigManager::getEntry("Plugin.HelmsleyVolume.ModelDir");
	std::cout << "Modl Dir: " << modelDir << std::endl;


	osgDB::Options* roomOptions = new osgDB::Options("noReverseFaces");
	osg::Node* room = osgDB::readNodeFile(modelDir + "testPrim.obj", roomOptions);

	_room = new SceneObject("room", false, false, false, false, false);
	_room->addChild(room);
	_room->setScale(800);
	_room->setPosition(osg::Vec3(12000, 0, 0));

	PluginHelper::registerSceneObject(_room, "HelmsleyVolume");
	_room->attachToScene();
	_nm = _room->getChildNode(0)->getNodeMask();

	_roomLocation = new SubMenu("Location", "Location");
	_roomLocation->setCallback(this);


	_privateLoc = new MenuButton("Private Lounge");
	_privateLoc->setCallback(this);
	_bedLoc = new MenuButton("Pediatric Room");
	_bedLoc->setCallback(this);
	_screenLoc = new MenuButton("Radiology Theater");
	_screenLoc->setCallback(this);

	_hideRoom = new MenuButton("Hide Room");
	_hideRoom->setCallback(this);

	
	_roomLocation->addItem(_privateLoc);
	_roomLocation->addItem(_bedLoc);
	_roomLocation->addItem(_screenLoc);
	_roomLocation->addItem(_hideRoom);

	_vMenu->addItem(_roomLocation);

	_xpos = new MenuRangeValue("X", -20000, 20000, 12000);
	_ypos = new MenuRangeValue("Y", -20000, 20000, 0);
	_zpos = new MenuRangeValue("Z", -20000, 20000, 0);
	_rotmenu = new MenuRangeValue("rot", -osg::PI, osg::PI, 0, osg::PI/2.0);

	_xpos->setCallback(this);
	_ypos->setCallback(this);
	_zpos->setCallback(this);
	_rotmenu->setCallback(this);

	_vMenu->addItem(_xpos);
	_vMenu->addItem(_ypos);
	_vMenu->addItem(_zpos);
	_vMenu->addItem(_rotmenu);

#endif
	
	osg::setNotifyLevel(osg::NotifySeverity::WARN);
	_splashscreen = new UIPopup();
	_splashscreen->setPosition(osg::Vec3(-800, 1000, 1850));
	_splashscreen->getRootElement()->setAbsoluteSize(osg::Vec3(1600, 1, 900));

	std::string splashdir = ConfigManager::getEntry("Plugin.HelmsleyVolume.ModelDir");
	UITexture* splashtex = new UITexture(splashdir + "3DMIP_Splash.png");

	_splashscreen->addChild(splashtex);
	_splashscreen->setActive(true, true);


#ifdef WITH_OPENVR
	osg::MatrixTransform* mt = PluginHelper::getHand(1);
	if (mt)
	{
		OpenVRDevice::instance()->updatePose();
		osg::Geode* hand = new osg::Geode();
		hand->addDrawable(OpenVRDevice::instance()->controllers[0].renderModel);
		mt->addChild(hand);
		std::cerr << "Set hand 1 to use rendermodel" << std::endl;
	}
	mt = PluginHelper::getHand(2);
	if (mt)
	{
		osg::Geode* hand = new osg::Geode();
		hand->addDrawable(OpenVRDevice::instance()->controllers[1].renderModel);
		mt->addChild(hand);
		std::cerr << "Set hand 2 to use rendermodel" << std::endl;
	}
#endif

	std::string fontfile = CalVR::instance()->getResourceDir();
	fontfile = fontfile + "/resources/ArenaCondensed.ttf";

	osgText::Font * font = osgText::readFontFile(fontfile);
	if (font)
	{
		UIUtil::setDefaultFont(font);
	}

	_instance = this;

	std::vector<osg::Camera*> cameras = std::vector<osg::Camera*>();
	cvr::CVRViewer::instance()->getCameras(cameras);
	for (int i = 0; i < cameras.size(); ++i)
	{
		cameras[i]->getGraphicsContext()->getState()->setUseModelViewAndProjectionUniforms(true);

	}


	_interactButton = cvr::ConfigManager::getInt("Plugin.HelmsleyVolume.InteractButton", 0);
	//_cuttingPlaneDistance = cvr::ConfigManager::getFloat("Plugin.HelmsleyVolume.CuttingPlaneDistance", 200.0f);
	//float size = cvr::ConfigManager::getFloat("Plugin.HelmsleyVolume.CuttingPlaneSize", 500.0f);


	screenshotTool = new ScreenshotTool("Screenshot Tool", false, true, false, false, true);
	PluginHelper::registerSceneObject(screenshotTool, "HelmsleyVolume");

	fileSelector = new FileSelector();


	//osg::setNotifyLevel(osg::NOTICE);
	std::cerr << "HelmsleyVolume init" << std::endl;


	SubMenu* fileMenu = new SubMenu("Files", "Files");
	fileMenu->setCallback(this);
	_vMenu->addItem(fileMenu);

	/*
	_cpButton = new MenuButton("Cutting Plane");
	_cpButton->setCallback(this);
	_mtButton = new MenuCheckbox("Measurement Tool", false);
	_mtButton->setCallback(this);
	_toolButton = nullptr;
	_stCheckbox = new MenuCheckbox("Screenshot Tool", false);
	_stCheckbox->setCallback(this);
	_vMenu->addItem(_cpButton);
	_vMenu->addItem(_mtButton);
	_vMenu->addItem(_stCheckbox);
	

	_selectionMenu = new PopupMenu("Interaction options", "", false);
	_selectionMenu->setVisible(false);

	_selectionMatrix = osg::Matrix();
	_selectionMatrix.makeTranslate(osg::Vec3(-300, 500, 300));
	_selectionMenu->setTransform(_selectionMatrix);

	*/
	/*
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
	*/

	//_toolMenu = new ToolMenu();

	createList(fileMenu, "Plugin.HelmsleyVolume.Files");

	MenuSystem::instance()->addMenuItem(_vMenu);

#ifdef WITH_OPENVR
	_resetHMD = new MenuButton("Reset HMD");
	_resetHMD->setCallback(this);
	MenuSystem::instance()->addMenuItem(_resetHMD);
#endif

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
	++_frameNum;
	int max = 900;
	int fade = 180;

	if (_frameNum > max)
	{
		//_splashscreen->setActive(false, false);
		MenuManager::instance()->removeMenuSystem(_splashscreen);
		// delete(_splashscreen);
	}
	else if(_frameNum > max - fade)
	{
		UITexture* splashtex = (UITexture*)_splashscreen->getRootElement()->getChild(0);
		splashtex->setTransparent(true);
		splashtex->setColor(osg::Vec4(1, 1, 1, (float)(max - _frameNum) / (float)fade));
	}
}

bool HelmsleyVolume::processEvent(InteractionEvent * e)
{
	if (e->getInteraction() == BUTTON_DOWN)
	{
		if (_tool == MEASUREMENT_TOOL && _lastMeasurementTool >= 0 && _lastMeasurementTool < _volumes.size())
		{
			//Measurement tool
			osg::Matrix mat = PluginHelper::getHandMat(e->asHandEvent()->getHand());
			//osg::Vec4d position = osg::Vec4(0, 0, 0, 1) * mat;
			//osg::Vec3f pos = osg::Vec3(position.x(), position.y(), position.z());

			if (e->getInteraction() == BUTTON_DOWN)
			{
				_measurementTools[_lastMeasurementTool]->setStart(mat.getTrans());
			}
		}
	}
	else if (e->getInteraction() == BUTTON_DRAG)
	{
		if (e->asTrackedButtonEvent() && e->asTrackedButtonEvent()->getButton() == _interactButton)
		{
			if (_tool == MEASUREMENT_TOOL && _lastMeasurementTool >= 0 && _lastMeasurementTool < _volumes.size())
			{
				//Measurement tool
				osg::Matrix mat = PluginHelper::getHandMat(e->asHandEvent()->getHand());
				//osg::Vec4d position = osg::Vec4(0, 0, 0, 1) * mat;
				//osg::Vec3f pos = osg::Vec3(position.x(), position.y(), position.z());

				_measurementTools[_lastMeasurementTool]->setEnd(mat.getTrans());
				_measurementTools[_lastMeasurementTool]->activate();
				return true;
			}
		}
	}
	else if (e->getInteraction() == BUTTON_UP)
	{
		if (e->asTrackedButtonEvent() && e->asTrackedButtonEvent()->getButton() == _interactButton)
		{
			if (_tool == CUTTING_PLANE)
			{
				//Cutting plane
				cuttingPlane->setNodeMask(0);

				return true; 
			}
			else if (_tool == MEASUREMENT_TOOL && _lastMeasurementTool >= 0 && _lastMeasurementTool < _volumes.size())
			{
				//Measurement tool
				if (_measurementTools[_lastMeasurementTool]->getLength() < 5.0)
				{
					_measurementTools[_lastMeasurementTool]->deactivate();
				}
			}
		}

	}
    
	return false;
}

void HelmsleyVolume::menuCallback(MenuItem* menuItem)
{
#ifdef WITH_OPENVR
	if (menuItem == _privateLoc)
	{
		_room->setPosition(osg::Vec3(-800, 3500, -420));
		_room->setRotation(osg::Quat(-osg::PI/2, osg::Vec3(0, 0, -1)));
	}
	else if (menuItem == _screenLoc)
	{
		_room->setPosition(osg::Vec3(-12000, -16000, 0));
		_room->setRotation(osg::Quat(osg::PI * 3.0/4.0, osg::Vec3(0, 0, -1)));
	}
	else if (menuItem == _bedLoc)
	{
		_room->setPosition(osg::Vec3(-7500, 9000, 0));
		_room->setRotation(osg::Quat(-osg::PI / 2, osg::Vec3(0, 0, -1)));
	}
	else if (menuItem == _hideRoom)
	{
		if (_hideRoom->getText() == "Hide Room") {
			_room->getChildNode(0)->setNodeMask(0);
			_hideRoom->setText("Show Room");
		}
		else {
			_room->getChildNode(0)->setNodeMask(_nm);
			_hideRoom->setText("Hide Room");
		}
	}
	else if (menuItem == _xpos)
	{
		osg::Vec3 pos = _room->getPosition();
		pos.x() = _xpos->getValue();
		_room->setPosition(pos);
	}
	else if (menuItem == _ypos)
	{
		osg::Vec3 pos = _room->getPosition();
		pos.y() = _ypos->getValue();
		_room->setPosition(pos);
	}
	else if (menuItem == _zpos)
	{
		osg::Vec3 pos = _room->getPosition();
		pos.z() = _zpos->getValue();
		_room->setPosition(pos);
	}
	else if (menuItem == _rotmenu)
	{
		_room->setRotation(osg::Quat(_rotmenu->getValue(), osg::Vec3(0, 0, -1)));
	}
#endif

	if (_buttonMap.find(menuItem) != _buttonMap.end())
	{
		bool found;
		std::string path = cvr::ConfigManager::getEntry(_buttonMap.at(menuItem), "", &found);
		if (!found) return;

		std::string maskpath = cvr::ConfigManager::getEntry(_buttonMap.at(menuItem) + ".Mask", "", &found);

		loadVolume(path, maskpath);
	}
	else if (std::find(_removeButtons.begin(), _removeButtons.end(), (MenuButton*)menuItem) != _removeButtons.end())
	{
		std::vector<MenuButton*>::iterator it = std::find(_removeButtons.begin(), _removeButtons.end(), (MenuButton*)menuItem);
		int index = std::distance(_removeButtons.begin(), it);

		removeVolume(index);
	}
	else if (std::find(_removeClippingPlaneButtons.begin(), _removeClippingPlaneButtons.end(), (MenuButton*)menuItem) != _removeClippingPlaneButtons.end())
	{
		std::vector<MenuButton*>::iterator it = std::find(_removeClippingPlaneButtons.begin(), _removeClippingPlaneButtons.end(), (MenuButton*)menuItem);
		int index = std::distance(_removeClippingPlaneButtons.begin(), it);

		removeCuttingPlane(index);
	}
	else if (menuItem == _cpButton)
	{
		if (_volumes.size())
		{
			createCuttingPlane(0);
		}
	}
	else if (menuItem == _mtButton)
	{
		if (_mtButton->getValue())
		{
			_tool = MEASUREMENT_TOOL;
			if (_toolButton && _toolButton != _mtButton)
			{
				_toolButton->setValue(false);
			}
			_toolButton = _mtButton;
		}
		else
		{
			_tool = NONE;
		}
	}
	else if (menuItem == _stCheckbox)
	{
		toggleScreenshotTool(_stCheckbox->getValue());
	}
	else if (menuItem == _resetHMD)
	{
		resetOrientation();
	}
}

void HelmsleyVolume::toggleScreenshotTool(bool on)
{
	if (on)
	{
		screenshotTool->attachToScene();
		screenshotTool->activate();
	}
	else
	{
		screenshotTool->detachFromScene();
		screenshotTool->deactivate();
	}
}

void HelmsleyVolume::activateMeasurementTool(int volume)
{
	_lastMeasurementTool = volume;
}

void HelmsleyVolume::deactivateMeasurementTool(int volume)
{
	_measurementTools[volume]->deactivate();
	_lastMeasurementTool = -1;
}

CuttingPlane* HelmsleyVolume::createCuttingPlane(unsigned int i)
{
	if (i >= _volumes.size())
	{
		return nullptr;
	}
	CuttingPlane* cp = new  CuttingPlane("Cutting Plane", false, true, false, true, true);
	cp->setVolume(_volumes[i]);
	cp->setSceneObject(_sceneObjects[i]);
	MenuButton* remove = new MenuButton("Remove Cutting Plane");
	cp->addMenuItem(remove);
	remove->setCallback(this);
	_removeClippingPlaneButtons.push_back(remove);
	PluginHelper::registerSceneObject(cp, "HelmsleyVolume");
	//cp->attachToScene();
	_sceneObjects[i]->addChild(cp);
	_cuttingPlanes.push_back(cp);
	return cp;
}

void HelmsleyVolume::removeCuttingPlane(unsigned int i)
{
	if (i < _cuttingPlanes.size() && _cuttingPlanes[i])
	{
		_volumes[i]->_PlaneNormal->set(osg::Vec3(0.f, 1.f, 0.f));
		_volumes[i]->_PlanePoint->set(osg::Vec3(0.f, -2.f, 0.f));
		_cuttingPlanes[i]->detachFromScene();
		delete(_cuttingPlanes[i]);
		_cuttingPlanes.erase(_cuttingPlanes.begin() + i);

		_removeClippingPlaneButtons.erase(_removeClippingPlaneButtons.begin() + i);
	}
}

void HelmsleyVolume::loadVolume(std::string path, std::string maskpath)
{
	SceneObject * so;
	so = new SceneObject("volume", false, true, true, true, false);
	so->setPosition(ConfigManager::getVec3("Plugin.HelmsleyVolume.Orientation.Volume.Position", osg::Vec3(0, 750, 500)));

	VolumeGroup * g = new VolumeGroup();
	std::cout << "volume group initiated" << std::endl;
	g->loadVolume(path, maskpath);
	so->addChild(g);
	std::cout << "volume loaded and added" << std::endl;
	MeasurementTool* tool = new MeasurementTool("Measurement Tool", false, false, false, false, false);
	tool->deactivate();
	so->addChild(tool);
	_measurementTools.push_back(tool);
	std::cout << "measurement tool added" << std::endl;


	PluginHelper::registerSceneObject(so, "HelmsleyVolume");
	std::cout << "registered scene object" << std::endl;
	so->attachToScene();
	so->setNavigationOn(false);
	so->addMoveMenuItem();
	so->addNavigationMenuItem();
	so->setShowBounds(true);
	std::cout << "scene Object set up" << std::endl;
	//Manually set the bounding box (since clipping plane / other things will be attached
	//so->setBoundsCalcMode(SceneObject::MANUAL);
	/*
	osg::BoundingBox bb;
	bb.init();
	ComputeBoundingBoxVisitor cbbv;
	cbbv.setBound(bb);
	g->accept(cbbv);
	bb = cbbv.getBound();
	so->setBoundingBox(bb);
	*/

	VolumeMenu* menu = new VolumeMenu(so, g);
	menu->init();

	std::cout << "regular menu created" << std::endl;

	NewVolumeMenu* newMenu = new NewVolumeMenu(so, g);
	newMenu->init();

	std::cout << "new menu initiated" << std::endl;
	_sceneObjects.push_back(so);
	_volumes.push_back(g);
	_contextMenus.push_back(menu);
	_worldMenus.push_back(newMenu);


	/*
	MenuButton* removeButton = new MenuButton("Remove Volume");
	so->addMenuItem(removeButton);
	removeButton->setCallback(this);
	_removeButtons.push_back(removeButton);
	*/
}

void HelmsleyVolume::removeVolume(int index)
{
	//Remove all cutting planes that are attached to the volume
	std::vector<CuttingPlane*>::iterator it = _cuttingPlanes.begin();
	while (it != _cuttingPlanes.end()) {
		if ((*it)->getVolume() == _volumes[index])
		{
			(*it)->detachFromScene();
			delete((*it));
			it = _cuttingPlanes.erase(it);
		}
		else
		{
			++it;
		}

	} 

	_sceneObjects[index]->detachFromScene();
	delete _contextMenus[index];
	delete _worldMenus[index];
	//delete _removeButtons[index];
	_volumes[index].release();
	//delete _sceneObjects[index];
	delete _sceneObjects[index];
	//delete _volumes[index]; //deleted automatically because no references left once sceneobject is deleted
	_contextMenus.erase(_contextMenus.begin() + index);
	_worldMenus.erase(_worldMenus.begin() + index);
	_volumes.erase(_volumes.begin() + index);
	_sceneObjects.erase(_sceneObjects.begin() + index);
	//_removeButtons.erase(_removeButtons.begin() + index);
}