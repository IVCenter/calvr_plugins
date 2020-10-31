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
#include <osg/Material>
#include <osg/TextureCubeMap>

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
	delete fileSelector;
	screenshotTool->detachFromScene();
	delete screenshotTool;
	centerLineTool->detachFromScene();
	delete centerLineTool;
	delete _room;
}

bool HelmsleyVolume::init()
{
	


	_vMenu = new SubMenu("HelmsleyVolume", "HelmsleyVolume");
	_vMenu->setCallback(this);

	
#ifdef WITH_OPENVR
	std::string modelDir = cvr::ConfigManager::getEntry("Plugin.HelmsleyVolume.ModelDir");


	osgDB::Options* roomOptions = new osgDB::Options("noReverseFaces");
	osg::Node* room = osgDB::readNodeFile(modelDir + "MIPCDVIZV3.obj", roomOptions);
	////////////////////////////Ryans forloop///////////////////
	//for (unsigned int i = 0; i < room->asGroup()->getNumChildren(); i++) {

	//	osg::Geode* g = dynamic_cast<osg::Geode*> (room->asGroup()->getChild(i));
	//	osg::Geometry* gm = g->getDrawable(0)->asGeometry();
	//	osg::Vec3Array* v = (osg::Vec3Array*) gm->getVertexArray();
	//	osg::Vec3Array* tv = (osg::Vec3Array*)(gm->getTexCoordArray(0));
	//	osg::Vec3Array* nv = (osg::Vec3Array*)(gm->getNormalArray());

	//	gm->setVertexAttribArray(0, v, osg::Array::BIND_PER_VERTEX);

	//	osg::ref_ptr<osg::Program> program(new osg::Program());

	//	if (g->getDrawable(0)->getOrCreateStateSet()->getTextureAttributeList().size() != 0) {

	//		gm->setVertexAttribArray(2, tv, osg::Array::BIND_PER_VERTEX);
	//		gm->setVertexAttribArray(1, nv, osg::Array::BIND_PER_VERTEX);

	//		program->addShader(new osg::Shader(osg::Shader::VERTEX, loadShaderFile("room.vert")));
	//		program->addShader(new osg::Shader(osg::Shader::FRAGMENT, loadShaderFile("texBRDF.frag")));

	//		osg::Uniform* textureUniform = new osg::Uniform(osg::Uniform::SAMPLER_2D, "tex");
	//		textureUniform->set(0);
	//		g->getDrawable(0)->getOrCreateStateSet()->addUniform(textureUniform);
	//		osg::Uniform* bumpUniform = new osg::Uniform(osg::Uniform::SAMPLER_2D, "normal");
	//		bumpUniform->set(1);
	//		g->getDrawable(0)->getOrCreateStateSet()->addUniform(bumpUniform);

	//		if (i >= 5 && i <= 6) {
	//			osg::Uniform* heightMap = new osg::Uniform("heightMap", 1);
	//			g->getDrawable(0)->getOrCreateStateSet()->addUniform(heightMap);
	//		}
	//		else {
	//			osg::Uniform* heightMap = new osg::Uniform("heightMap", 0);
	//			g->getDrawable(0)->getOrCreateStateSet()->addUniform(heightMap);
	//		}
	//	}
	//	else if (i == 25) {

	//		gm->setVertexAttribArray(1, nv, osg::Array::BIND_PER_VERTEX);

	//		program->addShader(new osg::Shader(osg::Shader::VERTEX, loadShaderFile("glass.vert")));
	//		program->addShader(new osg::Shader(osg::Shader::FRAGMENT, loadShaderFile("glass.frag")));
	//		osg::ref_ptr<osg::TextureCubeMap> cubemap = new osg::TextureCubeMap;

	//		osg::Image* imagePosX = osgDB::readImageFile(modelDir + "textures/blurrytrees.jpg");
	//		osg::Image* imageNegX = osgDB::readImageFile(modelDir + "textures/blurrytrees.jpg");
	//		osg::Image* imagePosY = osgDB::readImageFile(modelDir + "textures/blurrytrees.jpg");
	//		osg::Image* imageNegY = osgDB::readImageFile(modelDir + "textures/blurrytrees.jpg");
	//		osg::Image* imagePosZ = osgDB::readImageFile(modelDir + "textures/blurrytrees.jpg");
	//		osg::Image* imageNegZ = osgDB::readImageFile(modelDir + "textures/blurrytrees.jpg");

	//		cubemap->setImage(osg::TextureCubeMap::POSITIVE_X, imagePosX);
	//		cubemap->setImage(osg::TextureCubeMap::NEGATIVE_X, imageNegX);
	//		cubemap->setImage(osg::TextureCubeMap::POSITIVE_Y, imagePosY);
	//		cubemap->setImage(osg::TextureCubeMap::NEGATIVE_Y, imageNegY);
	//		cubemap->setImage(osg::TextureCubeMap::POSITIVE_Z, imagePosZ);
	//		cubemap->setImage(osg::TextureCubeMap::NEGATIVE_Z, imageNegZ);

	//		cubemap->setWrap(osg::Texture::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
	//		cubemap->setWrap(osg::Texture::WRAP_T, osg::Texture::CLAMP_TO_EDGE);
	//		cubemap->setWrap(osg::Texture::WRAP_R, osg::Texture::CLAMP_TO_EDGE);

	//		cubemap->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR);
	//		cubemap->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);

	//		cubemap->setResizeNonPowerOfTwoHint(true);

	//		g->getDrawable(0)->getOrCreateStateSet()->setTextureAttributeAndModes(0, cubemap, osg::StateAttribute::ON);

	//		osg::Uniform* environment = new osg::Uniform("skybox", 0);
	//		g->getDrawable(0)->getOrCreateStateSet()->addUniform(environment);
	//	}
	//	else {

	//		gm->setVertexAttribArray(1, nv, osg::Array::BIND_PER_VERTEX);

	//		program->addShader(new osg::Shader(osg::Shader::VERTEX, loadShaderFile("colored.vert")));
	//		program->addShader(new osg::Shader(osg::Shader::FRAGMENT, loadShaderFile("BRDF.frag")));
	//	}

	//	g->getDrawable(0)->getOrCreateStateSet()->setAttributeAndModes(program.get(), osg::StateAttribute::ON);

	//	osg::Material* mat = (osg::Material*)(g->getDrawable(0)->getOrCreateStateSet()->getAttribute(osg::StateAttribute::MATERIAL, 0));
	//	osg::Uniform* diffuse = new osg::Uniform("diffuseVal", mat->getDiffuse(osg::Material::Face::FRONT));
	//	g->getDrawable(0)->getOrCreateStateSet()->addUniform(diffuse);
	//	osg::Uniform* ambient = new osg::Uniform("ambientVal", mat->getAmbient(osg::Material::Face::FRONT));
	//	g->getDrawable(0)->getOrCreateStateSet()->addUniform(ambient);
	//	osg::Uniform* specular = new osg::Uniform("specularVal", mat->getSpecular(osg::Material::Face::FRONT));
	//	g->getDrawable(0)->getOrCreateStateSet()->addUniform(specular);
	//	osg::Uniform* shininess = new osg::Uniform("shininess", mat->getShininess(osg::Material::Face::FRONT));
	//	g->getDrawable(0)->getOrCreateStateSet()->addUniform(shininess);

	//	osg::Uniform* metalUni = new osg::Uniform("metalUni", mat->getSpecular(osg::Material::Face::FRONT).x());
	//	g->getDrawable(0)->getOrCreateStateSet()->addUniform(metalUni);

	//	osg::Uniform* roughUni = new osg::Uniform("roughUni", mat->getSpecular(osg::Material::Face::FRONT).y());
	//	g->getDrawable(0)->getOrCreateStateSet()->addUniform(roughUni);
	//}

	/////////////////////////////////////////////////////////
	_room = new SceneObject("room", false, false, false, false, false);
	_room->addChild(room);
	_room->setScale(800);
	_room->setPosition(osg::Vec3(-1030, 2125, 0));
	osg::Quat rot;
	rot.makeRotate(-osg::PI_2, 0, 0, 1);
	_room->setRotation(rot);
	PluginHelper::registerSceneObject(_room, "HelmsleyVolume");
	_room->attachToScene();
	_nm = _room->getChildNode(0)->getNodeMask();

	_roomLocation = new SubMenu("Room Options", "Location");
	_roomLocation->setCallback(this);

	_hideRoom = new MenuButton("Hide Room");
	_hideRoom->setCallback(this);

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

	
	centerLineTool = new CenterlineTool("CenterLine Tool", false, true, false, false, false);
	PluginHelper::registerSceneObject(centerLineTool, "HelmsleyVolume");
	



	fileSelector = new FileSelector();


	//osg::setNotifyLevel(osg::NOTICE);
	std::cerr << "HelmsleyVolume init" << std::endl;


	SubMenu* fileMenu = new SubMenu("Files", "Files");
	fileMenu->setCallback(this);
	_vMenu->addItem(fileMenu);


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
	int max = 500;
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
	if (menuItem == _hideRoom)
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

		removeVolume(index, false);
	}
	else if (std::find(_removeClippingPlaneButtons.begin(), _removeClippingPlaneButtons.end(), (MenuButton*)menuItem) != _removeClippingPlaneButtons.end())
	{
		std::vector<MenuButton*>::iterator it = std::find(_removeClippingPlaneButtons.begin(), _removeClippingPlaneButtons.end(), (MenuButton*)menuItem);
		int index = std::distance(_removeClippingPlaneButtons.begin(), it);

		removeCuttingPlane();
	}
	else if (menuItem == _cpButton)
	{
		if (_volumes.size())
		{
			createCuttingPlane();
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

void HelmsleyVolume::toggleCenterlineTool(bool on)
{
	if (on)
	{
		centerLineTool->attachToScene();
		centerLineTool->activate();
	}
	else
	{
		centerLineTool->detachFromScene();
		centerLineTool->deactivate();
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

void HelmsleyVolume::activateClippingPath() {
	CuttingPlane* cp;
	if (_cuttingPlanes.empty()) {
		 cp = createCuttingPlane();
	}
	else {
		cp = _cuttingPlanes[0];
	}

	
	osg::Vec3dArray* coords = _volumes[0]->getColonCoords();
	osg::Vec3dArray* coords2 = _volumes[0]->getIlleumCoords();
	

	osg::MatrixTransform* transform = new osg::MatrixTransform(_volumes[0]->_transform->getMatrix());
	
	osg::MatrixTransform* camTransform = new osg::MatrixTransform(_volumes[0]->getObjectToWorldMatrix() * _sceneObjects[0]->getObjectToWorldMatrix());
	
	centerLineTool->setCoords(coords, camTransform);
	centerLineTool->setCP(_cuttingPlanes[0]);
	cp->setCoords(coords, transform);
	centerLineTool->setCoords(coords2);
	cp->setCoords(coords2);

	
}


CuttingPlane* HelmsleyVolume::createCuttingPlane()
{
	if (_volumeIndex >= _volumes.size())
	{
		return nullptr;
	}

	
	CuttingPlane* cp = new  CuttingPlane("Cutting Plane", false, true, false, true, true);
	cp->setVolume(_volumes[_volumeIndex]);
	cp->setSceneObject(_sceneObjects[_volumeIndex]);
	MenuButton* remove = new MenuButton("Remove Cutting Plane");
	cp->addMenuItem(remove);
	remove->setCallback(this);
	_removeClippingPlaneButtons.push_back(remove);
	PluginHelper::registerSceneObject(cp, "HelmsleyVolume");
	_sceneObjects[_volumeIndex]->addChild(cp);
	cp->changePlane();
	_cuttingPlanes.push_back(cp);
	if (_volumeIndex == 0 && _cuttingPlanes.size() == 2) {
		CuttingPlane* temp = _cuttingPlanes[0];
		_cuttingPlanes[0] = _cuttingPlanes[1];
		_cuttingPlanes[1] = temp;
		
	}
	if (_cuttingPlanes.size() == 2) {
		_worldMenus[0]->toggleLinkOpacity(true);
	}


	return cp;
}

void HelmsleyVolume::removeCuttingPlane()
{

		_volumes[_volumeIndex]->_PlaneNormal->set(osg::Vec3(0.f, 1.f, 0.f));
		_volumes[_volumeIndex]->_PlanePoint->set(osg::Vec3(0.f, -2.f, 0.f));

		if (_cuttingPlanes.size() > 1) {
			_cuttingPlanes[_volumeIndex]->detachFromScene();
			delete(_cuttingPlanes[_volumeIndex]);
			_cuttingPlanes.erase(_cuttingPlanes.begin() + _volumeIndex);

			_removeClippingPlaneButtons.erase(_removeClippingPlaneButtons.begin() + _volumeIndex);
		}
		else {
			_cuttingPlanes[0]->detachFromScene();
			delete(_cuttingPlanes[0]);
			_cuttingPlanes.erase(_cuttingPlanes.begin() + 0);

			_removeClippingPlaneButtons.erase(_removeClippingPlaneButtons.begin() + 0);
		}
		_worldMenus[0]->setLinkOff();
		_worldMenus[0]->toggleLinkOpacity(false);
		
}

void HelmsleyVolume::toggleCenterLine(bool on) {
	//std::vector<osg::Geode*>* centerlines = _volumes[0]->getCenterLines();
	//if (on) {

	//	for (int i = 0; i < centerlines->size(); i++) {
	//		centerlines->at(i)->setNodeMask(0xffffffff);
	//	}
	//}
	//else {

	//	for (int i = 0; i < centerlines->size(); i++) {
	//		centerlines->at(i)->setNodeMask(0);
	//	}
	//}
}

void HelmsleyVolume::loadVolume(std::string path, std::string maskpath, bool onlyVolume)
{
	SceneObject * so;
	so = new SceneObject("volume", false, true, true, true, false);
	so->setPosition(ConfigManager::getVec3("Plugin.HelmsleyVolume.Orientation.Volume.Position", osg::Vec3(0, 750, 500)));

	VolumeGroup * g = new VolumeGroup();
	g->loadVolume(path, maskpath);
	so->addChild(g);
	
	MeasurementTool* tool = new MeasurementTool("Measurement Tool", false, false, false, false, false);
	tool->deactivate();
	so->addChild(tool);
	_measurementTools.push_back(tool);


	PluginHelper::registerSceneObject(so, "HelmsleyVolume");
	so->attachToScene();
	so->setNavigationOn(false);
	so->addMoveMenuItem();
	so->addNavigationMenuItem();
	so->setShowBounds(true);
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

	
	NewVolumeMenu* newMenu = new NewVolumeMenu(so, g);
	newMenu->init();
	_worldMenus.push_back(newMenu);
	
	
	

	_sceneObjects.push_back(so);
	_volumes.push_back(g);
	_contextMenus.push_back(menu);
	


	/*
	MenuButton* removeButton = new MenuButton("Remove Volume");
	so->addMenuItem(removeButton);
	removeButton->setCallback(this);
	_removeButtons.push_back(removeButton);
	*/
}

void HelmsleyVolume::loadSecondVolume(std::string path, std::string maskpath)
{

	SceneObject * so;
	VolumeGroup* g = new VolumeGroup();
	if (_sceneObjects.size() < 2) {
		so = new SceneObject("volume", false, true, true, true, false);
		_sceneObjects.push_back(so);
	}
	else {
		so = _sceneObjects[1];
		
	}
	so->setPosition(ConfigManager::getVec3("Plugin.HelmsleyVolume.Orientation.Volume.Position", osg::Vec3(300, 750, 500)));

	g->loadVolume(path, maskpath);
	so->addChild(g);
	PluginHelper::registerSceneObject(so, "HelmsleyVolume");
	so->attachToScene();
	so->setNavigationOn(false);
	so->addMoveMenuItem();
	so->addNavigationMenuItem();
	so->setShowBounds(true);
	VolumeMenu* menu = new VolumeMenu(so, g);
	menu->init();
	

	
	_volumes.push_back(g);
	_contextMenus.push_back(menu);

	_worldMenus[0]->setSecondVolume(g);
	_worldMenus[0]->toggleSwapOpacity();
	
	
	
}


void HelmsleyVolume::loadVolumeOnly(bool isPreset, std::string path, std::string maskpath) {
	VolumeGroup* g = new VolumeGroup();

	g->loadVolume(path, maskpath);
	_sceneObjects[0]->addChild(g);	//set new g on so
	_sceneObjects[0]->attachToScene();

	//VolumeMenu* menu = new VolumeMenu(so, g);	//set new g on menu
	_contextMenus[0]->setVolume(g);

	//NewVolumeMenu* newMenu = new NewVolumeMenu(so, g);// set new g on newmenu
	_worldMenus[0]->clearVolumes();
	HelmsleyVolume::instance()->setVolumeIndex(0);
	_worldMenus[0]->setNewVolume(g);
	_worldMenus[0]->toggleSwapOpacity();
	/*if(!isPreset)
		_worldMenus[0]->resetValues();*/

	_volumes.push_back(g);
}

void HelmsleyVolume::removeVolume(int index, bool onlyVolume)
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

void HelmsleyVolume::removeVolumeOnly(int index) {

	std::vector<CuttingPlane*>::iterator it = _cuttingPlanes.begin();
	while (it != _cuttingPlanes.end()) {
			(*it)->detachFromScene();
			delete((*it));
			it = _cuttingPlanes.erase(it);
	}

	for (int i = 0; i < _sceneObjects.size(); i++) {
		_sceneObjects[i]->detachFromScene();
		_sceneObjects[i]->removeChild(_volumes[0]);
		_volumes[0].release();
		_volumes.erase(_volumes.begin());
	}
	_worldMenus[0]->setLinkOff();
	_worldMenus[0]->toggleLinkOpacity(false);
	/*_sceneObjects[index]->detachFromScene();
	_sceneObjects[index]->removeChild(_volumes[0]);
	_volumes[0].release();
	_volumes.erase(_volumes.begin());*/
	//delete _volumes[index];
}

void HelmsleyVolume::removeSecondVolume() {
	if (_volumes.size() > 1) {
		std::vector<CuttingPlane*>::iterator it = _cuttingPlanes.begin();
		while (it != _cuttingPlanes.end()) {
			if ((*it)->getVolume() == _volumes[1])
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
		_sceneObjects[1]->detachFromScene();
		_sceneObjects[1]->removeChild(_volumes[1]);
		_volumes[1].release();
		_volumes.erase(_volumes.begin() + 1);
	}
}