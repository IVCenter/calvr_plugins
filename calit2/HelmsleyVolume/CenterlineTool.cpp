#include "CenterlineTool.h"
#include "HelmsleyVolume.h"
#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/NodeMask.h>

#include <sstream>
#include <iomanip>
#include <time.h>

void CenterlineTool::init()
{
	_cameraActive = false;
	


	std::string modelDir = cvr::ConfigManager::getEntry("Plugin.HelmsleyVolume.ModelDir");
	_tablet = osgDB::readNodeFile(modelDir + "CameraTool.obj");
	SceneObject* lightQuestionMark = new SceneObject("room", false, false, false, false, false);
	lightQuestionMark->addChild(_tablet);
	
	this->addChild(lightQuestionMark);


	osg::FrameBufferObject* fbo = new osg::FrameBufferObject();
	_image = new osg::Image();
	_image->allocateImage(1920, 1080, 1, GL_RGB, GL_UNSIGNED_BYTE);
	_image->setInternalTextureFormat(GL_RGB8);
	_texture = new osg::Texture2D();
	_texture->setResizeNonPowerOfTwoHint(false);
	_texture->setImage(_image);
	_texture->setTextureSize(1920, 1080);
	_texture->setInternalFormat(GL_RGBA);
	_texture->setNumMipmapLevels(0);
	_texture->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::LINEAR);
	_texture->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::LINEAR);
	fbo->setAttachment(osg::Camera::COLOR_BUFFER0, osg::FrameBufferAttachment(_texture));

	osg::ref_ptr<osg::Texture2D> depthbuffer = new osg::Texture2D();
	depthbuffer->setTextureSize(1920, 1080);
	depthbuffer->setResizeNonPowerOfTwoHint(false);
	depthbuffer->setInternalFormat(GL_DEPTH_COMPONENT);
	depthbuffer->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::LINEAR);
	depthbuffer->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::LINEAR);
	fbo->setAttachment(osg::Camera::DEPTH_BUFFER, osg::FrameBufferAttachment(depthbuffer));

	_camera = new osg::Camera();
	_camera->setAllowEventFocus(false);
	_camera->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	_camera->setClearColor(osg::Vec4(0.0, 0, 0.0, 1.0));
	_camera->setRenderOrder(osg::Camera::PRE_RENDER);
	_camera->setRenderTargetImplementation(osg::Camera::FRAME_BUFFER_OBJECT);
	_camera->attach(osg::Camera::COLOR_BUFFER0, _texture);
	_camera->attach(osg::Camera::DEPTH_BUFFER, depthbuffer);
	_camera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
	_camera->setCullMask(cvr::CULL_MASK);
	_camera->setCullMaskLeft(cvr::CULL_MASK_LEFT);
	_camera->setCullMaskRight(cvr::CULL_MASK_RIGHT);
	_camera->setViewport(0, 0, 1920, 1080);

	cvr::ScreenBase::addBuffer(_camera, fbo);


	//_display = new osg::ShapeDrawable(new osg::Box(osg::Vec3(0, 0, 2.5), 270-1, 480-1, 1));
	_display = new osg::Geometry();
	//osg::Vec3Array* coords = new osg::Vec3Array(4);
	//(*coords)[0] = osg::Vec3(-470.0 / 2, -4, -260.0 / 2);
	//(*coords)[1] = osg::Vec3(470.0 / 2, -4, -260.0 / 2);
	//(*coords)[2] = osg::Vec3(470.0 / 2, -4, 260.0 / 2);
	//(*coords)[3] = osg::Vec3(-470.0 / 2, -4, 260.0 / 2);

	osg::Vec3Array* coords = new osg::Vec3Array(4);
	(*coords)[0] = osg::Vec3(0.0, -4, -360.0);
	(*coords)[1] = osg::Vec3(640.0, -4, -360.0);
	(*coords)[2] = osg::Vec3(640.0, -4, 0.0);
	(*coords)[3] = osg::Vec3(0.0, -4, 0.0);


	_display->setVertexArray(coords);

	osg::Vec3Array* norms = new osg::Vec3Array(1);
	(*norms)[0] = osg::Vec3(0, 0, 1);
	_display->setNormalArray(norms, osg::Array::BIND_OVERALL);

	osg::Vec2Array* tcoords = new osg::Vec2Array(4);
	(*tcoords)[0].set(0.0f, 0.0f);
	(*tcoords)[1].set(1.0f, 0.0f);
	(*tcoords)[2].set(1.0f, 1.0f);
	(*tcoords)[3].set(0.0f, 1.0f);
	_display->setTexCoordArray(0, tcoords);

	_display->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 4));


	//_display->tex
	_display->getOrCreateStateSet()->setTextureAttributeAndModes(0, _texture, osg::StateAttribute::ON);
	_display->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

	osg::Geode* displaygeode = new osg::Geode();
	displaygeode->addChild(_display);
	this->addChild(displaygeode);


	_camera->addChild((osg::Node*)cvr::SceneManager::instance()->getScene());
	//setParams(60, 16.0 / 9.0);
	setParams(90, 16.0 / 9.0);
	activate();

	
	using namespace cvr;
	


	_cameraPop = new cvr::UIPopup();
	this->addChild(_cameraPop->getRoot());
	_cameraPop->setPosition(osg::Vec3(0, 4, 0));
	cvr::UIQuadElement* bknd = new cvr::UIQuadElement(UI_BACKGROUND_COLOR);
	bknd->setPercentSize(osg::Vec3(1, 1, 0.5));
	lightQuestionMark->setScale(1);
	osg::Vec3 pos = lightQuestionMark->getPosition();
	lightQuestionMark->setPosition(osg::Vec3(500.0, 10, -280.0));
	lightQuestionMark->setScale(2.2);
	

	_cameraPop->addChild(bknd);
	

	_colonButton = new CallbackButton();
	_colonButton->setCallback(this);
	_illeumButton = new CallbackButton();
	_illeumButton->setCallback(this);
	UIText* colText = new UIText("Colon", 40.0f, osgText::TextBase::CENTER_TOP);
	UIText* illText = new UIText("Illeum", 40.0f, osgText::TextBase::CENTER_TOP);
	_colonButton->addChild(colText);
	_illeumButton->addChild(illText);

	UIList* controlList = new UIList(UIList::LEFT_TO_RIGHT, UIList::CUT);
	UIList* organList = new UIList(UIList::TOP_TO_BOTTOM, UIList::CUT);
	_cameraPop->addChild(controlList);
	_cameraPop->addChild(organList);
	controlList->setPercentSize(osg::Vec3(1.0, 1.0, .1));
	controlList->setPercentPos(osg::Vec3(0.0, 0.0, -.4));
	organList->setPercentSize(osg::Vec3(.3, 1.0, .2));
	organList->setPercentPos(osg::Vec3(0.7, 0.0, 0.0));
	_imgDir = ConfigManager::getEntry("Plugin.HelmsleyVolume.ImageDir");
	_playButton = new ToolToggle(_imgDir + "play.png");
	_playButton->setCallback(this);

	_playButton->setPercentSize(osg::Vec3(0.2, 1.0, 1.0));
	controlList->addChild(_playButton);
	organList->addChild(_colonButton);
	organList->addChild(_illeumButton);
}

void CenterlineTool::uiCallback(UICallbackCaller* ui) {
	if (ui == _playButton) {
		if (_playButton->isOn()) {
			_playButton->setIcon(_imgDir + "pause.png");
			_updateCallback->play();
			_cp->play();
		}
		else {
			_playButton->setIcon(_imgDir + "play.png");
			_updateCallback->pause();
			_cp->pause();
		}
		_playButton->setDirty(true);
	}
	
	else if (ui == _colonButton) {
		_updateCallback->startFromCol();
		_cp->getUC()->startFromCol();
	}
	else if (ui == _illeumButton) {
		_updateCallback->startFromIll();
		_cp->getUC()->startFromIll();
	}
	
}

void CenterlineTool::setParams(double fov, double aspect)
{
	_camera->setProjectionMatrixAsPerspective(fov, aspect, cvr::ScreenBase::getNear(), cvr::ScreenBase::getFar());
}

void CenterlineTool::activate()
{
	if (!_cameraActive)
	{
		_cameraActive = true;
		dynamic_cast<osg::Group*>(cvr::CVRViewer::instance()->getSceneData())->addChild(_camera);
	}
}

void CenterlineTool::deactivate()
{
	if (_cameraActive)
	{
		_cameraActive = false;
		dynamic_cast<osg::Group*>(cvr::CVRViewer::instance()->getSceneData())->removeChild(_camera);
	}
}



