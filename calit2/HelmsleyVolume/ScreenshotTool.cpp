#include "ScreenshotTool.h"
#include "HelmsleyVolume.h"
#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/NodeMask.h>

#include <sstream>
#include <iomanip>
#include <time.h>
#include <filesystem>


#ifdef WIN32
#include "dirent.h"
#else
#include <dirent.h>
#endif

ScreenshotTool::~ScreenshotTool() {
	
}
void ScreenshotTool::init()
{
	_cameraActive = false;
	_saveThreads = std::vector<ScreenshotSaverThread*>();


	std::string modelDir = cvr::ConfigManager::getEntry("Plugin.HelmsleyVolume.ModelDir");
	/*_tablet = osgDB::readNodeFile(modelDir + "CameraTool.obj");
	this->addChild(_tablet);*/


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

	UIPopup* _cameraPop = new cvr::UIPopup();
	this->addChild(_cameraPop->getRoot());
	_cameraPop->setPosition(osg::Vec3(0, 4, 0));
	cvr::UIQuadElement* bknd = new cvr::UIQuadElement(UI_BACKGROUND_COLOR);
	bknd->setPercentSize(osg::Vec3(.65, 1, 0.5));


	this->setDisableBB(true);
	_cameraPop->addChild(bknd);

	UIList* buttonList = new UIList(UIList::LEFT_TO_RIGHT, UIList::CUT);
	_takePicture = new CallbackButton();
	UIQuadElement* bttnbknd = new UIQuadElement(UI_RED_ACTIVE_COLOR);
	bttnbknd->setTransparent(true);
	bttnbknd->setRounding(0, .2);
	bttnbknd->setBorderSize(.02);
	UIText* bttntxt = new UIText("Take Picture", 24.0f, osgText::TextBase::CENTER_CENTER);
	bttntxt->setPercentPos(osg::Vec3(0, -1, 0));
	_takePicture->setPercentSize(osg::Vec3(.5, 1, .2));
	_takePicture->addChild(bttnbknd);
	_takePicture->addChild(bttntxt);
	_takePicture->setCallback(this);




	buttonList->addChild(_takePicture);
	
	buttonList->setPercentPos(osg::Vec3(.125, 0, -.775));
	bknd->addChild(buttonList);

#ifdef CINE
	_renderCinematic = new CallbackButton();
	bttnbknd = new UIQuadElement(UI_INACTIVE_RED_COLOR);
	bttnbknd->setTransparent(true);
	bttnbknd->setRounding(0, .2);
	bttnbknd->setBorderSize(.02);
	bttntxt = new UIText("Take Cinematic", 24.0f, osgText::TextBase::CENTER_CENTER);
	bttntxt->setColor(UI_INACTIVE_WHITE_COLOR);
	bttntxt->setPercentPos(osg::Vec3(0, -1, 0));
	_renderCinematic->setPercentSize(osg::Vec3(.5, 1, .2));
	_renderCinematic->addChild(bttnbknd);
	_renderCinematic->addChild(bttntxt);
	_renderCinematic->setCallback(this);
	buttonList->addChild(_renderCinematic);
#endif


	
	_cameraMenu = new cvr::PopupMenu("Camera", "", false);
	_cameraMenu->setVisible(true);
	_cameraMenu->getRootObject()->getParent(0)->removeChild(_cameraMenu->getRootObject());
	_cameraMenu->setPosition(osg::Vec3(20, -5, 230));
	_cameraMenu->setMovable(false);
 	//this->addChild(_cameraMenu->getRootObject());
	_pictureButton = new cvr::MenuButton("Take Picture");
	_pictureButton->setCallback(this);
	_cameraMenu->addMenuItem(_pictureButton);


}

void ScreenshotTool::setParams(double fov, double aspect)
{
	_camera->setProjectionMatrixAsPerspective(fov, aspect, cvr::ScreenBase::getNear(), cvr::ScreenBase::getFar());
}

void ScreenshotTool::activate()
{
	if (!_cameraActive)
	{
		_cameraActive = true;
		dynamic_cast<osg::Group*>(cvr::CVRViewer::instance()->getSceneData())->addChild(_camera);
	}
}

void ScreenshotTool::deactivate()
{
	if (_cameraActive)
	{
		_cameraActive = false;
		dynamic_cast<osg::Group*>(cvr::CVRViewer::instance()->getSceneData())->removeChild(_camera);
	}
}

void ScreenshotTool::updateCallback(int handID, const osg::Matrix & mat)
{
	if (_cameraActive)
	{
		osg::Matrix localToWorld = getObjectToWorldMatrix();
		osg::Vec4 eye = osg::Vec4(0, 5, 0, 1) * localToWorld;
		osg::Vec4 center = osg::Vec4(0, 10, 0, 1) * localToWorld;
		osg::Vec4 up = osg::Vec4(0, 0, 1, 0) * localToWorld;
		//up.normalize();
		_camera->setViewMatrixAsLookAt(osg::Vec3(eye.x(), eye.y(), eye.z()),
			osg::Vec3(center.x(), center.y(), center.z()),
			osg::Vec3(up.x(), up.y(), up.z()));
	}
	while (_saveThreads.size())
	{
		ScreenshotSaverThread* s = _saveThreads[0];
		if (!s->isRunning())
		{
			_saveThreads.erase(_saveThreads.begin());
			delete s;
		}
	}
}

void ScreenshotTool::menuCallback(cvr::MenuItem* menuItem)
{
	if (menuItem == _pictureButton && !_pdc)
	{
		_camera->detach(osg::Camera::COLOR_BUFFER0);
		_camera->attach(osg::Camera::COLOR_BUFFER0, _image);
		_camera->dirtyAttachmentMap();

		//_camera->addPostDrawCallback()
		std::string path = cvr::ConfigManager::getEntry("Plugin.HelmsleyVolume.AppDir");
		path += "/Screenshots";
		DIR* dir = opendir(path.c_str());
		std::string filePath = "";
		if (dir == nullptr) {
			std::filesystem::create_directory(path);
		}

		//takePhoto(path + "/0.png");
		time_t t = std::time(nullptr);
		struct tm* tim = std::localtime(&t);
		char buf[32];

		strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%M-%S", tim);

		std::stringstream ss;
		ss << path << "\\" << buf << ".png";//std::put_time(tim, "%Y-%m-%d_%H-%M-%S") << ".png";
		_pdc = new ScreenshotCallback(this, ss.str());
		_camera->addPostDrawCallback(_pdc);

		std::cout << "picture stored in " << path << std::endl;
	}
	else
	{
		SceneObject::menuCallback(menuItem);
	}
}

void ScreenshotTool::uiCallback(UICallbackCaller* ui) {

	if (!_takingPhoto) {
		if (!_showingPhoto) {
			if (ui == _takePicture && !_pdc)
			{
				_camera->detach(osg::Camera::COLOR_BUFFER0);
				_camera->attach(osg::Camera::COLOR_BUFFER0, _image);
				_camera->dirtyAttachmentMap();

				//_camera->addPostDrawCallback()
				std::string path = cvr::ConfigManager::getEntry("Plugin.HelmsleyVolume.AppDir");
				path += "/Screenshots";
				DIR* dir = opendir(path.c_str());
				std::string filePath = "";
				if (dir == nullptr) {
					std::filesystem::create_directory(path);
				}
				//std::string path = cvr::ConfigManager::getEntry("Plugin.HelmsleyVolume.PictureLocation");
				//takePhoto(path + "/0.png");
				time_t t = std::time(nullptr);
				struct tm* tim = std::localtime(&t);
				char buf[32];

				strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%M-%S", tim);

				std::stringstream ss;
				ss << path << "\\" << buf << ".png";//std::put_time(tim, "%Y-%m-%d_%H-%M-%S") << ".png";
				std::cout << "picture stored in " << path << std::endl;
				_pdc = new ScreenshotCallback(this, ss.str());
				_camera->addPostDrawCallback(_pdc);
			}

#ifdef CINE
			else if (ui == _renderCinematic && false) { //disabled
				//create yaml
				((cvr::UIText*)_renderCinematic->getChild(1))->setText("Loading...");
				saveAsYaml();
				
			}
 		}
		else {
			if (ui == _takePicture && !_pictureSaved)//Save picture 
			{
				std::string path = cvr::ConfigManager::getEntry("Plugin.HelmsleyVolume.PictureLocation");
				time_t t = std::time(nullptr);
				struct tm* tim = std::localtime(&t);
				char buf[32];
				strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%M-%S", tim);
				std::stringstream ss;
				ss << path << "\\" << buf << ".png";
				std::rename(_imgCineFP.c_str(), ss.str().c_str());
				((cvr::UIText*)_takePicture->getChild(1))->setText("Picture Saved");
				((cvr::UIQuadElement*)_takePicture->getChild(0))->setColor(UI_INACTIVE_RED_COLOR);
				_pictureSaved = true;

			}
 			else if (ui == _renderCinematic && false)////DISABLED
			{
				_display->getOrCreateStateSet()->setTextureAttributeAndModes(0, _texture, osg::StateAttribute::ON);
				((cvr::UIText*)_takePicture->getChild(1))->setText("Take Picture"); 
				((cvr::UIQuadElement*)_takePicture->getChild(0))->setColor(UI_RED_ACTIVE_COLOR);
				((cvr::UIText*)_renderCinematic->getChild(1))->setText("Take Cinematic"); 
				_showingPhoto = false;
				_pictureSaved = false;
			}
#endif
		}
	}

}

#ifdef CINE
void ScreenshotTool::saveAsYaml() {
	_vm->saveYamlForCinematic();

}
#endif CINE

void ScreenshotTool::takePhoto(std::string filename)
{
	//osg::ref_ptr<osgDB::Options> options = new osgDB::Options();
	//options->
	osg::ref_ptr<osg::Image> imageToSave = _image;
	_camera->detach(osg::Camera::COLOR_BUFFER0);
	_camera->attach(osg::Camera::COLOR_BUFFER0, _texture);
	_camera->dirtyAttachmentMap();
	_camera->removePostDrawCallback(_pdc);
	_pdc.release();

	_image = new osg::Image();
	_image->allocateImage(1920, 1080, 1, GL_RGB, GL_UNSIGNED_BYTE);
	_image->setInternalTextureFormat(GL_RGB8);
	_texture->setImage(_image);


	ScreenshotSaverThread* s = new ScreenshotSaverThread(imageToSave, filename);
	_saveThreads.push_back(s);
	s->start();
	

}

#ifdef CINE

void ScreenshotTool::setPhoto(std::string imgLocation) {


	std::string seriesName = imgLocation.substr(imgLocation.find_last_of("/"));
	std::string patientDir = imgLocation.substr(0, imgLocation.find_last_of("/"));
	_imgCineFP = patientDir + seriesName + "cr0.png";


	osg::ref_ptr<osg::Texture2D> text = cvr::UIUtil::loadImage(_imgCineFP);
	_display->getOrCreateStateSet()->setTextureAttributeAndModes(0, text, osg::StateAttribute::ON);


	((cvr::UIText*)_takePicture->getChild(1))->setText("Save Image");//text
	((cvr::UIText*)_renderCinematic->getChild(1))->setText("OK");//text
	//std::remove(_imgCineFP.c_str());

	_showingPhoto = true;

}


#endif  

void ScreenshotSaverThread::run()
{
	if (!osgDB::writeImageFile(*_image, _filename))
	{
		std::cout << "Failed to write image file" << std::endl;
	}
}