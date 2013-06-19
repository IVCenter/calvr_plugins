#include "PanoViewObject.h"
#include "PanoViewLOD.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/NodeMask.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrUtil/OsgMath.h>
#include <PluginMessageType.h>

#include <osg/Depth>

#include <iostream>
#include <cstdio>
#include <fstream>
#include <cmath>

#ifdef WIN32
#define M_PI 3.141592653589793238462643
#endif

//#define PRINT_TIMING

using namespace cvr;

PanoViewObject::PanoViewObject(std::string name, std::string leftEyeFile, std::string rightEyeFile, float radius, int mesh, int depth, int size, float height, std::string vertFile, std::string fragFile) : SceneObject(name,false,false,false,true,false)
{
    _name = name;
    std::vector<std::string> left;
    std::vector<std::string> right;
    left.push_back(leftEyeFile);
    right.push_back(rightEyeFile);

    init(left,right,radius,mesh,depth,size,height,vertFile,fragFile);
}

PanoViewObject::PanoViewObject(std::string name, std::vector<std::string> & leftEyeFiles, std::vector<std::string> & rightEyeFiles, float radius, int mesh, int depth, int size, float height, std::string vertFile, std::string fragFile) : SceneObject(name,false,false,false,true,false)
{
    _name = name;
    init(leftEyeFiles,rightEyeFiles,radius,mesh,depth,size,height,vertFile,fragFile);
}

PanoViewObject::~PanoViewObject()
{
    if(_leftDrawable)
    {
        _leftDrawable->cleanup();
    }
}

void PanoViewObject::init(std::vector<std::string> & leftEyeFiles, std::vector<std::string> & rightEyeFiles, float radius, int mesh, int depth, int size, float height, std::string vertFile, std::string fragFile)
{
#ifndef WIN32
    _imageSearchPath = ConfigManager::getEntryConcat("value","Plugin.PanoViewLOD.ImageSearchPath",':',"");
#else
	_imageSearchPath = ConfigManager::getEntryConcat("value","Plugin.PanoViewLOD.ImageSearchPath",';',"");
#endif
    _floorOffset = ConfigManager::getFloat("value","Plugin.PanoViewLOD.FloorOffset",1500);

    std::string temp("PANOPATH=");
    temp = temp + _imageSearchPath;

    char * carray = new char[temp.size()+1];

    strcpy(carray,temp.c_str());

    putenv(carray);

    _leftGeode = new osg::Geode();
    _rightGeode = new osg::Geode();

    _leftGeode->setNodeMask(_leftGeode->getNodeMask() & (~CULL_MASK_RIGHT));
    _rightGeode->setNodeMask(_rightGeode->getNodeMask() & (~CULL_MASK_LEFT));
    _rightGeode->setNodeMask(_rightGeode->getNodeMask() & (~CULL_MASK));

    _pdi = new PanoDrawableInfo;
    _pdi->leftEyeFiles = leftEyeFiles;
    _pdi->rightEyeFiles = rightEyeFiles;

    _leftDrawable = new PanoDrawableLOD(_pdi,radius,mesh,depth,size,vertFile,fragFile);
    _rightDrawable = new PanoDrawableLOD(_pdi,radius,mesh,depth,size,vertFile,fragFile);

    _leftGeode->addDrawable(_leftDrawable);
    _rightGeode->addDrawable(_rightDrawable);

    addChild(_leftGeode);
    addChild(_rightGeode);

    _currentZoom = 0.0;

    _offset = ConfigManager::getVec3("Plugin.PanoViewLOD.Offset");

    _demoTime = 0.0;
    _demoChangeTime = ConfigManager::getDouble("value","Plugin.PanoViewLOD.DemoChangeTime",90.0);

    _removeOnClick = false;

    _coordChangeMat.makeRotate(M_PI/2.0,osg::Vec3(1,0,0));
    _spinMat.makeIdentity();
    float offset = height - _floorOffset + DEFAULT_PAN_HEIGHT;
    osg::Vec3 ovec(0,0,offset);
    _heightMat.makeTranslate(ovec + _offset);
    setTransform(_coordChangeMat * _spinMat * _tiltMat * _heightMat);

    _nextButton = _previousButton = NULL;

    if(leftEyeFiles.size() > 1)
    {
	_nextButton = new MenuButton("Next");
	_nextButton->setCallback(this);
	addMenuItem(_nextButton);
	_previousButton = new MenuButton("Previous");
	_previousButton->setCallback(this);
	addMenuItem(_previousButton);
    }

    _demoMode = new MenuCheckbox("Demo Mode", ConfigManager::getBool("value","Plugin.PanoViewLOD.DemoMode",false, NULL));
    _demoMode->setCallback(this);
    addMenuItem(_demoMode);

    _trackball = new MenuCheckbox("Trackball Mode", false);
    _trackball->setCallback(this);
    addMenuItem(_trackball);

    _radiusRV = new MenuRangeValue("Radius", 100, 100000, radius);
    _radiusRV->setCallback(this);
    addMenuItem(_radiusRV);

    _heightRV = new MenuRangeValue("Height", -2000, 2000, height);
    _heightRV->setCallback(this);
    addMenuItem(_heightRV);

    _alphaRV = new MenuRangeValue("Alpha",0.0,1.0,1.0);
    _alphaRV->setCallback(this);
    addMenuItem(_alphaRV);

    _leftDrawable->setAlpha(_alphaRV->getValue());
    _rightDrawable->setAlpha(_alphaRV->getValue());

    _zoomValuator = ConfigManager::getInt("value","Plugin.PanoViewLOD.ZoomValuator",0);
    _spinValuator = ConfigManager::getInt("value","Plugin.PanoViewLOD.SpinValuator",0);
    _spinScale = ConfigManager::getFloat("value","Plugin.PanoViewLOD.SpinScale",1.0);
    _zoomScale = ConfigManager::getFloat("value","Plugin.PanoViewLOD.ZoomScale",1.0);
    if(_zoomValuator == _spinValuator)
    {
	_sharedValuator = true;
    }
    else
    {
	_sharedValuator = false;
    }

    if(!_sharedValuator)
    {
	_spinCB = NULL;
	_zoomCB = NULL;
    }
    else
    {
	_spinCB = new MenuCheckbox("Spin Mode",true);
	_spinCB->setCallback(this);
	addMenuItem(_spinCB);

	_zoomCB = new MenuCheckbox("Zoom Mode",false);
	_zoomCB->setCallback(this);
	addMenuItem(_zoomCB);
    }

    _zoomResetButton = new MenuButton("Reset Zoom");
    _zoomResetButton->setCallback(this);
    addMenuItem(_zoomResetButton);

    _fadeActive = false;
    _fadeFrames = 0;
    _transitionType = NORMAL;
    _transitionStarted = false;

    _upArrow = _downArrow = _leftArrow = _rightArrow = false;

    _printValues = ConfigManager::getBool("value","Plugin.PanoViewLOD.PrintValues",false,NULL);

    osg::Depth * wdepth = new osg::Depth();
    wdepth->setWriteMask(false);
    _root->getOrCreateStateSet()->setAttributeAndModes(wdepth,osg::StateAttribute::ON);

    _transitionSkipFrames = 0;


    // load saved initial scales and locations
    _configPath = ConfigManager::getEntry("Plugin.PanoViewLOD.ConfigDir");

    _saveButton = new MenuButton("Save");
    _saveButton->setCallback(this);
    addMenuItem(_saveButton);

    std::ifstream cfile;
    cfile.open((_configPath + "/Init.cfg").c_str(), std::ios::in);

    if(!cfile.fail())
    {
      std::string line;
      while(!cfile.eof())
      {
         float rotate, zoom;
         char name[150];
         cfile >> name;
         if(cfile.eof())
         {
           break;
         }
         cfile >> rotate;
         cfile >> zoom;
         _locInit[std::string(name)] = std::pair<float, float>(rotate, zoom);
      }
    }
    cfile.close();

    if(_locInit.find(_name) != _locInit.end())
    {
        setRotate(_locInit[_name].first);

        float zoom = _locInit[_name].second;
	
        _currentZoom += zoom;
        if(_currentZoom < -2.0) _currentZoom = -2.0;
        if(_currentZoom > 0.5) _currentZoom = 0.5;

		updateZoom(PluginHelper::getHandMat(0));
    }

}

void PanoViewObject::setTransition(PanTransitionType transitionType, std::string transitionFilesDir, std::vector<std::string> & leftTransitionFiles, std::vector<std::string> & rightTransitionFiles, std::string configTag)
{
    _transitionType = transitionType;
    _leftTransitionFiles = leftTransitionFiles;
    _rightTransitionFiles = rightTransitionFiles;
    _transitionFilesDir = transitionFilesDir;

    if(transitionType == ZOOM)
    {
	_rotateStartDelay = ConfigManager::getFloat("rotateStart","Plugin.PanoViewLOD.ZoomTransition",0.0);
	_rotateStartDelay = ConfigManager::getFloat("rotateStart",configTag,_rotateStartDelay);
	_rotateInterval = ConfigManager::getFloat("rotateInterval","Plugin.PanoViewLOD.ZoomTransition",4.0);
	_rotateInterval = ConfigManager::getFloat("rotateInterval",configTag,_rotateInterval);
	_zoomStartDelay = ConfigManager::getFloat("zoomStart","Plugin.PanoViewLOD.ZoomTransition",4.0);
	_zoomStartDelay = ConfigManager::getFloat("zoomStart",configTag,_zoomStartDelay);
	_zoomInterval = ConfigManager::getFloat("zoomInterval","Plugin.PanoViewLOD.ZoomTransition",4.0);
	_zoomInterval = ConfigManager::getFloat("zoomInterval",configTag,_zoomInterval);
	_fadeStartDelay = ConfigManager::getFloat("fadeStart","Plugin.PanoViewLOD.ZoomTransition",8.0);
	_fadeStartDelay = ConfigManager::getFloat("fadeStart",configTag,_fadeStartDelay);
	_fadeInterval = ConfigManager::getFloat("fadeInterval","Plugin.PanoViewLOD.ZoomTransition",4.0);
	_fadeInterval = ConfigManager::getFloat("fadeInterval",configTag,_fadeInterval);
	_rotateRampUp = ConfigManager::getFloat("rotateRampUp","Plugin.PanoViewLOD.ZoomTransition",0.1*_rotateInterval);
	_rotateRampUp = ConfigManager::getFloat("rotateRampUp",configTag,_rotateRampUp);
	_rotateRampDown = ConfigManager::getFloat("rotateRampDown","Plugin.PanoViewLOD.ZoomTransition",0.1*_rotateInterval);
	_rotateRampDown = ConfigManager::getFloat("rotateRampDown",configTag,_rotateRampDown);
	_zoomRampUp = ConfigManager::getFloat("zoomRampUp","Plugin.PanoViewLOD.ZoomTransition",0.1*_zoomInterval);
	_zoomRampUp = ConfigManager::getFloat("zoomRampUp",configTag,_zoomRampUp);
	_zoomRampDown = ConfigManager::getFloat("zoomRampDown","Plugin.PanoViewLOD.ZoomTransition",0.1*_zoomInterval);
	_zoomRampDown = ConfigManager::getFloat("zoomRampDown",configTag,_zoomRampDown);

	if(_rotateRampUp + _rotateRampDown > _rotateInterval)
	{
	    _rotateRampUp = _rotateRampDown = 0.5 * _rotateInterval;
	}

	if(_zoomRampUp + _zoomRampDown > _zoomInterval)
	{
	    _zoomRampUp = _zoomRampDown = 0.5 * _zoomInterval;
	}

	_transitionStarted = _rotateDone = _zoomDone = _fadeDone = false;

	if(leftTransitionFiles.size())
	{
	    std::ifstream infile;
	    infile.open(leftTransitionFiles[0].c_str());
	    if(infile.fail())
	    {
		infile.open((transitionFilesDir + "/" + leftTransitionFiles[0]).c_str());
	    }

	    if(infile.fail())
	    {
		std::cerr << "Error: unable to open zoom transition file: " << leftTransitionFiles[0] << std::endl;
		_transitionType = NORMAL;
		return;
	    }

	    std::string line;
	    while(infile.good() && _zoomTransitionInfo.size() < _leftTransitionFiles.size())
	    {
		std::getline(infile,line);
		ZoomTransitionInfo zti;
		zti.rotationFromImage = zti.rotationToImage = zti.zoomValue = 0.0;
		zti.zoomDir = osg::Vec3(0,1,0);

		float x,y,z;
		int readValues = sscanf(line.c_str(),"%f %f %f %f %f %f", &zti.rotationFromImage, &zti.rotationToImage, &zti.zoomValue, &x, &y, &z);
		if(readValues < 3)
		{
		    std::cerr << "Warning: unable to parse transition values for line: " << line << std::endl;
		}
		else if(readValues > 3 && readValues < 6)
		{
		    std::cerr << "Warning: unable to parse transition zoom direction for line: " << line << std::endl;
		}
		else if(readValues == 6)
		{
		    zti.zoomDir.x() = x; 
		    zti.zoomDir.y() = y;
		    zti.zoomDir.z() = z;
		}

		zti.rotationFromImage *= M_PI / 180.0;
		zti.rotationToImage *= M_PI / 180.0;

		_zoomTransitionInfo.push_back(zti);
	    }

	    if(_zoomTransitionInfo.size() != _leftTransitionFiles.size())
	    {
		std::cerr << "Warning: Zoom transition entries count does not match set size.  Transitions: " << _zoomTransitionInfo.size() << " Set size: " << _leftTransitionFiles.size() << std::endl;

		while(_zoomTransitionInfo.size() < _leftTransitionFiles.size())
		{
		    ZoomTransitionInfo zti;
		    zti.rotationFromImage = zti.rotationToImage = zti.zoomValue = 0.0;
		    zti.zoomDir = osg::Vec3(0,1,0);

		    _zoomTransitionInfo.push_back(zti);
		}
	    }

#if 1
	    std::cerr << "Zoom TransitionInfo" << std::endl;
	    for(int i = 0; i < _zoomTransitionInfo.size(); i++)
	    {
		std::cerr << "Rotation From: " << _zoomTransitionInfo[i].rotationFromImage * 180.0 / M_PI << " To: " << _zoomTransitionInfo[i].rotationToImage * 180.0 / M_PI << " Zoom: " << _zoomTransitionInfo[i].zoomValue << " Zoom Direction x: " << _zoomTransitionInfo[i].zoomDir.x() << " y: " << _zoomTransitionInfo[i].zoomDir.y() << " z: " << _zoomTransitionInfo[i].zoomDir.z() << std::endl;
	    }
#endif

	}

	/*for(int i = 0; i < leftTransitionFiles.size(); i++)
	{
	    //TODO: read from files
	    ZoomTransitionInfo zti;
	    zti.rotationFromImage = 0.5 * ((float)i+1.0);
	    zti.rotationToImage = 0.2;
	    zti.zoomValue = -0.25;
	    _leftZoomTransitionInfo.push_back(zti);
	    _rightZoomTransitionInfo.push_back(zti);
	}*/
    }

    _leftDrawable->setTransitionType(transitionType);
    _rightDrawable->setTransitionType(transitionType);
}

void PanoViewObject::next()
{
    if(_transitionStarted)
    {
	return;
    }
    _fadeActive = true;
    _fadeFrames = 0;
    _leftDrawable->next();
    _rightDrawable->next();
    startTransition();
}

void PanoViewObject::previous()
{
    if(_transitionStarted)
    {
	return;
    }
    _fadeActive = true;
    _fadeFrames = 0;
    _leftDrawable->previous();
    _rightDrawable->previous();
    startTransition();
}

void PanoViewObject::setAlpha(float alpha)
{
    _alphaRV->setValue(alpha);
    _leftDrawable->setAlpha(_alphaRV->getValue());
    _rightDrawable->setAlpha(_alphaRV->getValue());
}

float PanoViewObject::getAlpha()
{
    return _alphaRV->getValue();
}

void PanoViewObject::setRotate(float rotate)
{
    _spinMat.makeRotate(rotate, osg::Vec3(0,0,1));
    setTransform(_coordChangeMat * _spinMat * _tiltMat * _heightMat);

    if(_currentZoom != 0.0)
    {
	updateZoom(_lastZoomMat);
    }
}

float PanoViewObject::getRotate()
{
    osg::Vec3d vec;
    double angle;
    _spinMat.getRotate().getRotate(angle,vec);
    if(vec.z() < 0 )
    {
        angle = (2.0 * M_PI) - angle;
    }
    return angle;
}

void PanoViewObject::menuCallback(cvr::MenuItem * item)
{
    if(item == _nextButton)
    {
	next();
    }

    if(item == _previousButton)
    {
	previous();
    }

    if(item == _radiusRV)
    {
	_leftDrawable->setRadius(_radiusRV->getValue());
	_rightDrawable->setRadius(_radiusRV->getValue());
    }

    if(item == _heightRV)
    {
	float offset = _heightRV->getValue() - _floorOffset + DEFAULT_PAN_HEIGHT;
	osg::Vec3 ovec(0,0,offset);
	_heightMat.makeTranslate(ovec + _offset);
	setTransform(_coordChangeMat * _spinMat * _tiltMat * _heightMat);
    }

    if(item == _alphaRV)
    {
	_leftDrawable->setAlpha(_alphaRV->getValue());
	_rightDrawable->setAlpha(_alphaRV->getValue());
    }

    if(item == _zoomResetButton)
    {
	_currentZoom = 0.0;

	_leftDrawable->setZoom(osg::Vec3(0,1,0),pow(10.0, _currentZoom));
	_rightDrawable->setZoom(osg::Vec3(0,1,0),pow(10.0, _currentZoom));
    }

    if(item == _saveButton)
    {
        float zoom = 1;

        _locInit[_name] = std::make_pair(getRotate(), _currentZoom);
        writeConfig();
    }

    if(item == _spinCB)
    {
	if(_spinCB->getValue())
	{
	    _zoomCB->setValue(false);
	}
    }

    if(item == _zoomCB)
    {
	if(_zoomCB->getValue())
	{
	    _spinCB->setValue(false);
	}
    }
    SceneObject::menuCallback(item);
}

void PanoViewObject::updateCallback(int handID, const osg::Matrix & mat)
{

    //std::cerr << "Update Callback." << std::endl;
#ifdef PRINT_TIMING

    //std::cerr << "Fade Time: " << _leftDrawable->getCurrentFadeTime() << std::endl;
    if(_fadeActive)
    {
	if(_leftDrawable->getCurrentFadeTime() > 0.0)
	{
	    _fadeFrames++;
	}
	else
	{
	    std::cerr << "Frames this fade: " << _fadeFrames << std::endl;
	    _fadeActive = false;
	}
    }

#endif

    if(_demoMode->getValue())
    {
	double time = PluginHelper::getLastFrameDuration();
	double val = (time / _demoChangeTime) * 2.0 * M_PI;
	osg::Matrix rot;
	rot.makeRotate(val, osg::Vec3(0,0,1));
	_spinMat = _spinMat * rot;
	setTransform(_coordChangeMat * _spinMat * _tiltMat * _heightMat);

	if(_currentZoom != 0.0)
	{
	    updateZoom(_lastZoomMat);
	}

	_demoTime += time;
	if(_demoTime > _demoChangeTime)
	{
	    _demoTime = 0.0;
	    next();
	}
    }
}

bool PanoViewObject::eventCallback(cvr::InteractionEvent * ie)
{
    if(ie->asTrackedButtonEvent())
    {
	TrackedButtonInteractionEvent * tie = ie->asTrackedButtonEvent();

	if(_removeOnClick && tie->getButton() == 0)
	{
	    PluginHelper::sendMessageByName("PanoViewLOD",PAN_UNLOAD,NULL);

	    _removeOnClick = false;
	    return true;
	}

	if(tie->getButton() == 2 && tie->getInteraction() == BUTTON_DOWN)
	{
	    next();
	    return true;
	}
	if(tie->getButton() == 3 && tie->getInteraction() == BUTTON_DOWN)
	{
	    previous();
	    return true;
	}
	if(_trackball->getValue() && tie->getButton() == 0 && tie->getInteraction() == BUTTON_DOWN)
	{
	    _tbDirValid = true;
	    _tbHand = tie->getHand();

	    osg::Vec3 start, end(0,100,0);
	    start = start * tie->getTransform();
	    end = end * tie->getTransform();
	    _tbDir = end - start;
	    _tbDir.normalize();
	    /*_tbDirValid = false;
	    osg::Vec3 startpoint(0,0,0), endpoint(0,1000.0,0), center(0,0,0);
	    startpoint = startpoint * tie->getTransform() * getWorldToObjectMatrix();
	    endpoint = endpoint * tie->getTransform() * getWorldToObjectMatrix();

	    _tbHand = tie->getHand();

	    osg::Vec3 isec1,isec2;
	    float w1,w2;
	    if(lineSphereIntersectionRef(startpoint,endpoint,center,_radiusRV->getValue(),isec1,w1,isec2,w2))
	    {
		osg::Vec3 isec;
		float w;
		if(w1 > w2)
		{
		    isec = isec1;
		    w = w1;
		}
		else
		{
		    isec = isec2;
		    w = w2;
		}

		if(w < 0)
		{
		    return false;
		}

		_tbDirValid = true;
		_tbDir = isec - center;
		_tbDir.normalize();
	    }*/
	    return true;
	}
	if(_trackball->getValue() && tie->getButton() == 0 && (tie->getInteraction() == BUTTON_DRAG || tie->getInteraction() == BUTTON_UP))
	{
	    if(tie->getHand() != _tbHand)
	    {
		return false;
	    }

	    osg::Vec3 start, end(0,100,0);
	    start = start * tie->getTransform();
	    end = end * tie->getTransform();
	    osg::Vec3 newDir = end - start;
	    newDir.normalize();

	    if(_tbDir.z() < 0.999 && newDir.z() < 0.999)
	    {
		osg::Vec3 tvec1, tvec2;
		tvec1 = _tbDir;
		tvec1.z() = 0.0;
		tvec1.normalize();

		tvec2 = newDir;
		tvec2.z() = 0.0;
		tvec2.normalize();

		osg::Matrix rot;
		rot.makeRotate(tvec1,tvec2);
		_spinMat = _spinMat * rot;
	    }

	    if(_tbDir.x() < 0.999 && newDir.x() < 0.999)
	    {
		osg::Vec3 tvec1, tvec2;
		tvec1 = _tbDir;
		tvec1.x() = 0.0;
		tvec1.normalize();

		tvec2 = newDir;
		tvec2.x() = 0.0;
		tvec2.normalize();

		osg::Matrix rot;
		rot.makeRotate(tvec1,tvec2);
		_tiltMat = _tiltMat * rot;
	    }

	    setTransform(_coordChangeMat * _spinMat * _tiltMat * _heightMat);
	    if(_currentZoom != 0.0)
	    {
		updateZoom(_lastZoomMat);
	    }

	    _tbDir = newDir;
	    /*osg::Vec3 startpoint(0,0,0), endpoint(0,1000.0,0), center(0,0,0);
	    startpoint = startpoint * tie->getTransform() * getWorldToObjectMatrix();
	    endpoint = endpoint * tie->getTransform() * getWorldToObjectMatrix();

	    //std::cerr << "Start x: " << startpoint.x() << " y: " << startpoint.y() << " z: " << startpoint.z() << std::endl;
	    //std::cerr << "End x: " << endpoint.x() << " y: " << endpoint.y() << " z: " << endpoint.z() << std::endl;

	    osg::Vec3 isec1,isec2;
	    float w1,w2;
	    if(lineSphereIntersectionRef(startpoint,endpoint,center,_radiusRV->getValue(),isec1,w1,isec2,w2))
	    {
		//std::cerr << "isec1 x: " << isec1.x() << " y: " << isec1.y() << " z: " << isec1.z() << " w: " << w1 << std::endl;
		//std::cerr << "isec2 x: " << isec2.x() << " y: " << isec2.y() << " z: " << isec2.z() << " w: " << w2 << std::endl;
		osg::Vec3 isec;
		float w;
		if(w1 > w2)
		{
		    isec = isec1;
		    w = w1;
		}
		else
		{
		    isec = isec2;
		    w = w2;
		}

		if(w < 0)
		{
		    return false;
		}

		//std::cerr << "isec x: " << isec.x() << " y: " << isec.y() << " z: " << isec.z() << std::endl;

		osg::Vec3 newDir = isec - center;
		newDir.normalize();

		if(_tbDirValid)
		{
		    if(_tbDir.y() < 0.999 && newDir.y() < 0.999)
		    {
			osg::Vec3 tvec1, tvec2;
			tvec1 = _tbDir;
			tvec1.y() = 0.0;
			tvec1.normalize();

			tvec2 = newDir;
			tvec2.y() = 0.0;
			tvec2.normalize();

			osg::Matrix rot;
			rot.makeRotate(tvec1,tvec2);
			//_tbM1 = _tbM1 * rot;
		    }

		    if(_tbDir.x() < 0.999 && newDir.x() < 0.999)
		    {
			osg::Vec3 tvec1, tvec2;
			tvec1 = _tbDir;
			tvec1.x() = 0.0;
			tvec1.normalize();

			tvec2 = newDir;
			tvec2.x() = 0.0;
			tvec2.normalize();

			osg::Matrix rot;
			rot.makeRotate(tvec1,tvec2);
			//_tbM2 = _tbM2 * rot;
		    }

		    //osg::Matrix rot;
		    //rot.makeRotate(_tbDir,newDir);
		    //_tbMat = rot * _tbMat;
		    //_tbMat = _tbM1 * _tbM2;
		    setTransform(_coordChangeMat * _spinMat * _tiltMat * _heightMat);
                    if(_currentZoom != 0.0)
                    {
                        updateZoom(_lastZoomMat);
                    }
		}
		else
		{
		    _tbDirValid = true;
		    _tbDir = newDir;
		}
	    }*/
	    return true;
	}
	/*if(tie->getButton() == 0 && tie->getInteraction() == BUTTON_DOWN)
	{
	    updateZoom(tie->getTransform());

	    return true;
	}
	if(tie->getButton() == 0 && (tie->getInteraction() == BUTTON_DRAG || tie->getInteraction() == BUTTON_UP))
	{
	    float val = -PluginHelper::getValuator(0,1);
	    if(fabs(val) > 0.25)
	    {
		_currentZoom += val * _zoomScale * PluginHelper::getLastFrameDuration() * 0.25;
		if(_currentZoom < -2.0) _currentZoom = -2.0;
		if(_currentZoom > 0.5) _currentZoom = 0.5;
	    }

	    updateZoom(tie->getTransform());

	    return true;
	}*/
	if(tie->getButton() == 4 && tie->getInteraction() == BUTTON_DOWN)
	{
	    _currentZoom = 0.0;

	    _leftDrawable->setZoom(osg::Vec3(0,1,0),pow(10.0, _currentZoom));
	    _rightDrawable->setZoom(osg::Vec3(0,1,0),pow(10.0, _currentZoom));

	    return true;
	}
    }
    else if(ie->asKeyboardEvent())
    {
	/*osg::Matrix rot;
	rot.makeRotate((M_PI / 50.0) * 0.6, osg::Vec3(0,0,1));
	_spinMat = _spinMat * rot;
	setTransform(_tbM1 * _tbM2 * _coordChangeMat * _spinMat * _heightMat);

	if(_currentZoom != 0.0)
	{
	    updateZoom(_lastZoomMat);
	}*/

	KeyboardInteractionEvent * kie = ie->asKeyboardEvent();
	if(kie->getKey() == 65362)
	{
	    _upArrow = kie->getInteraction() == KEY_DOWN;
	}
	else if(kie->getKey() == 65364)
	{
	    _downArrow = kie->getInteraction() == KEY_DOWN;
	}
	else if(kie->getKey() == 65361)
	{
	    _leftArrow = kie->getInteraction() == KEY_DOWN;
	}
	else if(kie->getKey() == 65363)
	{
	    _rightArrow = kie->getInteraction() == KEY_DOWN;
	}
    }
    else if(ie->asValuatorEvent())
    {
	//std::cerr << "Valuator id: " << ie->asValuatorEvent()->getValuator() << " value: " << ie->asValuatorEvent()->getValue() << std::endl;

	ValuatorInteractionEvent * vie = ie->asValuatorEvent();
	if(vie->getValuator() == _spinValuator)
	{
	    if(!_sharedValuator || _spinCB->getValue())
	    {
		float val = vie->getValue();
		if(fabs(val) < 0.15)
		{
		    return true;
		}

		if(val > 1.0)
		{
		    val = 1.0;
		}
		else if(val < -1.0)
		{
		    val = -1.0;
		}

		if(val < 0)
		{
		    val = -(val * val);
		}
		else
		{
		    val *= val;
		}

		osg::Matrix rot;
		rot.makeRotate((M_PI / 50.0) * val * _spinScale, osg::Vec3(0,0,1));
		_spinMat = _spinMat * rot;
		setTransform(_coordChangeMat * _spinMat * _tiltMat * _heightMat);

		if(_printValues)
		{
		    std::cerr << "Spin value: " << getRotate() * 180.0 / M_PI << std::endl;
		}

		if(_currentZoom != 0.0)
		{
		    updateZoom(_lastZoomMat);
		}
		return true;
	    }
	}
	if(vie->getValuator() == _zoomValuator)
	{
	    if(!_sharedValuator || _zoomCB->getValue())
	    {
		float val = vie->getValue();
		if(fabs(val) > 0.20)
		{
		    _currentZoom += val * _zoomScale * 0.017 * 0.25;
		    if(_currentZoom < -2.0) _currentZoom = -2.0;
		    if(_currentZoom > 0.5) _currentZoom = 0.5;
		}

		updateZoom(PluginHelper::getHandMat(vie->getHand()));
		return true;
	    }
	}
    }
    return false;
}

void PanoViewObject::preFrameUpdate()
{
    float spinChange = 0;
    float tiltChange = 0;

    static const float radsPerSec = 0.5;
    if(_upArrow && !_downArrow)
    {
	tiltChange -= radsPerSec * PluginHelper::getLastFrameDuration();
    }
    if(_downArrow && !_upArrow)
    {
	tiltChange += radsPerSec * PluginHelper::getLastFrameDuration();
    }

    if(_rightArrow && !_leftArrow)
    {
	spinChange += radsPerSec * PluginHelper::getLastFrameDuration();
    }
    if(_leftArrow && !_rightArrow)
    {
	spinChange -= radsPerSec * PluginHelper::getLastFrameDuration();
    }

    osg::Matrix rot;
    rot.makeRotate(spinChange, osg::Vec3(0,0,1));
    _spinMat = _spinMat * rot;

    rot.makeRotate(tiltChange, osg::Vec3(1,0,0));
    _tiltMat = _tiltMat * rot;

    setTransform(_coordChangeMat * _spinMat * _tiltMat * _heightMat);
    if(_currentZoom != 0.0)
    {
	updateZoom(_lastZoomMat);
    }

    if(_transitionType == ZOOM && _transitionStarted && !_transitionSkipFrames)
    {	
	_transitionTime += PluginHelper::getLastFrameDuration();

	if(!_rotateDone)
	{
	    float value;
	    if(_transitionTime <= _rotateStartDelay)
	    {
		value = 0.0;
	    }
	    else if(_transitionTime >= _rotateStartDelay + _rotateInterval)
	    {
		value = 1.0;
		_rotateDone = true;
	    }
	    else
	    {
		//removed to add rampup/down
		//value = (_transitionTime - _rotateStartDelay) / _rotateInterval;
		value = 0.0;

		float ratio = (2.0 * _rotateRampUp / M_PI);

		if(_transitionTime < _rotateStartDelay + _rotateRampUp)
		{
		    //std::cerr << "RampUp: ";
		    float angle = ((_transitionTime - _rotateStartDelay) / _rotateRampUp) * (M_PI / 2.0);
		    value += _rotateAmp * (1.0 - cos(angle)) * ratio;
		}
		else
		{
		    value += _rotateAmp * ratio;
		    if(_transitionTime < (_rotateStartDelay + _rotateInterval - _rotateRampDown))
		    {
			//std::cerr << "Middle: ";
			float length = _transitionTime - _rotateStartDelay - _rotateRampUp;
			value += _rotateAmp * length;
		    }
		    else
		    {
			//std::cerr << "RampDown: ";
			value += _rotateAmp * (_rotateInterval - _rotateRampUp - _rotateRampDown);
			float angle = ((_transitionTime - _rotateStartDelay - (_rotateInterval - _rotateRampDown)) / _rotateRampDown) * (M_PI / 2.0);
			value += _rotateAmp * sin(angle) * ratio;
		    }
		}
	    }

	    //if(value > 0.0)
	    //{
	    //	std::cerr << "Rotate Transition Value: " << value << " Time Value: " << (_transitionTime - _rotateStartDelay) / _rotateInterval << std::endl;
	    //}

	    float rotateValue = _rotateStart + value * (_rotateEnd - _rotateStart);

	    _pdi->fromTransitionTransform.makeRotate(rotateValue,osg::Vec3(0,1,0));
	}

	if(!_zoomDone)
	{
	    float value;
	    if(_transitionTime <= _zoomStartDelay)
	    {
		value = 0.0;
	    }
	    else if(_transitionTime >= _zoomStartDelay + _zoomInterval)
	    {
		value = 1.0;
		_zoomDone = true;
	    }
	    else
	    {
		//removed to add rampup/down
		//value = (_transitionTime - _zoomStartDelay) / _zoomInterval;
		value = 0.0;

		float ratio = (2.0 * _zoomRampUp / M_PI);

		if(_transitionTime < _zoomStartDelay + _zoomRampUp)
		{
		    //std::cerr << "RampUp: ";
		    float angle = ((_transitionTime - _zoomStartDelay) / _zoomRampUp) * (M_PI / 2.0);
		    value += _zoomAmp * (1.0 - cos(angle)) * ratio;
		}
		else
		{
		    value += _zoomAmp * ratio;
		    if(_transitionTime < (_zoomStartDelay + _zoomInterval - _zoomRampDown))
		    {
			//std::cerr << "Middle: ";
			float length = _transitionTime - _zoomStartDelay - _zoomRampUp;
			value += _zoomAmp * length;
		    }
		    else
		    {
			//std::cerr << "RampDown: ";
			value += _zoomAmp * (_zoomInterval - _zoomRampUp - _zoomRampDown);
			float angle = ((_transitionTime - _zoomStartDelay - (_zoomInterval - _zoomRampDown)) / _zoomRampDown) * (M_PI / 2.0);
			value += _zoomAmp * sin(angle) * ratio;
		    }
		}
	    }

	    //if(value > 0.0)
	    //{
	    //	std::cerr << "Zoom Transition Value: " << value << " Time Value: " << (_transitionTime - _zoomStartDelay) / _zoomInterval << std::endl;
	    //}

	    float zoomValue = value * _zoomEnd;

	    osg::Vec3 dir = _zoomTransitionDir;
	    dir = dir * getWorldToObjectMatrix() * osg::Matrix::inverse(_pdi->fromTransitionTransform);
	    osg::Vec3 point;
	    point = point * getWorldToObjectMatrix() * osg::Matrix::inverse(_pdi->fromTransitionTransform);
	    dir = dir - point;
	    dir.normalize();

	    //dir = osg::Vec3(0,0,-1);

	    for(std::map<int,sph_model*>::iterator it = _pdi->transitionModelMap.begin(); it!= _pdi->transitionModelMap.end(); it++)
	    {
		it->second->set_zoom(dir.x(),dir.y(),dir.z(),pow(10.0, zoomValue));
	    }
	}

	if(!_fadeDone)
	{
	    float value;
	    if(_transitionTime <= _fadeStartDelay)
	    {
		value = 0.0;
	    }
	    else if(_transitionTime >= _fadeStartDelay + _fadeInterval)
	    {
		value = 1.0;
		_fadeDone = true;
	    }
	    else
	    {
		value = (_transitionTime - _fadeStartDelay) / _fadeInterval;
	    }
	    _pdi->transitionFade = value;
	}

	if(_rotateDone && _zoomDone && _fadeDone)
	{
	    _transitionStarted = false;
	    _leftDrawable->transitionDone();
	    _rightDrawable->transitionDone();
	    setRotate(_finalRotate);
	}
    }
    else if(_transitionSkipFrames)
    {
	_transitionSkipFrames--;
    }
}

void PanoViewObject::updateZoom(osg::Matrix & mat)
{
    osg::Matrix m = osg::Matrix::inverse(_root->getMatrix());
    osg::Vec3 dir(0,1,0);
    osg::Vec3 point(0,0,0);
    dir = dir * mat * m;
    point = point * mat * m;
    dir = dir - point;
    dir.normalize();

    if(_printValues)
    {
	osg::Vec3 zoomDir = osg::Vec3(0,1,0) * mat;
	zoomDir = zoomDir - mat.getTrans();
	zoomDir.normalize();
	std::cerr << "Zoom value: " << _currentZoom << " x: " << zoomDir.x() << " y: " << zoomDir.y() << " z: " << zoomDir.z() << std::endl;
    }

    if(_leftDrawable)
    {
	_leftDrawable->setZoom(dir,pow(10.0, _currentZoom));
    }
    else
    {
	_rightDrawable->setZoom(dir,pow(10.0, _currentZoom));
    }

    _lastZoomMat = mat;
}

void PanoViewObject::startTransition()
{
    if(_leftDrawable->getSetSize() < 2)
    {
	return;
    }

    if(_transitionType == ZOOM)
    { 
	_pdi->transitionFade = 0.0;

	_transitionStarted = true;
	_transitionSkipFrames = 4;
	_transitionTime = 0.0;
	_rotateDone = _zoomDone = _fadeDone = false;

	int fromIndex = _leftDrawable->getLastIndex();
	int toIndex = _leftDrawable->getCurrentIndex();
	//std::cerr << "From Index: " << fromIndex << " To Index: " << toIndex << std::endl;

	_rotateStart = getRotate();
	_rotateEnd = _zoomTransitionInfo[fromIndex].rotationFromImage;

	while(fabs(_rotateEnd - _rotateStart) > M_PI)
	{
	    if(_rotateEnd > _rotateStart)
	    {
		_rotateEnd -= 2.0 * M_PI;
	    }
	    else
	    {
		_rotateEnd += 2.0 * M_PI;
	    }
	}

	float toRotation = _zoomTransitionInfo[fromIndex].rotationToImage;
	_finalRotate = toRotation;

	setRotate(0.0);

	_pdi->toTransitionTransform.makeRotate(toRotation,osg::Vec3(0,1,0));
	_pdi->fromTransitionTransform.makeRotate(_rotateStart,osg::Vec3(0,1,0));

	_currentZoom = 0.0;
	_leftDrawable->setZoom(osg::Vec3(0,1,0),pow(10.0, _currentZoom));
	_rightDrawable->setZoom(osg::Vec3(0,1,0),pow(10.0, _currentZoom));
	_zoomEnd = _zoomTransitionInfo[fromIndex].zoomValue;
	_zoomTransitionDir = _zoomTransitionInfo[fromIndex].zoomDir;

	static float ratio = 2.0 / M_PI;
	_rotateAmp = 1.0 / (_rotateRampUp*ratio + (_rotateInterval - _rotateRampUp - _rotateRampDown) + _rotateRampDown*ratio);
	_zoomAmp = 1.0 / (_zoomRampUp*ratio + (_zoomInterval - _zoomRampUp - _zoomRampDown) + _zoomRampDown*ratio);
    }
}

void PanoViewObject::writeConfig()
{

    if (!cvr::ComController::instance()->isMaster())
    {
        return;
    }

    std::cout << "Saved." << std::endl;

    std::ofstream cfile;
    cfile.open((_configPath + "/Init.cfg").c_str(), std::ios::trunc);

    if(!cfile.fail())
    {
        for(std::map<std::string, std::pair<float, float> >::iterator it = _locInit.begin();
        it != _locInit.end(); it++)
        {
            cfile << it->first << " ";
            cfile << it->second.first << " " << it->second.second << " " << std::endl;;
        }
    }
    cfile.close();
}
