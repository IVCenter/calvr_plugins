#include "PanoViewObject.h"

#include <config/ConfigManager.h>
#include <kernel/NodeMask.h>
#include <kernel/PluginHelper.h>

#include <iostream>

#define PRINT_TIMING

using namespace cvr;

PanoViewObject::PanoViewObject(std::string name, std::string leftEyeFile, std::string rightEyeFile, float radius, int mesh, int depth, int size, float height, std::string vertFile, std::string fragFile) : SceneObject(name,false,false,false,true,false)
{
    std::vector<std::string> left;
    std::vector<std::string> right;
    left.push_back(leftEyeFile);
    right.push_back(rightEyeFile);

    init(left,right,radius,mesh,depth,size,height,vertFile,fragFile);
}

PanoViewObject::PanoViewObject(std::string name, std::vector<std::string> & leftEyeFiles, std::vector<std::string> & rightEyeFiles, float radius, int mesh, int depth, int size, float height, std::string vertFile, std::string fragFile) : SceneObject(name,false,false,false,true,false)
{
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
    _imageSearchPath = ConfigManager::getEntry("value","Plugin.PanoViewLOD.ImageSearchPath","");
    _floorOffset = ConfigManager::getFloat("value","Plugin.PanoViewLOD.FloorOffset",0);

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

    _leftDrawable = new PanoDrawableLOD(leftEyeFiles,rightEyeFiles,radius,mesh,depth,size,vertFile,fragFile);
    _rightDrawable = new PanoDrawableLOD(leftEyeFiles,rightEyeFiles,radius,mesh,depth,size,vertFile,fragFile);

    _leftGeode->addDrawable(_leftDrawable);
    _rightGeode->addDrawable(_rightDrawable);

    addChild(_leftGeode);
    addChild(_rightGeode);

    _currentZoom = 0.0;

    _demoTime = 0.0;
    _demoChangeTime = ConfigManager::getDouble("value","Plugin.PanoViewLOD.DemoChangeTime",90.0);

    _coordChangeMat.makeRotate(M_PI/2.0,osg::Vec3(1,0,0));
    _spinMat.makeIdentity();
    float offset = height - _floorOffset;
    _heightMat.makeTranslate(osg::Vec3(0,0,offset));
    setTransform(_coordChangeMat * _spinMat * _heightMat);

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

    _radiusRV = new MenuRangeValue("Radius", 100, 100000, radius);
    _radiusRV->setCallback(this);
    addMenuItem(_radiusRV);

    _heightRV = new MenuRangeValue("Height", -1000, 10000, height);
    _heightRV->setCallback(this);
    addMenuItem(_heightRV);

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
}

void PanoViewObject::next()
{
    _fadeActive = true;
    _fadeFrames = 0;
    _leftDrawable->next();
    _rightDrawable->next();
}

void PanoViewObject::previous()
{
    _fadeActive = true;
    _fadeFrames = 0;
    _leftDrawable->previous();
    _rightDrawable->previous();
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
	float offset = _heightRV->getValue() - _floorOffset;
	_heightMat.makeTranslate(osg::Vec3(0,0,offset));
	setTransform(_coordChangeMat * _spinMat * _heightMat);
    }

    if(item == _zoomResetButton)
    {
	_currentZoom = 0.0;

	_leftDrawable->setZoom(osg::Vec3(0,1,0),pow(10.0, _currentZoom));
	_rightDrawable->setZoom(osg::Vec3(0,1,0),pow(10.0, _currentZoom));
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
	setTransform(_coordChangeMat * _spinMat * _heightMat);

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
	if(tie->getButton() == 0 && tie->getInteraction() == BUTTON_DOWN)
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
	}
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
	osg::Matrix rot;
	rot.makeRotate((M_PI / 50.0) * 0.6, osg::Vec3(0,0,1));
	_spinMat = _spinMat * rot;
	setTransform(_coordChangeMat * _spinMat * _heightMat);

	if(_currentZoom != 0.0)
	{
	    updateZoom(_lastZoomMat);
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
		setTransform(_coordChangeMat * _spinMat * _heightMat);

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

void PanoViewObject::updateZoom(osg::Matrix & mat)
{
    osg::Matrix m = osg::Matrix::inverse(_root->getMatrix());
    osg::Vec3 dir(0,1,0);
    osg::Vec3 point(0,0,0);
    dir = dir * mat * m;
    point = point * mat * m;
    dir = dir - point;
    dir.normalize();

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
