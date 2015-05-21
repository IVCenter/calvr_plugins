#include "VVLocal.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/ScreenBase.h>
#include <cvrKernel/CVRViewer.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/PluginHelper.h>

#include <osgDB/WriteFile>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>
#include <sstream>

using namespace cvr;

VVLocal::VVLocal()
{
    _error = false;
    _ssDir = ConfigManager::getEntry("value","Plugin.VroomView.ScreenShotDir","");
    _maxDim = ConfigManager::getFloat("value","Plugin.VroomView.MaxDim",4096.0);
    _targetDim = ConfigManager::getFloat("value","Plugin.VroomView.TargetDim",8192.0);
}

VVLocal::~VVLocal()
{
}

void VVLocal::takeScreenShot(std::string label)
{
    float pWidth,pHeight;
    float ratio = SceneManager::instance()->getTiledWallWidth() / SceneManager::instance()->getTiledWallHeight();

    if(ratio > 1.0)
    {
	pWidth = _targetDim;
	pHeight = _targetDim * (1.0 / ratio);
    }
    else
    {
	pHeight = _targetDim;
	pWidth = _targetDim * (1.0 / ratio);
    }

    int rows = (pHeight-1) / _maxDim;
    rows++;
    int cols = (pWidth-1) / _maxDim;
    cols++;
    pHeight /= rows;
    pWidth /= cols;

    _rows = rows;
    _cols = cols;

    time_t ctime = time(NULL);
    struct tm timetm = *localtime(&ctime);
    char timestr[256];
    timestr[255] = '\0';
    strftime(timestr, 255, "%F-%H_%M_%S", &timetm);

    float width = SceneManager::instance()->getTiledWallWidth();
    float height = SceneManager::instance()->getTiledWallHeight();

    std::stringstream labelBaseName;
    if(label != "")
    {
	labelBaseName << label;
    }
    else
    {
	labelBaseName << "ScreenShot-" << timestr;
    }
    _baseName = labelBaseName.str();

    for(int i = 0; i < rows; i++)
    {
	for(int j = 0; j < cols; j++)
	{
	    SubImageInfo * sii = new SubImageInfo;

	    std::stringstream labelBase;
	    if(label != "")
	    {
		labelBase << label << "_" << i << "_" << j;
	    }
	    else
	    {
		labelBase << "ScreenShot-" << timestr << "_" << i << "_" << j;
	    }

	    sii->label = _ssDir + "/" + labelBase.str() + ".tif";
	    _fileNames.push_back(sii->label);

	    sii->width = width / ((float)cols);
	    sii->height = height / ((float)rows);

	    osg::Vec3 center;
	    center.x() = (sii->width * (((float)j) + 0.5)) - (width / 2.0);
	    center.z() = (sii->height * (((float)-i) - 0.5)) + (height / 2.0);

	    sii->center = center * SceneManager::instance()->getTiledWallTransform();

	    sii->image = new osg::Image();
	    sii->image->allocateImage(pWidth,pHeight,GL_RGBA,GL_RGBA,GL_FLOAT);
	    sii->image->setInternalTextureFormat(4);

	    sii->depthTex = new osg::Texture2D();
	    sii->depthTex->setTextureSize(pWidth,pHeight);
	    sii->depthTex->setInternalFormat(GL_DEPTH_COMPONENT);
	    sii->depthTex->setFilter(osg::Texture2D::MIN_FILTER,osg::Texture2D::NEAREST);
	    sii->depthTex->setFilter(osg::Texture2D::MAG_FILTER,osg::Texture2D::NEAREST);
	    sii->depthTex->setResizeNonPowerOfTwoHint(false);
	    sii->depthTex->setUseHardwareMipMapGeneration(false);


	    sii->camera = new osg::Camera();
	    sii->camera->setAllowEventFocus(false);
	    sii->camera->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	    sii->camera->setClearColor(osg::Vec4(0.0,0,0,1.0));
	    sii->camera->setRenderOrder(osg::Camera::PRE_RENDER);
	    sii->camera->setRenderTargetImplementation(osg::Camera::FRAME_BUFFER_OBJECT);
	    sii->camera->attach(osg::Camera::COLOR_BUFFER0, sii->image, 0, 0);
	    sii->camera->attach(osg::Camera::DEPTH_BUFFER,sii->depthTex);
	    sii->camera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
	    sii->camera->setViewport(0,0,pWidth,pHeight);

	    sii->camera->addChild((osg::Node*)SceneManager::instance()->getScene());

	    sii->takeImage = false;
	    _imageInfoList.push_back(sii);
	}
    }
}

void VVLocal::preFrame()
{
    if(_error)
    {
	return;
    }

    if(!processSubImage())
    {
	_error = true;
	return;
    }
}

bool VVLocal::processSubImage()
{
    bool combine = false;

    for(std::vector<SubImageInfo*>::iterator it = _imageInfoList.begin(); it != _imageInfoList.end(); )
    {
	if(!(*it)->takeImage)
	{
	    takeSubImage(*it);
	}
	else
	{
	    combine = true;

	    dynamic_cast<osg::Group*>(CVRViewer::instance()->getSceneData())->removeChild((*it)->camera);
	}

	it++;
    }

    if(combine)
    {
	SaveThread * thread = new SaveThread(_imageInfoList,_fileNames,_baseName,_ssDir,_rows,_cols);
	thread->startThread();

	for(int i = 0; i < _imageInfoList.size(); ++i)
	{
	    delete _imageInfoList[i];
	}
	_imageInfoList.clear();
	_fileNames.clear();
	_baseName = "";
    }

    return true;
}

void VVLocal::takeSubImage(SubImageInfo * info)
{
    info->takeImage = true;

    dynamic_cast<osg::Group*>(CVRViewer::instance()->getSceneData())->addChild(info->camera);

    setSubImageParams(info,info->center,info->width,info->height);
}

void VVLocal::setSubImageParams(SubImageInfo * info, osg::Vec3 pos, float width, float height)
{
    osg::Matrix centerTrans;
    centerTrans.makeTranslate(-pos);
    osg::Vec3 camPos = PluginHelper::getHeadMat(0).getTrans();
    camPos = camPos * centerTrans;
    osg::Matrix camTrans;
    camTrans.makeTranslate(-camPos);
    
    osg::Matrix view = centerTrans * camTrans * osg::Matrix::lookAt(osg::Vec3(0,0,0),osg::Vec3(0,1,0),osg::Vec3(0,0,1));

    float top, bottom, left, right;
    float screenDist = -camPos.y();

    top = ScreenBase::getNear() * (height / 2.0 - camPos.z()) / screenDist;
    bottom = ScreenBase::getNear() * (-height / 2.0 - camPos.z()) / screenDist;
    right = ScreenBase::getNear() * (width / 2.0 - camPos.x()) / screenDist;
    left = ScreenBase::getNear() * (-width / 2.0 - camPos.x()) / screenDist;

    osg::Matrix proj;
    proj.makeFrustum(left,right,bottom,top,ScreenBase::getNear(),ScreenBase::getFar());

    info->camera->setViewMatrix(view);
    info->camera->setProjectionMatrix(proj);
}

SaveThread::SaveThread(std::vector<SubImageInfo*> & infoList, std::vector<std::string> & nameList, std::string baseName, std::string ssDir, int rows, int cols)
{
    _baseName = baseName;
    _nameList = nameList;
    _ssDir = ssDir;
    _rows = rows;
    _cols = cols;

    for(int i = 0; i < infoList.size(); ++i)
    {
	_imageList.push_back(infoList[i]->image);
    }
}

void SaveThread::run()
{
    for(int i = 0; i < _imageList.size(); ++i)
    {
	std::cerr << "Writing: " << _nameList[i] << std::endl;
	osgDB::writeImageFile(*_imageList[i].get(),_nameList[i]);
    }

    std::string outfile = _ssDir + "/" + _baseName + ".tif";
    std::stringstream combineScript;
    combineScript << "montage ";
    for(int i = 0; i < _nameList.size(); ++i)
    {
	combineScript << _nameList[i] << " ";
    }
    combineScript << "-mode Concatenate -tile " << _cols << "x" << _rows << " " << outfile;

    std::cerr << "script: " << combineScript.str() << std::endl;
    system(combineScript.str().c_str());
    for(int i = 0; i < _nameList.size(); ++i)
    {
	std::string rmss = "rm ";
	rmss += _nameList[i];
	system(rmss.c_str());
    }

    _imageList.clear();
    _nameList.clear();
}
