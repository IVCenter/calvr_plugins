#include "VVLocal.h"

#include <cvrKernel/ScreenBase.h>
#include <cvrKernel/CVRViewer.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/PluginHelper.h>

#include <osgDB/WriteFile>

#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

using namespace cvr;

VVLocal::VVLocal()
{
    _error = false;
}

VVLocal::~VVLocal()
{
}

void VVLocal::takeScreenShot(std::string label)
{
    SubImageInfo * sii = ne SubImageInfo;
    sii->image = new osg::Image();
    sii->image->allocateImage(1920,1080,GL_RGBA,GL_RGBA,GL_FLOAT);
    sii->image->setInternalTextureFormat(4);

    _subDepthTex = new osg::Texture2D();
    _subDepthTex->setTextureSize(1920,1080);
    _subDepthTex->setInternalFormat(GL_DEPTH_COMPONENT);
    _subDepthTex->setFilter(osg::Texture2D::MIN_FILTER,osg::Texture2D::NEAREST);
    _subDepthTex->setFilter(osg::Texture2D::MAG_FILTER,osg::Texture2D::NEAREST);
    _subDepthTex->setResizeNonPowerOfTwoHint(false);
    _subDepthTex->setUseHardwareMipMapGeneration(false);


    _subCamera = new osg::Camera();
    _subCamera->setAllowEventFocus(false);
    _subCamera->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    _subCamera->setClearColor(osg::Vec4(1.0,0,0,1.0));
    _subCamera->setRenderOrder(osg::Camera::PRE_RENDER);
    _subCamera->setRenderTargetImplementation(osg::Camera::FRAME_BUFFER_OBJECT);
    _subCamera->attach(osg::Camera::COLOR_BUFFER0, _subImage, 0, 0);
    _subCamera->attach(osg::Camera::DEPTH_BUFFER,_subDepthTex);
    _subCamera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);

    _subCamera->addChild((osg::Node*)SceneManager::instance()->getScene());
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
    for(std::vector<SubImageInfo*>::iterator it = _imageInfoList.begin(); it != _imageInfoList.end(); )
    {
	if(!(*it)->takeImage)
	{
	    takeSubImage(*it);
	}
	else
	{
	    osgDB::writeImageFile(*(*it)->image.get(),"/home/aprudhom/testImage.tif");

	    // send data and cleanup
	    dynamic_cast<osg::Group*>(CVRViewer::instance()->getSceneData())->removeChild((*it)->camera);
	    delete (*it);
	    it = _imageInfoList.erase(it);
	    continue;
	}

	it++;
    }

    return true;
}

void VVLocal::takeSubImage(SubImageInfo * info)
{
    info->takeImage = true;

    dynamic_cast<osg::Group*>(CVRViewer::instance()->getSceneData())->addChild(info->camera);

    osg::Vec3 center;
    center = info->center * SceneManager::instance()->getTiledWallTransform();
    setSubImageParams(info,center,info->width,info->height);
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
