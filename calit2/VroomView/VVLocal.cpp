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
    SubImageInfo * sii = new SubImageInfo;
    sii->image = new osg::Image();
    sii->image->allocateImage(1920,1080,GL_RGBA,GL_RGBA,GL_FLOAT);
    sii->image->setInternalTextureFormat(4);

    sii->depthTex = new osg::Texture2D();
    sii->depthTex->setTextureSize(1920,1080);
    sii->depthTex->setInternalFormat(GL_DEPTH_COMPONENT);
    sii->depthTex->setFilter(osg::Texture2D::MIN_FILTER,osg::Texture2D::NEAREST);
    sii->depthTex->setFilter(osg::Texture2D::MAG_FILTER,osg::Texture2D::NEAREST);
    sii->depthTex->setResizeNonPowerOfTwoHint(false);
    sii->depthTex->setUseHardwareMipMapGeneration(false);


    sii->camera = new osg::Camera();
    sii->camera->setAllowEventFocus(false);
    sii->camera->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    sii->camera->setClearColor(osg::Vec4(1.0,0,0,1.0));
    sii->camera->setRenderOrder(osg::Camera::PRE_RENDER);
    sii->camera->setRenderTargetImplementation(osg::Camera::FRAME_BUFFER_OBJECT);
    sii->camera->attach(osg::Camera::COLOR_BUFFER0, sii->image, 0, 0);
    sii->camera->attach(osg::Camera::DEPTH_BUFFER,sii->depthTex);
    sii->camera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);

    sii->camera->addChild((osg::Node*)SceneManager::instance()->getScene());
    sii->takeImage = false;
    _imageInfoList.push_back(sii);
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
	    osgDB::writeImageFile(*(*it)->image.get(),"/Users/aprudhomme/testImage.tif");

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
