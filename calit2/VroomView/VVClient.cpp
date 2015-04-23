#include "VVClient.h"

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

VVClient::VVClient(cvr::CVRSocket * socket)
{
    _error = false;
    _con = socket;
}

VVClient::~VVClient()
{
    delete _con;
}

void VVClient::preFrame()
{
    if(_error)
    {
	return;
    }

    if(!processSocket())
    {
	_error = true;
	return;
    }

    if(!processSubImage())
    {
	_error = true;
	return;
    }
}

bool VVClient::processSocket()
{
    fd_set readset;
    FD_ZERO(&readset);
    FD_SET(_con->getSocketFD(),&readset);
    if(select(_con->getSocketFD()+1,&readset,NULL,NULL,NULL) >= 0)
    {
	if(FD_ISSET(_con->getSocketFD(),&readset))
	{
	    // process input
	    


	}
    }

    return true;
}

bool VVClient::processSubImage()
{
    for(std::vector<SubImageInfo*>::iterator it = _imageInfoList.begin(); it != _imageInfoList.end(); )
    {
	if(!(*it)->takeImage)
	{
	    takeSubImage(*it);
	}
	else
	{
	    //osgDB::writeImageFile(*(*it)->image.get(),"/home/aprudhom/testImage.tif");

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

void VVClient::takeSubImage(SubImageInfo * info)
{
    info->takeImage = true;

    dynamic_cast<osg::Group*>(CVRViewer::instance()->getSceneData())->addChild(info->camera);

    osg::Vec3 center;
    center = info->center * SceneManager::instance()->getTiledWallTransform();
    setSubImageParams(info,center,info->width,info->height);
}

void VVClient::setSubImageParams(SubImageInfo * info, osg::Vec3 pos, float width, float height)
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
