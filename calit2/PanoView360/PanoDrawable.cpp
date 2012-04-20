#include "PanoDrawable.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/ScreenConfig.h>
#include <cvrKernel/ScreenBase.h>
#include <cvrKernel/ComController.h>
#include <cvrKernel/NodeMask.h>

#include <osgDB/ReadFile>

#include <iostream>
#include <cstring>

#ifdef __APPLE__
#include <glu.h>
#else
#include <GL/glu.h>
#endif

#define GL_CLAMP_TO_EDGE 0x812F

using namespace osg;
using namespace std;
using namespace cvr;

std::map<int, int> PanoDrawable::_contextinit;
std::map<int, std::vector<std::vector< GLuint * > > > PanoDrawable::rtextures;
std::map<int, std::vector<std::vector< GLuint * > > > PanoDrawable::ltextures;
OpenThreads::Mutex PanoDrawable::_initLock;
OpenThreads::Mutex PanoDrawable::_leftLoadLock;
OpenThreads::Mutex PanoDrawable::_rightLoadLock;
OpenThreads::Mutex PanoDrawable::_singleLoadLock;
OpenThreads::Mutex PanoDrawable::_rcLock;
bool PanoDrawable::_deleteDone = false;

PanoDrawable::PanoDrawable(float radius_in, float viewanglev_in, float viewangleh_in, float camHeight_in, int segmentsPerTexture_in, int maxTextureSize_in)
{
    _deleteDone = false;
    _doDelete = false;
    radius = radius_in;
    viewanglev = viewanglev_in;
    viewangleh = viewangleh_in;
    camHeight = camHeight_in;

    badinit = 0;

    _rotation = 0.0;

    floorOffset = ConfigManager::getFloat("Plugin.PanoView360.FloorOffset", 0.0);
    _renderOnMaster = ConfigManager::getBool("Plugin.PanoView360.RenderOnMaster",false);
    _highRamLoad = ConfigManager::getBool("Plugin.PanoView360.HighRamLoad",false);
    _useSingleLock = ConfigManager::getBool("Plugin.PanoView360.UseSingleLock",true);

    std::cerr << "High ram load value: " << _highRamLoad << std::endl;

    if(viewanglev < 10)
    {
	viewanglev = 10;
    }
    if(viewanglev > 180)
    {
	viewanglev = 180;
    }

    if(viewangleh < 10)
    {
	viewangleh = 10;
    }
    if(viewangleh > 360)
    {
	viewangleh = 360;
    }

    segmentsPerTexture = segmentsPerTexture_in;
    maxTextureSize = maxTextureSize_in;
    rows = cols = 0;
    mono = 0;
    flip = 0;
    setUseDisplayList(false);
    _eyeMask = 0;
}


PanoDrawable::~PanoDrawable()
{
    /*
    if(init)
    {
        for(int i = 0; i < rows; i++)
        {
            for(int j = 0; j < cols; j++)
            {
                glDeleteTextures(1, rtextures[i][j]);
                delete rtextures[i][j];
                if(!mono)
                {
                    glDeleteTextures(1, ltextures[i][j]);
                    delete ltextures[i][j];
                }
            }
        }
    }
    */
    if(_highRamLoad)
    {
	for(int i = 0; i < rtiles.size(); i++)
	{
	    for(int j = 0; j < rtiles[i].size(); j++)
	    {
		if(rtiles[i][j])
		{
		    delete[] rtiles[i][j];
		}
	    }
	}
	rtiles.clear();

	for(int i = 0; i < ltiles.size(); i++)
	{
	    for(int j = 0; j < ltiles[i].size(); j++)
	    {
		if(ltiles[i][j])
		{
		    delete[] ltiles[i][j];
		}
	    }
	}
	ltiles.clear();
    }
}

void PanoDrawable::updateRotate(float f)
{
    if(f < 0.1 && f > -0.1)
    {
        return;
    }

    if(f >= 10.0 || f <= -10)
    {
        return;
    }

    //cerr << "UpdateRotate: " << f << endl;

    float ff = f;
    if(ff > 1.0)
    {
        ff = 1.0;
    }
    else if(ff < -1.0)
    {
        ff = -1.0;
    }

    ff *= -1.0f;

    _rotation += (M_PI / 50.0) * ff;

    if(_rotation > (M_PI * 2.0f))
    {
        _rotation -= (M_PI * 2.0f);
    }
    else if(_rotation < 0.0)
    {
        _rotation += (M_PI * 2.0f);
    }
}

void PanoDrawable::deleteTextures()
{
    if(_highRamLoad)
    {
	for(int i = 0; i < rtiles.size(); i++)
	{
	    for(int j = 0; j < rtiles[i].size(); j++)
	    {
		if(rtiles[i][j])
		{
		    delete[] rtiles[i][j];
		}
	    }
	}
	rtiles.clear();

	for(int i = 0; i < ltiles.size(); i++)
	{
	    for(int j = 0; j < ltiles[i].size(); j++)
	    {
		if(ltiles[i][j])
		{
		    delete[] ltiles[i][j];
		}
	    }
	}
	ltiles.clear();
    }

    /*if(!init)
    {
	_deleteDone = true;
	return;
    }*/

    /*if(_maxContext == 0)
    {
	for(int i = 0; i < rows; i++)
        {
            for(int j = 0; j < cols; j++)
            {
                glDeleteTextures(1, rtextures[i][j]);
                delete rtextures[i][j];
                if(!mono)
                {
                    glDeleteTextures(1, ltextures[i][j]);
                    delete ltextures[i][j];
                }
            }
        }
	_deleteDone = true;
    }
    else
    {*/
	_doDelete = true;
    //}
}

bool PanoDrawable::deleteDone()
{
    return _deleteDone;
}

PanoDrawable::PanoDrawable(const PanoDrawable&,const osg::CopyOp&) : Drawable()
{

}

void PanoDrawable::setImage(std::string file_path)
{
    setImage(file_path, file_path);
}

void PanoDrawable::setFlip(int f)
{
    flip = f;
}

void PanoDrawable::setImage(std::string file_path_r, std::string file_path_l)
{

    if(file_path_r == file_path_l)
    {
        mono = 1;
    }
    rfile = file_path_r;
    lfile = file_path_l;
    _contextinit.clear();
    rtextures.clear();
    ltextures.clear();
}


bool PanoDrawable::initTexture(eye e, int context) const
{
    if(e == RIGHT)
    {
	if(_useSingleLock)
	{
	    _singleLoadLock.lock();
	}
	else
	{
	    _rightLoadLock.lock();
	}

	if(!_highRamLoad || !rtiles.size())
	{
	    osg::Image* rimage =osgDB::readImageFile(rfile);
	    if(!rimage)
	    {
		std::cerr << "PanoDrawable: Unable to load right eye image: " << rfile << endl;
		if(_useSingleLock)
		{
		    _singleLoadLock.unlock();
		}
		else
		{
		    _rightLoadLock.unlock();
		}
		return false;
	    }
	    if(flip)
	    {
		rimage->flipVertical();
	    }
	    unsigned char * ridata = rimage->data();
	    width = rimage->s();
	    height = rimage->t();

	    if(height % maxTextureSize != 0 || width % maxTextureSize != 0)
	    {
		std::cerr << "PanoDrawable: Image dimensions not multiple of " << maxTextureSize << endl;
		if(_useSingleLock)
		{
		    _singleLoadLock.unlock();
		}
		else
		{
		    _rightLoadLock.unlock();
		}
		return false;
	    }

	    for(int i = 0; i < height / maxTextureSize; i++)
	    {
		rtextures[context].push_back(std::vector<GLuint *>());
		for(int j = 0; j < width / maxTextureSize; j++)
		{
		    rtextures[context][i].push_back(new GLuint);
		}
	    }

	    _rcLock.lock();
	    rows = height / maxTextureSize;
	    cols = width / maxTextureSize;
	    _rcLock.unlock();

	    unsigned char * tiledata;

	    if(!_highRamLoad)
	    {
		tiledata = new unsigned char[maxTextureSize * maxTextureSize * 3];
	    }

	    int rowsize = maxTextureSize * 3;

	    for(int i = 0; i < rtextures[context].size(); i++)
	    {
		if(_highRamLoad)
		{
		    rtiles.push_back(std::vector<unsigned char *>());
		}
		for(int j = 0; j < rtextures[context][i].size(); j++)
		{
		    if(_highRamLoad)
		    {
			tiledata = new unsigned char[maxTextureSize * maxTextureSize * 3];
			rtiles[i].push_back(tiledata);
		    }

		    ridata = rimage->data();
		    ridata = ridata + (i * width * maxTextureSize * 3) + (j * rowsize);
		    for(int k = 0; k < maxTextureSize; k++)
		    {
			memcpy((void*)(tiledata + (k*rowsize)), (void *)ridata, rowsize);
			ridata += width * 3;
		    }

		    if(!_highRamLoad)
		    {
			glGenTextures(1, rtextures[context][i][j]);
			glBindTexture(GL_TEXTURE_2D, *(rtextures[context][i][j]));
			glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
			glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR_MIPMAP_LINEAR);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			gluBuild2DMipmaps(GL_TEXTURE_2D, 4, maxTextureSize, maxTextureSize, GL_RGB, GL_UNSIGNED_BYTE, tiledata);
		    }
		}
	    }

	    if(rimage)
	    {
		delete[] rimage->data();
		rimage->unref();
	    }

	    if(!_highRamLoad)
	    {
		delete[] tiledata;
	    }

	}

	if(_useSingleLock)
	{
	    _singleLoadLock.unlock();
	}
	else
	{
	    _rightLoadLock.unlock();
	}

	if(_highRamLoad)
	{
	    if(!rtextures[context].size())
	    {
		for(int i = 0; i < rtiles.size(); i++)
		{
		    rtextures[context].push_back(std::vector<GLuint *>());
		    for(int j = 0; j < rtiles[i].size(); j++)
		    {
			rtextures[context][i].push_back(new GLuint);
		    }
		}
	    }

	    for(int i = 0; i < rtiles.size(); i++)
	    {
		for(int j = 0; j < rtiles[i].size(); j++)
		{
		    glGenTextures(1, rtextures[context][i][j]);
		    glBindTexture(GL_TEXTURE_2D, *(rtextures[context][i][j]));
		    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
		    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR_MIPMAP_LINEAR);
		    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		    gluBuild2DMipmaps(GL_TEXTURE_2D, 4, maxTextureSize, maxTextureSize, GL_RGB, GL_UNSIGNED_BYTE, rtiles[i][j]);
		}
	    }
	}

	return true;
    }
    else
    {
	if(_useSingleLock)
	{
	    _singleLoadLock.lock();
	}
	else
	{
	    _leftLoadLock.lock();
	}

	if(!_highRamLoad || !ltiles.size())
	{
	    osg::Image* limage =osgDB::readImageFile(lfile);
	    if(!limage)
	    {
		std::cerr << "PanoDrawable: Unable to load left eye image: " << lfile << endl;
		if(_useSingleLock)
		{
		    _singleLoadLock.unlock();
		}
		else
		{
		    _leftLoadLock.unlock();
		}
		return false;
	    }
	    if(flip)
	    {
		limage->flipVertical();
	    }
	    unsigned char * lidata = limage->data();
	    width = limage->s();
	    height = limage->t();

	    if(height % maxTextureSize != 0 || width % maxTextureSize != 0)
	    {
		std::cerr << "PanoDrawable: Image dimensions not multiple of " << maxTextureSize << endl;
		if(_useSingleLock)
		{
		    _singleLoadLock.unlock();
		}
		else
		{
		    _leftLoadLock.unlock();
		}
		return false;
	    }

	    for(int i = 0; i < height / maxTextureSize; i++)
	    {
		ltextures[context].push_back(std::vector<GLuint *>());
		for(int j = 0; j < width / maxTextureSize; j++)
		{
		    ltextures[context][i].push_back(new GLuint);
		}
	    }

	    _rcLock.lock();
	    rows = height / maxTextureSize;
	    cols = width / maxTextureSize;
	    _rcLock.unlock();

	    unsigned char * tiledata;
	    if(!_highRamLoad)
	    {
		tiledata = new unsigned char[maxTextureSize * maxTextureSize * 3];
	    }

	    int rowsize = maxTextureSize * 3;

	    for(int i = 0; i < ltextures[context].size(); i++)
	    {
		if(_highRamLoad)
		{
		    ltiles.push_back(std::vector<unsigned char *>());
		}
		for(int j = 0; j < ltextures[context][i].size(); j++)
		{
		    if(_highRamLoad)
		    {
			tiledata = new unsigned char[maxTextureSize * maxTextureSize * 3];
			ltiles[i].push_back(tiledata);
		    }
		    lidata = limage->data();
		    lidata = lidata + (i * width * maxTextureSize * 3) + (j * rowsize);
		    for(int k = 0; k < maxTextureSize; k++)
		    {
			memcpy((void*)(tiledata + (k*rowsize)), (void *)lidata, rowsize);
			lidata += width * 3;
		    }

		    if(!_highRamLoad)
		    {
			glGenTextures(1, ltextures[context][i][j]);
			glBindTexture(GL_TEXTURE_2D, *(ltextures[context][i][j]));
			glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
			glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR_MIPMAP_LINEAR);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			gluBuild2DMipmaps(GL_TEXTURE_2D, 4, maxTextureSize, maxTextureSize, GL_RGB, GL_UNSIGNED_BYTE, tiledata);
		    }
		}
	    }

	    if(!_highRamLoad)
	    {
		delete[] tiledata;
	    }
	    if(limage)
	    {
		delete[] limage->data();
		limage->unref();
	    }

	}

	if(_useSingleLock)
	{
	    _singleLoadLock.unlock();
	}
	else
	{
	    _leftLoadLock.unlock();
	}

	if(_highRamLoad)
	{
	    if(!ltextures[context].size())
	    {
		for(int i = 0; i < ltiles.size(); i++)
		{
		    ltextures[context].push_back(std::vector<GLuint *>());
		    for(int j = 0; j < ltiles[i].size(); j++)
		    {
			ltextures[context][i].push_back(new GLuint);
		    }
		}
	    }

	    for(int i = 0; i < ltiles.size(); i++)
	    {
		for(int j = 0; j < ltiles[i].size(); j++)
		{
		    glGenTextures(1, ltextures[context][i][j]);
		    glBindTexture(GL_TEXTURE_2D, *(ltextures[context][i][j]));
		    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
		    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR_MIPMAP_LINEAR);
		    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		    gluBuild2DMipmaps(GL_TEXTURE_2D, 4, maxTextureSize, maxTextureSize, GL_RGB, GL_UNSIGNED_BYTE, ltiles[i][j]);
		}
	    }
	}

	return true;
    }
}

float PanoDrawable::getRadius()
{
    return radius;
}

void PanoDrawable::setRadius(float r)
{
    radius = r;
}

int PanoDrawable::getSegmentsPerTexture()
{
    return segmentsPerTexture;
}

void PanoDrawable::setSegmentsPerTexture(int spt)
{
    segmentsPerTexture = spt;
}

int PanoDrawable::getMaxTextureSize()
{
    return maxTextureSize;
}

void PanoDrawable::setMaxTextureSize(int mts)
{
    (void)mts;
    return;
    /*if(init)
    {
        for(int i = 0; i < rtiles.size(); i++)
        {
            for(int j = 0; j < rtiles[0].size(); j++)
            {
                delete rtiles[i][j];
                glDeleteTextures(1, rtextures[i][j]);
                delete rtextures[i][j];
                if(!mono)
                {
                    delete ltiles[i][j];
                    glDeleteTextures(1, ltextures[i][j]);
                    delete ltextures[i][j];
                }
            }
        }
    }

    init = 0;

    maxTextureSize = mts;
    setImage(rfile, lfile);*/
}

void PanoDrawable::getViewAngle(float & a, float & b)
{
    a = viewanglev;
    b = viewangleh;
}

void PanoDrawable::setViewAngle(float a, float b)
{
    if(a < 10)
    {
	viewanglev = 10;
    }
    else if(a > 180)
    {
	viewanglev = 180;
    }
    else
    {
	viewanglev = a; 
    }

    if(b < 10)
    {
	viewangleh = 10;
    }
    else if(b > 360)
    {
	viewangleh = 360;
    }
    else
    {
	viewangleh = b; 
    }
}

float PanoDrawable::getCamHeight()
{
    return camHeight;
}

void PanoDrawable::setCamHeight(float h)
{
    camHeight = h;
}

void PanoDrawable::drawImplementation(RenderInfo& ri) const
{
    if(ComController::instance()->isMaster() && !_renderOnMaster)
    {
	return;
    }

    int context = ri.getContextID();

    _initLock.lock();
    if(badinit)
    {
	if(_doDelete)
	{
	    if(_contextinit[ri.getContextID()] >  0)
	    {
		for(int i = 0; i < rows; i++)
		{
		    for(int j = 0; j < cols; j++)
		    {
			if(_contextinit[ri.getContextID()] & RIGHT)
			{
			    glDeleteTextures(1, rtextures[context][i][j]);
			    delete rtextures[context][i][j];
			}
			if(_contextinit[ri.getContextID()] & LEFT)
			{
			    glDeleteTextures(1, ltextures[context][i][j]);
			    delete ltextures[context][i][j];
			}
		    }
		}
		_contextinit[ri.getContextID()] = -1;
	    }
	    bool tempb = true;
	    for(map<int, int>::iterator it = _contextinit.begin(); it != _contextinit.end(); it++)
	    {
		if(it->second > 0)
		{
		    tempb = false;
		}
	    }
	    _deleteDone = tempb;
	}
	_initLock.unlock();
	return;
    } 

    /*string host;
    int vx, vy, context;

    context = ri.getContextID();
    
    vx = (int)ri.getCurrentCamera()->getViewport()->x();
    vy = (int)ri.getCurrentCamera()->getViewport()->y();

    char hostname[51];
    gethostname(hostname, 50);
    host = hostname;*/

    int eye = 0;

    if(!getNumParents())
    {
	_initLock.unlock();
	return;
    }

    osg::Node::NodeMask nm = getParent(0)->getNodeMask();
    //std::cerr << "Node Mask: " << nm << std::endl;
    if((nm & CULL_MASK) || (nm & CULL_MASK_LEFT) )
    {
	//std::cerr << "LEFT" << std::endl;
	if(ScreenBase::getEyeSeparation() >= 0.0)
	{
	    eye = LEFT;
	}
	else
	{
	    eye = RIGHT;
	}
    }
    else
    {
	//std::cerr << "RIGHT" << std::endl;
	if(ScreenBase::getEyeSeparation() >= 0.0)
	{
	    eye = RIGHT;
	}
	else
	{
	    eye = LEFT;
	}
    }

    if(_contextinit[ri.getContextID()] >= 0)
    {
	if(!(_contextinit[ri.getContextID()] & eye))
	{
	    _initLock.unlock();
	    bool val = initTexture((PanoDrawable::eye)eye, context);
	    _initLock.lock();
	    if(val)
	    {
		_contextinit[ri.getContextID()] |= eye;
	    }
	    else
	    {
		badinit = 1;
		_initLock.unlock();
		return;
	    }
	}
    }

    if(_doDelete)
    {
	if(_contextinit[ri.getContextID()] >  0)
	{
	    for(int i = 0; i < rows; i++)
	    {
		for(int j = 0; j < cols; j++)
		{
		    if(_contextinit[ri.getContextID()] & RIGHT)
		    {
			glDeleteTextures(1, rtextures[context][i][j]);
			delete rtextures[context][i][j];
		    }
		    if(_contextinit[ri.getContextID()] & LEFT)
		    {
			glDeleteTextures(1, ltextures[context][i][j]);
			delete ltextures[context][i][j];
		    }
		}
	    }
	    _contextinit[ri.getContextID()] = -1;
	}
	bool tempb = true;
	for(map<int, int>::iterator it = _contextinit.begin(); it != _contextinit.end(); it++)
	{
	    if(it->second > 0)
	    {
		tempb = false;
	    }
	}
	_deleteDone = tempb;
	_initLock.unlock();
	return;
    }

    _initLock.unlock();

    _rcLock.lock();
    drawShape((PanoDrawable::eye)eye, context);
    _rcLock.unlock();
}

void PanoDrawable::drawShape(PanoDrawable::eye eye, int context) const
{
}
