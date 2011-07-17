#include "CylinderDrawable.h"

#include <config/ConfigManager.h>
#include <kernel/ScreenConfig.h>
#include <kernel/ScreenBase.h>
#include <kernel/ComController.h>
#include <kernel/NodeMask.h>

#define GL_CLAMP_TO_EDGE 0x812F

using namespace osg;
using namespace std;
using namespace cvr;

std::map<int, int> CylinderDrawable::_contextinit;
std::map<int, std::vector<std::vector< GLuint * > > > CylinderDrawable::rtextures;
std::map<int, std::vector<std::vector< GLuint * > > > CylinderDrawable::ltextures;
OpenThreads::Mutex CylinderDrawable::_initLock;

CylinderDrawable::CylinderDrawable(float radius_in, float viewangle_in, float viewangleh_in, float camHeight_in, int segmentsPerTexture_in, int maxTextureSize_in)
{
    _deleteDone = false;
    _doDelete = false;
    radius = radius_in;
    viewangle = viewangle_in;
    viewangleh = viewangleh_in;
    camHeight = camHeight_in;

    currenteye = firsteye;

    badinit = 0;

    floorOffset = ConfigManager::getFloat("Plugin.PanoView360.FloorOffset", 0.0);
    _renderOnMaster = ConfigManager::getBool("Plugin.PanoView360.RenderOnMaster",false);

    if(viewangle < 10)
    {
	viewangle = 10;
    }
    if(viewangle > 170)
    {
	viewangle = 170;
    }

    if(viewangleh < 10)
    {
        viewangleh = 10;
    }
    else if(viewangleh > 360)
    {
        viewangleh = 360;
    }

    segmentsPerTexture = segmentsPerTexture_in;
    maxTextureSize = maxTextureSize_in;
    rows = cols = 0;
    init = 0;
    mono = 0;
    flip = 0;
    _maxContext = 0;
    _rotation = 0.0;
    setUseDisplayList(false);
}


CylinderDrawable::~CylinderDrawable()
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
}

void CylinderDrawable::updateRotate(float f)
{
    if(f < 0.1 && f > -0.1)
    {
        return;
    }
    float ff = f;

    if(fabs(ff) >= 10.0)
    {
        return;
    }

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

void CylinderDrawable::deleteTextures()
{
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
    else*/
    {
	_doDelete = true;
    }
}

bool CylinderDrawable::deleteDone()
{
    return _deleteDone;
}

CylinderDrawable::CylinderDrawable(const CylinderDrawable&,const osg::CopyOp&) : PanoDrawable()
{

}

void CylinderDrawable::setImage(std::string file_path)
{
    setImage(file_path, file_path);
}

void CylinderDrawable::setFlip(int f)
{
    flip = f;
}

void CylinderDrawable::setImage(std::string file_path_r, std::string file_path_l)
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


bool CylinderDrawable::initTexture(eye e, int context) const
{
    if(e == RIGHT)
    {
	osg::Image* rimage =osgDB::readImageFile(rfile);
	if(!rimage)
	{
	    std::cerr << "CylinderDrawable: Unable to load right eye image: " << rfile << endl;
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
	    std::cerr << "CylinderDrawable: Image dimensions not multiple of " << maxTextureSize << endl;
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

	rows = height / maxTextureSize;
	cols = width / maxTextureSize;

	unsigned char * tiledata = new unsigned char[maxTextureSize * maxTextureSize * 3];

	int rowsize = maxTextureSize * 3;

	for(int i = 0; i < rows; i++)
	{
	    for(int j = 0; j < cols; j++)
	    {
		ridata = rimage->data();
		ridata = ridata + (i * width * maxTextureSize * 3) + (j * rowsize);
		for(int k = 0; k < maxTextureSize; k++)
		{
		    memcpy((void*)(tiledata + (k*rowsize)), (void *)ridata, rowsize);
		    ridata += width * 3;
		}
		glGenTextures(1, rtextures[context][i][j]);
		glBindTexture(GL_TEXTURE_2D, *(rtextures[context][i][j]));
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR_MIPMAP_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		gluBuild2DMipmaps(GL_TEXTURE_2D, 4, maxTextureSize, maxTextureSize, GL_RGB, GL_UNSIGNED_BYTE, tiledata);
	    }
	}

	if(rimage)
	{
	    delete[] rimage->data();
	    rimage->unref();
	}

	delete[] tiledata;
	return true;
    }
    else
    {
	osg::Image* limage =osgDB::readImageFile(lfile);
	if(!limage)
	{
	    std::cerr << "CylinderDrawable: Unable to load left eye image: " << lfile << endl;
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
	    std::cerr << "CylinderDrawable: Image dimensions not multiple of " << maxTextureSize << endl;
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

	rows = height / maxTextureSize;
	cols = width / maxTextureSize;

	unsigned char * tiledata = new unsigned char[maxTextureSize * maxTextureSize * 3];

	int rowsize = maxTextureSize * 3;

        //cerr << "Rows: " << rows << " Cols: " << cols << endl;

        //cerr << "Width: " << width << " Height: " << height << endl;
	for(int i = 0; i < rows; i++)
	{
	    for(int j = 0; j < cols; j++)
	    {
		lidata = limage->data();
		lidata = lidata + (i * width * maxTextureSize * 3) + (j * rowsize);
		for(int k = 0; k < maxTextureSize; k++)
		{ 
                    //cerr << "k: " << k << endl;
		    memcpy((void*)(tiledata + (k*rowsize)), (void *)lidata, rowsize);
		    lidata += width * 3;
		}
		glGenTextures(1, ltextures[context][i][j]);
		glBindTexture(GL_TEXTURE_2D, *(ltextures[context][i][j]));
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR_MIPMAP_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		gluBuild2DMipmaps(GL_TEXTURE_2D, 4, maxTextureSize, maxTextureSize, GL_RGB, GL_UNSIGNED_BYTE, tiledata);
	    }
	}

	delete[] tiledata;
	if(limage)
	{
	    delete[] limage->data();
	    limage->unref();
	}
	return true;
    }
}

float CylinderDrawable::getRadius()
{
    return radius;
}

void CylinderDrawable::setRadius(float r)
{
    radius = r;
}

int CylinderDrawable::getSegmentsPerTexture()
{
    return segmentsPerTexture;
}

void CylinderDrawable::setSegmentsPerTexture(int spt)
{
    segmentsPerTexture = spt;
}

int CylinderDrawable::getMaxTextureSize()
{
    return maxTextureSize;
}

void CylinderDrawable::setMaxTextureSize(int mts)
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

void CylinderDrawable::getViewAngle(float & a, float & b)
{
    a = viewangle;
    b = viewangleh;
}

void CylinderDrawable::setViewAngle(float a, float b)
{
    if(a < 10)
    {
	viewangle = 10;
    }
    else if(a > 170)
    {
	viewangle = 170;
    }
    else
    {
	viewangle = a; 
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

float CylinderDrawable::getCamHeight()
{
    return camHeight;
}

void CylinderDrawable::setCamHeight(float h)
{
    camHeight = h;
}

void CylinderDrawable::drawImplementation(RenderInfo& ri) const
{
    if(ComController::instance()->isMaster() && !_renderOnMaster)
    {
	return;
    }

    _initLock.lock();
    if(badinit)
    {
	_initLock.unlock();
	return;
    }

    int context = ri.getContextID();

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
	    if(initTexture((CylinderDrawable::eye)eye, context))
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
	if(_contextinit[ri.getContextID()] > 0)
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

    //cerr << "in drawimplementation.\n";

    //int rows = rtiles.size();
    //int cols = rtiles[0].size();

    float radsPerTile = (2.0 * PI * (viewangleh / 360.0)) / ((float)cols);
    float radsPerSegment = radsPerTile / ((float)segmentsPerTexture);

    //float tlength = (2.0 * PI * radius) /((float)cols);
    //float theight = tlength *(((float)rows) / ((float)cols));
    float theight = tan((viewangle/360.0)* PI) * radius * 2.0;
    

    glPushAttrib(GL_ALL_ATTRIB_BITS);  

    float AdcamHeight = camHeight + floorOffset + (theight / 2.0);


    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    //glDisable(GL_DEPTH_TEST);
    //glDepthFunc(GL_ALWAYS);
    //glEnable(GL_BLEND);
    //glEnable(GL_POLYGON_SMOOTH);
    //glBlendFunc(GL_ONE, GL_ZERO);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    //glHint(GL_POINT_SMOOTH, GL_NICEST);
    //glHint(GL_LINE_SMOOTH, GL_NICEST);
    //glHint(GL_POLYGON_SMOOTH, GL_NICEST);

    //glEnable(GL_POINT_SMOOTH);
    //glEnable(GL_LINE_SMOOTH);
    //glEnable(GL_POLYGON_SMOOTH);

    //glClearColor(0.0, 0.0, 0.0, 0.0);
    //glClear(GL_COLOR_BUFFER_BIT);

    if(currenteye)
    {
	currenteye = 0;
    }
    else
    {
	currenteye = 1;
    }

    for(int i = 0; i < rows; i++)
    {
        //float rads = 0;
	float rads = _rotation;
        for(int j = 0; j < cols; j++)
        {
	    if(eye == RIGHT || (ScreenConfig::instance()->getEyeSeparationMultiplier() == 0.0))
            {
		if(eye == LEFT && !(_contextinit[ri.getContextID()] == BOTH))
		{
		    glPopAttrib();
		    return;
		}
                glBindTexture(GL_TEXTURE_2D, *(rtextures[context][i][j]));
            }
            else
            {
                glBindTexture(GL_TEXTURE_2D, *(ltextures[context][i][j]));
            }
            glBegin(GL_QUADS);
            for(int k = 0; k < segmentsPerTexture; k++)
            {
                //glTexCoord2f(((float)k) / ((float)segmentsPerTexture), ((float)(rows - (i + 1))) / ((float)rows)); 
                glTexCoord2f(((float)k) / ((float)segmentsPerTexture), 0.0);
                glVertex3f(-(cos(rads)*radius), sin(rads)*radius,  ((((float)(i)) / ((float)rows))*theight) - AdcamHeight);	// Bottom Left Of The Texture and Quad
                //glTexCoord2f(((float)k+1.0) / ((float)segmentsPerTexture), ((float)(rows - (i + 1))) / ((float)rows));
                glTexCoord2f(((float)k+1.0) / ((float)segmentsPerTexture), 0.0); 
                glVertex3f(-(cos(rads+radsPerSegment)*radius), sin(rads+radsPerSegment)*radius,  ((((float)(i)) / ((float)rows))*theight) - AdcamHeight);	// Bottom Right Of The Texture and Quad
                //glTexCoord2f(((float)k+1.0) / ((float)segmentsPerTexture), ((float)(rows - (i))) / ((float)rows)); 
                glTexCoord2f(((float)k+1.0) / ((float)segmentsPerTexture), 1.0);
                glVertex3f(-(cos(rads+radsPerSegment)*radius), sin(rads+radsPerSegment)*radius,  ((((float)(i+1.0)) / ((float)rows))*theight) - AdcamHeight);	// Top Right Of The Texture and Quad
                //glTexCoord2f(((float)k) / ((float)segmentsPerTexture), ((float)(rows - (i))) / ((float)rows)); 
                glTexCoord2f(((float)k) / ((float)segmentsPerTexture), 1.0);
                glVertex3f(-(cos(rads)*radius), sin(rads)*radius,  ((((float)(i+1.0)) / ((float)rows))*theight) - AdcamHeight);	// Top Left Of The Texture and Quad
                rads += radsPerSegment;
            }
            glEnd();
        }
    }
    glPopAttrib();
}
