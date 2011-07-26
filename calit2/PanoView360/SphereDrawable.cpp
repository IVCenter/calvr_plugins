#include "SphereDrawable.h"

#include <config/ConfigManager.h>
#include <kernel/ScreenConfig.h>
#include <kernel/ScreenBase.h>
#include <kernel/ComController.h>
#include <kernel/NodeMask.h>

#define GL_CLAMP_TO_EDGE 0x812F

using namespace osg;
using namespace std;
using namespace cvr;

std::map<int, int> SphereDrawable::_contextinit;
std::map<int, std::vector<std::vector< GLuint * > > > SphereDrawable::rtextures;
std::map<int, std::vector<std::vector< GLuint * > > > SphereDrawable::ltextures;
OpenThreads::Mutex SphereDrawable::_initLock;
bool SphereDrawable::_deleteDone = false;

SphereDrawable::SphereDrawable(float radius_in, float viewanglev_in, float viewangleh_in, float camHeight_in, int segmentsPerTexture_in, int maxTextureSize_in)
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


SphereDrawable::~SphereDrawable()
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

void SphereDrawable::updateRotate(float f)
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

void SphereDrawable::deleteTextures()
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
    else
    {*/
	_doDelete = true;
    //}
}

bool SphereDrawable::deleteDone()
{
    return _deleteDone;
}

SphereDrawable::SphereDrawable(const SphereDrawable&,const osg::CopyOp&) : PanoDrawable()
{

}

void SphereDrawable::setImage(std::string file_path)
{
    setImage(file_path, file_path);
}

void SphereDrawable::setFlip(int f)
{
    flip = f;
}

void SphereDrawable::setImage(std::string file_path_r, std::string file_path_l)
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


bool SphereDrawable::initTexture(eye e, int context) const
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

	for(int i = 0; i < rows; i++)
	{
	    for(int j = 0; j < cols; j++)
	    {
		lidata = limage->data();
		lidata = lidata + (i * width * maxTextureSize * 3) + (j * rowsize);
		for(int k = 0; k < maxTextureSize; k++)
		{
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

float SphereDrawable::getRadius()
{
    return radius;
}

void SphereDrawable::setRadius(float r)
{
    radius = r;
}

int SphereDrawable::getSegmentsPerTexture()
{
    return segmentsPerTexture;
}

void SphereDrawable::setSegmentsPerTexture(int spt)
{
    segmentsPerTexture = spt;
}

int SphereDrawable::getMaxTextureSize()
{
    return maxTextureSize;
}

void SphereDrawable::setMaxTextureSize(int mts)
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

void SphereDrawable::getViewAngle(float & a, float & b)
{
    a = viewanglev;
    b = viewangleh;
}

void SphereDrawable::setViewAngle(float a, float b)
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

float SphereDrawable::getCamHeight()
{
    return camHeight;
}

void SphereDrawable::setCamHeight(float h)
{
    camHeight = h;
}

void SphereDrawable::drawImplementation(RenderInfo& ri) const
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

    /*for(int i = 0; i < (_eyeMap[host][context]).size(); i++)
    {
	if(_eyeMap[host][context][i].first.first == vx && _eyeMap[host][context][i].first.second == vy)
	{
	    eye = _eyeMap[host][context][i].second;
	}
    }

    if(eye == 0)
    {
	cerr << "Unable to determine eye for host: " << host << " context: " << context << " vx: " << vx << " vy: " << vy << endl;
	badinit = 1;
	_initLock.unlock();
	return;
    }*/

    if(_contextinit[ri.getContextID()] >= 0)
    {
	if(!(_contextinit[ri.getContextID()] & eye))
	{
	    if(initTexture((SphereDrawable::eye)eye, context))
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

    float radsPerTileH = (viewangleh * (M_PI / 180.0f)) / ((float)cols);
    float radsPerTileV = (viewanglev * (M_PI / 180.0f)) / ((float)rows);
    float radsPerSegmentH = radsPerTileH / ((float)segmentsPerTexture);
    float radsPerSegmentV = radsPerTileV / ((float)segmentsPerTexture);

    //float theight = tan((viewangle/360.0)* PI) * radius * 2.0;

    glPushAttrib(GL_ALL_ATTRIB_BITS);  

    float AdcamHeight = camHeight + floorOffset;


    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);

    float radsv = ((180.0f - viewanglev) / 2.0f) * (M_PI / 180.0f); 
    for(int i = rows - 1; i >= 0; i--)
    {
        float radsh = _rotation;
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
	    float temprads = radsv;
            for(int k = 0; k < segmentsPerTexture; k++)
            {
		radsv = temprads;
		for(int m = 0; m < segmentsPerTexture; m++)
		{
                    float h = radsh;
                    float hpo = radsh + radsPerSegmentH;
                    if(h > (2.0f * M_PI))
                    {
                        h -= (2.0f * M_PI);
                    }
                    if(hpo > (2.0f * M_PI))
                    {
                        hpo -= (2.0f * M_PI);
                    }
		    glTexCoord2f(((float)k) / ((float)segmentsPerTexture), (1.0f - ((float)(m+1)) / ((float)segmentsPerTexture)));
                    //glTexCoord2f((1.0f - ((float)(m+1)) / ((float)segmentsPerTexture)), ((float)k) / ((float)segmentsPerTexture));
		    glVertex3f(radius * cos(-h) * sin(radsv + radsPerSegmentV), radius * sin(-h) * sin(radsv + radsPerSegmentV), (radius * cos(radsv + radsPerSegmentV)) - AdcamHeight);	// Bottom Left Of The Texture and Quad
		    glTexCoord2f(((float)k+1.0) / ((float)segmentsPerTexture), (1.0f - ((float)(m+1)) / ((float)segmentsPerTexture))); 
                    //glTexCoord2f((1.0f - ((float)(m+1)) / ((float)segmentsPerTexture)), ((float)k+1.0) / ((float)segmentsPerTexture));
		    glVertex3f(radius * cos(-hpo) * sin(radsv + radsPerSegmentV), radius * sin(-hpo) * sin(radsv + radsPerSegmentV), (radius * cos(radsv + radsPerSegmentV)) - AdcamHeight);	// Bottom Right Of The Texture and Quad
		    glTexCoord2f(((float)k+1.0) / ((float)segmentsPerTexture), (1.0f - ((float)m) / ((float)segmentsPerTexture)));
                    //glTexCoord2f((1.0f - ((float)m) / ((float)segmentsPerTexture)), ((float)k+1.0) / ((float)segmentsPerTexture));
		    glVertex3f(radius * cos(-hpo) * sin(radsv), radius * sin(-hpo) * sin(radsv), (radius * cos(radsv)) - AdcamHeight);	// Top Right Of The Texture and Quad
		    glTexCoord2f(((float)k) / ((float)segmentsPerTexture), (1.0f - ((float)m) / ((float)segmentsPerTexture)));
                    //glTexCoord2f((1.0f - ((float)m) / ((float)segmentsPerTexture)), ((float)k) / ((float)segmentsPerTexture));
		    glVertex3f(radius * cos(-h) * sin(radsv), radius * sin(-h) * sin(radsv), (radius * cos(radsv)) - AdcamHeight);	// Top Left Of The Texture and Quad
		    radsv += radsPerSegmentV;
		}
		radsh += radsPerSegmentH;
            }
            glEnd();
	    radsv = temprads;
        }
	radsv += radsPerTileV;
    }
    glPopAttrib();
}
