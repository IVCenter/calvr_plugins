/***************************************************************
* File Name: CAVEGeode.h
*
* Class Name: CAVEGeode
*
***************************************************************/

#ifndef _CAVE_GEODE_H_
#define _CAVE_GEODE_H_


// C++
#include <iostream>
#include <list>
#include <string>

// Open scene graph
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/StateSet>
#include <osg/Texture2D>

#include <osgDB/ReadFile>


/***************************************************************
* Class: CAVEGeode
***************************************************************/
class CAVEGeode: public osg::Geode
{
  public:
    CAVEGeode();
    ~CAVEGeode();

    /* change color, texture and transparency values for all geode types */
    void applyColorTexture( const osg::Vec3 &diffuse, const osg::Vec3 &specular, const float &alpha,
			    const std::string &texFilename);
    void applyAlpha(const float &alpha);

    void applyAudioInfo(const std::string &audioInfo) { mAudioInfo = audioInfo; }
    const std::string &getAudioInfo() { return mAudioInfo; }

    static const std::string getDataDir();

  protected:

    /* color & texture properties */
    float mAlpha;
    osg::Vec3 mDiffuse, mSpecular;
    std::string mTexFilename;
    std::string mAudioInfo;

    /* virtual function definition for device handlers */
    virtual void movedon() = 0;
    virtual void movedoff() = 0;
    virtual void pressed() = 0;
    virtual void released() = 0;
};


#endif





