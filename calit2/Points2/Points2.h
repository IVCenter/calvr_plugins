#ifndef _POINTS_H
#define _POINTS_H

#include <queue>
#include <vector>

// CVR
#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/ScreenBase.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/SceneObject.h>
#include <cvrKernel/Navigation.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrMenu/MenuSystem.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrKernel/CVRViewer.h>
#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/FileHandler.h>

// OSG
#include <osg/Group>
#include <osg/Vec3>
#include <osg/Uniform>
#include <osgDB/ReadFile>
#include <osgDB/FileUtils>
#include <OpenThreads/Mutex>

//osgCompute
#include <iostream>
#include <sstream>
#include <osg/ArgumentParser>
#include <osg/Texture2D>
#include <osg/Viewport>
#include <osg/AlphaFunc>
#include <osg/PolygonMode>
#include <osg/Geometry>
#include <osg/Point>
#include <osg/Array>
#include <osg/PointSprite>
#include <osg/Geometry>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/FileUtils>
#include <osgDB/Registry>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>
/*
#include <osgCuda/Computation>
#include <osgCuda/Memory>
#include <osgCuda/Geometry>

#include "PtclMover.h"
#include "PtclEmitter.h"
//..........................
//CUDA
#include "CudaParticle.h"
#include "CudaHelper.h"
*/
class Points2 : public cvr::CVRPlugin, public cvr::MenuCallback, public cvr::FileLoadCallback, public cvr::PerContextCallback
{
  protected:
    osg::ref_ptr<osg::Program> pgm1;
    float initialPointScale;
 
    // container to hold pdf data
    struct PointObject
    {
       std::string name;
       cvr::SceneObject* scene;
       osg::Geode* points;
       osg::Uniform* pointScale;
       osg::Uniform* objectScale;
    }; 

    void readXYZ(std::string& filename, osg::Vec3Array* points, osg::Vec4Array* colors);
    void readXYB(std::string& filename, osg::Vec3Array* points, osg::Vec4Array* colors);

    // context map
    std::map<struct PointObject*,cvr::MenuRangeValue*> _sliderMap;
    std::map<struct PointObject*,cvr::MenuButton*> _deleteMap;
    std::vector<struct PointObject*> _loadedPoints;

    osg::Uniform* objectScale;

  public:
    Points2();
    virtual ~Points2();
    bool init();
    virtual bool loadFile(std::string file);
    bool loadFile(std::string file, osg::Group * grp);
    void menuCallback(cvr::MenuItem * item);
    void preFrame();
    void message(int type, char *&data, bool collaborative=false);
    void initParticles();
    osg::Geode* getBoundingBox();
/*
//osgCompute
osg::Geode* getGeode();
osg::ref_ptr<osgCompute::Computation> getComputation();
osg::ref_ptr<osgCompute::ResourceVisitor> getVisitor( osg::FrameStamp* fs );
//CudaTest
	virtual void perContextCallback(int contextid, PerContextCallback::PCCType type) const;
int max_age;
float gravity;
float anim;
int disappear_age;
int showFrameNo;
int lastShowFrameNo;
int showStartTime;
int showTime;
int lastShowTime;
int startTime;
int nowTime;
int frNum;
int colorFreq;
float alphaControl;
int draw_water_sky;
int contextid;
        float h_injectorData[INJT_DATA_MUNB][INJT_DATA_ROWS][INJT_DATA_ROW_ELEM];
        float h_reflectorData[REFL_DATA_MUNB][REFL_DATA_ROWS][REFL_DATA_ROW_ELEM];
        float * h_particleData;
        float * h_debugData;
        osg::ref_ptr<osg::Geometry> _particleGeo;
        osg::ref_ptr<osg::Geode> _particleGeode;
        osg::ref_ptr<osg::Vec3Array> _positionArray;
        osg::ref_ptr<osg::Vec4Array> _colorArray;
        osg::ref_ptr<osg::DrawArrays> _primitive;
        mutable std::map<int,bool> _cudaContextSet;
        mutable OpenThreads::Mutex _callbackLock;
        mutable std::map<int,bool> _callbackInit;
        bool _callbackAdded;
        bool _callbackActive;
        mutable std::map<int,CUdeviceptr> d_debugDataMap;
        mutable std::map<int,CUdeviceptr> d_particleDataMap;
*/
};
#endif
