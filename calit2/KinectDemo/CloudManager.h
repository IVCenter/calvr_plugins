#ifndef _ANIMATION_MANAGER_MULT
#define _ANIMATION_MANAGER_MULT

#include <osgDB/ReadFile>
#include <osgDB/FileUtils>
#include <osgUtil/DelaunayTriangulator>
#include <cvrKernel/SceneManager.h>
#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuRangeValue.h>

#include <shared/PubSub.h>
#include <protocol/skeletonframe.pb.h>
#include <protocol/colormap.pb.h>
#include <protocol/depthmap.pb.h>
#include <protocol/pointcloud.pb.h>
#include <zmq.hpp>
#include "kUtils.h"
//#include "Skeleton.h"
#include <unordered_map>

#include <iostream>
#include <string>
#include <map>

#include <OpenThreads/Thread>


//  zmq::context_t contextCloud(1);

class CloudManager : public OpenThreads::Thread {

public:

    CloudManager(std::string server);
    ~CloudManager();
    //contextCloud(1);
    SubSocket<RemoteKinect::PointCloud>* cloudT_socket;
    void update();
    bool isCacheDone();
    RemoteKinect::PointCloud* packet;

    virtual void run();
    void quit();
    int firstRunStatus();
    osg::ref_ptr<osg::Vec4Array> kinectColours;
    osg::ref_ptr<osg::Vec3Array> kinectVertices;
    osg::ref_ptr<osg::Vec3Array> kinectNormals;
    osg::ref_ptr<osg::Geometry> tnodeGeom;
    osg::ref_ptr<osg::Vec4Array> user1Colours;
    osg::ref_ptr<osg::Vec3Array> user1Vertices;
    osg::ref_ptr<osg::Vec4Array> newColours;
    osg::ref_ptr<osg::Vec3Array> newVertices;
    osg::ref_ptr<osg::Vec3Array> newNormals;
    std::vector<int> idArray;
    std::vector<osg::ref_ptr<osg::Vec3Array> > userVerticesArray;
    std::vector<osg::ref_ptr<osg::Vec4Array> > userColoursArray;
    std::vector<osg::ref_ptr<osg::Vec3Array> > lHandVerticesArray;
    std::vector<osg::ref_ptr<osg::Vec4Array> > lHandColoursArray;

    std::vector<osg::ref_ptr<osg::DrawArrays> > drawArrays;
    std::vector<osg::ref_ptr<osg::DrawArrays> > drawArraysHand;
    bool useKColor;
    bool userColor;
    bool should_quit;
    int max_users;
    std::vector<float> userRadius;
    float radiusMin;
    std::vector<int> userSize;
    std::vector<bool> userOn;
    std::vector<int> lastUserSize;
    std::vector<int> lHandSize;
    std::vector<int> lastlHandSize;
protected:

    bool _cacheDone;
    bool _next;
    bool pause;
    int _firstRun;
    int minDistHSV, maxDistHSV;
    int minDistHSVDepth, maxDistHSVDepth;
    std::unordered_map<float, osg::Vec4f> distanceColorMap;
    std::unordered_map<float, osg::Vec4f> userColorMap;
    osg::Vec4f getColorRGB(int dist);
    osg::Vec4f getColorUser(int uid, int dist);
    void processNewCloud();
    void reduceUsersArray();
    void reduceHandArray();
    void updateUserVertices(int id, int j, osg::Vec3 ppos);
    void updateUserColours(int id, int j, osg::Vec4 tempColour);
    void updateHandVertices(int id, int j, int hand, osg::Vec3 ppos);
    void updateHandColours(int id, int j, int hand,  osg::Vec4 tempColour);
    void triangulatePointCloud();
    void generateHandArray();
    std::string kinectServer;
};

#endif
