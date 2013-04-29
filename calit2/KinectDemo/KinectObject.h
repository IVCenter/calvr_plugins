#include "CloudManager.h"
#include <osg/PolygonMode>
#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/SceneObject.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/MenuCheckbox.h>
#include "Skeleton.h"
#include <cvrKernel/PluginHelper.h>
#include <zmq.hpp>

class KinectObject : public cvr::SceneObject
{
public:
    KinectObject(std::string name, std::string cloud_server, std::string skeleton_server, osg::Vec3 position);
    CloudManager* cm;
    osg::Group* group;
    osg::Geometry* geom;
    osg::Geode* kgeode;
    osg::Switch* switchNode;
    osg::Program* pgm1;
    osg::Node* _modelFileNode1;
    osg::DrawArrays* drawArray;
    float initialPointScale;
    bool _firstRun;
    bool _cloudIsOn;

    float kinectX;
    float kinectY;
    float kinectZ;
    std::string cloudServer;
    std::string skeletonServer;

    RemoteKinect::SkeletonFrame* skel_frame;
    SubSocket<RemoteKinect::SkeletonFrame>* skel_socket;
    std::map<int, Skeleton> mapIdSkel;

    //osg::ref_ptr<osg::Vec4Array> kinectColours;
    //osg::ref_ptr<osg::Vec3Array> kinectVertices;
    osg::Vec4Array* kinectColours;
    osg::Vec3Array* kinectVertices;

    void cloudInit();
    void cloudOn();
    void cloudUpdate();
    void cloudOff();

    void skeletonOn();
    void skeletonUpdate();
    void skeletonOff();
    std::map<int, Skeleton>* skeletonGetMap();
    zmq::context_t* context;
protected:

};
