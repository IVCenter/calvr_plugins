/**
 * @file PluginMessageType.h
 *
 * File that contains the type values used for message passing in
 * Plugins.
 */

#ifndef CALVR_PLUGIN_MESSAGE_TYPE_H
#define CALVR_PLUGIN_MESSAGE_TYPE_H

#include <cvrKernel/SceneObject.h>

#include <osg/MatrixTransform>

#include <string>

// PathRecorder
enum PathRecorderMessageType
{
    PR_SET_RECORD = 0,
    PR_SET_PLAYBACK,
    PR_SET_ACTIVE_ID,
    PR_SET_PLAYBACK_SPEED,
    PR_START,
    PR_PAUSE,
    PR_STOP,
    PR_GET_TIME,
    PR_GET_START_MAT,
    PR_GET_START_SCALE,
    PR_IS_STOPPED
};

// MenuBasic
enum MenuMessageType
{
    MB_HEAD_TRACKING = 0,
    MB_STEREO
};

// OsgEarth
enum OsgEarthMessageType
{
    OE_ADD_MODEL = 0,
    OE_TRANSFORM_POINTER
};

// OsgEarth message struct (lat (degrees), lon (degrees), height (meters above surface)
// and PluginName (name of plugin sending message)
struct OsgEarthRequest
{
    float lat;
    float lon;
    float height;
    char pluginName[4096];
    osg::MatrixTransform* trans;
};

enum PointsMessageType
{
    POINTS_LOAD_REQUEST=0
};

struct PointsLoadInfo
{
    std::string file;
    osg::ref_ptr<osg::Group> group;
    bool useShader;
};

enum PanoViewLODMessageType
{
    PAN_LOAD_REQUEST=0,
    PAN_HEIGHT_REQUEST,
    PAN_SET_ALPHA,
    PAN_UNLOAD
};

struct PanLoadRequest
{
    std::string name;
    float rotationOffset;
    std::string plugin;
    int pluginMessageType;
    bool loaded;
};

struct PanHeightRequest
{
    std::string name;
    float height;
};

enum PointsWithPanMessageType
{
    PWP_PAN_UNLOADED=0
};

enum OsgPdfMessageType
{
    PDF_LOAD_REQUEST=0
};

struct OsgPdfLoadRequest
{
    osg::Matrixd transform;
    float width;
    std::string path;
    bool loaded;
    bool tiledWallObject;
    cvr::SceneObject * object;
};

// ModelLoader
 
enum ModelLoaderMessageType
{
    ML_LOAD_REQUEST=0,
    ML_REMOVE_ALL
};

struct ModelLoaderLoadRequest
{
    char fileLabel[1024];
    osg::Matrixd transform;
};

// StructView

enum StructViewMessageType
{
    SV_ENABLE=0,
    SV_DISABLE,
    SV_LAYER_ON,
    SV_LAYER_OFF
};

#endif
