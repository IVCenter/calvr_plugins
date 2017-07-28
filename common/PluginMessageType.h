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
    OE_TRANSFORM_POINTER,
    OE_MENU
};

// OsgEarth message struct (lat (degrees), lon (degrees), height (meters above surface)
// and PluginName (name of plugin sending message)
struct OsgEarthRequest
{
    float lat;
    float lon;
    float height;
    std::string plugin;
    osg::MatrixTransform* trans;
};

struct OsgEarthMenuRequest
{
    std::string plugin;
    cvr::SubMenu* oe_menu;
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
    PAN_IMAGE_LOAD_REQUEST,
    PAN_HEIGHT_REQUEST,
    PAN_SET_ALPHA,
    PAN_SET_ROTATE,
    PAN_SET_TILT,
    PAN_SET_DEMOMODE,
    PAN_NEXT,
    PAN_UNLOAD
};

struct PanLoadRequest
{
    std::string name;
    float rotationOffset;
    bool enableDemoMode;
    std::string plugin;
    int pluginMessageType;
    bool enableRemoval;
    // to load images that arent part of the menu
    std::string leftFile;
    std::string rightFile;
    bool loaded;
};

struct PanHeightRequest
{
    std::string name;
    float height;
};

enum CatalystMessageType
{
    CAT_PAN_UNLOADED=0
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

// OsgVnc
enum OsgVncMessageType
{
    VNC_GOOGLE_QUERY=0,
    VNC_HIDE,
    VNC_SCALE,
    VNC_POSITION
};

struct OsgVncRequest
{
    std::string query;
    bool hide;
    float scale;
    osg::Vec3f position;
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

// LayoutManager

class Layout
{
public:
    virtual std::string Name(void) = 0;
    virtual void Cleanup(void) = 0;
    virtual bool Start(void) = 0;
    virtual bool Update(void) = 0;
};

struct LayoutManagerAddLayoutData
{
    Layout* layout;
};

enum LayoutManagerMessageType
{
    LM_ADD_LAYOUT = 0,
    LM_START_LAYOUT
};

// Video
enum VideoMessageType
{
    VIDEO_LOAD,
    VIDEO_STREAM,
    VIDEO_STOP
};


struct VideoSceneObject : public cvr::SceneObject
{
    virtual void play() = 0;
    virtual void stop() = 0;
    
    VideoSceneObject(
        std::string name, bool navigation, bool movable, bool clip, 
        bool contextMenu, bool showBounds) : cvr::SceneObject(
            name, navigation, movable, clip, contextMenu, showBounds) {}
            
    virtual ~VideoSceneObject() {}
};

struct VideoMessageData
{
    // why == VIDEO_LOAD:
    //  - input: path (path to video file to load)
    //  - output: obj (allocated by plugin)
    
    // why == VIDEO_STOP:
    //  - input: obj (as provided by an earlier VIDEO_LOAD)
    //  - unused: path
    
    VideoMessageType why;
    std::string path;   
    VideoSceneObject* obj;
    int width, height;
    unsigned int gid;
};

enum BerkeleyMessageType
{
    BERK_USER=0,
    BERK_NO_USER,
    BERK_NAV
};

struct BerkeleyRequest
{
   int user;
   float rx;
   float ry;
   float my;
};


#endif

