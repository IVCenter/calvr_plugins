/**
 * @file PluginMessageType.h
 *
 * File that contains the type values used for message passing in
 * Plugins.
 */

#ifndef CALVR_PLUGIN_MESSAGE_TYPE_H
#define CALVR_PLUGIN_MESSAGE_TYPE_H

#include <osg/MatrixTransform>

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

#endif
