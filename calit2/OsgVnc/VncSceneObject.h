#ifndef VNC_SCENE_OBJECT_H
#define VNC_SCENE_OBJECT_H

#include <cvrKernel/TiledWallSceneObject.h>
#include <cvrKernel/InteractionManager.h>
#include <osgWidget/VncClient>

class VncSceneObject : public cvr::TiledWallSceneObject
{
    public:
        VncSceneObject(std::string name, osgWidget::VncClient * client, bool vncEvents, bool navigation, bool movable, bool clip,
                bool contextMenu, bool showBounds = false);
        ~VncSceneObject();
        
        virtual bool processEvent(cvr::InteractionEvent * ie);
        virtual void updateCallback(int handID, const osg::Matrix & mat);

    protected:
        bool _active;           // whether the vnc window is currently intersected
        bool _vncEvents;        // where vncEvents should be sent back the server
        osg::Vec3 _intersect;   // intersection point (translated into screen co-ordinates)
        float _width;           // width of vnc window texture
        float _height;          // height of vnc window texture
        float _windowScale;           // the scale used to compute window co-ordinates

        osgWidget::VncClient * _client;
        osgWidget::VncImage * _image;   // image to forward events too
        osg::BoundingBox _bound;        // bound of the vnc window

        VncSceneObject(std::string name, bool navigation, bool movable, bool clip,
                        bool contextMenu, bool showBounds);
};
#endif
