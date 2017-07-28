#ifndef NETWORK_WALL_SCENE_OBJECT
#define NETWORK_WALL_SCENE_OBJECT

#include <cvrKernel/TiledWallSceneObject.h>
#include <cvrKernel/InteractionEvent.h>
#include <cvrInput/TrackingManager.h>

class NetworkWallSceneObject : public cvr::TiledWallSceneObject
{
    public:
        NetworkWallSceneObject(std::string name, bool navigation, bool movable,
                bool clip, bool contextMenu, bool showBounds = false);
        virtual ~NetworkWallSceneObject();
        virtual bool processEvent(cvr::InteractionEvent * ie);
    
    protected:
	float _scaleIncrement;
	float _currentScale;
	float _maxScale;
	osg::Matrix _differenceMat;
	osg::Vec3 _soPoint;
};
#endif

