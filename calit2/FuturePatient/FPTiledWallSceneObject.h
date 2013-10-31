#ifndef FP_TILED_WALL_SCENE_OBJECT
#define FP_TILED_WALL_SCENE_OBJECT

#include <cvrKernel/SceneObject.h>

class FPTiledWallSceneObject : public cvr::SceneObject
{
    public:
        FPTiledWallSceneObject(std::string name, bool navigation, bool movable, bool clip,
                bool contextMenu, bool showBounds = false);
        virtual ~FPTiledWallSceneObject();

        void setTiledWallMovement(bool b);
        bool getTiledWallMovement()
        {
            return _tiledWallMovement;
        }

        virtual bool processEvent(cvr::InteractionEvent * ie);
    protected:
        virtual void moveCleanup();

        bool _tiledWallMovement;
        osg::Vec3 _movePoint;
};

#endif
