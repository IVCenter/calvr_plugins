#ifndef PHYSICS_ENGINE
#define PHYSICS_ENGINE

#include <osg/Referenced>
#include <PxPhysicsAPI.h>
#include <extensions/PxExtensionsAPI.h>
#include <vector>
#include <map>

namespace osgPhysx
{

/** The engine instance to be used globally */
class Engine : public osg::Referenced
{
public:
    static Engine* instance();
    bool init();
    physx::PxPhysics* getPhysicsSDK() { return _physicsSDK; }

    /** Add scene to the engine */
    bool addScene(const std::string& name);
    bool addScene( const std::string& name, physx::PxScene* s );
    bool removeScene( const std::string& name, bool doRelease );
    
    physx::PxScene* getScene( const std::string& name );
    const physx::PxScene* getScene( const std::string& name ) const;
    
    typedef std::map<std::string, physx::PxScene*> SceneMap;
    SceneMap& getSceneMap() { return _sceneMap; }
    const SceneMap& getSceneMap() const { return _sceneMap; }
    
    /** Add actor object to specified scene */
    bool addActor( const std::string& scene, physx::PxActor* actor );
//    bool addActor( const std::string& scene, physx::PxRigidActor* actor, const physx::PxFilterData& filter );
//    bool addActor( const std::string& scene, physx::PxParticleBase* ps, const physx::PxFilterData& filter );
    bool removeActor( const std::string& scene, physx::PxActor* actor );
    
    typedef std::vector<physx::PxActor*> ActorList;
    typedef std::map<physx::PxScene*, ActorList> ActorMap;
    ActorMap& getAllActors() { return _actorMap; }
    physx::PxActor * getActorAt(const std::string & scene, int loc);

    /** Get or create a new cooking object */
    physx::PxCooking* getOrCreateCooking( physx::PxCookingParams* params=0, bool forceCreating=false );

    /** Update the physics system every frame */
    void update( double step =  _defaultTimestep);
    
    /** Clear all saved data */
    void clear();
    
protected:
    Engine();
    virtual ~Engine();
    
    void releaseActors( physx::PxScene* scene );
    
    SceneMap _sceneMap;
    ActorMap _actorMap;

    physx::PxTolerancesScale _defaultToleranceScale;

    static physx::PxReal _defaultTimestep;
    physx::PxPhysics* _physicsSDK;

    physx::PxCooking* _cooking;
};

}

#endif
