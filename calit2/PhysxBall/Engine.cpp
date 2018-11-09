#include "Engine.h"
#include <osg/io_utils>
//#include "PhysicsUtil.h"
#include <algorithm>
#include <iostream>

using namespace osgPhysx;
using namespace physx;

PxReal Engine::_defaultTimestep = 1.0f/60.0f;

static physx::PxDefaultErrorCallback _defaultErrorCallback;
static physx::PxDefaultAllocator _defaultAllocatorCallback;
static physx::PxSimulationFilterShader _defaultFilterShader = physx::PxDefaultSimulationFilterShader;

Engine* Engine::instance()
{
    static osg::ref_ptr<Engine> s_registry = new Engine;
    return s_registry.get();
}

Engine::Engine()
{
    _physicsSDK = nullptr;
    _cooking = nullptr;
}

Engine::~Engine()
{
    clear();
    PxCloseExtensions();
    _physicsSDK->release();
    if ( _cooking ) _cooking->release();
}

bool Engine::init()
{
    PxFoundation* foundation = nullptr;
    foundation = PxCreateFoundation( PX_PHYSICS_VERSION, _defaultAllocatorCallback, _defaultErrorCallback );
    if ( !foundation )
    {
        OSG_WARN << "Unable to initialize PhysX foundation." << std::endl;
        return false;
    }

    _defaultToleranceScale.length = 50;//units in cm, 50cm
    _defaultToleranceScale.speed = 981;

    _physicsSDK = PxCreatePhysics( PX_PHYSICS_VERSION, *foundation, PxTolerancesScale() );
    if ( !_physicsSDK ) 
    {
        OSG_WARN << "Unable to initialize PhysX SDK." << std::endl;
        return false;
    }
#if(PX_PHYSICS_VERSION >= 34)
    // PX_C_EXPORT bool PX_CALL_CONV 	PxInitExtensions (physx::PxPhysics &physics, physx::PxPvd *pvd) since 3.4
    if (!PxInitExtensions(*_physicsSDK, nullptr)) {
        return false;
    }
#else
    if (!PxInitExtensions(*mPhysics)){
        return false;
    }
}
#endif
    return true;
}


bool Engine::addScene( const std::string& name, PxScene* s )
{
    if ( !s || _sceneMap.find(name)!=_sceneMap.end() ) return false;
    _sceneMap[name] = s;
    return true;
}

bool Engine::removeScene( const std::string& name, bool doRelease )
{
    SceneMap::iterator itr = _sceneMap.find( name );
    if ( itr==_sceneMap.end() ) return false;
    
    if ( doRelease )
    {
        releaseActors( itr->second );
        itr->second->release();
    }
    _sceneMap.erase( itr );
    return true;
}

PxScene* Engine::getScene( const std::string& name )
{
    SceneMap::iterator itr = _sceneMap.find( name );
    if ( itr==_sceneMap.end() ) return NULL;
    return itr->second;
}

const PxScene* Engine::getScene( const std::string& name ) const
{
    SceneMap::const_iterator itr = _sceneMap.find( name );
    if ( itr==_sceneMap.end() ) return NULL;
    return itr->second;
}

bool Engine::addActor( const std::string& s, PxActor* actor )
{
    PxScene* scene = getScene(s);
    if ( !scene || !actor ) return false;
    scene->addActor( *actor );
    _actorMap[scene].push_back( actor );
    return true;
}
PxActor * Engine::getActorAt(const std::string & s, int loc){
    PxScene* scene = getScene(s);

    ActorMap::iterator itr = _actorMap.find( scene );
    if ( itr==_actorMap.end() ) return nullptr;
    return itr->second[loc];
}
bool Engine::removeActor( const std::string& s, PxActor* actor )
{
    PxScene* scene = getScene(s);
    if ( !scene || !actor ) return false;
    
    ActorMap::iterator itr = _actorMap.find( scene );
    if ( itr==_actorMap.end() ) return false;
    
    ActorList& actors = itr->second;
    ActorList::iterator fitr = std::find( actors.begin(), actors.end(), actor );
    if ( fitr==actors.end() ) return false;
    
    scene->removeActor( *actor );
    actors.erase( fitr );
    if ( !actors.size() ) _actorMap.erase( itr );
    return true;
}

PxCooking* Engine::getOrCreateCooking( PxCookingParams* params, bool forceCreating )
{
    if ( forceCreating && _cooking )
    {
        _cooking->release();
        _cooking = NULL;
    }
    
    if ( !_cooking )
    {
        if ( params )
            _cooking = PxCreateCooking( PX_PHYSICS_VERSION, _physicsSDK->getFoundation(), *params );
        else
        {
            physx::PxTolerancesScale sc;
            physx::PxCookingParams defParams( sc );
            _cooking = PxCreateCooking( PX_PHYSICS_VERSION, _physicsSDK->getFoundation(), defParams );
        }
    }
    return _cooking;
}

void Engine::update( double step )
{
    for ( SceneMap::iterator itr=_sceneMap.begin(); itr!=_sceneMap.end(); ++itr )
    {
        PxScene* scene = itr->second;
        scene->simulate( step );
        while( !scene->fetchResults() ) { /* do nothing but wait */ }
    }
}

void Engine::clear()
{
    for ( SceneMap::iterator itr=_sceneMap.begin(); itr!=_sceneMap.end(); ++itr )
    {
        PxScene* scene = itr->second;
        releaseActors( scene );
        scene->release();
    }
    _sceneMap.clear();
    _actorMap.clear();
}

void Engine::releaseActors( PxScene* scene )
{
    ActorMap::iterator itr = _actorMap.find( scene );
    if ( itr==_actorMap.end() ) return;

    ActorList& actors = itr->second;
    for ( unsigned int i=0; i<actors.size(); ++i )
    {
        scene->removeActor( *(actors[i]) );
    }
    _actorMap.erase( itr );
}

bool Engine::addScene(const std::string& name) {
    PxSceneDesc * _sceneDesc = new PxSceneDesc(_defaultToleranceScale);
    _sceneDesc->gravity = PxVec3(.0f, .0f, -9.81f);

    if(!_sceneDesc->cpuDispatcher)
    {
        PxDefaultCpuDispatcher* mCpuDispatcher = PxDefaultCpuDispatcherCreate(1);
        if(!mCpuDispatcher) return false;
        _sceneDesc->cpuDispatcher = mCpuDispatcher;
    }
    if(!_sceneDesc->filterShader)
        _sceneDesc->filterShader  = _defaultFilterShader;

    PxScene * scene = _physicsSDK->createScene(*_sceneDesc);
    if(!scene) return false;

    scene->setVisualizationParameter(PxVisualizationParameter::eSCALE, 1.0);
    scene->setVisualizationParameter(PxVisualizationParameter::eCOLLISION_SHAPES, 1.0f);
    return addScene(name, scene);
}