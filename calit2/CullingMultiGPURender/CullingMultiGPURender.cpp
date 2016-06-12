/**
 * @file CullingMultiGPURender.cpp
 * Driving class of our algorithm.
 *
 * Implementation of the plugin class for our graphics environment. 
 *
 * @author Andrew Prudhomme (aprudhomme@ucsd.edu)
 *
 * @date 09/15/2010
 */

#include "CullingMultiGPURender.h"
#include "CustomStatsHandler.h"

#include <cvrKernel/SceneManager.h>
#include <cvrKernel/CVRViewer.h>

#include <cvrConfig/ConfigManager.h>

#include <cvrInput/TrackingManager.h>

#include <osgViewer/View>
#include <osgViewer/ViewerEventHandlers>

#include <iostream>

#include <time.h>

using namespace cvr;
using namespace std;
using namespace osg;
using namespace osgViewer;

CVRPLUGIN(CullingMultiGPURender)

CullingMultiGPURender::CullingMultiGPURender()
{
}

CullingMultiGPURender::~CullingMultiGPURender()
{
}

/**
 * Initialization function for the plugin called once when loaded.
 *
 * Reads configuration from config file, initializes animation and renderer objects, 
 * sets up callbacks
 *
 * @return init status, true if good, false otherwise
 */
bool CullingMultiGPURender::init()
{

    int w,h;
    bool geo;
    DepthBits db;
    std::string filename;
    TextureCopyType copyType;

    // window size for multigpu rendering
    w = ConfigManager::getInt("Plugin.CullingMultiGPURender.Width");
    h = ConfigManager::getInt("Plugin.CullingMultiGPURender.Height");

    // geometry shaded normals or not
    geo = ConfigManager::getBool("Plugin.CullingMultiGPURender.GeometryShader");

    // file to load
    filename = ConfigManager::getEntry("Plugin.CullingMultiGPURender.FileName");

    // how many bits to use for depth buffers
    int bits = ConfigManager::getInt("Plugin.CullingMultiGPURender.DepthBuffer");

    // method used to copy textures
    std::string tcopy = ConfigManager::getEntry("Plugin.CullingMultiGPURender.TextureCopy");

    // convert bits into enum value
    if(bits == 16)
    {
	db = D16;
    }
    else if(bits == 24)
    {
	db = D24;
    }
    else
    {
	db = D32;
    }

    // convert texture copy type into enum value
    if(tcopy == "PBOS")
    {
	copyType = PBOS;
    }
    else if(tcopy == "CUDA_COPY")
    {
	copyType = CUDA_COPY;
    }
    else
    {
	copyType = READ_PIX;
    }

    // create the renderer
    _renderer = new MultiGPURenderer(w,h,geo,copyType,db);

    // find out how many gpus we should be using
    _numGPUs = _renderer->getNumGPUs();

    // get access to a camera for modelviewprojection calculation and eye position
    Viewer::Contexts contexts;
    CVRViewer::instance()->getContexts(contexts);
    GraphicsWindow* gw = dynamic_cast<GraphicsWindow*>(*contexts.begin());
    osg::GraphicsContext::Cameras& cameras = gw->getCameras();
    _camera = *cameras.begin();

    // initalize the ChcAnimate frame work
    //std::string filename("/home/covise/data/covise/philip/chcplusframe/convert2/fourcarsolidfixPlot"); // temp placement
    //filename = "/fastdata/honda/veryLargeFinal"; // temp placement
    //filename = "/data/covise/philip/chcplusframe/convert2/temp/Large15Deep"; // temp placement
    //filename = "/fastdata/honda/LargeDeep"; // temp placement
    //filename = "/home/aprudhom/data/honda/small-chc/smallPlot"; // andrew's system 
    //filename = "/remote/temp/veryLargeFinal"; // temp placement

    printf("Number of gpus found is %d\n", _renderer->getNumGPUs());
    // create and init animation management class
    _chcAnimate = new ChcAnimate(filename, _renderer->getNumGPUs());
    _renderer->setPartMap( (std::map<int, PartInfo*> *) _chcAnimate->getGeometryMap());

    // register parts list per gpu with renderer
    for(int i = 0; i < _renderer->getNumGPUs(); i++)
    {
	_renderer->setPartList(i, *_chcAnimate->getPartList(i));
    }

    // create drawable for osg render traversal callback
    _drawable = new CallbackDrawable(_renderer, _chcAnimate, _renderer->getNumGPUs());
    _geode = new Geode();

    _geode->addDrawable(_drawable.get());
    _geode->setDataVariance(osg::Object::STATIC);

    SceneManager::instance()->getObjectsRoot()->addChild(_geode);
    //SceneManager::instance()->getScene()->addChild(_geode);

    // setup predraw hook on osg cameras
    setupDrawHook();

    // replace osg stats graph with our custom one for different stat measurments
    osgViewer::View::EventHandlers eh = CVRViewer::instance()->getEventHandlers();

    std::cerr << "Viewer has " << eh.size() << " event handlers." << std::endl;

    for(osgViewer::View::EventHandlers::iterator it = eh.begin(); it != eh.end(); it++)
    {
	if(dynamic_cast<osgViewer::StatsHandler*>((*it).get()))
	{
	    std::cerr << "Found Stats Handler." << std::endl;
	    CVRViewer::instance()->removeEventHandler((*it).get());
	}
    }

    CustomStatsHandler * csh = new CustomStatsHandler(_renderer->getNumGPUs());
    csh->setKeyEventTogglesOnScreenStats((int)'S');
    csh->setKeyEventPrintsOutStats((int)'P');
    CVRViewer::instance()->addEventHandler(csh);

    return true;
}

/**
 * CalVR plugin interface function called once per frame before the draw or predraw callbacks.
 *
 * Used to update matrices for chc calculations
 */
void CullingMultiGPURender::preFrame()
{

    /*timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = 10000000;
    nanosleep(&ts,NULL);*/

    // update eye and viewproj matrix //TODO
    osg::Vec3 eyePos, center, up;
    osg::Matrixd scale;
    scale.makeScale(SceneManager::instance()->getObjectScale(), SceneManager::instance()->getObjectScale(), SceneManager::instance()->getObjectScale());
    osg::Matrixd model = SceneManager::instance()->getObjectTransform()->getMatrix();

    osg::Matrixd modelViewProjection = scale * model * _camera->getViewMatrix() * _camera->getProjectionMatrix();
    
    // translate eyePos to object space
    _camera->getViewMatrixAsLookAt(eyePos, center, up);
    //eyePos = TrackingManager::instance()->getHeadMat().getTrans() * osg::Matrix::inverse(scale * model);
    eyePos = eyePos * osg::Matrix::inverse(scale * model);
   
    _chcAnimate->updateViewParameters((double*)eyePos.ptr() ,(double*) modelViewProjection.ptr());
    _chcAnimate->update();
}

/**
 *  Adds predraw callbacks to the osg cameras to provide callback to our algorithm
 */
void CullingMultiGPURender::setupDrawHook()
{
    int index = 0;
    Viewer::Contexts contexts;
    CVRViewer::instance()->getContexts(contexts);
    for(Viewer::Contexts::iterator citr = contexts.begin(); citr != contexts.end(); ++citr)
    {
	GraphicsWindow* gw = dynamic_cast<GraphicsWindow*>(*citr);
	if(gw->getTraits()->screenNum >= _numGPUs)
	{
	    continue;
	}
	osg::GraphicsContext::Cameras& cameras = gw->getCameras();
	for(osg::GraphicsContext::Cameras::iterator citr = cameras.begin(); citr != cameras.end(); ++citr)
	{
	    PreDrawHook * pdh = new PreDrawHook(new ChcPreDrawCallback(index++, _renderer->getNumGPUs(), _chcAnimate));
	    osg::Camera* camera = *citr;
	    camera->setPreDrawCallback(pdh);
	    _drawHookList.push_back(pdh);
	}
    }
}
