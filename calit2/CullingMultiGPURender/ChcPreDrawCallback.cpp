#include <GL/glew.h>
#include <kernel/SceneManager.h>
#include <kernel/CVRViewer.h>
#include <osgViewer/View>
#include "ChcPreDrawCallback.h"
#include "ChcAnimate.h"

#include <config/ConfigManager.h>
#include "CudaHelper.h"

using namespace cvr;
using namespace osgViewer;

OpenThreads::Barrier ChcPreDrawCallback::_occlusionBarrier;

ChcPreDrawCallback::ChcPreDrawCallback(int gpuNum, int totalGpuNum, ChcAnimate* chcAnimate) : _chcAnimate(chcAnimate), _gpuNum(gpuNum), _totalGpuNum(totalGpuNum)
{
    _parts = _chcAnimate->getPartList(gpuNum);
    _pmap = _chcAnimate->getGeometryMap();
    _initGlew = false;

    _cudaCopy = cvr::ConfigManager::getBool("Plugin.CullingMultiGPURender.CudaCopy",false);

    // need to update the modelview and projection matrixs
    Viewer::Contexts contexts;
    CVRViewer::instance()->getContexts(contexts);
    GraphicsWindow* gw = dynamic_cast<GraphicsWindow*>(*contexts.begin());
    osg::GraphicsContext::Cameras& cameras = gw->getCameras();
    _camera = *cameras.begin();

    _cudaInit = false;
    _bufferManInit = false;
}


// call back used to transfer data from memory to video card
void ChcPreDrawCallback::preDrawCallback(osg::Camera * cam)
{
    //std::cerr << "In predrawcallback" << std::endl;
    if(!_initGlew)
    {
	    GLenum err = glewInit();
	    if(GLEW_OK != err) {
		// problem: glewInit failed, something is seriously wrong
		fprintf(stderr,"Error: %s\n",glewGetErrorString(err));
		exit(1);
	    }

	    if(!GLEW_ARB_occlusion_query) {
		printf("I require the GL_ARB_occlusion_query OpenGL extension to work.\n");
		exit(1);
	    }

	    _initGlew = true;
    }
 
    if(_cudaCopy && !_cudaInit)
    {
	//glewInit();
	//cudaSetDevice(_gpuNum);
	cudaGLSetGLDevice(_gpuNum);
	_cudaInit = true;
	printCudaErr();
	std::cerr << "Cuda Device set to: " << _gpuNum << std::endl;
    }

    osg::Stats * stats = NULL;
    if(cam)
    {
	stats = cam->getStats();
    }

    if(stats && !stats->collectStats("mgpu"))
    {
	stats = NULL;
    }

    /*if(_gpuNum)
    {
	for(int i = 0; i < (int) _parts->size(); i++)
	{
	    std::map<int, Geometry * >::iterator it;
	    it = _pmap->find(_parts->at(i));
	    if( it != _pmap->end() )
	    {
		//initalize buffers if they havent been set
		it->second->InitalizeBuffers(_gpuNum);
	    }	
	}
    }
    else
    {
	std::map<int, Geometry * >::iterator it;
	for(it = _pmap->begin(); it != _pmap->end(); it++)
	{
	    it->second->InitalizeBuffers(_gpuNum);
	}
    }*/

    if(!_bufferManInit)
    {
	_bufferManager = new BufferManager(*_pmap,*_parts, _gpuNum, _cudaCopy);
	for(int i = 0; i < (int) _parts->size(); i++)
	{
	    std::map<int, Geometry * >::iterator it;
	    it = _pmap->find(_parts->at(i));
	    if( it != _pmap->end() )
	    {
		//initalize buffers if they havent been set
		it->second->InitalizeBuffers(_gpuNum);
		it->second->setBufferManager(_bufferManager);
	    }	
	}

	if(!_gpuNum)
	{
	    std::map<int, Geometry * >::iterator it;
	    for(it = _pmap->begin(); it != _pmap->end(); it++)
	    {
		it->second->InitalizeBuffers(_gpuNum);
		it->second->setBufferManager0(_bufferManager);
	    }
	}

	//std::cerr << "Block on init barrier. " << _totalGpuNum << std::endl;
	_occlusionBarrier.block(_totalGpuNum);
	//std::cerr << "Unblocked init barrier." << std::endl;
	_bufferManInit = true;
    }

    _bufferManager->setFrame(_chcAnimate->getFrame());
    _bufferManager->setNextFrame(_chcAnimate->getNextFrame());

    
    for(int i = 0; i < (int) _parts->size(); i++)
    {
        std::map<int, Geometry * >::iterator it;
        it = _pmap->find(_parts->at(i));
        if( it != _pmap->end() )
        {
	   it->second->setFrameNumbers(_gpuNum,_chcAnimate->getFrame(),_chcAnimate->getNextFrame());
		//it->second->setFrameNumber(i, _chcAnimate->getFrame());
		//it->second->setNextFrameNumber(_chcAnimate->getNextFrame());
	}	
    }

    _occlusionBarrier.block(_totalGpuNum);

    // do occlusion testing after buffers are initalized
    // occulsion query (execute only in first context each frame)
    if( _gpuNum == 0 )
    {
	/*for(int i = 0; i < (int) _parts->size(); i++)
	{
	    std::map<int, Geometry * >::iterator it;
	    it = _pmap->find(_parts->at(i));
	    if( it != _pmap->end() )
	    {
		it->second->setFrameNumbers(i,_chcAnimate->getFrame(),_chcAnimate->getNextFrame());
		//it->second->setFrameNumber(i, _chcAnimate->getFrame());
		//it->second->setNextFrameNumber(_chcAnimate->getNextFrame());
	    }	
	}*/

	osg::Matrixd scale;
	scale.makeScale(SceneManager::instance()->getObjectScale(), SceneManager::instance()->getObjectScale(), SceneManager::instance()->getObjectScale());
	osg::Matrixd model = SceneManager::instance()->getObjectTransform()->getMatrix();

	// set the view size
	osg::Viewport* view = _camera->getViewport();
	glViewport((GLint)view->x(), (GLint)view->y(), (GLsizei)view->width(), (GLsizei)view->height());

	glMatrixMode (GL_PROJECTION);
	glPushMatrix();
	glLoadMatrixd(_camera->getProjectionMatrix().ptr());

	glMatrixMode (GL_MODELVIEW);
	glPushMatrix();
	glLoadMatrixd((scale * model * _camera->getViewMatrix()).ptr());	

	glEnable(GL_DEPTH_TEST);

	// occulsion testing to occur in first context only
	_chcAnimate->setNextFrame();

	glMatrixMode (GL_PROJECTION);
	glPopMatrix();

	glMatrixMode (GL_MODELVIEW);
	glPopMatrix();
    }


    //set up barrier
    //std::cerr << "Block on occlusion barrier. " << _totalGpuNum << std::endl;
    _occlusionBarrier.block(_totalGpuNum);
    //std::cerr << "Unblocked occlusion barrier." << std::endl;

    //cudaStream_t cStream;

    if(_cudaCopy)
    {
//	cudaStreamCreate(&cStream);
    }

    // loop through call geometry set frame ( copy data from memory to video card )
    // SetFrame also adds next frame geometry loads from disk
    //std::cerr << "doing set frame" << std::endl;

    double fsStart, fsEnd;

    if(stats)
    {
	fsStart = osg::Timer::instance()->delta_s(CVRViewer::instance()->getStartTick(), osg::Timer::instance()->tick());
    }

    for(int i = 0; i < (int) _parts->size(); i++)
    {
	std::map<int, Geometry * >::iterator it;
	it = _pmap->find(_parts->at(i));
	if( it != _pmap->end() )
	{

	    if(it->second->isOn())
	    {
		//std::cerr << "found on part " << it->second->getPartNumber() << std::endl;
		// indicate that geometry was drawn in multidraw
		it->second->setDrawn(true);

		//if(!_cudaCopy)
		//{
		    it->second->SetFrame(_gpuNum,true);
		//}
		//else
		//{
		//    it->second->SetFrame(_gpuNum, cStream, true);
		//}
	    }
	    
	}	
    }

    if(stats)
    {
	fsEnd = osg::Timer::instance()->delta_s(CVRViewer::instance()->getStartTick(), osg::Timer::instance()->tick());
	stats->setAttribute(CVRViewer::instance()->getViewerFrameStamp()->getFrameNumber(), "SetFrame begin time", fsStart);
	stats->setAttribute(CVRViewer::instance()->getViewerFrameStamp()->getFrameNumber(), "SetFrame end time", fsEnd);
	stats->setAttribute(CVRViewer::instance()->getViewerFrameStamp()->getFrameNumber(), "SetFrame time taken", fsEnd-fsStart);
    }

    //std::cerr << "starting next fetch." << std::endl;
    for(int i = 0; i < (int) _parts->size(); i++)
    {
	std::map<int, Geometry * >::iterator it;
	it = _pmap->find(_parts->at(i));
	if( it != _pmap->end() )
	{

	    if(it->second->isOn())
	    {
		it->second->startNextFetch();
	    }
	}	
    }

    double lfdStart, lfdEnd;

    if(stats)
    {
	lfdStart = osg::Timer::instance()->delta_s(CVRViewer::instance()->getStartTick(), osg::Timer::instance()->tick());
    }

    _bufferManager->loadFrameData();

    if(_cudaCopy)
    {
//	cudaStreamSynchronize(cStream);
//	cudaStreamDestroy(cStream);
	cudaThreadSynchronize();
    }
    else
    {
	if(stats)
	{
	    glFinish();
	}
    }

    if(stats)
    {
	lfdEnd = osg::Timer::instance()->delta_s(CVRViewer::instance()->getStartTick(), osg::Timer::instance()->tick());
	stats->setAttribute(CVRViewer::instance()->getViewerFrameStamp()->getFrameNumber(), "LoadFrameData begin time", lfdStart);
	stats->setAttribute(CVRViewer::instance()->getViewerFrameStamp()->getFrameNumber(), "LoadFrameData end time", lfdEnd);
	stats->setAttribute(CVRViewer::instance()->getViewerFrameStamp()->getFrameNumber(), "LoadFrameData time taken", lfdEnd-lfdStart);
    }
}

ChcPreDrawCallback::ChcPreDrawCallback()
{
}

ChcPreDrawCallback::~ChcPreDrawCallback()
{
}

