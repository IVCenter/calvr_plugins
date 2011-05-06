/**
 * @file CallbackDrawable.cpp
 * Osg drawable class that we use to provide the draw callback to the multigpu renderer 
 *
 * @author Andrew Prudhomme (aprudhomme@ucsd.edu)
 *
 * @date 09/15/2010
 */

#include "CallbackDrawable.h"

#include <kernel/CVRViewer.h>

#include "FetchQueue.h"

#include <iostream>

using namespace osg;
using namespace cvr;

/**
 * @param renderer parallel rendering object
 * @param animate animation manager
 * @param numGPUs number of gpus to use for parallel rendering
 */
CallbackDrawable::CallbackDrawable(MultiGPURenderer * renderer, ChcAnimate * animate, int numGPUs)
{
    _renderer = renderer;
    _chcAnimate = animate;
    _numGPUs = numGPUs;
    setUseDisplayList(false);
}

CallbackDrawable::CallbackDrawable(const CallbackDrawable&,const osg::CopyOp& copyop)
{
}

CallbackDrawable::~CallbackDrawable()
{
}

/**
 * Draw call from the osg rendering traversal.
 *
 * We use it to trigger our parallel draw and do postdraw cleanup
 *
 * @param ri osg render state info
 */
void CallbackDrawable::drawImplementation(osg::RenderInfo& ri) const
{

    if(ri.getContextID() >= _numGPUs)
    {
	return;
    }

    // determine of draw stats should be collected in the camera
    osg::Stats * stats = NULL;
    if(ri.getCurrentCamera())
    {
	stats = ri.getCurrentCamera()->getStats();
    }

    if(stats && !stats->collectStats("mgpu"))
    {
	stats = NULL;
    }

    // do the parallel draw
    //std::cerr << "Draw context: " << ri.getContextID() << std::endl;
    if(_renderer)
    {
	_renderer->draw(ri.getContextID(), ri.getCurrentCamera());
    }
    
    double postStart, postEnd;

    if(ri.getContextID() == 0 && stats)
    {
	glFinish();
	postStart = osg::Timer::instance()->delta_s(CVRViewer::instance()->getStartTick(), osg::Timer::instance()->tick());
    }

    // do GLOcclusion test in first context only
    if( ri.getContextID() == 0 )
    {
	//FetchQueue::getInstance()->lock();
	_chcAnimate->turnOffGeometry();
	_chcAnimate->postRender();
	//FetchQueue::getInstance()->unlock();
    }

    if(ri.getContextID() == 0 && stats)
    {
	glFinish();
	postEnd = osg::Timer::instance()->delta_s(CVRViewer::instance()->getStartTick(), osg::Timer::instance()->tick());
	stats->setAttribute(CVRViewer::instance()->getViewerFrameStamp()->getFrameNumber(), "PostCHC begin time", postStart);
	stats->setAttribute(CVRViewer::instance()->getViewerFrameStamp()->getFrameNumber(), "PostCHC end time", postEnd);
	stats->setAttribute(CVRViewer::instance()->getViewerFrameStamp()->getFrameNumber(), "PostCHC time taken", postEnd-postStart);
    }

    // wait for primary gpu to finish occulsion testing
    _postBarrier.block(_numGPUs);

    // do cleanup of parts drawn during occlusion testing
    _chcAnimate->postRenderPerThread(ri.getContextID());
}

osg::BoundingBox CallbackDrawable::computeBound() const
{
    // arbitrarily large
    Vec3 size2(10000, 10000, 10000);
    _boundingBox.init();
    _boundingBox.set(-size2[0], -size2[1], -size2[2], size2[0], size2[1], size2[2]);
    return _boundingBox;
}

void CallbackDrawable::updateBoundingBox()
{
    computeBound();
    dirtyBound();
}
