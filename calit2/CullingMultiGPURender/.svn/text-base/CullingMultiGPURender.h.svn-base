/**
 * @file CullingMultiGPURender.h
 * Driving class of our algorithm.
 *
 * Implementation of the plugin class for our graphics environment. 
 *
 * @author Andrew Prudhomme (aprudhomme@ucsd.edu)
 *
 * @date 09/15/2010
 */

#ifndef CULLING_MULTI_GPU_RENDER_H
#define CULLING_MULTI_GPU_RENDER_H

#include <kernel/CVRPlugin.h>

#include <osg/Geode>

#include "CallbackDrawable.h"
#include "PreDrawHook.h"
#include "MultiGPURenderer.h"
#include "ChcAnimate.h"

/**
 * Interface class for this CalVR plugin.
 *
 * Implements interface functions that are called by our VR framework.
 */
class CullingMultiGPURender : public cvr::CVRPlugin
{
    public:
        CullingMultiGPURender();
        virtual ~CullingMultiGPURender();

        bool init();
        void preFrame();

    protected:

        void setupDrawHook();

        int _numGPUs;           ///< number of GPUs being used for parallel rendering

        MultiGPURenderer * _renderer;   ///< the multiGPU renderer

        osg::ref_ptr<CallbackDrawable> _drawable;   ///< osg drawable to provide render callback
        osg::ref_ptr<osg::Geode> _geode;            ///< geode for osg drawable

        std::vector<osg::ref_ptr<PreDrawHook> > _drawHookList;  ///< list of objects that provide the predraw callbacks from osg

        ChcAnimate * _chcAnimate;       ///< the animation managment class
        osg::Camera * _camera;          ///< osg camera to get view and projection matrices from
};

#endif
