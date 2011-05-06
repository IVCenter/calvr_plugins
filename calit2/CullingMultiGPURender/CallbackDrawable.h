/**
 * @file CallbackDrawable.h
 * Osg drawable class that we use to provide the draw callback to the multigpu renderer 
 *
 * @author Andrew Prudhomme (aprudhomme@ucsd.edu)
 *
 * @date 09/15/2010
 */

#ifndef CALLBACK_DRAWABLE_H
#define CALLBACK_DRAWABLE_H

#include <osg/Drawable>

#include "ChcAnimate.h"
#include "MultiGPURenderer.h"
/**
 * Osg drawable class that we use to provide the draw callback to the multigpu renderer
 */
class CallbackDrawable : public osg::Drawable
{
    public:
        CallbackDrawable(MultiGPURenderer * renderer, ChcAnimate * animate, int numGPUs);
        virtual ~CallbackDrawable();

        /// copy constructor
        CallbackDrawable(const CallbackDrawable&,const osg::CopyOp& copyop=osg::CopyOp::SHALLOW_COPY);

        virtual void drawImplementation(osg::RenderInfo& ri) const;

        /// osg function used to clone the object
        virtual Object* cloneType() const { return new CallbackDrawable(_renderer, _chcAnimate, _numGPUs); }
        /// osg function used to clone the object
        virtual Object* clone(const osg::CopyOp& copyop) const { return new CallbackDrawable(*this,copyop); }
        /// used to calculate bounds of drawable
        virtual osg::BoundingBox computeBound() const;
        /// update the drawable bounds
        virtual void updateBoundingBox();

    protected:
        MultiGPURenderer * _renderer;               ///< our renderer
        ChcAnimate * _chcAnimate;                   ///< the animation manager

        int _numGPUs;                               ///< number of gpus used for the parallel draw

        mutable OpenThreads::Barrier _postBarrier;  ///< used to synchronize multiple render threads
};

#endif
