#ifndef CALLBACK_DRAWABLE_H
#define CALLBACK_DRAWABLE_H

#include <osg/Drawable>

#include "FlowPagedRenderer.h"

class CallbackDrawable : public osg::Drawable
{
    public:
        CallbackDrawable(FlowPagedRenderer * renderer, osg::BoundingBox bounds);
        virtual ~CallbackDrawable();

        /// copy constructor
        CallbackDrawable(const CallbackDrawable&,const osg::CopyOp& copyop=osg::CopyOp::SHALLOW_COPY);

        virtual void drawImplementation(osg::RenderInfo& ri) const;

        /// osg function used to clone the object
        virtual Object* cloneType() const { return new CallbackDrawable(_renderer,_bbox); }
        /// osg function used to clone the object
        virtual Object* clone(const osg::CopyOp& copyop) const { return new CallbackDrawable(*this,copyop); }
        /// used to calculate bounds of drawable
        virtual osg::BoundingBox computeBound() const;
        /// update the drawable bounds
        virtual void updateBoundingBox();

    protected:
        osg::BoundingBox _bbox;
        FlowPagedRenderer * _renderer;
};

#endif
