#ifndef CALLBACK_DRAWABLE_H
#define CALLBACK_DRAWABLE_H

#include <osg/Drawable>

#include <vector>
#include <GL/gl.h>

class CudaTestDrawable : public osg::Drawable
{
    public:
        CudaTestDrawable();
        virtual ~CudaTestDrawable();
        CudaTestDrawable(const CudaTestDrawable&,const osg::CopyOp& copyop=osg::CopyOp::SHALLOW_COPY);

        virtual void drawImplementation(osg::RenderInfo& ri) const;

        virtual Object* cloneType() const { return new CudaTestDrawable(); }
        virtual Object* clone(const osg::CopyOp& copyop) const { return new CudaTestDrawable(*this,copyop); }
        virtual osg::BoundingBox computeBound() const;
        virtual void updateBoundingBox();

    protected:
        void init() const;

        void copyData() const;

        mutable bool _init;
        mutable std::vector<GLuint> _vbolist;
};

#endif
