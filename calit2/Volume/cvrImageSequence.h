#ifndef CVRIMAGESEQUENCE
#define CVRIMAGESEQUENCE 1

#include <osg/ImageSequence>

using namespace osg;

class cvrImageSequence : public osg::ImageSequence
{
    public:
        cvrImageSequence();
        cvrImageSequence (const ImageSequence &ImageSequence, const CopyOp &copyop=CopyOp::SHALLOW_COPY);
        virtual ~cvrImageSequence();
        virtual Object* cloneType() const { return NULL; }
        virtual Object* clone(const osg::CopyOp& copyop) const { return new cvrImageSequence(*this,copyop); }
        virtual bool isSameKindAs(const Object* obj) const { return dynamic_cast<const cvrImageSequence*>(obj)!=NULL; }
        virtual const char* libraryName() const { return "cvr"; }
        virtual const char* className() const { return "cvrImageSequence"; }

        // added functions
        double getCurrentTime() { return _seekTime; }; 
        int getFrame(double time) { return imageIndex(time); };
};
#endif
