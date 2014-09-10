#ifndef CVR_OSG_IMAGE_SEQUENCE
#define CVR_OSG_IMAGE_SEQUENCE

#include <osg/ImageSequence>

class CVRImageSequence : public osg::ImageSequence
{
    public:
        CVRImageSequence();
        CVRImageSequence(const CVRImageSequence & sequence, const osg::CopyOp& copyop=osg::CopyOp::SHALLOW_COPY);
        virtual ~CVRImageSequence();

        virtual void update(osg::NodeVisitor* nv);

    protected:
};

#endif
