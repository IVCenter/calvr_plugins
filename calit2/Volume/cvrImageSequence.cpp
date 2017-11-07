#include "cvrImageSequence.h"

using namespace osg;

cvrImageSequence::cvrImageSequence() : osg::ImageSequence()
{
}

cvrImageSequence::cvrImageSequence(const ImageSequence &ImageSequence, const CopyOp &copyop)
    : osg::ImageSequence(ImageSequence, copyop)
{
}

cvrImageSequence::~cvrImageSequence()
{
}
