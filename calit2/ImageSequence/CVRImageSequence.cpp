#include "CVRImageSequence.h"

#include <osg/NodeVisitor>

CVRImageSequence::CVRImageSequence() : osg::ImageSequence()
{
}

CVRImageSequence::CVRImageSequence(const CVRImageSequence & sequence, const osg::CopyOp& copyop) : osg::ImageSequence(sequence,copyop)
{
}

CVRImageSequence::~CVRImageSequence()
{
}

void CVRImageSequence::update(osg::NodeVisitor* nv)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);

    // if imageDataList is empty then there is nothing update can do.
    if (_imageDataList.empty()) return;

    osg::NodeVisitor::ImageRequestHandler* irh = nv->getImageRequestHandler();

    if(_status = PLAYING)
    {
	int nextImage = (_previousAppliedImageIndex+1) % _imageDataList.size();
	if(!_imageDataList[nextImage]._image)
	{
	    osg::ref_ptr<osg::Image> image = irh->readImageFile(_imageDataList[nextImage]._filename, _readOptions.get());
	    if(image.valid())
	    {
		_setImage(nextImage, image.get());
	    }
	}

	if(_imageDataList[nextImage]._image)
	{
	    setImageToChild(nextImage);
	}
    }
}
