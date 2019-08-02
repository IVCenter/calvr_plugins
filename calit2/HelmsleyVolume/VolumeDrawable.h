#ifndef HELMSLEY_VOLUME_DRAWABLE_H
#define HELMSLEY_VOLUME_DRAWABLE_H

#include <osg/Drawable>
#include <osg/BoundingSphere>
#include <osg/BoundingBox>
#include <osg/Program>
#include <osg/Shader>

class VolumeDrawable : public osg::Drawable
{
public:
	VolumeDrawable();
	virtual ~VolumeDrawable();
	VolumeDrawable(const VolumeDrawable &, const osg::CopyOp& copyop = osg::CopyOp::SHALLOW_COPY);

	virtual void drawImplementation(osg::RenderInfo &renderInfo) const;
	virtual osg::BoundingSphere computeBound() const;
	virtual osg::BoundingBox computeBoundingBox() const;

	virtual Object* cloneType() const { return NULL; }
	virtual Object* clone(const osg::CopyOp& copyop) const { return new VolumeDrawable(*this, copyop); }
	virtual bool isSameKindAs(const Object* obj) const { return dynamic_cast<const VolumeDrawable*>(obj) != NULL; }
	virtual const char* libraryName() const { return "HelmsleyVolume"; }
	virtual const char* className() const { return "VolumeDrawable"; }


	//non-osg functions
	std::string loadShaderFile(std::string filename) const;
	void init();

protected:
	osg::ref_ptr<osg::Program> _program;
	osg::ref_ptr<osg::Program> _computeProgram;

	osg::ref_ptr<osg::StateSet> _volumeState;

};


#endif
