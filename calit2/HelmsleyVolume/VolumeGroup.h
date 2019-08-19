#ifndef HELMSLEY_VOLUME_GROUP_H
#define HELMSLEY_VOLUME_GROUP_H

#include <osg/Node>
#include <osg/Geode>
#include <osg/BoundingSphere>
#include <osg/BoundingBox>
#include <osg/Program>
#include <osg/ShapeDrawable>
#include <osg/DispatchCompute>
#include <osg/BindImageTexture>
#include <osg/Texture3D>
#include <osg/DispatchCompute>
#include <osg/CullFace>
#include <osg/PositionAttitudeTransform>
#include <iostream>


class VolumeGroup : public osg::Group
{
public:
	VolumeGroup();
	virtual ~VolumeGroup();
	VolumeGroup(const VolumeGroup &, const osg::CopyOp& copyop = osg::CopyOp::SHALLOW_COPY);

	virtual Object* cloneType() const { return NULL; }
	virtual Object* clone(const osg::CopyOp& copyop) const { return new VolumeGroup(*this, copyop); }
	virtual bool isSameKindAs(const Object* obj) const { return dynamic_cast<const VolumeGroup*>(obj) != NULL; }
	virtual const char* libraryName() const { return "HelmsleyVolume"; }
	virtual const char* className() const { return "VolumeGeode"; }


	//non-osg functions
	std::string loadShaderFile(std::string filename) const;
	void init();
	void loadVolume(std::string path);
	void precompute();
	void setDirty(bool d=true) { _dirty = d; };

	osg::ref_ptr<osg::Uniform> _PlanePoint;
	osg::ref_ptr<osg::Uniform> _PlaneNormal;
	osg::ref_ptr<osg::Uniform> _StepSize;
	std::map<std::string, osg::ref_ptr<osg::Uniform>> _computeUniforms;

	osg::Matrix getObjectToWorldMatrix();
	osg::Matrix getWorldToObjectMatrix();


protected:

	bool _dirty;

	osg::ref_ptr<osg::Program> _program;
	osg::ref_ptr<osg::Program> _computeProgram;

	osg::ref_ptr<osg::DispatchCompute> _computeNode;
	osg::ref_ptr<osg::ShapeDrawable> _cube;
	osg::ref_ptr<osg::PositionAttitudeTransform> _pat;
	
	osg::ref_ptr<osg::Texture3D> _volume;
	osg::ref_ptr<osg::Texture3D> _baked;

	
	//osg::ref_ptr<osg::Uniform> _Volume;
	//osg::ref_ptr<osg::Uniform> _DepthTexture;
};



class VolumeDrawCallback : public osg::Drawable::DrawCallback
{
public:

	VolumeGroup* group;

	VolumeDrawCallback(VolumeGroup* g) : group(g) {}

	virtual void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const
	{
		if (!group)
		{
			return;
		}

		drawable->drawImplementation(renderInfo);

	}
};

class ComputeDrawCallback : public osg::Drawable::DrawCallback
{
public:

	ComputeDrawCallback() {}

	virtual void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const
	{
		drawable->drawImplementation(renderInfo);
		renderInfo.getState()->get<osg::GLExtensions>()->glMemoryBarrier(GL_ALL_BARRIER_BITS);
	}
};


#endif
