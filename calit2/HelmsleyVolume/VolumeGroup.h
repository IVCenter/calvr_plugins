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
#include <osg/Texture2D>
#include <osg/DispatchCompute>
#include <osg/BufferIndexBinding>
#include <osg/StateAttributeCallback>
#include <osg/BufferObject>
#include <osg/CullFace>
#include <osg/PositionAttitudeTransform>
#include <osg/FrameBufferObject>
#include <osg/LineWidth>
#include <iostream>
#include <cvrKernel/ScreenConfig.h>
#include <cvrKernel/ScreenBase.h>




class VolumeGroup : public osg::Group
{
	friend class VolumeDrawCallback;
public:
	VolumeGroup();
	virtual ~VolumeGroup();
	VolumeGroup(const VolumeGroup &, const osg::CopyOp& copyop = osg::CopyOp::SHALLOW_COPY);

	virtual Object* cloneType() const { return NULL; }
	virtual Object* clone(const osg::CopyOp& copyop) const { return new VolumeGroup(*this, copyop); }
	virtual bool isSameKindAs(const Object* obj) const { return dynamic_cast<const VolumeGroup*>(obj) != NULL; }
	virtual const char* libraryName() const { return "HelmsleyVolume"; }
	virtual const char* className() const { return "VolumeGeode"; }


	void init();
	void loadVolume(std::string path, std::string maskpath = "");
	void loadMask(std::string path, osg::Image* volume);
	bool hasMask()
	{
		return _hasMask;
	}
	void precompute();
	void setDirtyAll()
	{
		for (std::map<osg::GraphicsContext*, bool>::iterator it = _dirty.begin(); it != _dirty.end(); ++it)
		{
			it->second = true;
		}
	}
	void setDirty(osg::GraphicsContext* gc, bool d=true) { _dirty[gc] = d; };
	bool isDirty(osg::GraphicsContext* gc) { return _dirty[gc]; };
	osg::Drawable* getDrawable() { return _cube; };
	osg::Vec3dArray* getColonCoords() { return _colonCoords; }

	std::vector<osg::Geode*>* getCenterLines() { return _centerLineGeodes; }
	osg::DispatchCompute* getCompute() { return _computeNode; };
	
	void dirtyVolumeShader() { _program->dirtyProgram(); };
	void dirtyComputeShader() { _computeProgram->dirtyProgram(); };
	

	void flipCull();

	osg::ref_ptr<osg::Uniform> _PlanePoint;
	osg::ref_ptr<osg::Uniform> _PlaneNormal;
	osg::ref_ptr<osg::Uniform> _StepSize;
	osg::ref_ptr<osg::Uniform> _RelativeViewport;
	std::map<std::string, osg::ref_ptr<osg::Uniform> > _computeUniforms;

	osg::Matrix getObjectToWorldMatrix();
	osg::Matrix getWorldToObjectMatrix();

	osg::ref_ptr<osg::MatrixTransform> _transform;
	osg::ref_ptr<osg::MatrixTransform> _lineTransform;
	osg::ref_ptr<osg::FrameBufferObject> _resolveFBO;
	

protected:
	bool _hasMask;

	std::map<osg::GraphicsContext*, bool> _dirty;

	osg::ref_ptr<osg::Program> _program;
	osg::ref_ptr<osg::Program> _computeProgram;
	osg::ref_ptr<osg::Program> _minMaxProgram;

	osg::ref_ptr<osg::ShaderStorageBufferObject>  _ssbo;
	osg::ref_ptr<osg::ShaderStorageBufferBinding> _ssbb;
	osg::ref_ptr<osg::UIntArray> data;

	osg::ref_ptr<osg::DispatchCompute> _computeNode;
	osg::ref_ptr<osg::DispatchCompute> _minMaxNode;
	osg::ref_ptr<osg::ShapeDrawable> _cube;
	
	std::vector<osg::Geode*>* _centerLineGeodes;
	osg::Vec3dArray* _colonCoords;
	osg::Vec3dArray* _illeumCoords;
	osg::ref_ptr<osg::Texture3D> _volume;
	osg::ref_ptr<osg::Texture3D> _baked;

	osg::ref_ptr<osg::Texture2D> _depthTexture;
	osg::ref_ptr<osg::Texture2D> _colorTexture;

	osg::ref_ptr<osg::CullFace> _side;
	
};



class VolumeDrawCallback : public osg::Drawable::DrawCallback
{

public:

	VolumeGroup* group;

	VolumeDrawCallback(VolumeGroup* g) : group(g) {}

	virtual void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const;
};

class ComputeDrawCallback : public osg::Drawable::DrawCallback
{
public:

	VolumeGroup* group;

	ComputeDrawCallback(VolumeGroup* g) : group(g) {}

	virtual void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const
	{
		//compute needs to run once per graphics context
		if (group->isDirty(renderInfo.getCurrentCamera()->getGraphicsContext()))
		{
			
			//std::cout << "computing in gc " << renderInfo.getState()->getContextID() << "\n";
			//std::cout << "0: " << renderInfo.getState()->getLastAppliedTextureAttribute(0, osg::StateAttribute::TEXTURE)->getName()
			//          << ", 1: " << renderInfo.getState()->getLastAppliedTextureAttribute(1, osg::StateAttribute::TEXTURE)->getName() << std::endl;
			
			drawable->drawImplementation(renderInfo);
			renderInfo.getState()->get<osg::GLExtensions>()->glMemoryBarrier(GL_ALL_BARRIER_BITS);
			group->setDirty(renderInfo.getCurrentCamera()->getGraphicsContext(), false);
		}
	}
};

class MinMaxDrawCallback : public osg::Drawable::DrawCallback
{
public:

	VolumeGroup* group;

	
	MinMaxDrawCallback(VolumeGroup* g) : group(g) {}

	virtual void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const
	{
		//compute needs to run once per graphics context
		

			//std::cout << "minmax changed" << "\n";
			//std::cout << "0: " << renderInfo.getState()->getLastAppliedTextureAttribute(0, osg::StateAttribute::TEXTURE)->getName()
			//          << ", 1: " << renderInfo.getState()->getLastAppliedTextureAttribute(1, osg::StateAttribute::TEXTURE)->getName() << std::endl;
			
			/*drawable->drawImplementation(renderInfo);
			renderInfo.getState()->get<osg::GLExtensions>()->glMemoryBarrier(GL_ALL_BARRIER_BITS);*/
	
			
			
		
	}
};

class ShaderStorageBufferCallback : public osg::StateAttributeCallback
{
public:
	void operator() (osg::StateAttribute* attr, osg::NodeVisitor* nv)
	{
		//if you need to process the data in your app-code , better leaving it on GPU and processing there, uploading per frame will make it slow
		//std::cout << "ssbo callback reached" << std::endl;
		osg::ShaderStorageBufferBinding* ssbb = static_cast<osg::ShaderStorageBufferBinding*>(attr);
		osg::UIntArray* array = static_cast<osg::UIntArray*>(ssbb->getBufferData());

		



		int someValue = array->at(0);
		int someValue2 = array->at(1);
		//std::cout << "someValue now: " << someValue << std::endl;
	//	std::cout << "someValue ow: " << someValue2 << std::endl;
		//data transfer performance test
		    array->dirty();
	}
};





#endif
