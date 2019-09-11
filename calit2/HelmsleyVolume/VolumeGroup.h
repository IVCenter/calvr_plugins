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
#include <osg/CullFace>
#include <osg/PositionAttitudeTransform>
#include <osg/FrameBufferObject>
#include <iostream>
#include <cvrKernel/ScreenConfig.h>
#include <cvrKernel/ScreenBase.h>


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
	static std::string loadShaderFile(std::string filename);

	void init();
	void loadVolume(std::string path);
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
	
	void dirtyVolumeShader() { _program->dirtyProgram(); };
	void dirtyComputeShader() { _computeProgram->dirtyProgram(); };

	osg::ref_ptr<osg::Uniform> _PlanePoint;
	osg::ref_ptr<osg::Uniform> _PlaneNormal;
	osg::ref_ptr<osg::Uniform> _StepSize;
	std::map<std::string, osg::ref_ptr<osg::Uniform> > _computeUniforms;

	osg::Matrix getObjectToWorldMatrix();
	osg::Matrix getWorldToObjectMatrix();

	osg::ref_ptr<osg::PositionAttitudeTransform> _pat;
	osg::ref_ptr<osg::FrameBufferObject> _resolveFBO;


protected:

	std::map<osg::GraphicsContext*, bool> _dirty;

	osg::ref_ptr<osg::Program> _program;
	osg::ref_ptr<osg::Program> _computeProgram;

	osg::ref_ptr<osg::DispatchCompute> _computeNode;
	osg::ref_ptr<osg::ShapeDrawable> _cube;

	osg::ref_ptr<osg::Texture3D> _volume;
	osg::ref_ptr<osg::Texture3D> _baked;

	osg::ref_ptr<osg::Texture2D> _depthTexture;
	osg::ref_ptr<osg::Texture2D> _colorTexture;
	
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
			std::cerr << "group doesn't exist!" << std::endl;
			return;
		}
		//osg::setNotifyLevel(osg::DEBUG_INFO);

		//std::cout << renderInfo.getCurrentCamera()->isRenderToTextureCamera() << std::endl;

		//osg::Texture* dt = renderInfo.getCurrentCamera()->getBufferAttachmentMap()[osg::Camera::COLOR_BUFFER0]._texture;
		cvr::ScreenBase* screen = cvr::ScreenConfig::instance()->getScreen(cvr::ScreenConfig::instance()->findScreenNumber(renderInfo.getCurrentCamera()));
		screen->resolveBuffers(renderInfo.getCurrentCamera(), group->_resolveFBO, renderInfo.getState(), GL_DEPTH_BUFFER_BIT);


		//const osg::Texture* dt = group->_resolveFBO->getAttachment(osg::Camera::COLOR_BUFFER0).getTexture();
		//renderInfo.getState()->applyTextureAttribute(1, dt);
		//renderInfo.getState()->applyTextureMode(1, GL_TEXTURE_2D, osg::StateAttribute::ON);

		/*
		if (renderInfo.getCurrentCamera()->getName().substr(0, 6).compare("OpenVR") == 0)
		{
			//osg::setNotifyLevel(osg::NOTICE);
			//throw "bleh";
			const osg::StateAttribute* sa = renderInfo.getState()->getLastAppliedTextureAttribute(0, osg::StateAttribute::TEXTURE);
			const osg::Texture* t = sa->asTexture();
		}
		else
		{
			const osg::StateAttribute* sa = renderInfo.getState()->getLastAppliedTextureAttribute(0, osg::StateAttribute::TEXTURE);
			const osg::Texture* t = sa->asTexture();
			osg::GraphicsContext* gc = renderInfo.getCurrentCamera()->getGraphicsContext();
		}


		//std::cout << renderInfo.getState()->getLastAppliedTextureAttribute(1, osg::StateAttribute::TEXTURE)->getName() << std::endl;

		*/
		drawable->drawImplementation(renderInfo);
		//renderInfo.getState()->applyTextureAttribute(1, NULL);
		//renderInfo.getState()->applyTextureMode(1, GL_TEXTURE_2D, osg::StateAttribute::ON);
	}
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

			//std::cout << "computing\n";
			//std::cout << "0: " << renderInfo.getState()->getLastAppliedTextureAttribute(0, osg::StateAttribute::TEXTURE)->getName()
			//          << ", 1: " << renderInfo.getState()->getLastAppliedTextureAttribute(1, osg::StateAttribute::TEXTURE)->getName() << std::endl;
			
			drawable->drawImplementation(renderInfo);
			renderInfo.getState()->get<osg::GLExtensions>()->glMemoryBarrier(GL_ALL_BARRIER_BITS);
			group->setDirty(renderInfo.getCurrentCamera()->getGraphicsContext(), false);
		}
	}
};


#endif
