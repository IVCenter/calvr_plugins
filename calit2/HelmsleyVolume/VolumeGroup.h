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

	osg::ref_ptr<osg::Uniform> _MVP;
	osg::ref_ptr<osg::Uniform> _ViewToObject;
	osg::ref_ptr<osg::Uniform> _ViewInverse;
	osg::ref_ptr<osg::Uniform> _InverseProjection;
	osg::ref_ptr<osg::Uniform> _CameraPosition;
	osg::ref_ptr<osg::Uniform> _PlanePoint;
	osg::ref_ptr<osg::Uniform> _PlaneNormal;
	osg::ref_ptr<osg::Uniform> _StepSize;


protected:
	osg::ref_ptr<osg::Program> _program;
	osg::ref_ptr<osg::Program> _computeProgram;

	osg::ref_ptr<osg::StateSet> _volumeState;
	osg::ref_ptr<osg::StateSet> computeState;

	osg::ref_ptr<osg::Image> _volume;
	osg::ref_ptr<osg::ShapeDrawable> _cube;
	osg::ref_ptr<osg::PositionAttitudeTransform> _pat;
	

	
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
		//renderInfo.getState()->setCheckForGLErrors(osg::State::CheckForGLErrors::ONCE_PER_ATTRIBUTE);
		//group->_InverseProjection->set(osg::Matrix::inverse(renderInfo.getState()->getProjectionMatrix()));
		//group->_ViewToObject->set(osg::Matrix::inverse(renderInfo.getState()->getModelViewMatrix()));
		//group->_ViewInverse->set(osg::Matrix::inverse(renderInfo.getCurrentCamera()->getViewMatrix()));

		osg::Vec3f camPos = osg::Vec3f(0, 0, 0);
		osg::Vec3f camLook = osg::Vec3f(0, 0, 0);
		osg::Vec3f camUp = osg::Vec3f(0, 0, 0);
		renderInfo.getCurrentCamera()->getViewMatrixAsLookAt(camPos, camLook, camUp);
		//osg::NodePathList npl = group->getParentalNodePaths();
		
		//std::cout << "pos: " << camPos.x() << ", " << camPos.y() << ", " << camPos.z() << "  |  " << npl.size() << std::endl;

		//group->_CameraPosition->set(camPos);


		//group->_PlanePoint->set(osg::Vec3f(0, 0, 0));
		//group->_PlaneNormal->set(osg::Vec3f(0, 0, 0));

		drawable->drawImplementation(renderInfo);
	}
};


#endif
