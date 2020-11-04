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
#include <thread>
#include <cvrKernel/ScreenConfig.h>
#include <cvrKernel/ScreenBase.h>

//#define NUMGRAYVALS 255u
#define CLIPLIMIT3D .85f
#define ORGANCOUNT 8

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


	////////////////CLAHE
	typedef  osg::ref_ptr<osg::ShaderStorageBufferBinding> t_ssbb;
	typedef  osg::ref_ptr<osg::AtomicCounterBufferBinding> t_acbb;
	///////CLAHE Variables
	osg::Vec3i _numSB_3D = osg::Vec3i(4, 4, 2);
	osg::Vec3i _sizeSB = osg::Vec3i(0, 0, 0);
	float _clipLimit3D = 0.85f;
	float _minClipValue = 0.0;
	osg::Vec3i _volDims = osg::Vec3i(0, 0, 0);
	unsigned int _numGrayVals = 255u;
	unsigned int _histSize = _numSB_3D.x() * _numSB_3D.y() * _numSB_3D.z() * _numGrayVals;
	unsigned int _numHist = _numSB_3D.x() * _numSB_3D.y() * _numSB_3D.z();
	//CLAHE Variables///////

	///////Main Methods
	t_acbb precompMinMax();
 	std::pair<t_ssbb, t_ssbb> precompHist();
 	t_ssbb precompExcess(t_ssbb ssbbHist, t_ssbb ssbbHistMax);
 	void precompHistClip(t_ssbb ssbbHist, t_ssbb ssbbHistMax, t_ssbb ssbbExcess, t_acbb acbbminmax);
 	void precompLerp(t_ssbb ssbbHist);
	//Main Methods//////////

	////Extra Methods
	void setNumBins(unsigned int numBins) { _numGrayVals = numBins; }
	void genClahe();
	//Extra Methods//
	//CLAHE//////////////
	unsigned int getHistMax();

	osg::ref_ptr<osg::ShaderStorageBufferBinding> getHistBB();


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
	osg::Vec3dArray* getIlleumCoords() { return _illeumCoords; }
	void precompTotalHistogram();


	std::vector<osg::ref_ptr<osg::Geode>>* getCenterLines() { return _centerLineGeodes; }
	osg::DispatchCompute* getCompute() { return _computeNode; };
	
	void dirtyVolumeShader() { _program->dirtyProgram(); };
	void dirtyComputeShader() { _computeProgram->dirtyProgram(); };
	

	void flipCull();

	osg::ref_ptr<osg::Uniform> _PlanePoint;
	osg::ref_ptr<osg::Uniform> _PlaneNormal;
	osg::ref_ptr<osg::Uniform> _StepSize;
	osg::ref_ptr<osg::Uniform> _testScale;
	osg::ref_ptr<osg::Uniform> _maxSteps;
	osg::ref_ptr<osg::Uniform> _RelativeViewport;
	std::map<std::string, osg::ref_ptr<osg::Uniform> > _computeUniforms;

	osg::Matrix getObjectToWorldMatrix();
	osg::Matrix getWorldToObjectMatrix();

	osg::ref_ptr<osg::MatrixTransform> _transform;
	osg::ref_ptr<osg::MatrixTransform> _lineTransform;
	osg::ref_ptr<osg::FrameBufferObject> _resolveFBO;


	struct Values {
		std::vector<std::vector<float>> opacityData;//0=Center, 1=BottomWidth, 2=TopWidth, 3= Height, 4 = Lowest
		std::vector<float> contrastData;
		int tf;
		int tentCount;

		///////////////////Cutting Plane///////////////////////
		osg::Vec3 cpPos; 
		osg::Quat cpRot;
		
		///////////////////Masks////////////////////////
		std::vector<bool> masks; //Order: o,c,k,b,s,i,a,v

		//////////////////Tools/////////////////////////
		bool cpToggle = false;

		bool saved = false;

	};

	Values values;
	

protected:
	bool _hasMask;
	std::string _minMaxShader;
	std::string _excessShader;
	std::string _histShader;
	std::string _clipShader;
	std::string _clipShader2;
	std::string _lerpShader;

	std::string _totalHistShader;


	std::map<osg::GraphicsContext*, bool> _dirty;

	osg::ref_ptr<osg::Program> _program;
	osg::ref_ptr<osg::Program> _computeProgram;


	osg::ref_ptr<osg::ShaderStorageBufferObject>  _ssbo;
	osg::ref_ptr<osg::ShaderStorageBufferBinding> _ssbb;
	osg::ref_ptr<osg::UIntArray> data;

	osg::ref_ptr<osg::DispatchCompute> _computeNode;
	osg::ref_ptr<osg::DispatchCompute> sourceNode;
	osg::ref_ptr<osg::DispatchCompute> _histNode;
	osg::ref_ptr<osg::DispatchCompute> _excessNode;
	osg::ref_ptr<osg::DispatchCompute> _minMaxNode;
	osg::ref_ptr<osg::DispatchCompute> _clipHist1Node;
	osg::ref_ptr<osg::DispatchCompute> _lerpNode;

	osg::ref_ptr<osg::DispatchCompute> _totalHistNode;
	osg::ref_ptr<osg::ShapeDrawable> _cube;


	//osg::ref_ptr<osg::ShaderStorageBufferBinding> ssbbHist;
//	osg::ref_ptr<osg::ShaderStorageBufferBinding> ssbbExcess;

	std::vector<osg::ref_ptr<osg::Geode>>* _centerLineGeodes;
	osg::ref_ptr<osg::Vec3dArray> _colonCoords;
	osg::ref_ptr<osg::Vec3dArray> _illeumCoords;
	osg::ref_ptr<osg::Texture3D> _volume;
	osg::ref_ptr<osg::Texture3D> _claheVolume;
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

			if(_claheDirty[0] != 1)
				group->setDirty(renderInfo.getCurrentCamera()->getGraphicsContext(), false);
			
		}
	}

	uint16_t* _claheDirty = new uint16_t(1);
};



class AtomicCallback : public osg::Drawable::DrawCallback
{
public:
	VolumeGroup* group;
	AtomicCallback(VolumeGroup* g) : group(g)
	{
	}

	virtual void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const
	{
		if (group->isDirty(renderInfo.getCurrentCamera()->getGraphicsContext()) && stop[0] != 1)
		{
			drawable->drawImplementation(renderInfo);
			renderInfo.getState()->get<osg::GLExtensions>()->glMemoryBarrier(GL_ALL_BARRIER_BITS);
			/*osg::ref_ptr<osg::UIntArray> _atomicCounterArray = new osg::UIntArray();
			_atomicCounterArray->push_back(0);
			_atomicCounterArray->push_back(0);
			_atomicCounterArray->push_back(0);
			_acbb->readData(*renderInfo.getState(), *_atomicCounterArray);
			
			unsigned int value = osg::maximum(1u, _atomicCounterArray->front());
			
			std::cout << "Pixel Count: " << value << std::endl;
			std::cout << "Min: " << _atomicCounterArray->at(1) << std::endl;
			std::cout << "Max: " << _atomicCounterArray->at(2) << std::endl;*/
			
			stop[0] = 1;
		}
	}

	uint16_t* stop = new uint16_t(2);
	osg::ref_ptr<osg::AtomicCounterBufferBinding> _acbb;

};



class ReadShaderStorageBufferCallback : public osg::Drawable::DrawCallback
{
public:
	VolumeGroup* group;

	ReadShaderStorageBufferCallback(VolumeGroup* g) : group(g)
	{
	}

	virtual void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const
	{
		
		if (group->isDirty(renderInfo.getCurrentCamera()->getGraphicsContext()) && stop[0] != 1)
		{
			std::cout << "Hist executed." << std::endl;
			std::cout << "ReadShaderBufferDataCalback executed." << std::endl << std::flush;
			drawable->drawImplementation(renderInfo);
			renderInfo.getState()->get<osg::GLExtensions>()->glMemoryBarrier(GL_ALL_BARRIER_BITS);


			stop[0] = 1;
		}
	}

	uint16_t* stop = new uint16_t(2);
	int _buffersize;
	int _buffersize2;
	osg::ref_ptr<osg::ShaderStorageBufferBinding> _ssbb;
	osg::ref_ptr<osg::ShaderStorageBufferBinding> _ssbb2;
};

class TotalHistCallback : public osg::Drawable::DrawCallback
{
public:
	VolumeGroup* group;

	TotalHistCallback(VolumeGroup* g) : group(g)
	{
	}

	virtual void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const
	{
		
		if (group->isDirty(renderInfo.getCurrentCamera()->getGraphicsContext()) && stop[0] != 1)
		{
 			std::cout << "Total hist executed." << std::endl << std::flush;
			drawable->drawImplementation(renderInfo);
			renderInfo.getState()->get<osg::GLExtensions>()->glMemoryBarrier(GL_ALL_BARRIER_BITS);

			osg::ref_ptr<osg::UIntArray> _atomicCounterArray = new osg::UIntArray;
			_atomicCounterArray->push_back(0);
			_acbb->readData(*renderInfo.getState(), *_atomicCounterArray);

			unsigned int value = osg::maximum(1u, _atomicCounterArray->front());

			std::cout << "Max Bin Value: " << value << std::endl;
			histMax[0] = value;
			stop[0] = 1;
			group->setDirty(renderInfo.getCurrentCamera()->getGraphicsContext(), false);
		}
	}

 	osg::ref_ptr<osg::AtomicCounterBufferBinding> _acbb;


	

	uint16_t* stop = new uint16_t(2);
	uint32_t* histMax = new uint32_t(0);
	int _buffersize;
	
 	osg::ref_ptr<osg::ShaderStorageBufferBinding> _ssbb;
 };

class ExcessSSB : public osg::Drawable::DrawCallback
{
public:
	VolumeGroup* group;

	ExcessSSB(VolumeGroup* g) : group(g)
	{
	}

	virtual void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const
	{
		if (group->isDirty(renderInfo.getCurrentCamera()->getGraphicsContext()) && stop[0] != 1)
		{
			std::cout << "Excess executed." << std::endl << std::flush;
			osg::ref_ptr<osg::UIntArray> uintArray = new osg::UIntArray(_buffersize);

			drawable->drawImplementation(renderInfo);
			renderInfo.getState()->get<osg::GLExtensions>()->glMemoryBarrier(GL_ALL_BARRIER_BITS);


			stop[0] = 1;
		}
	}

	uint16_t* stop = new uint16_t(2);
	int _buffersize;
	osg::ref_ptr<osg::ShaderStorageBufferBinding> _ssbb;
};
#define GLEX() renderInfo.getState()->get<osg::GLExtensions>()
class Clip1SSB : public osg::Drawable::DrawCallback
{
public:
	VolumeGroup* group;

	Clip1SSB(VolumeGroup* g) : group(g)
	{
	}

	virtual void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const;
	

	static void mapHistogram(uint32_t minVal, uint32_t maxVal, uint32_t numPixelsSB, uint32_t numBins, uint32_t* localHist) {

		float sum = 0;
		const float scale = ((float)(maxVal - minVal)) / (float)numPixelsSB;

		// for each bin
		for (unsigned int i = 0; i < numBins; i++) {

			// add the histogram value for this contextual region to the sum 
			sum += localHist[i];

			// normalize the cdf
			localHist[i] = (unsigned int)(std::min(minVal + sum * scale, (float)maxVal));
		}
	}


	
	uint16_t* stop = new uint16_t(2);
	int _buffersize;
	osg::ref_ptr<osg::ShaderStorageBufferBinding> _ssbbExcess; //excess
	osg::ref_ptr<osg::ShaderStorageBufferBinding> _ssbbHist; //excess
	osg::ref_ptr<osg::ShaderStorageBufferBinding> _ssbbHistMax; //excess
	osg::ref_ptr<osg::Program> _clipshader2;
	int numPixels;
	unsigned int _numGrayVals;
	osg::Vec3i volDims;
	osg::ref_ptr<osg::AtomicCounterBufferBinding> _acbbminMax;

};

class LerpSSB : public osg::Drawable::DrawCallback
{
public:
	VolumeGroup* group;

	LerpSSB(VolumeGroup* g) : group(g)
	{
	}

	virtual void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const
	{
		if (group->isDirty(renderInfo.getCurrentCamera()->getGraphicsContext()) && stop[0] != 1)
		{
			
			drawable->drawImplementation(renderInfo);
			renderInfo.getState()->get<osg::GLExtensions>()->glMemoryBarrier(GL_ALL_BARRIER_BITS);
			std::cout << "CLAHE READY" << std::endl;
			
			stop[0] = 1;
			_claheDirty[0] = 0;
			//group->setDirty(renderInfo.getCurrentCamera()->getGraphicsContext(), false);
		}
	}

	uint16_t* stop = new uint16_t(2);
	uint16_t* _claheDirty;
	int _buffersize;
	osg::ref_ptr<osg::ShaderStorageBufferBinding> _ssbb;
};





#endif
