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
#include "MarchingCubesLUTs.h"
#include "defines.h"

//#define NUMGRAYVALS 255u
#define CLIPLIMIT3D .85f
#define ORGANCOUNT 8
#define TOOLCOUNT 8

enum class ORGANID {
	COLON = 4u,
	ILLEUM = 16u,
	AORTA = 32u,
	VEIN = 64u
};

enum RENDERBIN_ORDER : int {
	UNDEFINED,
	MINMAX,
	LUT,
	CLAHEHIST,
	EXCESS,
	CLIP,
	LERP,
	VOLUME,
	CUBE,
	MCS,
	NONCLAHEHIST
};






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

#ifdef DEBUGCODE
	void loadRawVolume(std::string path);
#endif // DEBUGCODE



	void loadVolume(std::string path, std::string maskpath = "");
	void loadMask(std::string path, osg::Image* volume);
	void loadAttnMaps(std::string path, osg::Image* volume);
	bool hasMask()
	{
		return _hasMask;
	}
	bool _statusDirty = false;

	////////////////CLAHE
	typedef  osg::ref_ptr<osg::ShaderStorageBufferBinding> t_ssbb;
	typedef  osg::ref_ptr<osg::AtomicCounterBufferBinding> t_acbb;
	///////CLAHE Variables
	
	float _claheRes = 4.0f;
	osg::Vec3i _numSB_3D = osg::Vec3i(4, 4, 2);
	osg::Vec3i _sizeSB = osg::Vec3i(0, 0, 0);


	float _clipLimit3D = 0.85f;
	float _minClipValue = 0.0;
	osg::Vec3i _volDims = osg::Vec3i(0, 0, 0);
 	unsigned int _numGrayVals = 255u;
	unsigned int _histSize = _numSB_3D.x() * _numSB_3D.y() * _numSB_3D.z() * _numGrayVals;
	unsigned int _numHist = _numSB_3D.x() * _numSB_3D.y() * _numSB_3D.z();

	bool _clahePrecomped = false;

	///////CLAHE Methods
	t_acbb precompMinMax();
	t_acbb setupMinmaxSSBO();
	t_ssbb precompLUT(t_acbb minMax);
	t_ssbb setupLUTSSBO(t_acbb minMax);
 	std::pair<t_ssbb, t_ssbb> precompHist();
 	std::pair<t_ssbb, t_ssbb> setupHistSSBO();
 	t_ssbb precompExcess(t_ssbb ssbbHist, t_ssbb ssbbHistMax);
 	t_ssbb setupExcessSSBO(t_ssbb ssbbHist, t_ssbb ssbbHistMax);

 	void precompHistClip(t_ssbb ssbbHist, t_ssbb ssbbHistMax, t_ssbb ssbbExcess, t_acbb acbbminmax);
 	void setupClip(t_ssbb ssbbHist, t_ssbb ssbbHistMax, t_ssbb ssbbExcess, t_acbb acbbminmax);

 	void precompLerp(t_ssbb ssbbHist);
 	void setupLerp(t_ssbb ssbbHist);
	//MC
	MarchingCubesLUTs _luts;
	osg::ref_ptr<osg::Vec3Array> _mcVertices = nullptr;
	osg::ref_ptr<osg::Geometry> geo = nullptr;
	osg::ref_ptr<osg::Vec3Array> _va = nullptr;
	t_ssbb _ssbbTV;
	bool _mcrInitialized = false;
	bool _mcIsReady = false;
	bool _mcPrecomped = false;
	short unsigned int _mcRes = 16;
	ORGANID _mcOrgan = ORGANID::COLON;

	void setMCVertices(osg::ref_ptr<osg::Vec3Array> floats) { _mcVertices = floats; }
	bool isMCInitialized() { return _mcrInitialized; }
	bool toggleMC();
	void intializeMC();
	void genMCs();
	void readyMCUI();
	osg::ref_ptr<osg::Geometry> getMCGeom();
	unsigned int getMCVertCount();
	osg::ref_ptr<osg::Vec3Array> getVA();
	osg::ref_ptr<osg::ShaderStorageBufferBinding> getMCSSBB();
	void printSTLFile();

	//Selection Methods
	void lockSelection(bool lock);
	void removeSelection(bool remove);
	void disableSelection(bool disable);
	//Clahe Methods
	void setNumBins(unsigned int numBins) { 
		_numGrayVals = numBins;
		_histSize = _numSB_3D.x() * _numSB_3D.y() * _numSB_3D.z() * _numGrayVals;
	}
	void setClipLimit(float clipLimit) {
		_clipLimit3D = clipLimit;
	}
	void setClaheRes(float res);
	void genClahe();
	void setCLAHEUseSelection(bool use);

 
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
	void precompMarchingCubes();


	std::vector<osg::ref_ptr<osg::Geode>>* getCenterLines() { return _centerLineGeodes; }
	osg::DispatchCompute* getCompute() { return _computeNode; };
	
	void dirtyVolumeShader() { _program->dirtyProgram(); };
	void dirtyComputeShader() { _computeProgram->dirtyProgram(); };
	
	void readyCenterLine(std::string path);
	void flipCull();

	osg::ref_ptr<osg::Uniform> _PlanePoint;
	osg::ref_ptr<osg::Uniform> _PlaneNormal;
	osg::ref_ptr<osg::Uniform> _StepSize;
 	osg::ref_ptr<osg::Uniform> _maxSteps;
	osg::ref_ptr<osg::Uniform> _RelativeViewport;
	std::map<std::string, osg::ref_ptr<osg::Uniform> > _computeUniforms;

	osg::Matrix getObjectToWorldMatrix();
	osg::Matrix getWorldToObjectMatrix();

	osg::ref_ptr<osg::MatrixTransform> _transform;
	osg::ref_ptr<osg::MatrixTransform> _lineTransform;
	osg::ref_ptr<osg::FrameBufferObject> _resolveFBO;
	
	osg::ref_ptr<osg::Geode> _mcrGeode = nullptr;
	bool _mcrDirty = false;
	void* mcr;

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

		bool toolToggles[(int)TOOLID::COUNT];
	
		bool saved = false;

	};

	Values values;
	bool _UIDirty = false;

#ifdef VOLKIT
 	void setUseVolkit(bool use) {
		_useVolkit = use;
	};
#endif 
	//Getters
	osg::Vec3 getScale() { return _scale; }

protected:
	bool _hasMask;
	bool _claheAvailable = false;
	osg::Vec3 _scale = osg::Vec3(0, 0, 0);

	std::string _minMaxShader;
	std::string _lutShader;
	std::string _excessShader;
	std::string _histShader;
	std::string _clipShader;
	std::string _clipShader2;
	std::string _lerpShader;

	std::string _totalHistShader;
	std::string _marchingCubesShader;


	std::map<osg::GraphicsContext*, bool> _dirty;
	

	osg::ref_ptr<osg::Program> _program;
	osg::ref_ptr<osg::Program> _computeProgram;


	osg::ref_ptr<osg::ShaderStorageBufferObject>  _ssbo;
	osg::ref_ptr<osg::ShaderStorageBufferBinding> _ssbb;
	osg::ref_ptr<osg::UIntArray> data;

	osg::ref_ptr<osg::DispatchCompute> _computeNode;
	osg::ref_ptr<osg::DispatchCompute> _minMaxNode;
	osg::ref_ptr<osg::DispatchCompute> _lutNode;
	osg::ref_ptr<osg::DispatchCompute> _histNode;
	osg::ref_ptr<osg::DispatchCompute> _excessNode;
 	osg::ref_ptr<osg::DispatchCompute> _clipHist1Node;
	osg::ref_ptr<osg::DispatchCompute> _lerpNode;

	osg::ref_ptr<osg::DispatchCompute> _totalHistNode;
	osg::ref_ptr<osg::DispatchCompute> _marchingCubeNode;
	osg::ref_ptr<osg::ShapeDrawable> _cube;
	


	//osg::ref_ptr<osg::ShaderStorageBufferBinding> ssbbHist;
//	osg::ref_ptr<osg::ShaderStorageBufferBinding> ssbbExcess;

	std::vector<osg::ref_ptr<osg::Geode>>* _centerLineGeodes;
	std::shared_ptr<void> _illeumLine;
	std::shared_ptr<void> _colonLine;
	osg::ref_ptr<osg::Vec3dArray> _colonCoords = nullptr;
	osg::ref_ptr<osg::Vec3dArray> _illeumCoords;
	osg::ref_ptr<osg::Texture3D> _volume;
	osg::ref_ptr<osg::Texture3D> _claheVolume;
	osg::ref_ptr<osg::Texture3D> _baked;

	osg::ref_ptr<osg::Texture2D> _depthTexture;
	osg::ref_ptr<osg::Texture2D> _colorTexture;

	osg::ref_ptr<osg::CullFace> _side;
	
	//Volkit
#ifdef VOLKIT
	bool _useVolkit = false;
	std::string _fileName = "";
#endif 
	

 
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
			
			
			drawable->drawImplementation(renderInfo);
			renderInfo.getState()->get<osg::GLExtensions>()->glMemoryBarrier(GL_ALL_BARRIER_BITS);

		
			
		}
	}

	uint16_t* _claheDirty = new uint16_t(1);
};



class MinMaxCallback : public osg::Drawable::DrawCallback
{
public:
	VolumeGroup* group;
	MinMaxCallback(VolumeGroup* g) : group(g)
	{
	}

	virtual void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const
	{
		if (group->isDirty(renderInfo.getCurrentCamera()->getGraphicsContext()) && stop[0] != 1)
		{
 
			drawable->drawImplementation(renderInfo);
			renderInfo.getState()->get<osg::GLExtensions>()->glMemoryBarrier(GL_ALL_BARRIER_BITS);
			auto t1 = std::chrono::high_resolution_clock::now();
			auto ms_int = std::chrono::time_point_cast<std::chrono::milliseconds>(t1);
			auto epoch = ms_int.time_since_epoch();
			auto value = std::chrono::duration_cast<std::chrono::milliseconds>(epoch);
			const_cast<long&>(_t1) = value.count();
			
			stop[0] = 1;
			
		}
	}

	uint16_t* stop = new uint16_t(2);
	osg::ref_ptr<osg::AtomicCounterBufferBinding> _acbb;
	long _t1;
 
};



class LUTCallback : public osg::Drawable::DrawCallback
{
public:
	VolumeGroup* group;
	LUTCallback(VolumeGroup* g) : group(g)
	{
	}

	virtual void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const
	{
		if (group->isDirty(renderInfo.getCurrentCamera()->getGraphicsContext()) && stop[0] != 1)
		{
 
			drawable->drawImplementation(renderInfo);
			renderInfo.getState()->get<osg::GLExtensions>()->glMemoryBarrier(GL_ALL_BARRIER_BITS);
			 
			stop[0] = 1;
			
		}
	}

	uint16_t* stop = new uint16_t(2);
	osg::ref_ptr<osg::ShaderStorageBufferBinding> _ssbb;
	unsigned int _buffersize;
};



class CLAHEHistCallback : public osg::Drawable::DrawCallback
{
public:
	VolumeGroup* group;

	CLAHEHistCallback(VolumeGroup* g) : group(g)
	{
	}

	virtual void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const
	{
		
		if (group->isDirty(renderInfo.getCurrentCamera()->getGraphicsContext()) && stop[0] != 1)
		{
			
			drawable->drawImplementation(renderInfo);
			renderInfo.getState()->get<osg::GLExtensions>()->glMemoryBarrier(GL_ALL_BARRIER_BITS);

			///////////////DEBUGGING
			{
				osg::ref_ptr<osg::UIntArray> uintArray = new osg::UIntArray(_buffersize);
				osg::GLBufferObject* glBufferObject = _ssbb->getBufferData()->getBufferObject()->getOrCreateGLBufferObject(renderInfo.getState()->getContextID());
				//std::cout << glBufferObject << std::endl;

				GLint previousID = 1;
				glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING, &previousID);

				if ((GLuint)previousID != glBufferObject->getGLObjectID())
					glBufferObject->_extensions->glBindBuffer(GL_SHADER_STORAGE_BUFFER, glBufferObject->getGLObjectID());

				GLubyte* data = (GLubyte*)glBufferObject->_extensions->glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY_ARB);
				//std::cout << data << std::endl;
				if (data)
				{
					size_t size = osg::minimum<int>(_ssbb->getSize(), uintArray->getTotalDataSize());
					memcpy((void*)&(uintArray->front()), data + _ssbb->getOffset(), size);
					glBufferObject->_extensions->glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
				}

				if ((GLuint)previousID != glBufferObject->getGLObjectID())
					glBufferObject->_extensions->glBindBuffer(GL_SHADER_STORAGE_BUFFER, previousID);


				unsigned int value = uintArray->front();
				//std::cout << "Hist before clip" << value << std::endl;

				/*for (int i = 0; i < 100; i++) {
					std::cout << uintArray->at(i) << std::endl;

				}*/


			}
			//DEBUGGING/////////////////////////////
			/////////////////DEBUGGING
			//{
			//	osg::ref_ptr<osg::UIntArray> uintArray = new osg::UIntArray(_buffersize2);
			//	osg::GLBufferObject* glBufferObject = _ssbb2->getBufferData()->getBufferObject()->getOrCreateGLBufferObject(renderInfo.getState()->getContextID());
			//	//std::cout << glBufferObject << std::endl;

			//	GLint previousID = 1;
			//	glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING, &previousID);

			//	if ((GLuint)previousID != glBufferObject->getGLObjectID())
			//		glBufferObject->_extensions->glBindBuffer(GL_SHADER_STORAGE_BUFFER, glBufferObject->getGLObjectID());

			//	GLubyte* data = (GLubyte*)glBufferObject->_extensions->glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY_ARB);
			//	//std::cout << data << std::endl;
			//	if (data)
			//	{
			//		size_t size = osg::minimum<int>(_ssbb2->getSize(), uintArray->getTotalDataSize());
			//		memcpy((void*)&(uintArray->front()), data + _ssbb2->getOffset(), size);
			//		glBufferObject->_extensions->glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
			//	}

			//	if ((GLuint)previousID != glBufferObject->getGLObjectID())
			//		glBufferObject->_extensions->glBindBuffer(GL_SHADER_STORAGE_BUFFER, previousID);


			//	/*unsigned int value = uintArray->front();
			//	std::cout << "Hist Check before Lerp " << value << std::endl;*/
			//	std::cout << "testing histmax..." << std::endl;
			//	for (int i = 0; i < _buffersize2; i++) {
			//		std::cout << uintArray->at(i) << std::endl;
			//	}

			//}
			////DEBUGGING/////////////////////////////


			stop[0] = 1;
 
		}
	}

	uint16_t* stop = new uint16_t(1);
	int _buffersize;
	int _buffersize2;
	osg::ref_ptr<osg::ShaderStorageBufferBinding> _ssbb;
	osg::ref_ptr<osg::ShaderStorageBufferBinding> _ssbb2;
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
 			/*osg::ref_ptr<osg::UIntArray> uintArray = new osg::UIntArray(_buffersize);*/

			drawable->drawImplementation(renderInfo);
			renderInfo.getState()->get<osg::GLExtensions>()->glMemoryBarrier(GL_ALL_BARRIER_BITS);

			



			stop[0] = 1;
		}
	}

	uint16_t* stop = new uint16_t(1);
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


	
	uint16_t* stop = new uint16_t(1);
	int _buffersize;
	osg::ref_ptr<osg::ShaderStorageBufferBinding> _ssbbExcess; //excess
	osg::ref_ptr<osg::ShaderStorageBufferBinding> _ssbbHist; //excess
	osg::ref_ptr<osg::ShaderStorageBufferBinding> _ssbbHistMax; //excess
	osg::ref_ptr<osg::Program> _clipshader2;
	int numPixels;
	unsigned int _numGrayVals;
	float _clipLimit;
	
	osg::Vec3i volDims;
 	osg::Vec3i _sb3D;
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
 			
			
			//group->setDirty(renderInfo.getCurrentCamera()->getGraphicsContext(), false);
 			auto t2 = std::chrono::high_resolution_clock::now();
			auto ms_int = std::chrono::time_point_cast<std::chrono::milliseconds>(t2);
			auto epoch = ms_int.time_since_epoch();
			auto value = std::chrono::duration_cast<std::chrono::milliseconds>(epoch);

			std::cout << "time: " << value.count() - _minmaxCallback->_t1   << " milliseconds"<< std::endl;
 			 
			stop[0] = 1;
			_claheDirty[0] = 0;
		}
	}

	uint16_t* stop = new uint16_t(1);
	uint16_t* _claheDirty;
	int _buffersize;
	osg::ref_ptr<osg::ShaderStorageBufferBinding> _ssbb;
	MinMaxCallback* _minmaxCallback;


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
			drawable->drawImplementation(renderInfo);
			renderInfo.getState()->get<osg::GLExtensions>()->glMemoryBarrier(GL_ALL_BARRIER_BITS);

			/*osg::ref_ptr<osg::UIntArray> _atomicCounterArray = new osg::UIntArray(1);
			
			_acbb->readData(*renderInfo.getState(), *_atomicCounterArray);

			unsigned int value = osg::maximum(1u, _atomicCounterArray->front());*/

			histMax[0] = 1000u;
			stop[0] = 1;
			group->setDirty(renderInfo.getCurrentCamera()->getGraphicsContext(), false);

		}
	}

	osg::ref_ptr<osg::AtomicCounterBufferBinding> _acbb;




	uint16_t* stop = new uint16_t(0);
	uint32_t* histMax = new uint32_t(0);
	

	osg::ref_ptr<osg::ShaderStorageBufferBinding> _ssbb;
};



class MarchingCubeCallback : public osg::Drawable::DrawCallback
{
public:
	VolumeGroup* group;

	MarchingCubeCallback(VolumeGroup* g) : group(g)
	{
	}

	virtual void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const
	{

		if (group->isDirty(renderInfo.getCurrentCamera()->getGraphicsContext()) && stop[0] != 1)
		{
			drawable->drawImplementation(renderInfo);
			renderInfo.getState()->get<osg::GLExtensions>()->glMemoryBarrier(GL_ALL_BARRIER_BITS);
			_VA->dirty();
			
			osg::ref_ptr<osg::UIntArray> _atomicCounterArray = new osg::UIntArray();
			_atomicCounterArray->push_back(0);
			_acbb->readData(*renderInfo.getState(), *_atomicCounterArray);
			renderInfo.getState()->get<osg::GLExtensions>()->glMemoryBarrier(GL_ALL_BARRIER_BITS);
			
			unsigned int value = _atomicCounterArray->front();
			//std::cout << "value: " << value << std::endl;
			vertexCount[0] = value;


 			{
				osg::ref_ptr<osg::Vec3Array> uintArray = new osg::Vec3Array(_buffersize/3);
				osg::GLBufferObject* glBufferObject = _ssbb->getBufferData()->getBufferObject()->getOrCreateGLBufferObject(renderInfo.getState()->getContextID());
 				GLint previousID = 1;
				glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING, &previousID);

				if ((GLuint)previousID != glBufferObject->getGLObjectID())
					glBufferObject->_extensions->glBindBuffer(GL_SHADER_STORAGE_BUFFER, glBufferObject->getGLObjectID());

				GLubyte* data = (GLubyte*)glBufferObject->_extensions->glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY_ARB);
				if (data)
				{
					size_t size = osg::minimum<int>(_ssbb->getSize(), uintArray->getTotalDataSize());
					memcpy((void*)&(uintArray->front()), data + _ssbb->getOffset(), size);
					glBufferObject->_extensions->glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
				}

				if ((GLuint)previousID != glBufferObject->getGLObjectID())
					glBufferObject->_extensions->glBindBuffer(GL_SHADER_STORAGE_BUFFER, previousID);

				
				group->setMCVertices(uintArray);
			}


			
			stop[0] = 1;
			ready[0] = 1;
			
			group->readyMCUI();

			group->setDirty(renderInfo.getCurrentCamera()->getGraphicsContext(), false);

 		}
	}

	//const_cast
 	uint16_t* stop = new uint16_t(1);
	uint16_t* ready = new uint16_t(0);
	uint32_t* vertexCount = new uint32_t(0);
	int _buffersize;
	osg::ref_ptr<osg::Vec3dArray> vertices;
	osg::ref_ptr<osg::Vec3Array> _VA;
	MarchingCubesLUTs luts;
	osg::ref_ptr<osg::ShaderStorageBufferBinding> _ssbb;
	osg::Vec3i _volDims;
	osg::ref_ptr<osg::Geometry> _geomToPass;
	osg::ref_ptr<osg::AtomicCounterBufferBinding> _acbb;


};




#endif
