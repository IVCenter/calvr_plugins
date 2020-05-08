#pragma once
#include <osg/Program>
#include "HelmsleyVolume.h"
class CLAHE
{
public:
	CLAHE();
	void init();
	void initNodes();
	inline void loadNodes(osg::Program* prog);
	void loadShader(std::string name, std::string filename, osg::Program* prog);

	float _maxClipLimit = 1.0f;
	float _minClipLimit = 0.1f;

	osg::ref_ptr<osg::DispatchCompute> _minMaxNode;
	osg::ref_ptr<osg::DispatchCompute> _LUTNode;
	osg::ref_ptr<osg::DispatchCompute> _histNode;
	osg::ref_ptr<osg::DispatchCompute> _excessNode;
	osg::ref_ptr<osg::DispatchCompute> _clip1Node;
	osg::ref_ptr<osg::DispatchCompute> _clip2Node;
	osg::ref_ptr<osg::DispatchCompute> _lerpNode;
	osg::ref_ptr<osg::DispatchCompute> _flerpNode;
	
protected:
	osg::ref_ptr<osg::Program> _minMaxShader;
	osg::ref_ptr<osg::Program>	_LUTshader;
	osg::ref_ptr<osg::Program> _histShader;
	osg::ref_ptr<osg::Program> _excessShader;
	osg::ref_ptr<osg::Program> _clipShaderPass1;
	osg::ref_ptr<osg::Program> _clipShaderPass2;
	osg::ref_ptr<osg::Program>_lerpShader;
	osg::ref_ptr<osg::Program> _focusedLerpShader;


	std::string minMaxFn = HelmsleyVolume::loadShaderFile("minMax.comp");
	std::string lutFn = HelmsleyVolume::loadShaderFile("LUT.comp");
	std::string histFn = HelmsleyVolume::loadShaderFile("hist.comp");
	std::string excessFn = HelmsleyVolume::loadShaderFile("excess.comp");
	std::string clipHistFn = HelmsleyVolume::loadShaderFile("clipHist.comp");
	std::string clipHist2Fn = HelmsleyVolume::loadShaderFile("clipHist_p2.comp");
	std::string lerpFn = HelmsleyVolume::loadShaderFile("lerp.comp");
	std::string flerpFn = HelmsleyVolume::loadShaderFile("lerp_focused.comp");



	
	
};