#include "CLAHE.h"


CLAHE::CLAHE() {
	init();
}
void CLAHE::init() {

	_minMaxShader = new osg::Program;
	_LUTshader = new osg::Program;
	_histShader = new osg::Program;
	_excessShader = new osg::Program;
	_clipShaderPass1 = new osg::Program;
	_clipShaderPass2 = new osg::Program;
	_lerpShader = new osg::Program;
	_focusedLerpShader = new osg::Program;
	
	
	
	CLAHE::loadShader("minMax", minMaxFn, _minMaxShader);
	CLAHE::loadShader("LUT", lutFn, _LUTshader);
	CLAHE::loadShader("hist", histFn, _histShader);
	CLAHE::loadShader("excess", excessFn, _excessShader);
	CLAHE::loadShader("clip1", clipHistFn, _clipShaderPass1);
	CLAHE::loadShader("clip2", clipHist2Fn, _clipShaderPass2);
	CLAHE::loadShader("lerp", lerpFn, _lerpShader);
	CLAHE::loadShader("flerp", flerpFn, _focusedLerpShader);

	initNodes();
}

 void CLAHE::loadShader(std::string name, std::string filename, osg::Program* prog) {
	prog->setName(name);
	prog->addShader(new osg::Shader(osg::Shader::COMPUTE, filename));
}

 void CLAHE::initNodes() {
	_minMaxNode = new osg::DispatchCompute(0, 0, 0);
	_minMaxNode->setDataVariance(osg::Object::DYNAMIC);

	//loadNodes(_minMaxShader);
}

 inline void CLAHE::loadNodes(osg::Program* prog) {

}