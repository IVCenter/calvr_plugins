#include "ShaderToyCVR.h"

#include <cvrMenu/MenuSystem.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrConfig/ConfigManager.h>

#include <osg/Depth>
#include <osg/CullFace>
#include <osgViewer/ViewerBase>
#include <osgUtil/CullVisitor>

#include <ctime>
#include <fstream>
#include <string>
#include <sstream>

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#else
#include <sys/types.h>
#include <dirent.h>
#endif

using namespace osg;
using namespace std;
using namespace cvr;

CVRPLUGIN(ShaderToyCVR)

ShaderToyCVR::ShaderToyCVR() {}
ShaderToyCVR::~ShaderToyCVR() {}

static const string vertSource =
	"#version 400\n"
	"uniform mat4 osg_ModelViewProjectionMatrix;\n"
	"uniform vec3 iCameraPosition;\n"
	"layout(location = 0) in vec4 osg_Vertex;\n"
	"out vec3 _WorldRay;"
	"void main() {\n"
	"	_WorldRay = osg_Vertex.xyz;\n"
	"	gl_Position = osg_ModelViewProjectionMatrix * osg_Vertex;\n"
	"}";

static const string fragSource =
	"#version 400\n"
	"uniform vec3 iResolution;"
	"uniform float iTime;"
	"uniform float iTimeDelta;"
	"uniform int iFrame;"
	"uniform float iChannelTime[4];"
	"uniform vec3 iChannelResolution[4];"
	"uniform vec4 iMouse;"
	"uniform sampler2D iChannel0;"
	"uniform sampler2D iChannel1;"
	"uniform sampler2D iChannel2;"
	"uniform sampler2D iChannel3;"
	"uniform vec4 iDate;"
	"uniform float iSampleRate;"
	"uniform vec3 iCameraPosition;"
	"in vec3 _WorldRay;\n";

static const string fragMain = 
	"void main() {\n"
	"	vec4 c;\n"
	"	//mainImage(c, gl_FragCoord.xy);\n"
	"	mainVR(c, gl_FragCoord.xy, iCameraPosition.xzy, normalize(_WorldRay.xzy));\n"
	"	gl_FragColor = c;\n"
	"	gl_FragColor.a = 1.0;\n"
	"}";

struct CameraPositionCallback : public Uniform::Callback {
	osg::Camera* _camera;
	CameraPositionCallback(Camera* camera) : _camera(camera) {}
	virtual void operator()(osg::Uniform* uniform, osg::NodeVisitor* nv) {
		Vec3 camp = Vec3f(0.f, 0.f, 0.f) * _camera->getInverseViewMatrix();
		uniform->set(camp * computeWorldToLocal(nv->getNodePath()) * .01f);
	}
};

void ShaderToyCVR::LoadShader(const string& name) {
	ifstream t;
	t.open((mShaderDir + name).c_str());
	printf("Loading shader %s\n", (mShaderDir + name).c_str());

	string src = fragSource;

	stringstream ss;
	ss << t.rdbuf();
	src += ss.str();
	src += fragMain;
	
	mShader->removeShader(mActiveShader);
	mShader->addShader(mActiveShader = new Shader(Shader::FRAGMENT, src));
}

void ShaderToyCVR::CreateCube() {
	vertices = new Vec3Array();
	indices = new DrawElementsUInt(PrimitiveSet::TRIANGLES, 0);

	vertices->setName("ShaderToy Cube Vertices");
	const float s = 1000000.f;
	vertices->push_back(Vec3f(-s, -s,  s));
	vertices->push_back(Vec3f( s, -s,  s));
	vertices->push_back(Vec3f( s,  s,  s));
	vertices->push_back(Vec3f(-s,  s,  s));
	vertices->push_back(Vec3f(-s, -s, -s));
	vertices->push_back(Vec3f( s, -s, -s));
	vertices->push_back(Vec3f( s,  s, -s));
	vertices->push_back(Vec3f(-s,  s, -s));

	indices->setName("ShaderToy Cube Indices");
	indices->push_back(0); indices->push_back(2); indices->push_back(1);
	indices->push_back(2); indices->push_back(0); indices->push_back(3);
	indices->push_back(1); indices->push_back(6); indices->push_back(5);
	indices->push_back(6); indices->push_back(1); indices->push_back(2);
	indices->push_back(7); indices->push_back(5); indices->push_back(6);
	indices->push_back(5); indices->push_back(7); indices->push_back(4);
	indices->push_back(4); indices->push_back(3); indices->push_back(0);
	indices->push_back(3); indices->push_back(4); indices->push_back(7);
	indices->push_back(4); indices->push_back(1); indices->push_back(5);
	indices->push_back(1); indices->push_back(4); indices->push_back(0);
	indices->push_back(3); indices->push_back(6); indices->push_back(2);
	indices->push_back(6); indices->push_back(3); indices->push_back(7);

	Geometry* geometry = new Geometry();
	geometry->setVertexArray(vertices);
	geometry->addPrimitiveSet(indices);

	Geode* geode = new Geode();
	geode->addDrawable(geometry);

	mShader = new Program();
	mShader->setName("ShaderToy");
	mShader->addShader(new Shader(Shader::VERTEX, vertSource));
	mShader->addShader(mActiveShader = new Shader(Shader::FRAGMENT, fragSource + "void main(){gl_FragColor = vec4(0,1,0,1);}"));

	iCameraPosition = new Uniform(Uniform::FLOAT_VEC3, "iCameraPosition");

	iResolution = new Uniform(Uniform::FLOAT_VEC3, "iResolution");
	iTime = new Uniform(Uniform::FLOAT, "iTime");
	iTimeDelta = new Uniform(Uniform::FLOAT, "iTimeDelta");
	iFrame = new Uniform(Uniform::INT, "iFrame");
	iChannelTime = new Uniform(Uniform::FLOAT, "iChannelTime", 4);
	iChannelResolution = new Uniform(Uniform::FLOAT_VEC3, "iChannelResolution", 4);
	iMouse = new Uniform(Uniform::FLOAT_VEC4, "iMouse");
	iChannel0 = new Uniform(Uniform::SAMPLER_2D, "iChannel0");
	iChannel1 = new Uniform(Uniform::SAMPLER_2D, "iChannel1");
	iChannel2 = new Uniform(Uniform::SAMPLER_2D, "iChannel2");
	iChannel3 = new Uniform(Uniform::SAMPLER_2D, "iChannel3");
	iDate = new Uniform(Uniform::FLOAT_VEC4, "iDate");
	iSampleRate = new Uniform(Uniform::FLOAT, "iSampleRate");

	mState = geode->getOrCreateStateSet();
	mState->setAttributeAndModes(new Depth(Depth::LEQUAL, 0.0, 1.0, false));
	//mState->setAttributeAndModes(new CullFace(CullFace::Mode::BACK));
	mState->setAttributeAndModes(mShader, StateAttribute::ON);

	mState->addUniform(iCameraPosition);

	mState->addUniform(iResolution);
	mState->addUniform(iTime);
	mState->addUniform(iTimeDelta);
	mState->addUniform(iFrame);
	mState->addUniform(iChannelTime);
	mState->addUniform(iChannelResolution);
	mState->addUniform(iMouse);
	mState->addUniform(iChannel0);
	mState->addUniform(iChannel1);
	mState->addUniform(iChannel2);
	mState->addUniform(iChannel3);
	mState->addUniform(iDate);
	mState->addUniform(iSampleRate);

	mSceneObject = new SceneObject("ShaderToy", true, false, false, true, false);
	mSceneObject->addChild(geode);
	PluginHelper::registerSceneObject(mSceneObject, "ShaderToy");
	mSceneObject->attachToScene();
}

bool ShaderToyCVR::init() {
	mSubMenu = new SubMenu("ShaderToy", "ShaderToy");
	mSubMenu->setCallback(this);

	mShaderDir = ConfigManager::getEntry("Plugin.ShaderToyCVR.ShaderDir");

	#ifdef WIN32

	WIN32_FIND_DATA ffd;
	HANDLE hFind = FindFirstFile((mShaderDir + "*.frag").c_str(), &ffd);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			if (ffd.cFileName[0] == '.') continue;

			if (!(ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
				printf("Found shader %s\n", ffd.cFileName);
				auto btn = new MenuButton(ffd.cFileName);
				btn->setCallback(this);
				mSubMenu->addItem(btn);
				mShaderButtons.push_back(btn);
			}
		} while (FindNextFile(hFind, &ffd) != 0);

		FindClose(hFind);
	} else
		printf("Failed to open shader directory\n");

	#else

	DIR* dirp = opendir(mShaderDir.c_str());
	struct dirent* dp;
	while ((dp = readdir(dirp)) != NULL) {
		if (strrchr(dp->d_name, '.') != "frag") continue;

		printf("Found shader %s\n", dp->d_name);
		auto btn = new MenuButton(dp->d_name);
		btn->setCallback(this);
		mSubMenu->addItem(btn);
		mShaderButtons.push_back(btn);
	}
	closedir(dirp);

	#endif

	MenuSystem::instance()->addMenuItem(mSubMenu);

	CreateCube();

	osgViewer::ViewerBase::Cameras cameras;
	CVRViewer::instance()->getCameras(cameras);
	for (osgViewer::ViewerBase::Cameras::iterator it = cameras.begin(); it != cameras.end(); it++) {
		(*it)->getGraphicsContext()->getState()->setUseModelViewAndProjectionUniforms(true);
		iCameraPosition->setUpdateCallback(new CameraPositionCallback(*it));
	}

	return true;
}

void ShaderToyCVR::menuCallback(MenuItem* menuItem) {
	for (unsigned int i = 0; i < mShaderButtons.size(); i++) {
		if (menuItem == mShaderButtons[i]) {
			mStartTime = PluginHelper::getProgramDuration();
			mFrame = 0;
			LoadShader(mShaderButtons[i]->getText());
			break;
		}
	}
}

void ShaderToyCVR::preFrame() {
	mFrame++;

	int w = 1600, h = 900;

	//osgViewer::ViewerBase::Contexts contexts;
	//CVRViewer::instance()->getContexts(contexts);
	//for (const auto& c : contexts) {
	//	w = c->getTraits()->width;
	//	h = c->getTraits()->height;
	//}

	mMouse.set(PluginHelper::getMouseX(), PluginHelper::getMouseY(), 0, 0);

	time_t nowt = time(0);
	tm* now = localtime(&nowt);

	iResolution->set(Vec3(w, h, 1));
	iTime->set((float)(PluginHelper::getProgramDuration() - mStartTime));
	iTimeDelta->set((float)PluginHelper::getLastFrameDuration());
	iFrame->set(mFrame);
	iDate->set(Vec4(now->tm_year, now->tm_mon, now->tm_mday, now->tm_sec));
	iSampleRate->set(44100.0f);
	iMouse->set(mMouse);
}