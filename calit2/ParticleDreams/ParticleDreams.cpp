#include "ParticleDreams.h"
#include "CudaParticle.h"
#include "CudaHelper.h"

#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/Navigation.h>
#include <cvrKernel/CVRViewer.h>
#include <cvrKernel/CVRStatsHandler.h>
#include <cvrConfig/ConfigManager.h>

#include <osg/PointSprite>
#include <osg/BlendFunc>
#include <osg/Depth>
#include <osgDB/FileUtils>
#include <osgDB/ReadFile>

#include <cuda_gl_interop.h>

#include <sys/time.h>

float ftToM(float feet)
{
    return feet * 0.3048;
}

double getTimeInSecs()
{
    struct timeval tv;
    gettimeofday(&tv,NULL);

    return tv.tv_sec + (tv.tv_usec / 1000000.0);
}

struct MyComputeBounds : public osg::Drawable::ComputeBoundingBoxCallback
{
    MyComputeBounds() {}
    MyComputeBounds(const MyComputeBounds & mcb, const osg::CopyOp &) {}
    virtual osg::BoundingBox computeBound(const osg::Drawable &) const
    {
	return _bound;
    }

    osg::BoundingBox _bound;
};

using namespace cvr;

CVRPLUGIN(ParticleDreams)

ParticleDreams::ParticleDreams()
{
}

ParticleDreams::~ParticleDreams()
{
	oasclient::ClientInterface::shutdown();
}

bool ParticleDreams::init()
{
    _myMenu = new SubMenu("Particle Dreams");

    _enable = new MenuCheckbox("Enable",false);
    _enable->setCallback(this);
    _myMenu->addItem(_enable);

    _dataDir = ConfigManager::getEntry("value","Plugin.ParticleDreams.DataDir","") + "/";

    PluginHelper::addRootMenuItem(_myMenu);

    hand_id = ConfigManager::getInt("value","Plugin.ParticleDreams.HandID",0);

    return true;
}

void ParticleDreams::menuCallback(MenuItem * item)
{
    if(item == _enable)
    {
	if(_enable->getValue())
	{

	    CVRViewer::instance()->getStatsHandler()->addStatTimeBar(CVRStatsHandler::CAMERA_STAT,"PDCuda Time:","PD Cuda duration","PD Cuda start","PD Cuda end",osg::Vec3(0,1,0),"PD stats");
	    //CVRViewer::instance()->getStatsHandler()->addStatTimeBar(CVRStatsHandler::CAMERA_STAT,"PDCuda Copy:","PD Cuda Copy duration","PD Cuda Copy start","PD Cuda Copy end",osg::Vec3(0,0,1),"PD stats");

	    initPart();
	    initGeometry();
	    initSound();
	}
    }
}

void ParticleDreams::preFrame()
{
    if(_enable->getValue())
    {
	//do driver thread part of step
	
	double intigrate_time =1;
	//timeing
	showTime=getTimeInSecs() - showStartTime;
	showFrameNo++;
	
	nowTime = getTimeInSecs();
	
	//first print out meaningless
	if ( (nowTime - startTime) > intigrate_time)
	{
		
	    if (FR_RATE_PRINT >0) printf("%f ms %f FR/sec  ",intigrate_time/frNum*1000,frNum/intigrate_time);
	    startTime = nowTime;  frNum =0;
	}
	frNum++;

	updateHand();

	sceneChange =0;

	if ((but4old ==0)&&(but4 == 1)&&(but1))
	{
	    sceneOrder =( sceneOrder+1)%5;sceneChange=1;
	}
	if (nextSean ==1) { sceneOrder =( sceneOrder+1)%5;sceneChange=1;nextSean =0;}
	//reordering seenes

	if (sceneOrder ==0)sceneNum =4;
	if (sceneOrder ==1)sceneNum =1;
	if (sceneOrder ==2)sceneNum =2;
	if (sceneOrder ==3)sceneNum =0;
	if (sceneOrder ==4)sceneNum =3;

	//if((sceneChange==1) && (witch_scene ==3)){scene_data_3_kill_audio();}
	//if((sceneChange==1) && (witch_scene ==0)){scene_data_0_kill_audio();}
	//if((sceneChange==1) && (witch_scene ==1)){scene_data_1_kill_audio();}
	//if((sceneChange==1) && (witch_scene ==2)){scene_data_2_kill_audio();}
	//if((sceneChange==1) && (witch_scene ==4)){scene_data_4_kill_audio();}

	if (sceneChange)
	{
		if (witch_scene == 0) 		scene_data_0_kill_audio();
		else if (witch_scene == 1)	scene_data_1_kill_audio();
		else if (witch_scene == 2)	scene_data_2_kill_audio();
		else if (witch_scene == 3)	scene_data_3_kill_audio();
		else if (witch_scene == 4)	scene_data_4_kill_audio();
	}

	if (sceneNum ==0)
	{//paint on walls
	    if (sceneChange ==1){scene0Start =1;sceneChange =0;witch_scene =0;}
	    scene_data_0_host();
	}
	if (sceneNum ==1)
	{//sprial fountens
	    if (sceneChange ==1){scene1Start =1;sceneChange =0;witch_scene =1;}
	    scene_data_1_host();
	}
	if (sceneNum ==2)
	{//4 waterfalls
	    if (sceneChange ==1){scene2Start =1;sceneChange =0;witch_scene =2;}
	    scene_data_2_host();
	}
	if (sceneNum ==3)
	{//painting skys
	    if (sceneChange ==1){scene3Start =1;sceneChange =0;witch_scene =3;}
	    scene_data_3_host();
	}

	if (sceneNum ==4)
	{//rain
	    if (sceneChange ==1){scene4Start =1;sceneChange =0;witch_scene =4;}
	    scene_data_4_host();
	}

	for ( int n =1;n < h_injectorData[0][0][0] +1;n++)
	{
	    // kludge to handel gimbel lock for velociys straight up			
	    if (h_injectorData[n][3][0] ==0 && h_injectorData[n][3][2] == 0){ h_injectorData[n][3][0] += .0001;}
	}

	but4old = but4;
	but3old = but3;
	but2old = but2;
	but1old = but1;
	triggerold = trigger;
    }
}

bool ParticleDreams::processEvent(InteractionEvent * event)
{
    TrackedButtonInteractionEvent * tie = event->asTrackedButtonEvent();
    if(tie)
    {
	if(tie->getHand() == hand_id)
	{
	    if((tie->getInteraction() == BUTTON_DOWN || tie->getInteraction() == BUTTON_DOUBLE_CLICK) && tie->getButton() <= 4)
	    {
		if(tie->getButton() == 0)
		{
		    trigger = 1;
		}
		else if(tie->getButton() == 1)
		{
		    but1 = 1;
		}
		else if(tie->getButton() == 2)
		{
		    but2 = 1;
		}
		else if(tie->getButton() == 3)
		{
		    but3 = 1;
		}
		else if(tie->getButton() == 4)
		{
		    but4 = 1;
		}
	    }
	    else if(tie->getInteraction() == BUTTON_UP && tie->getButton() <= 4)
	    {
		if(tie->getButton() == 0)
		{
		    trigger = 0;
		}
		else if(tie->getButton() == 1)
		{
		    but1 = 0;
		}
		else if(tie->getButton() == 2)
		{
		    but2 = 0;
		}
		else if(tie->getButton() == 3)
		{
		    but3 = 0;
		}
		else if(tie->getButton() == 4)
		{
		    but4 = 0;
		}
	    }
	}
    }

    return false;
}

void ParticleDreams::perContextCallback(int contextid) const
{
    if(CVRViewer::instance()->done())
    {
	//TODO: add cuda cleanup
	return;
    }
    //std::cerr << "ContextID: " << contextid << std::endl;
    _callbackLock.lock();
    if(!_callbackInit[contextid])
    {

	int cudaDevice = ScreenConfig::instance()->getCudaDevice(contextid);
	cudaGLSetGLDevice(cudaDevice);
	cudaSetDevice(cudaDevice); 
	//std::cerr << "CudaDevice: " << cudaDevice << std::endl;

	printCudaErr();
	osg::VertexBufferObject * vbo = _particleGeo->getOrCreateVertexBufferObject();
	vbo->setUsage(GL_DYNAMIC_DRAW);
	osg::GLBufferObject * glbo = vbo->getOrCreateGLBufferObject(contextid);
	//std::cerr << "Context: " << contextid << " VBO id: " << glbo->getGLObjectID() << " size: " << vbo->computeRequiredBufferSize() << std::endl;
	checkRegBufferObj(glbo->getGLObjectID());
	printCudaErr();

	if(cuMemAlloc(&d_debugDataMap[contextid], 128 * sizeof(float)) == CUDA_SUCCESS)
	{
	    cuMemcpyHtoD(d_debugDataMap[contextid], h_debugData, 128 * sizeof(float));
	    printCudaErr();
	}
	else
	{
	    std::cerr << "d_debugData cuda alloc failed." << std::endl;
	    printCudaErr();
	}

	size_t psize = PDATA_ROW_SIZE * CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT * sizeof(float);
	if(cuMemAlloc(&d_particleDataMap[contextid], psize) == CUDA_SUCCESS)
	{
	    cuMemcpyHtoD(d_particleDataMap[contextid], h_particleData, psize);
	    printCudaErr();
	}
	else
	{
	    std::cerr << "d_particleData cuda alloc failed." << std::endl;
	    printCudaErr();
	}



	_callbackInit[contextid] = true;
    }
    _callbackLock.unlock();

    osg::Stats * stats = NULL;

    osgViewer::ViewerBase::Contexts contexts;
    CVRViewer::instance()->getContexts(contexts);

    for(osgViewer::ViewerBase::Contexts::iterator citr = contexts.begin(); citr != contexts.end();
                ++citr)
    {
	if((*citr)->getState()->getContextID() != contextid)
	{
	    continue;
	}

	osg::GraphicsContext::Cameras& cameras = (*citr)->getCameras();
	for(osg::GraphicsContext::Cameras::iterator camitr = cameras.begin(); camitr != cameras.end();++camitr)
	{
	    if((*camitr)->getStats())
	    {
		stats = (*camitr)->getStats();
		break;
	    }
	}

	if(stats)
	{
	    break;
	}
    }

    double cudastart, cudaend;
    double cudacopystart, cudacopyend;

    if(stats && ! stats->collectStats("PD stats"))
    {
	stats = NULL;
    }

    if(stats)
    {
	cudastart = osg::Timer::instance()->delta_s(CVRViewer::instance()->getStartTick(), osg::Timer::instance()->tick());
    }

    if (sceneNum ==0)
    {//paint on walls
	scene_data_0_context(contextid);
    }
    if (sceneNum ==1)
    {//sprial fountens
	scene_data_1_context(contextid);
    }
    if (sceneNum ==2)
    {//4 waterfalls
	scene_data_2_context(contextid);
    }
    if (sceneNum ==3)
    {//painting skys
	scene_data_3_context(contextid);
    }

    if (sceneNum ==4)
    {//rain
	scene_data_4_context(contextid);
    }

    //if(stats)
    //{
    //	cudacopystart = osg::Timer::instance()->delta_s(CVRViewer::instance()->getStartTick(), osg::Timer::instance()->tick());
    //}

    //cudaMemcpyToSymbol("injdata",h_injectorData,sizeof(h_injectorData));
    //cudaMemcpyToSymbol("refldata",h_reflectorData,sizeof(h_reflectorData));
    setReflData((void*)h_reflectorData,sizeof(h_reflectorData));
    setInjData((void*)h_injectorData,sizeof(h_injectorData));

    //if(stats)
    //{
    //	cudacopyend = osg::Timer::instance()->delta_s(CVRViewer::instance()->getStartTick(), osg::Timer::instance()->tick());
    //}

    //process audio fades
    //if ((SOUND_SERV ==1)&& (::host->root() == 1)){	audioProcess();}

    CUdeviceptr d_vbo;
    GLuint vbo = _particleGeo->getOrCreateVertexBufferObject()->getOrCreateGLBufferObject(contextid)->getGLObjectID();

    checkMapBufferObj((void**)&d_vbo,vbo);

    float * d_colorptr = (float*)d_vbo;
    d_colorptr += 3*_positionArray->size();

    launchPoint1((float3*)d_vbo,(float4*)d_colorptr,(float*)d_particleDataMap[contextid],(float*)d_debugDataMap[contextid],CUDA_MESH_WIDTH,CUDA_MESH_HEIGHT,max_age,disappear_age,alphaControl,anim,gravity,colorFreq,0.0);


    printCudaErr();

    cudaThreadSynchronize();

    checkUnmapBufferObj(vbo);

    if(stats)
    {
	cudaend = osg::Timer::instance()->delta_s(CVRViewer::instance()->getStartTick(), osg::Timer::instance()->tick());
        stats->setAttribute(CVRViewer::instance()->getViewerFrameStamp()->getFrameNumber(), "PD Cuda start", cudastart);
        stats->setAttribute(CVRViewer::instance()->getViewerFrameStamp()->getFrameNumber(), "PD Cuda end", cudaend);
        stats->setAttribute(CVRViewer::instance()->getViewerFrameStamp()->getFrameNumber(), "PD Cuda duration", cudaend-cudastart);
	
	//stats->setAttribute(CVRViewer::instance()->getViewerFrameStamp()->getFrameNumber(), "PD Cuda Copy start", cudacopystart);
        //stats->setAttribute(CVRViewer::instance()->getViewerFrameStamp()->getFrameNumber(), "PD Cuda Copy end", cudacopyend);
        //stats->setAttribute(CVRViewer::instance()->getViewerFrameStamp()->getFrameNumber(), "PD Cuda Copy duration", cudacopyend-cudacopystart);
    }
}

void ParticleDreams::initPart()
{
    max_age = 2000;
    gravity = 0.0001;
    anim = 0;
    disappear_age = 2000;
    showFrameNo = 0;
    lastShowFrameNo = -1;
    showStartTime = 0;
    showTime = 0;
    lastShowTime = -1;
    startTime = 0;
    nowTime = 0;
    frNum = 1;
    colorFreq = 16;
    draw_water_sky = 1;
    //TODO: get from config file
    hand_id = 0;
    state =0;
    trigger =0;
    but4 =0;
    but3 =0;
    but2 =0;
    but1 =0;
    // init seenes
    scene0Start =0;
    scene1Start =0;
    scene2Start =0;
    scene3Start =0;
    scene4Start =1;//// must be set to starting
    scene0ContextStart =0;
    scene1ContextStart =0;
    scene2ContextStart =0;
    scene3ContextStart =0;
    scene4ContextStart =0;
    sceneNum =0;
    sceneOrder = 0;
    nextSean =0;
    witch_scene =4;// must be set to starting
    old_witch_scene = -1;
    sceneChange =0;
    modulator[0] = 1.0;
    modulator[1] = 1.0;
    modulator[2] = 1.0;
    modulator[3] = 1.0;

    // zero out h_reflectorData
    for (int reflNum =0;reflNum < REFL_DATA_MUNB;reflNum++)
    {
	for (int rownum =0;rownum < REFL_DATA_ROWS;rownum++)
	{ 
	    h_reflectorData[reflNum ][rownum][0]=0;
	    h_reflectorData[reflNum ][rownum][1]=0;
	    h_reflectorData[reflNum ][rownum][2]=0;
	}
    }

    for (int injNum =0;injNum < INJT_DATA_MUNB;injNum++)
    {
	for (int rownum =0;rownum < INJT_DATA_ROWS;rownum++)
	{ 
	    h_injectorData[injNum ][rownum][0]=0;
	    h_injectorData[injNum ][rownum][1]=0;
	    h_injectorData[injNum ][rownum][2]=0;
	}
    }

    sizeDebug = 128;

    h_debugData = new float[sizeDebug];
    for (int i = 0; i < 128; ++i)
    {
	h_debugData[i]=0;
	old_refl_hits[i] = 0;
	refl_hits[i] = 0;
    }

    int rowsize = PDATA_ROW_SIZE;
    size_t size = rowsize * CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT;

    srand(1);

    h_particleData = new float[size];

    for (int i = 0; i < CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT; ++i)
    {
        // set age to random ages < max age to permit a respawn of the particle
        h_particleData[PDATA_ROW_SIZE*i] = rand() % max_age; // age
 
    }

    // init velocity
    for (int i = 0; i < CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT; ++i)
    { 
        h_particleData[PDATA_ROW_SIZE * i + 1] = -10000;
        h_particleData[PDATA_ROW_SIZE * i + 2] = -10000;
        h_particleData[PDATA_ROW_SIZE * i + 3] = -10000;
    }

    for (int i = 0; i < CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT; ++i)
    {
        // gen 3 random numbers for each partical
        h_particleData[PDATA_ROW_SIZE * i +4] = 0.0002 * (rand()%10000) -1.0 ;
        h_particleData[PDATA_ROW_SIZE * i +5] = 0.0002 * (rand()%10000) -1.0 ;
        h_particleData[PDATA_ROW_SIZE * i +6] = 0.0002 * (rand()%10000) -1.0 ;
        //printf ( "rnd num %f %f %f \n", h_particleData[PDATA_ROW_SIZE * i +4],h_particleData[PDATA_ROW_SIZE * i +5],h_particleData[PDATA_ROW_SIZE * i +6]);
    }

    CVRViewer::instance()->addPerContextPostFinishCallback(this);
}

void ParticleDreams::initGeometry()
{
    _particleObject = new PDObject("Particle Dreams",false,false,false,false,false);

    _particleGeode = new osg::Geode();
    _particleGeo = new osg::Geometry();

    _particleGeo->setUseDisplayList(false);
    _particleGeo->setUseVertexBufferObjects(true);

    MyComputeBounds * mcb = new MyComputeBounds();
    _particleGeo->setComputeBoundingBoxCallback(mcb);
    mcb->_bound = osg::BoundingBox(osg::Vec3(-100000,-100000,-100000),osg::Vec3(100000,100000,100000));

    _positionArray = new osg::Vec3Array(CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT);
    for(int i = 0; i < CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT; i++)
    {
	//_positionArray->at(i) = osg::Vec3((rand()%2000)-1000.0,(rand()%2000)-1000.0,(rand()%2000)-1000.0);
	_positionArray->at(i) = osg::Vec3(0,0,0);
    }

    _colorArray = new osg::Vec4Array(CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT);
    for(int i = 0; i < CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT; i++)
    {
	_colorArray->at(i) = osg::Vec4(0.0,0.0,0.0,0.0);
    }

    _particleGeo->setVertexArray(_positionArray);
    _particleGeo->setColorArray(_colorArray);
    _particleGeo->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    _particleGeo->dirtyBound();

    _primitive = new osg::DrawArrays(osg::PrimitiveSet::POINTS,0,CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT);
    _particleGeo->addPrimitiveSet(_primitive);

    osg::PointSprite * sprite = new osg::PointSprite();
    osg::BlendFunc * blend = new osg::BlendFunc();
    blend->setFunction(osg::BlendFunc::SRC_ALPHA, osg::BlendFunc::ONE);
    osg::Depth * depth = new osg::Depth();
    depth->setWriteMask(false);

    osg::StateSet * stateset = _particleGeo->getOrCreateStateSet();
    stateset->setTextureAttributeAndModes(0, sprite, osg::StateAttribute::ON);
    stateset->setAttributeAndModes(blend, osg::StateAttribute::ON);
    stateset->setAttributeAndModes(depth, osg::StateAttribute::ON);
    stateset->setMode(GL_BLEND,osg::StateAttribute::ON);
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
   
    _spriteVert = osg::Shader::readShaderFile(osg::Shader::VERTEX, osgDB::findDataFile(_dataDir + "glsl/sprite.vert"));
    _spriteFrag = osg::Shader::readShaderFile(osg::Shader::FRAGMENT, osgDB::findDataFile(_dataDir + "glsl/sprite.frag"));
    _spriteProgram = new osg::Program();
    _spriteProgram->setName("Sprite");
    _spriteProgram->addShader(_spriteVert);
    _spriteProgram->addShader(_spriteFrag);
    stateset->setAttribute(_spriteProgram);
    stateset->setMode(GL_VERTEX_PROGRAM_POINT_SIZE, osg::StateAttribute::ON);

    _spriteTexture = new osg::Texture2D();
    osg::ref_ptr<osg::Image> image = osgDB::readImageFile(_dataDir + "glsl/sprite.png");
    if(image)
    {
	_spriteTexture->setImage(image);
	_spriteTexture->setWrap(osg::Texture::WRAP_S,osg::Texture::CLAMP_TO_EDGE);
	_spriteTexture->setWrap(osg::Texture::WRAP_T,osg::Texture::CLAMP_TO_EDGE);
	_spriteTexture->setFilter(osg::Texture::MIN_FILTER,osg::Texture::LINEAR_MIPMAP_LINEAR);
	_spriteTexture->setFilter(osg::Texture::MAG_FILTER,osg::Texture::LINEAR);
	_spriteTexture->setResizeNonPowerOfTwoHint(false);
	stateset->setTextureAttributeAndModes(0,_spriteTexture,osg::StateAttribute::ON);
    }
    else
    {
	std::cerr << "Unable to read sprite texture: " << _dataDir + "glsl/sprite.png" << std::endl;
    }

    _particleGeode->addDrawable(_particleGeo);
    _particleObject->addChild(_particleGeode);
    PluginHelper::registerSceneObject(_particleObject);
    _particleObject->attachToScene();
    _particleObject->setNavigationOn(true);

    osg::Matrix m, ms, mt;
    m.makeRotate((90.0/180.0)*M_PI,osg::Vec3(1.0,0,0));
    ms.makeScale(osg::Vec3(1000.0,1000.0,1000.0));
    mt.makeTranslate(osg::Vec3(0,0,-Navigation::instance()->getFloorOffset()));
    _particleObject->setTransform(m*ms*mt);
}

void ParticleDreams::initSound()
{
    std::string ipAddrStr;
    unsigned short port;
    std::string pathToAudioFiles;

    ipAddrStr =  ConfigManager::getEntry("ipAddr","Plugin.ParticleDreams.SoundServer","127.0.0.1");
;
    port = ConfigManager::getInt("port", "Plugin.ParticleDreams.SoundServer", 31231);

    if (!oasclient::ClientInterface::initialize(ipAddrStr, port))
    {
        std::cerr << "Unable to connect to sound server at " << ipAddrStr << ", port " << port << std::endl;
        soundEnabled = false;
        return;
    }

    soundEnabled = true;

    // LOAD pathToAudioFiles HERE
    pathToAudioFiles = _dataDir + "/sound/";

    oasclient::Listener::getInstance().setGain(0.7);
    
    std::cerr << pathToAudioFiles << std::endl;
    chimes.initialize(pathToAudioFiles, "chimes.wav");

    pinkNoise.initialize(pathToAudioFiles, "cdtds.31.pinkNoise.wav");
    pinkNoise.setGain(0); pinkNoise.setLoop(true); pinkNoise.play(1.0);

    dan_texture_09.initialize(pathToAudioFiles, "dan_texture_09.wav");
    dan_texture_09.setGain(0); dan_texture_09.setLoop(true); dan_texture_09.play(1.0);

    texture_12.initialize(pathToAudioFiles, "dan_texture_12.wav");
    texture_12.setGain(0); texture_12.setLoop(true); texture_12.play(1.0);

    short_sound_01a.initialize(pathToAudioFiles, "dan_short_sound_01a.wav");
    short_sound_01a.play(1);

    texture_17_swirls3.initialize(pathToAudioFiles, "dan_texture_17_swirls3.wav");
    rain_at_sea.initialize(pathToAudioFiles, "dan_texture_18_rain_at_sea.wav");
    dan_texture_13.initialize(pathToAudioFiles, "dan_texture_13.wav");
    dan_texture_05.initialize(pathToAudioFiles, "dan_texture_05.wav");
    dan_short_sound_04.initialize(pathToAudioFiles, "dan_short_sound_04.wav");
    dan_ambiance_2.initialize(pathToAudioFiles, "dan_ambiance_2.wav");
    dan_ambiance_1.initialize(pathToAudioFiles, "dan_ambiance_1.wav");
    dan_5min_ostinato.initialize(pathToAudioFiles, "dan_10120607_5_min_ostinato.WAV");
    dan_10120603_Rez1.initialize(pathToAudioFiles, "dan_10120603_Rez.1.wav");
    dan_mel_amb_slower.initialize(pathToAudioFiles, "dan_10122604_mel_amb_slower.wav");
    harmonicAlgorithm.initialize(pathToAudioFiles, "harmonicAlgorithm.wav");
    dan_rain_at_sea_loop.initialize(pathToAudioFiles, "dan_rain_at_sea_loop.wav");

    dan_10122606_sound_spray.initialize(pathToAudioFiles, "dan_10122606_sound_spray.wav");
    dan_10122606_sound_spray.setGain(0); dan_10122606_sound_spray.setLoop(true); dan_10122606_sound_spray.play(1);

    dan_10122608_sound_spray_low.initialize(pathToAudioFiles, "dan_10122608_sound_spray_low.wav");
    dan_10120600_rezS3_rez2.initialize(pathToAudioFiles, "dan_10120600_RezS.3_Rez.2.wav");
}

void ParticleDreams::updateHand()
{
    osg::Vec3 offset(0.0,0.150,0.0);

    osg::Matrix m = PluginHelper::getHandMat(hand_id) * _particleObject->getWorldToObjectMatrix();
    osg::Vec3 handdir = osg::Vec3(0,1.0,0) * m;
    handdir = handdir - m.getTrans();
    handdir.normalize();
    osg::Vec3 handpos = m.getTrans() + offset;
    m.setTrans(handpos);

    wandPos[0] = handpos.x();
    wandPos[1] = handpos.y();
    wandPos[2] = handpos.z();
    wandVec[0] = handdir.x();
    wandVec[1] = handdir.y();
    wandVec[2] = handdir.z();
}

void ParticleDreams::pdata_init_age(int mAge)
{
    for (int i = 0; i < CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT; ++i)
    {
        // set age to random ages < max age to permit a respawn of the particle
        h_particleData[PDATA_ROW_SIZE*i] = rand() % mAge; // age
 
    }

}
void ParticleDreams::pdata_init_velocity(float vx,float vy,float vz)
{
    for (int i = 0; i < CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT; ++i)
    { 
        h_particleData[PDATA_ROW_SIZE * i + 1] = vx;
        h_particleData[PDATA_ROW_SIZE * i + 2] = vy;
        h_particleData[PDATA_ROW_SIZE * i + 3] = vz;
    }

}
void ParticleDreams::pdata_init_rand()
{
          
    for (int i = 0; i < CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT; ++i)
    {
        // gen 3 random numbers for each partical
        h_particleData[PDATA_ROW_SIZE * i +4] = 0.0002 * (rand()%10000) -1.0 ;
        h_particleData[PDATA_ROW_SIZE * i +5] = 0.0002 * (rand()%10000) -1.0 ;
        h_particleData[PDATA_ROW_SIZE * i +6] = 0.0002 * (rand()%10000) -1.0 ;
        //printf ( "rnd num %f %f %f \n", h_particleData[PDATA_ROW_SIZE * i +4],h_particleData[PDATA_ROW_SIZE * i +5],h_particleData[PDATA_ROW_SIZE * i +6]);

    }

}

void ParticleDreams::copy_reflector( int sorce, int destination)
{
    for (int row =0;row < REFL_DATA_ROWS;row++)	
    {
        for (int ele =0;ele <3;ele++)
        {
            h_reflectorData[ destination][row][ele] = h_reflectorData[sorce ][row][ele];
        }
    }
}

void ParticleDreams::copy_injector( int sorce, int destination)
{
    //printf ( "s, d , %i %i \n",sorce,destination);
    for (int row =0;row < INJT_DATA_ROWS;row++)	
    {
        for (int ele =0;ele <3;ele++)
        {
            h_injectorData[ destination][row][ele] = h_injectorData[sorce ][row][ele];
            //printf ( " %f ", h_injectorData[ destination][row][ele]);
        }
        //printf("\n");
    }
    //printf("\n");
}

int ParticleDreams::load6wallcaveWalls(int firstRefNum)
{
    float caverad = ftToM(5.0);
    int reflNum;
    float damping =.5;
    float no_traping =0;
    reflNum = firstRefNum;
    h_reflectorData[reflNum ][0][0]=1;h_reflectorData[reflNum ][0][1]=0;// type, age ie colormod, ~  0 is off 1 is plane reflector
    h_reflectorData[reflNum ][1][0]=ftToM(5);    h_reflectorData[reflNum ][1][1]= 0.0;h_reflectorData[reflNum ][1][2]=0;//x,y,z position
    h_reflectorData[reflNum ][2][0]= -1.0;  h_reflectorData[reflNum ][2][1]=0;    h_reflectorData[reflNum ][2][2]=0;//x,y,z normal
    h_reflectorData[reflNum ][3][0]=caverad; h_reflectorData[reflNum ][3][1]=0.00; h_reflectorData[reflNum ][3][2]=0;//reflector radis ,~,~ 
    h_reflectorData[reflNum ][4][0]=0.000;h_reflectorData[reflNum ][4][1]=0.000;h_reflectorData[reflNum ][4][2]=0.000;//t,u,v jiter  not implimented = speed 
    h_reflectorData[reflNum ][5][0]= damping; h_reflectorData[reflNum ][5][1]=no_traping;  h_reflectorData[reflNum ][5][2]=0.0;//reflectiondamping , no_traping ~
    h_reflectorData[reflNum ][6][0]=0;    h_reflectorData[reflNum ][6][1]=0;    h_reflectorData[reflNum ][6][2]=0;// not implemented yet centrality of rnd distribution speed dt tu ~


    reflNum = firstRefNum;
    //front
    h_reflectorData[reflNum ][1][0]=0;    h_reflectorData[reflNum ][1][1]= caverad;h_reflectorData[reflNum ][1][2]= -caverad;//x,y,z position
    h_reflectorData[reflNum ][2][0]=0;  h_reflectorData[reflNum ][2][1]=0;    h_reflectorData[reflNum ][2][2]=1;//x,y,z normal
//	h_reflectorData[reflNum ][2][0]=1;  h_reflectorData[reflNum ][2][1]=1;    h_reflectorData[reflNum ][2][2]=1;//x,y,z normal

    copy_reflector( reflNum, reflNum +1);
    reflNum++;//back
    h_reflectorData[reflNum ][1][0]=0;    h_reflectorData[reflNum ][1][1]= caverad;h_reflectorData[reflNum ][1][2]=caverad;//x,y,z position
    h_reflectorData[reflNum ][2][0]=0;  h_reflectorData[reflNum ][2][1]=0;    h_reflectorData[reflNum ][2][2]=-1;//x,y,z normal

    copy_reflector( reflNum, reflNum +1);
    reflNum++;//right
    h_reflectorData[reflNum ][1][0]=caverad;    h_reflectorData[reflNum ][1][1]= caverad;h_reflectorData[reflNum ][1][2]=0;//x,y,z position
    h_reflectorData[reflNum ][2][0]=-1;  h_reflectorData[reflNum ][2][1]=0;    h_reflectorData[reflNum ][2][2]=0;//x,y,z normal

    copy_reflector( reflNum, reflNum +1);

    reflNum++;//left
    h_reflectorData[reflNum ][1][0]=-caverad;    h_reflectorData[reflNum ][1][1]= caverad;h_reflectorData[reflNum ][1][2]=0;//x,y,z position
    h_reflectorData[reflNum ][2][0]=1;  h_reflectorData[reflNum ][2][1]=0;    h_reflectorData[reflNum ][2][2]=0;//x,y,z normal

    copy_reflector( reflNum, reflNum +1);
    reflNum++;//top
    h_reflectorData[reflNum ][1][0]=0;    h_reflectorData[reflNum ][1][1]= 2*caverad;h_reflectorData[reflNum ][1][2]=0;//x,y,z position
    h_reflectorData[reflNum ][2][0]=0;  h_reflectorData[reflNum ][2][1]=-1;    h_reflectorData[reflNum ][2][2]=0;//x,y,z normal

    copy_reflector( reflNum, reflNum +1);
    reflNum++;//bottom
    h_reflectorData[reflNum ][1][0]=0;    h_reflectorData[reflNum ][1][1]= -0;h_reflectorData[reflNum ][1][2]=0;//x,y,z position
    h_reflectorData[reflNum ][2][0]=0;  h_reflectorData[reflNum ][2][1]=1;    h_reflectorData[reflNum ][2][2]=0;//x,y,z normal

    
    return reflNum;

}

void ParticleDreams::scene_data_0_host()
{
    gravity = .1;
    gravity = 0.000001;
    //gravity = 0.0001;
    colorFreq =16;
    max_age = 2000;
    disappear_age =2000;
    alphaControl =0;
    static float time_in_sean;
    time_in_sean = time_in_sean + 1.0/TARGET_FR_RATE;

    if (scene0Start == 1)
    {
	size_t size = PDATA_ROW_SIZE * CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT * sizeof (float);
	//pdata_init_age( max_age);
	pdata_init_velocity(0, -0.005, 0);
	pdata_init_rand();
	//cuMemcpyHtoD(d_particleData, h_particleData, size);
	time_in_sean =0 * TARGET_FR_RATE;
	//::user->home();

	if (DEBUG_PRINT >0)printf("scene0Start \n");
	/*if ((SOUND_SERV ==1)&& (::host->root() == 1))
	{
	    audioPlay(dan_ambiance_2,1.0);audioGain(dan_ambiance_2,1);

	}*/
	if (soundEnabled)
	{
		dan_ambiance_2.setGain(1);
		dan_ambiance_2.play(1);
	}

	scene0ContextStart = 1;
    }
    else
    {
	scene0ContextStart = 0;
    }


    if (time_in_sean >90)nextSean=1;          

    anim = showFrameNo * .001;

    draw_water_sky =0;
    
    int injNum ;	
    h_injectorData[0][0][0] =1;// number of injectors ~ ~   ~ means dont care
    //injector 1
    injNum =1;

    /*if ((SOUND_SERV ==1)&& (::host->root() == 1))
    {
	//audioGain(texture_12,trigger);
	//if ((triggerold == 0) && (trigger ==1)){audioPlay(short_sound_01a,0.1);audioGain(texture_12,1);}
	if ((triggerold == 0) && (trigger ==1)){audioFadeUp( texture_12, 1, 1, short_sound_01a);}
	if ((triggerold == 1) && (trigger ==0)){audioFadeOut( texture_12, 10, -1);}

    }*/

    if (soundEnabled)
    {
    	if (triggerold == 0 && trigger == 1)
    	{
    		short_sound_01a.play();
    		texture_12.fade(1, 1);
    	}
    	else if (triggerold == 1 && trigger == 0)
    	{
    		texture_12.fade(0, 10);
    	}
    }
    injNum =1;
    h_injectorData[injNum][1][0]=1;h_injectorData[injNum][1][1]=trigger;// type, injection ratio ie streem volume, ~
    h_injectorData[injNum][2][0]=wandPos[0];h_injectorData[injNum][2][1]=wandPos[1];h_injectorData[injNum][2][2]=wandPos[2];//x,y,z position
    h_injectorData[injNum][3][0]=wandVec[0];h_injectorData[injNum][3][1]=wandVec[1];h_injectorData[injNum][3][2]=wandVec[2];//x,y,z velocity direction
    h_injectorData[injNum][4][0]=0.00;h_injectorData[injNum][4][1]=0.00;h_injectorData[injNum][4][2]=.0;//x,y,z size
    h_injectorData[injNum][5][0]=0.010;h_injectorData[injNum][5][1]=0.010;h_injectorData[injNum][5][2]=0.000;//t,u,v jiter v not implimented = speed 
    h_injectorData[injNum][6][0]=.1;h_injectorData[injNum][6][1]=0.1;h_injectorData[injNum][6][2]=0.0;//speed jiter ~ ~
    h_injectorData[injNum][7][0]=5;h_injectorData[injNum][7][1]=5;h_injectorData[injNum][7][2]=5;//centrality of rnd distribution speed dt tu ~
    //if (trigger){printf (" wandPos[0 ,1,2] wandVec[0,1,2] %f %f %f    %f %f %f \n", wandPos[0],wandPos[1],wandPos[2],wandVec[0],wandVec[1],wandVec[2]);}
    // load starcave wall reflectors
    //h_reflectorData[0][0][0] =loadStarcaveWalls(1);
    if (time_in_sean >5)h_reflectorData[0][0][0] =load6wallcaveWalls(1);

    scene0Start =0;
}

void ParticleDreams::scene_data_1_host()
{
    draw_water_sky =0;
// particalsysparamiteres--------------
    gravity = .00005;	
    max_age = 2000;
    disappear_age =2000;
    colorFreq =64 *max_age/2000.0 ;
    alphaControl =0;//turns alph to transparent beloy y=0
// reload  rnd < max_age in to pdata
    static float time_in_sean;
    time_in_sean = time_in_sean + 1.0/TARGET_FR_RATE;


    if (scene1Start == 1)
    {
        //size_t size = PDATA_ROW_SIZE * CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT * sizeof (float);
        //pdata_init_age( max_age);
        //pdata_init_velocity(-10000, -10000, -10000);
        //pdata_init_rand();
        //cuMemcpyHtoD(d_particleData, h_particleData, size);
        time_in_sean =0 * TARGET_FR_RATE;
        //::user->home();
        //printf( "in start sean3 \n");
        if (DEBUG_PRINT >0)printf("scene0Start \n");
        /*if ((SOUND_SERV ==1)&& (::host->root() == 1))
        {
            audioPlay(dan_5min_ostinato,1.0);audioGain(dan_5min_ostinato,0.5);
                    
        }*/
        if (soundEnabled)
        {
        	dan_5min_ostinato.setGain(0.5);
        	dan_5min_ostinato.play(1);
        }
            
        //printf( " time %f \n", ::user->get_t());
	scene1ContextStart = 1;
    }
    else
    {
	scene1ContextStart = 0;
    }

//printf ("time_in_sean 1 %f\n",time_in_sean);
    if (time_in_sean >110)nextSean=1;
    if (DEBUG_PRINT >0)printf( "in sean1 \n");
//printf( "in sean1 \n"); 

    // anim += 0.001;// .0001
    static float rotRate = .05;
    anim = showFrameNo * rotRate;
    rotRate += .000001;

    //anim += 3.14159/4;
    //tracker data
    //printf("  devid %d \n",devid );
    // printf("pos  P %f %f %f\n", P[0], P[1], P[2]);
    //printf(" direc  V %f %f %f\n", V[0], V[1], V[2]);


//	 injector data 

//	 injector data 
    int injNum ;	
    // number of injectors ~ ~   ~ means dont care
    //injector 1
    injNum =1;
	
    h_injectorData[injNum][1][0]=2;h_injectorData[injNum][1][1]=1.0;// type, injection ratio ie streem volume, ~
    h_injectorData[injNum][2][0]=0;h_injectorData[injNum][2][1]=ftToM(0.1);h_injectorData[injNum][2][2]=0;//x,y,z position
    h_injectorData[injNum][3][0]=0.02 * (sin(time_in_sean*2*M_PI));h_injectorData[injNum][3][1]=0.5;h_injectorData[injNum][3][2]=0.02 * (cos(time_in_sean*2*M_PI));//x,y,z velocity
    //h_injectorData[injNum][3][0]=0.02 *0.0;h_injectorData[injNum][3][1]=10;h_injectorData[injNum][3][2]=0.02 * -1;//x,y,z velocity
	
    h_injectorData[injNum][4][0]=0.03;h_injectorData[injNum][4][1]=0.03;h_injectorData[injNum][4][2]=0.03;//x,y,z size
    h_injectorData[injNum][5][0]=0.0000;h_injectorData[injNum][5][1]=0.00000;h_injectorData[injNum][5][2]=0.000;//t,u,v jiter v not implimented = speed 
    //h_injectorData[injNum][5][0]=0.000;h_injectorData[injNum][5][1]=0.000;h_injectorData[injNum][5][2]=0.000;//t,u,v jiter v not implimented = speed 
    h_injectorData[injNum][6][0]= .03;h_injectorData[injNum][6][1]=0.0;h_injectorData[injNum][6][2]=0.0;//speed jiter ~ ~
    h_injectorData[injNum][7][0]=5;h_injectorData[injNum][7][1]=5;h_injectorData[injNum][7][2]=5;//centrality of rnd distribution speed dt tu ~
    int reflect_on;
    reflect_on =0;
    float speed= 1.0/30; //one rotation every 
    speed = 1.0/45;	
    float stime =0 ;
    float t,rs,as,mr=9.0;
	
    int numobj =5;
    h_injectorData[0][0][0] =numobj;



    for(int i=1;i<=numobj;i++)
    {
        if (time_in_sean > 1.0/speed)reflect_on =1;	
        if (time_in_sean > 1.5/speed)
        {
            reflect_on =1;
			
            h_injectorData[injNum][3][0]= 0;h_injectorData[injNum][3][1]=.5;h_injectorData[injNum][3][2]=0.0;//x,y,z velocity
            if (time_in_sean > 2.0/speed)
            {
                h_injectorData[injNum][3][0]= -ftToM(rs*sin(as))/8 ;h_injectorData[injNum][3][1]=.5;h_injectorData[injNum][3][2]=ftToM(rs*cos(as))/8;//x,y,z velocity
                h_injectorData[injNum][6][0]= .026;h_injectorData[injNum][6][1]=0.0;h_injectorData[injNum][6][2]=0.0;//speed jiter ~ ~
                           
            }
        }	 	
        copy_injector(1, i);
        stime= i*(1/speed/numobj);
        if (time_in_sean >stime)
        {
            injNum =i; t = time_in_sean - stime;rs =2*speed*(2*M_PI)*t;as =speed*(2*M_PI)*t;if (rs >mr)rs=mr;if (rs < 2)rs = 2;
            //rs =mr;
            h_injectorData[injNum][2][0]=ftToM(rs*sin(as));h_injectorData[injNum][2][2]=-ftToM(rs*cos(as))  ;

					
        }		
    }	

    h_reflectorData[0][0][0] =2;// number of reflectors ~ ~   ~ means dont care
	
	
    int reflNum = 1;
    h_reflectorData[reflNum ][0][0]=reflect_on;h_reflectorData[reflNum ][0][1]=0;// type, age ie colormod,, ~  0 is off 1 is plane reflector
    h_reflectorData[reflNum ][1][0]=0;    h_reflectorData[reflNum ][1][1]= 0.0;h_reflectorData[reflNum ][1][2]=0;//x,y,z position
    h_reflectorData[reflNum ][2][0]=0.0;  h_reflectorData[reflNum ][2][1]=1;    h_reflectorData[reflNum ][2][2]=0;//x,y,z normal
    h_reflectorData[reflNum ][3][0]=ftToM(10.00); h_reflectorData[reflNum ][3][1]=0.00; h_reflectorData[reflNum ][3][2]=0;//reflector radis ,~,~ 
    h_reflectorData[reflNum ][4][0]=0.000;h_reflectorData[reflNum ][4][1]=0.000;h_reflectorData[reflNum ][4][2]=0.000;//t,u,v jiter  not implimented = speed 
    h_reflectorData[reflNum ][5][0]= 0.4; h_reflectorData[reflNum ][5][1]=0.0;  h_reflectorData[reflNum ][5][2]=0.0;//reflectiondamping , no_traping ~
    h_reflectorData[reflNum ][6][0]=0;    h_reflectorData[reflNum ][6][1]=0;    h_reflectorData[reflNum ][6][2]=0;// not implemented yet centrality of rnd distribution speed dt tu ~
	
    reflNum = 2;
//BOB XFORM THIS
    float x = wandPos[0];
    float y = wandPos[1];
    float z = wandPos[2];
    float dx = wandVec[0]/2;
    float dy = wandVec[1]/2;
    float dz = wandVec[2]/2;

    dx = wandMat[4];
    dy = wandMat[5];
    dz = wandMat[6];


    h_reflectorData[reflNum ][0][0]=trigger;h_reflectorData[reflNum ][0][1]=1;// type, age ie colormod, ~  0 is off 1 is plane reflector  0 is off 1 is plane reflector
    h_reflectorData[reflNum ][1][0]=x;    h_reflectorData[reflNum ][1][1]= y;h_reflectorData[reflNum ][1][2]=z;//x,y,z position
    h_reflectorData[reflNum ][2][0]=dx;  h_reflectorData[reflNum ][2][1]=dy;    h_reflectorData[reflNum ][2][2]=dz;//x,y,z normal
    h_reflectorData[reflNum ][3][0]=ftToM(0.5); h_reflectorData[reflNum ][3][1]=0.00; h_reflectorData[reflNum ][3][2]=0;//reflector radis ,~,~ 
    h_reflectorData[reflNum ][4][0]=0.000;h_reflectorData[reflNum ][4][1]=0.000;h_reflectorData[reflNum ][4][2]=0.000;//t,u,v jiter  not implimented = speed 
    h_reflectorData[reflNum ][5][0]= 1; h_reflectorData[reflNum ][5][1]= 1.00;  h_reflectorData[reflNum ][5][2]=0.0;//reflectiondamping , no_traping ~
    h_reflectorData[reflNum ][6][0]=0;    h_reflectorData[reflNum ][6][1]=0;    h_reflectorData[reflNum ][6][2]=0;// not implemented yet centrality of rnd distribution speed dt tu ~
     
    /*if ((SOUND_SERV ==1)&& (::host->root() == 1))
    {

        if ((REFL_HITS ==1 ) && (trigger))
        {
               float ag =h_debugData[reflNum]/500.0;
               if (ag >.5) ag=.5;
            audioGain(dan_10122606_sound_spray,ag);
            //printf ("spiral Fountens hits in scene 1 %f  ln hits %f\n",h_debugData[reflNum],log((h_debugData[reflNum])));
        }
        if ((triggerold ==1) && (trigger ==0)) audioGain(dan_10122606_sound_spray,0);

    }*/

    if (soundEnabled)
    {
    	if ((REFL_HITS == 1) && trigger)
    	{
    		float newGain = h_debugData[reflNum]/500.0;
    		if (newGain > 0.5) newGain = 0.5;
    		dan_10122606_sound_spray.setGain(newGain);
    	}
    	if ((triggerold == 1) && (trigger == 0))
    		dan_10122606_sound_spray.setGain(0);
    }


    scene1Start =0;
}

void ParticleDreams::scene_data_2_host()
{
    //4 waterFalls

    draw_water_sky =0;
    // particalsysparamiteres--------------
    gravity = .005;	
    gravity = .0001;	
    max_age = 2000;
    disappear_age =2000;
    colorFreq =64 *max_age/2000.0 ;
    alphaControl =0;//turns alph to transparent beloy y=0
    static float time_in_sean;
    time_in_sean = time_in_sean + 1.0/TARGET_FR_RATE;

    // reload  rnd < max_age in to pdata

    if (scene2Start == 1)
    {
	//size_t size = PDATA_ROW_SIZE * CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT * sizeof (float);
	//pdata_init_age( max_age);
	//pdata_init_velocity(-10000, -10000, -10000);
	//pdata_init_rand();
	//cuMemcpyHtoD(d_particleData, h_particleData, size);
	if (DEBUG_PRINT >0)printf( "in start sean2 \n");
	time_in_sean =0 * TARGET_FR_RATE;
	//::user->home();
	if (DEBUG_PRINT >0)printf("scene0Start \n");
	/*if ((SOUND_SERV ==1)&& (::host->root() == 1))
	{
	    //audioFadeUp( harmonicAlgorithm, 15, 1, -1);
	    audioPlay(harmonicAlgorithm,1.0);audioGain(harmonicAlgorithm,1);

	}*/

	if (soundEnabled)
	{
		harmonicAlgorithm.setGain(1);
		harmonicAlgorithm.play(1);
	}

	//printf( " time %f \n", ::user->get_t());
	scene2ContextStart = 1;
    }
    else
    {
	scene2ContextStart = 0;
    }


    if (time_in_sean >90)nextSean=1;
    //printf ("time_in_sean 2 %f\n",time_in_sean);
    if (DEBUG_PRINT >0)printf( "in sean2 \n");
    //printf( "in sean2 \n");

    // anim += 0.001;// .0001
    static float rotRate = .05;
    anim = showFrameNo * rotRate;
    rotRate += .000001;

    //anim += 3.14159/4;
    //tracker data
    //printf("  devid %d \n",devid );
    // printf("pos  P %f %f %f\n", P[0], P[1], P[2]);
    //printf(" direc  V %f %f %f\n", V[0], V[1], V[2]);


    //	 injector data 

    //	 injector data 
    int injNum ;	
    h_injectorData[0][0][0] =1;// number of injectors ~ ~   ~ means dont care
    //injector 1
    /*
       injNum =1;

       h_injectorData[injNum][1][0]=1;h_injectorData[injNum][1][1]=1.0;// type, injection ratio ie streem volume, ~
       h_injectorData[injNum][2][0]=0;h_injectorData[injNum][2][1]=ftToM(0.00);h_injectorData[injNum][2][2]=0;//x,y,z position
       h_injectorData[injNum][3][0]=0.02 * (sin(anim));h_injectorData[injNum][3][1]=10;h_injectorData[injNum][3][2]=0.02 * (cos(anim));//x,y,z velocity
    //h_injectorData[injNum][3][0]=0.02 *0.0;h_injectorData[injNum][3][1]=0;h_injectorData[injNum][3][2]=0.02 * -1;//x,y,z velocity

    h_injectorData[injNum][4][0]=0.00;h_injectorData[injNum][4][1]=0.00;h_injectorData[injNum][4][2]=.0;//x,y,z size
    h_injectorData[injNum][5][0]=0.0010;h_injectorData[injNum][5][1]=0.0010;h_injectorData[injNum][5][2]=0.000;//t,u,v jiter v not implimented = speed 
    //h_injectorData[injNum][5][0]=0.000;h_injectorData[injNum][5][1]=0.000;h_injectorData[injNum][5][2]=0.000;//t,u,v jiter v not implimented = speed 
    h_injectorData[injNum][6][0]= 1.1;h_injectorData[injNum][6][1]=0.0;h_injectorData[injNum][6][2]=0.0;//speed jiter ~ ~
    h_injectorData[injNum][7][0]=5;h_injectorData[injNum][7][1]=5;h_injectorData[injNum][7][2]=5;//centrality of rnd distribution speed dt tu ~

    if ((SOUND_SERV ==1)&& (::host->root() == 1))
    {
    audioPos (texture_17_swirls3, 30* h_injectorData[injNum][3][0], 0, -30* h_injectorData[injNum][3][2]);

    }
     */

    h_injectorData[0][0][0] =4;// number of injectors ~ ~   ~ means dont care

    // injector 1
    injNum =1;//front
    h_injectorData[injNum][1][0]=2;h_injectorData[injNum][1][1]=1.0;// type, injection ratio ie streem volume, ~
    h_injectorData[injNum][2][0]=0;h_injectorData[injNum][2][1]=ftToM(6);h_injectorData[injNum][2][2]=ftToM(-2);//x,y,z position
    h_injectorData[injNum][3][0]=0.0;h_injectorData[injNum][3][1]=0.010;h_injectorData[injNum][3][2]=0.01;//x,y,z velocity drection
    h_injectorData[injNum][4][0]=ftToM(0.25);h_injectorData[injNum][4][1]=ftToM(0.25);h_injectorData[injNum][4][2]=ftToM(0.0);//x,y,z size
    h_injectorData[injNum][5][0]=0.000;h_injectorData[injNum][5][1]=0.000;h_injectorData[injNum][5][2]=0.000;//t,u,v jiter v not implimented = speed 
    h_injectorData[injNum][6][0]=0.2000;h_injectorData[injNum][6][1]=0.0;h_injectorData[injNum][6][2]=0.0;//speed jiter ~ ~
    h_injectorData[injNum][7][0]=5;h_injectorData[injNum][7][1]=5;h_injectorData[injNum][7][2]=5;//centrality of rnd distribution speed dt tu ~

    copy_injector(1, 2);
    injNum =2;//back
    h_injectorData[injNum][2][0]=0;h_injectorData[injNum][2][1]=ftToM(6);h_injectorData[injNum][2][2]=ftToM(2);//x,y,z position
    h_injectorData[injNum][3][0]=0.0;h_injectorData[injNum][3][1]=0.010;h_injectorData[injNum][3][2]=-0.01;//x,y,z velocity drection

    copy_injector(1, 3);
    injNum =3;//back
    h_injectorData[injNum][2][0]=ftToM(2);h_injectorData[injNum][2][1]=ftToM(6);h_injectorData[injNum][2][2]=ftToM(0);//x,y,z position
    h_injectorData[injNum][3][0]=-0.010;h_injectorData[injNum][3][1]=0.010;h_injectorData[injNum][3][2]=0;//x,y,z velocity drection
    h_injectorData[injNum][4][0]=ftToM(0.25);h_injectorData[injNum][4][1]=ftToM(0.25);h_injectorData[injNum][4][2]=ftToM(0.25);//x,y,z size
    copy_injector(1, 4);
    injNum =4;//back
    h_injectorData[injNum][2][0]=ftToM(-2);h_injectorData[injNum][2][1]=ftToM(6);h_injectorData[injNum][2][2]=ftToM(0);//x,y,z position
    h_injectorData[injNum][3][0]=0.010;h_injectorData[injNum][3][1]=0.010;h_injectorData[injNum][3][2]=0;//x,y,z velocity drection
    h_injectorData[injNum][4][0]=ftToM(0.25);h_injectorData[injNum][4][1]=ftToM(0.25);h_injectorData[injNum][4][2]=ftToM(0.25);//x,y,z size


    /*
    //injector 3
    //sound for inj3
    injNum =3;

    if ((SOUND_SERV ==1)&& (::host->root() == 1))
    {
    //audioGain(texture_12,trigger);

    //if ((triggerold == 0) && (trigger ==1)){audioPlay(short_sound_01a,0.1);audioGain(texture_12,1);}

    if ((triggerold == 0) && (trigger ==1)){audioGain(short_sound_01a,10.0);audioFadeUp( texture_12, 1, 1, short_sound_01a);}
    if ((triggerold == 1) && (trigger ==0)){audioFadeOut( texture_12, 10, -1);}
    audioPos (texture_12, 1* h_injectorData[injNum][2][0], 1* h_injectorData[injNum][1][0], -1* h_injectorData[injNum][2][2]);
    audioPos (short_sound_01a, 1* h_injectorData[injNum][2][0], 1* h_injectorData[injNum][1][0], -1* h_injectorData[injNum][2][2]);


    }
    h_injectorData[injNum][1][0]=1;h_injectorData[injNum][1][1]=trigger*4.0;// type, injection ratio ie streem volume, ~
    h_injectorData[injNum][2][0]=wandPos[0];h_injectorData[injNum][2][1]=wandPos[1];h_injectorData[injNum][2][2]=wandPos[2];//x,y,z position
    h_injectorData[injNum][3][0]=wandVec[0];h_injectorData[injNum][3][1]=wandVec[1];h_injectorData[injNum][3][2]=wandVec[2];//x,y,z velocity direction
    //h_injectorData[injNum][3][0]=0.02 * (sin(anim));h_injectorData[injNum][3][1]=0;h_injectorData[injNum][3][2]=0.02 * (cos(anim));//x,y,z velocity
    //h_injectorData[injNum][3][0]=0.02 *0.0;h_injectorData[injNum][3][1]=0;h_injectorData[injNum][3][2]=0.02 * -1;//x,y,z velocity
    h_injectorData[injNum][4][0]=0.00;h_injectorData[injNum][4][1]=0.00;h_injectorData[injNum][4][2]=.0;//x,y,z size
    h_injectorData[injNum][5][0]=0.010;h_injectorData[injNum][5][1]=0.010;h_injectorData[injNum][5][2]=0.000;//t,u,v jiter v not implimented = speed 
    h_injectorData[injNum][6][0]=0.1;h_injectorData[injNum][6][1]=0.0;h_injectorData[injNum][6][2]=0.0;//speed jiter ~ ~
    h_injectorData[injNum][7][0]=5;h_injectorData[injNum][7][1]=5;h_injectorData[injNum][7][2]=5;//centrality of rnd distribution speed dt tu ~
    //
    if (trigger){if (DEBUG_PRINT >0) {printf(" wandPos[0 ,1,2] wandVec[0,1,2] %f %f %f    %f %f %f \n", wandPos[0],wandPos[1],wandPos[2],wandVec[0],wandVec[1],wandVec[2]);}}

    //cuMemcpyHtoD(d_injectorData, h_injectorData, sizei);
    ///	 injector data 
    //float reflectorNum =0;
    //int reflectI;
    //reflectI = (int) reflectorNum *3*8;//8 sets of 3 vallues
     */
    h_reflectorData[0][0][0] =1;// number of reflectors ~ ~   ~ means dont care
    int reflNum;
    reflNum = 1;
    float x = wandPos[0];
    float y = wandPos[1];
    float z = wandPos[2];
    float dx = wandVec[0]/2;
    float dy = wandVec[1]/2;
    float dz = wandVec[2]/2;

    dx = wandMat[4];
    dy = wandMat[5];
    dz = wandMat[6];


    h_reflectorData[reflNum ][0][0]=trigger;h_reflectorData[reflNum ][0][1]=1;// type, age ie colormod, ~  0 is off 1 is plane reflector  0 is off 1 is plane reflector
    h_reflectorData[reflNum ][1][0]=x;    h_reflectorData[reflNum ][1][1]= y;h_reflectorData[reflNum ][1][2]=z;//x,y,z position
    h_reflectorData[reflNum ][2][0]=dx;  h_reflectorData[reflNum ][2][1]=dy;    h_reflectorData[reflNum ][2][2]=dz;//x,y,z normal
    h_reflectorData[reflNum ][3][0]=ftToM(0.5); h_reflectorData[reflNum ][3][1]=0.00; h_reflectorData[reflNum ][3][2]=0;//reflector radis ,~,~ 
    h_reflectorData[reflNum ][4][0]=0.000;h_reflectorData[reflNum ][4][1]=0.000;h_reflectorData[reflNum ][4][2]=0.000;//t,u,v jiter  not implimented = speed 
    h_reflectorData[reflNum ][5][0]= 1; h_reflectorData[reflNum ][5][1]= 1.00;  h_reflectorData[reflNum ][5][2]=0.0;//reflectiondamping , no_traping ~
    h_reflectorData[reflNum ][6][0]=0;    h_reflectorData[reflNum ][6][1]=0;    h_reflectorData[reflNum ][6][2]=0;// not implemented yet centrality of rnd distribution speed dt tu ~

    /*if ((SOUND_SERV ==1)&& (::host->root() == 1))
    {

	if ((REFL_HITS ==1 ) && (trigger))
	{
	    float ag =h_debugData[reflNum]/500.0;
	    if (ag >.5) ag=.5;
	    audioGain(dan_10122606_sound_spray,ag);

	    audioGain(dan_10122606_sound_spray,ag);
	    //printf ("4 waterFalls hits in scene %f  ln hits %f\n",h_debugData[reflNum],log((h_debugData[reflNum])));
	}
	if ((triggerold ==1) && (trigger ==0)) audioGain(dan_10122606_sound_spray,0);

    }*/

    if (soundEnabled)
    {
    	if ((REFL_HITS == 1) && trigger)
    	{
    		float newGain = h_debugData[reflNum]/500.0;
    		if (newGain > 0.5) newGain = 0.5;

    		dan_10122606_sound_spray.setGain(newGain);
    	}

    	if (triggerold == 1 && !trigger)
    		dan_10122606_sound_spray.setGain(0);
    }

    scene2Start =0;
}

void ParticleDreams::scene_data_3_host()
{
    //painting skys

    draw_water_sky =1;
    // particalsysparamiteres--------------
    gravity = .000005;	
    max_age = 2000;
    disappear_age =2000;
    colorFreq =64 *max_age/2000.0 ;
    alphaControl =1;//turns alph to transparent beloy y=0
    static float time_in_sean;
    static float rotRate;

    time_in_sean = time_in_sean + 1.0/TARGET_FR_RATE;

    // reload  rnd < max_age in to pdata

    if (scene3Start == 1)
    {
		size_t size = PDATA_ROW_SIZE * CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT * sizeof (float);
		pdata_init_age( max_age);
		pdata_init_velocity(-10000, -10000, -10000);
		pdata_init_rand();
		///cuMemcpyHtoD(d_particleData, h_particleData, size);
		if (DEBUG_PRINT >0)printf( "in start sean3 \n");
		time_in_sean =0 * TARGET_FR_RATE;
		//::user->home();
		rotRate = .05;
		///::user->set_t(86400);// set to 00:00 monday  old
		///::user->set_t(1);// set to 0 days 1 sec
		///::user->set_t(63711);// set to 0 days 1 sec

		//::user->time(43200/2.0);// set to dawn  old
		//::user->set_t(43200/2.0);

		//::user->pass(43200/2.0);
		//printf( " time %f \n", ::user->get_t());

		/*if ((SOUND_SERV ==1)&& (::host->root() == 1))
		{
			audioLoop(rain_at_sea,1);audioPlay(rain_at_sea,1.0);audioGain(rain_at_sea,1);
			audioLoop(texture_17_swirls3,1);audioPlay(texture_17_swirls3,1.0);audioGain(texture_17_swirls3,1);

		}*/

		if (soundEnabled)
		{
			rain_at_sea.setLoop(true);
			rain_at_sea.setGain(1);
			rain_at_sea.play(1);

			texture_17_swirls3.setLoop(true);
			texture_17_swirls3.setGain(1);
			texture_17_swirls3.play(1);
		}
		scene3ContextStart = 1;
    }
    else
    {
	scene3ContextStart = 0;
    }
    //printf ("time_in_sean 3 %f\n",time_in_sean);

    if (DEBUG_PRINT >0)printf( "in sean3 \n");
    // printf ( "seantime time %f %f\n",time_in_sean,::user->get_t());
    //lerp(in, beginIN, endIn, beginOut, endOut)

    ///if((time_in_sean > 0)&& (time_in_sean <= 30)) user->set_t(lerp(time_in_sean, 0, 30, 63400, 74000));
    ///if((time_in_sean > 30)&& (time_in_sean <= 90)) user->set_t(lerp(time_in_sean, 30, 90, 74000, 107000));
    //if((time_in_sean > 30)&& (time_in_sean <= 40)) user->set_t(lerp(time_in_sean, 30, 40, 74000, 110000));
    //     ::user->set_t(107000);// set to 0 days 1 sec


    // anim += 0.001;// .0001

    anim = showFrameNo * rotRate;
    rotRate += .000001;

    //anim += 3.14159/4;
    //tracker data
    //printf("  devid %d \n",devid );
    // printf("pos  P %f %f %f\n", P[0], P[1], P[2]);
    //printf(" direc  V %f %f %f\n", V[0], V[1], V[2]);


    //	 injector data 

    //	 injector data 
    int injNum ;	
    h_injectorData[0][0][0] =3;// number of injectors ~ ~   ~ means dont care
    //injector 1
    injNum =1;

    h_injectorData[injNum][1][0]=1;h_injectorData[injNum][1][1]=1.0;// type, injection ratio ie streem volume, ~
    h_injectorData[injNum][2][0]=0;h_injectorData[injNum][2][1]=ftToM(8.00);h_injectorData[injNum][2][2]=0;//x,y,z position
    h_injectorData[injNum][3][0]=1 * (sin(anim/5));h_injectorData[injNum][3][1]=0.000;h_injectorData[injNum][3][2]=1 * (cos(anim/5));//x,y,z velocity direction
    h_injectorData[injNum][3][0]=0.02 * (sin(anim));h_injectorData[injNum][3][1]=0;h_injectorData[injNum][3][2]=0.02 * (cos(anim));//x,y,z velocity
    //h_injectorData[injNum][3][0]=0.02 *0.0;h_injectorData[injNum][3][1]=0;h_injectorData[injNum][3][2]=0.02 * -1;//x,y,z velocity

    h_injectorData[injNum][4][0]=0.00;h_injectorData[injNum][4][1]=0.00;h_injectorData[injNum][4][2]=.0;//x,y,z size
    h_injectorData[injNum][5][0]=0.0010;h_injectorData[injNum][5][1]=0.0010;h_injectorData[injNum][5][2]=0.000;//t,u,v jiter v not implimented = speed 
    //h_injectorData[injNum][5][0]=0.000;h_injectorData[injNum][5][1]=0.000;h_injectorData[injNum][5][2]=0.000;//t,u,v jiter v not implimented = speed 
    h_injectorData[injNum][6][0]= 1.1;h_injectorData[injNum][6][1]=0.0;h_injectorData[injNum][6][2]=0.0;//speed jiter ~ ~
    h_injectorData[injNum][7][0]=5;h_injectorData[injNum][7][1]=5;h_injectorData[injNum][7][2]=5;//centrality of rnd distribution speed dt tu ~

    /*if ((SOUND_SERV ==1)&& (::host->root() == 1))
    {
	if (ENABLE_SOUND_POS_UPDATES) audioPos (texture_17_swirls3, 30* h_injectorData[injNum][3][0], 0, -30* h_injectorData[injNum][3][2]);

    }*/

    if (soundEnabled && ENABLE_SOUND_POS_UPDATES)
    {
    	texture_17_swirls3.setPosition(30 * h_injectorData[injNum][3][0], 0, -30 * h_injectorData[injNum][3][2]);
    }

    //
    // injector 2
    injNum =2;
    h_injectorData[injNum][1][0]=1;h_injectorData[injNum][1][1]=0.0;// type, injection ratio ie streem volume, ~
    h_injectorData[injNum][2][0]=0;h_injectorData[injNum][2][1]=2.5;h_injectorData[injNum][2][2]=0;//x,y,z position
    h_injectorData[injNum][3][0]=0.0;h_injectorData[injNum][3][1]=1.000;h_injectorData[injNum][3][2]=0;//x,y,z velocity drection
    h_injectorData[injNum][4][0]=0.25;h_injectorData[injNum][4][1]=0.25;h_injectorData[injNum][4][2]=0;//x,y,z size
    h_injectorData[injNum][5][0]=0.000;h_injectorData[injNum][5][1]=0.000;h_injectorData[injNum][5][2]=0.000;//t,u,v jiter v not implimented = speed 
    h_injectorData[injNum][6][0]=.00;h_injectorData[injNum][6][1]=0.5;h_injectorData[injNum][6][2]=0.0;//speed jiter ~ ~
    h_injectorData[injNum][7][0]=5;h_injectorData[injNum][7][1]=5;h_injectorData[injNum][7][2]=5;//centrality of rnd distribution speed dt tu ~

    //injector 3
    //sound for inj3
    injNum =3;

    /*if ((SOUND_SERV ==1)&& (::host->root() == 1))
    {
	//audioGain(texture_12,trigger);

	//if ((triggerold == 0) && (trigger ==1)){audioPlay(short_sound_01a,0.1);audioGain(texture_12,1);}

	if ((triggerold == 0) && (trigger ==1)){audioGain(short_sound_01a,10.0);audioFadeUp( texture_12, 1, 1, short_sound_01a);}
	if ((triggerold == 1) && (trigger ==0)){audioFadeOut( texture_12, 10, -1);}

	if (ENABLE_SOUND_POS_UPDATES)
	{  
	    audioPos (texture_12, 1* h_injectorData[injNum][2][0], 1* h_injectorData[injNum][1][0], -1* h_injectorData[injNum][2][2]);
	    audioPos (short_sound_01a, 1* h_injectorData[injNum][2][0], 1* h_injectorData[injNum][1][0], -1* h_injectorData[injNum][2][2]);
	} 

    }*/

    if (soundEnabled)
    {
    	if (triggerold == 0 && trigger == 1)
    	{
    		short_sound_01a.setGain(10);
    		short_sound_01a.play();
    		texture_12.fade(1, 1);
    	}
    	if (triggerold == 1 && trigger == 0)
    	{
    		texture_12.fade(0, 10);
    	}

    	if (ENABLE_SOUND_POS_UPDATES)
    	{
    		texture_12.setPosition(	1 * h_injectorData[injNum][2][0],
    								1 * h_injectorData[injNum][1][0],
    							   -1 * h_injectorData[injNum][2][2]);

    		short_sound_01a.setPosition(1 * h_injectorData[injNum][2][0],
    									1 * h_injectorData[injNum][1][0],
    								   -1 * h_injectorData[injNum][2][2]);
    	}
    }

    h_injectorData[injNum][1][0]=1;h_injectorData[injNum][1][1]=trigger*4.0;// type, injection ratio ie streem volume, ~
    h_injectorData[injNum][2][0]=wandPos[0];h_injectorData[injNum][2][1]=wandPos[1];h_injectorData[injNum][2][2]=wandPos[2];//x,y,z position
    h_injectorData[injNum][3][0]=wandVec[0];h_injectorData[injNum][3][1]=wandVec[1];h_injectorData[injNum][3][2]=wandVec[2];//x,y,z velocity direction
    //h_injectorData[injNum][3][0]=0.02 * (sin(anim));h_injectorData[injNum][3][1]=0;h_injectorData[injNum][3][2]=0.02 * (cos(anim));//x,y,z velocity
    //h_injectorData[injNum][3][0]=0.02 *0.0;h_injectorData[injNum][3][1]=0;h_injectorData[injNum][3][2]=0.02 * -1;//x,y,z velocity
    h_injectorData[injNum][4][0]=0.00;h_injectorData[injNum][4][1]=0.00;h_injectorData[injNum][4][2]=.0;//x,y,z size
    h_injectorData[injNum][5][0]=0.010;h_injectorData[injNum][5][1]=0.010;h_injectorData[injNum][5][2]=0.000;//t,u,v jiter v not implimented = speed 
    h_injectorData[injNum][6][0]=0.1;h_injectorData[injNum][6][1]=0.0;h_injectorData[injNum][6][2]=0.0;//speed jiter ~ ~
    h_injectorData[injNum][7][0]=5;h_injectorData[injNum][7][1]=5;h_injectorData[injNum][7][2]=5;//centrality of rnd distribution speed dt tu ~
    //
    if (trigger){if (DEBUG_PRINT >0) {printf(" wandPos[0 ,1,2] wandVec[0,1,2] %f %f %f    %f %f %f \n", wandPos[0],wandPos[1],wandPos[2],wandVec[0],wandVec[1],wandVec[2]);}}

    scene3Start =0;
}

void ParticleDreams::scene_data_4_host()
{
    //falling rain

    draw_water_sky =0;
    // particalsysparamiteres--------------
    gravity = .001;	
    gravity = .00003;	
    max_age = 2000;
    disappear_age =2000;
    colorFreq =64 *max_age/2000.0 ;
    alphaControl =0;//turns alph to transparent beloy y=0
    static float time_in_sean;
    time_in_sean = time_in_sean + 1.0/TARGET_FR_RATE;

    // reload  rnd < max_age in to pdata

    if (scene4Start == 1)
    {
	//size_t size = PDATA_ROW_SIZE * CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT * sizeof (float);
	//pdata_init_age( max_age);
	//pdata_init_velocity(-10000, -10000, -10000);
	//pdata_init_rand();
	//cuMemcpyHtoD(d_particleData, h_particleData, size);

	//::user->home();
	if (DEBUG_PRINT >0)printf( "in start sean4 \n");
	time_in_sean =0 * TARGET_FR_RATE;
	/*if ((SOUND_SERV ==1)&& (::host->root() == 1))
	{

	    audioLoop(dan_rain_at_sea_loop,1);
	    audioPlay(dan_rain_at_sea_loop,1.0);audioGain(dan_rain_at_sea_loop,1.0);
	    //printf(" playcode exicuted\n");
	}*/

	if (soundEnabled)
	{
		dan_rain_at_sea_loop.setLoop(1);
		dan_rain_at_sea_loop.setGain(1);
		dan_rain_at_sea_loop.play();
	}
	//printf( " time %f \n", ::user->get_t());
	scene4ContextStart = 1;
    }
    else
    {
	scene4ContextStart = 0;
    }
    if (time_in_sean >20)nextSean=1;
    //printf ("time_in_sean 4 %f\n",time_in_sean);
    if (DEBUG_PRINT >0)printf( "in sean4 \n");
    //printf( "in sean4 \n");

    // anim += 0.001;// .0001
    static float rotRate = .05;
    anim = showFrameNo * rotRate;
    rotRate += .000001;

    //anim += 3.14159/4;
    //tracker data
    //printf("  devid %d \n",devid );
    // printf("pos  P %f %f %f\n", P[0], P[1], P[2]);
    //printf(" direc  V %f %f %f\n", V[0], V[1], V[2]);


    //	 injector data 
    h_reflectorData[0][0][0] =1;// turn off all reflectors

    //	 injector data 
    int injNum ;	
    h_injectorData[0][0][0] =1;// number of injectors ~ ~   ~ means dont care
    //injector 1
    /*
       injNum =1;

       h_injectorData[injNum][1][0]=1;h_injectorData[injNum][1][1]=1.0;// type, injection ratio ie streem volume, ~
       h_injectorData[injNum][2][0]=0;h_injectorData[injNum][2][1]=ftToM(0.00);h_injectorData[injNum][2][2]=0;//x,y,z position
       h_injectorData[injNum][3][0]=0.02 * (sin(anim));h_injectorData[injNum][3][1]=10;h_injectorData[injNum][3][2]=0.02 * (cos(anim));//x,y,z velocity
    //h_injectorData[injNum][3][0]=0.02 *0.0;h_injectorData[injNum][3][1]=0;h_injectorData[injNum][3][2]=0.02 * -1;//x,y,z velocity

    h_injectorData[injNum][4][0]=0.00;h_injectorData[injNum][4][1]=0.00;h_injectorData[injNum][4][2]=.0;//x,y,z size
    h_injectorData[injNum][5][0]=0.0010;h_injectorData[injNum][5][1]=0.0010;h_injectorData[injNum][5][2]=0.000;//t,u,v jiter v not implimented = speed 
    //h_injectorData[injNum][5][0]=0.000;h_injectorData[injNum][5][1]=0.000;h_injectorData[injNum][5][2]=0.000;//t,u,v jiter v not implimented = speed 
    h_injectorData[injNum][6][0]= 1.1;h_injectorData[injNum][6][1]=0.0;h_injectorData[injNum][6][2]=0.0;//speed jiter ~ ~
    h_injectorData[injNum][7][0]=5;h_injectorData[injNum][7][1]=5;h_injectorData[injNum][7][2]=5;//centrality of rnd distribution speed dt tu ~

    if ((SOUND_SERV ==1)&& (::host->root() == 1))
    {
    audioPos (texture_17_swirls3, 30* h_injectorData[injNum][3][0], 0, -30* h_injectorData[injNum][3][2]);

    }
     */

    h_injectorData[0][0][0] =1;// number of injectors ~ ~   ~ means dont care

    // injector 1
    injNum =1;
    h_injectorData[injNum][1][0]=2;h_injectorData[injNum][1][1]=1.0;// type, injection ratio ie streem volume, ~
    h_injectorData[injNum][2][0]=0;h_injectorData[injNum][2][1]=ftToM(10);h_injectorData[injNum][2][2]=ftToM(-2);//x,y,z position
    h_injectorData[injNum][3][0]=0.0;h_injectorData[injNum][3][1]=0.001;h_injectorData[injNum][3][2]=0.00;//x,y,z velocity drection
    h_injectorData[injNum][4][0]=ftToM(1);h_injectorData[injNum][4][1]=ftToM(1);h_injectorData[injNum][4][2]=ftToM(1);//x,y,z size
    h_injectorData[injNum][5][0]=0.000;h_injectorData[injNum][5][1]=0.000;h_injectorData[injNum][5][2]=0.000;//t,u,v jiter v not implimented = speed 
    h_injectorData[injNum][6][0]=0.2000;h_injectorData[injNum][6][1]=0.0;h_injectorData[injNum][6][2]=0.0;//speed jiter ~ ~
    h_injectorData[injNum][7][0]=5;h_injectorData[injNum][7][1]=5;h_injectorData[injNum][7][2]=5;//centrality of rnd distribution speed dt tu ~

    copy_injector(1, 2);
    injNum =2;//back
    h_injectorData[injNum][2][0]=0;h_injectorData[injNum][2][1]=ftToM(6);h_injectorData[injNum][2][2]=ftToM(2);//x,y,z position
    h_injectorData[injNum][3][0]=0.0;h_injectorData[injNum][3][1]=0.010;h_injectorData[injNum][3][2]=-0.01;//x,y,z velocity drection

    copy_injector(1, 3);
    injNum =3;//back
    h_injectorData[injNum][2][0]=ftToM(2);h_injectorData[injNum][2][1]=ftToM(6);h_injectorData[injNum][2][2]=ftToM(0);//x,y,z position
    h_injectorData[injNum][3][0]=-0.010;h_injectorData[injNum][3][1]=0.010;h_injectorData[injNum][3][2]=0;//x,y,z velocity drection
    h_injectorData[injNum][4][0]=ftToM(0.25);h_injectorData[injNum][4][1]=ftToM(0.25);h_injectorData[injNum][4][2]=ftToM(0.25);//x,y,z size
    copy_injector(1, 4);
    injNum =4;//back
    h_injectorData[injNum][2][0]=ftToM(-2);h_injectorData[injNum][2][1]=ftToM(6);h_injectorData[injNum][2][2]=ftToM(0);//x,y,z position
    h_injectorData[injNum][3][0]=0.010;h_injectorData[injNum][3][1]=0.010;h_injectorData[injNum][3][2]=0;//x,y,z velocity drection
    h_injectorData[injNum][4][0]=ftToM(0.25);h_injectorData[injNum][4][1]=ftToM(0.25);h_injectorData[injNum][4][2]=ftToM(0.25);//x,y,z size



    h_reflectorData[0][0][0] =1;// number of reflectors ~ ~   ~ means dont care
    int reflNum;
    reflNum = 1;
    float x = wandPos[0];
    float y = wandPos[1];
    float z = wandPos[2];
    float dx = wandVec[0]/2;
    float dy = wandVec[1]/2;
    float dz = wandVec[2]/2;

    dx = wandMat[4];
    dy = wandMat[5];
    dz = wandMat[6];

    h_reflectorData[reflNum ][0][0]=trigger;h_reflectorData[reflNum ][0][1] =1;// type, age ie colormod, ~  0 is off 1 is plane reflector  0 is off 1 is plane reflector
    h_reflectorData[reflNum ][1][0]=x;    h_reflectorData[reflNum ][1][1]= y;h_reflectorData[reflNum ][1][2]=z;//x,y,z position
    h_reflectorData[reflNum ][2][0]=dx;  h_reflectorData[reflNum ][2][1]=dy;    h_reflectorData[reflNum ][2][2]=dz;//x,y,z normal
    h_reflectorData[reflNum ][3][0]=ftToM(0.5); h_reflectorData[reflNum ][3][1]=0.00; h_reflectorData[reflNum ][3][2]=0;//reflector radis ,~,~ 
    h_reflectorData[reflNum ][4][0]=0.000;h_reflectorData[reflNum ][4][1]=0.000;h_reflectorData[reflNum ][4][2]=0.000;//t,u,v jiter  not implimented = speed 
    h_reflectorData[reflNum ][5][0]= 1; h_reflectorData[reflNum ][5][1]= 1.00;  h_reflectorData[reflNum ][5][2]=0.0;//reflectiondamping , no_traping ~
    h_reflectorData[reflNum ][6][0]=0;    h_reflectorData[reflNum ][6][1]=0;    h_reflectorData[reflNum ][6][2]=0;// not implemented yet centrality of rnd distribution speed dt tu ~

    /*if ((SOUND_SERV ==1)&& (::host->root() == 1))
    {

	if ((REFL_HITS ==1 ) && (trigger))
	{
	    float ag =h_debugData[reflNum]/500.0;
	    if (ag >.5) ag=.5;

	    audioGain(dan_10122606_sound_spray,ag);
	    //printf ("falling rain hits in scene %f  ln hits %f\n",h_debugData[reflNum],log((h_debugData[reflNum])));
	}
	if ((triggerold ==1) && (trigger ==0)) audioGain(dan_10122606_sound_spray,0);
    }*/

    if (soundEnabled)
    {
    	if (REFL_HITS == 1 && trigger)
    	{
    		float newGain = h_debugData[reflNum] / 500.0;
    		if (newGain > 0.5) newGain = 0.5;

    		dan_10122606_sound_spray.setGain(newGain);
    	}

    	if (triggerold == 1 && !trigger)
    		dan_10122606_sound_spray.setGain(0);
    }
    scene4Start =0;
}

void ParticleDreams::scene_data_0_context(int contextid) const
{
    if(scene0ContextStart)
    {
	size_t size = PDATA_ROW_SIZE * CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT * sizeof (float);
	cuMemcpyHtoD(d_particleDataMap[contextid], h_particleData, size);
    }
}

void ParticleDreams::scene_data_1_context(int contextid) const
{
    if(scene1ContextStart)
    {
    }
}

void ParticleDreams::scene_data_2_context(int contextid) const
{
    if(scene2ContextStart)
    {
    }
}

void ParticleDreams::scene_data_3_context(int contextid) const
{
    if(scene3ContextStart)
    {
	size_t size = PDATA_ROW_SIZE * CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT * sizeof (float);
	cuMemcpyHtoD(d_particleDataMap[contextid], h_particleData, size);
    }
}

void ParticleDreams::scene_data_4_context(int contextid) const
{
    if(scene4ContextStart)
    {
    }
}

void ParticleDreams::scene_data_0_kill_audio()
{
	h_reflectorData[0][0][0] = 0;
	h_injectorData[0][0][0] = 0;

	if (soundEnabled)
	{
		dan_ambiance_2.fade(0, 150);
		dan_10122606_sound_spray.setGain(0);
	}
}

void ParticleDreams::scene_data_1_kill_audio()
{
	h_reflectorData[0][0][0] = 0;
	h_injectorData[0][0][0] = 0;

	if (soundEnabled)
	{
		dan_5min_ostinato.fade(0, 150);
		dan_10122606_sound_spray.setGain(0);
	}
}

void ParticleDreams::scene_data_2_kill_audio()
{
	h_reflectorData[0][0][0] = 0;
	h_injectorData[0][0][0] = 0;

	if (soundEnabled)
	{
		harmonicAlgorithm.fade(0, 150);
		dan_10122606_sound_spray.setGain(0);
	}
}

void ParticleDreams::scene_data_3_kill_audio()
{
	h_reflectorData[0][0][0] = 0;
	h_injectorData[0][0][0] = 0;

	if (soundEnabled)
	{
		rain_at_sea.fade(0, 150);
		texture_17_swirls3.fade(0, 150);
		dan_10122606_sound_spray.setGain(0);
		dan_10120600_rezS3_rez2.setGain(0);
	}
}

void ParticleDreams::scene_data_4_kill_audio()
{
	h_reflectorData[0][0][0] = 0;
	h_injectorData[0][0][0] = 0;

	if (soundEnabled)
	{
		dan_rain_at_sea_loop.fade(0, 150);
		dan_10122606_sound_spray.setGain(0);
	}
}
