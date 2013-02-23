#include "AlgebraInMotion.h"
#include "CudaParticle.h"
#include "CudaHelper.h"

#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/Navigation.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/CVRViewer.h>
#include <cvrKernel/CVRStatsHandler.h>
#include <cvrConfig/ConfigManager.h>

#include <osg/PointSprite>
#include <osg/BlendFunc>

#include <osg/Depth>
#include <osg/ShapeDrawable>
#include <osgDB/FileUtils>
#include <osgDB/ReadFile>
#include <string>
#include <cuda_gl_interop.h>
//ContextChange 2 lines below fr2 scr

#include <cuda.h>
#include <cudaGL.h>

#include <sys/time.h>
#include <osg/Texture2D>
#include <osg/Material>
#include <osgText/Font3D>
#include <osgText/Text3D>
#include <osg/io_utils>

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

CVRPLUGIN(AlgebraInMotion)

AlgebraInMotion::AlgebraInMotion()
{
    _pointerHeading = 0.0;
    _pointerPitch = 0.0;
    _pointerRoll = 0.0;
}

AlgebraInMotion::~AlgebraInMotion()
{
	oasclient::ClientInterface::shutdown();
}

bool AlgebraInMotion::init()
{
    _myMenu = new SubMenu("Algebra In Motion");

    _enable = new MenuCheckbox("Enable",false);
    _enable->setCallback(this);
    _myMenu->addItem(_enable);

    _gravityRV = new MenuRangeValue("Gravity",0.0,0.1,.003);
    _gravityRV->setCallback(this);

    _speedRV = new MenuRangeValue("speed",0.0,0.4,.2);
    _speedRV->setCallback(this);


    _rotateInjCB = new MenuCheckbox("rotate injector",true);
    _rotateInjCB->setCallback(this);
    _reflectorCB = new MenuCheckbox("reflector on",true);
    _reflectorCB->setCallback(this);

    _dataDir = ConfigManager::getEntry("value","Plugin.AlgebraInMotion.DataDir","") + "/";

    PluginHelper::addRootMenuItem(_myMenu);

    hand_id = ConfigManager::getInt("value","Plugin.AlgebraInMotion.HandID",0);

	_TargetSystem = ConfigManager::getEntry("value","Plugin.AlgebraInMotion.TargetSystem","TourCaveCalit2");
	// TourCaveCalit2 TourCaveSaudi NexCaveCalit2 StarCave Cave2

//	ContextChange block comented out id fr 1 screen version
	
	_DisplaySystem = ConfigManager::getEntry("value","Plugin.AlgebraInMotion.DisplaySystem","Simulator");
//	_DisplaySystem = ConfigManager::getEntry("value","Plugin.AlgebraInMotion.DisplaySystem","TourCaveCalit2");
	// TourCaveCalit2 TourCaveSaudi NexCaveCalit2 StarCave Cave2 Simulator
	

   return true;
}

void AlgebraInMotion::menuCallback(MenuItem * item)
{
    if(item == _enable)
    {
	if(_enable->getValue())
	{

	    CVRViewer::instance()->getStatsHandler()->addStatTimeBar(CVRStatsHandler::CAMERA_STAT,"AIMCuda Time:","PD Cuda duration","PD Cuda start","PD Cuda end",osg::Vec3(0,1,0),"PD stats");
	    //CVRViewer::instance()->getStatsHandler()->addStatTimeBar(CVRStatsHandler::CAMERA_STAT,"PDCuda Copy:","PD Cuda Copy duration","PD Cuda Copy start","PD Cuda Copy end",osg::Vec3(0,0,1),"PD stats");

        SceneManager::instance()->setHidePointer(true);

	    initPart();
	    initGeometry();
	    initSound();
	}

/*
        if(item == _rotateInjCB)
        {
            if(_rotateInjCB->getValue())
            {
            }
            else
            {
            }
        }
*/

    }
}

void AlgebraInMotion::preFrame()
{

    if(_enable->getValue())
    {
	//do driver thread part of step
	
    if(SceneManager::instance()->getMenuOpenObject() == _particleObject)
    {
        SceneManager::instance()->setHidePointer(false);
    }
    else
    {
        SceneManager::instance()->setHidePointer(true);
    }

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

	//if ((but2old ==0)&&(but2 == 1)&&(but1))
	if (skipTonextScene ==1)
	{	std::cout << "skipTonextScene ==1 " << std::endl;
	    sceneOrder =( sceneOrder+1)%4;sceneChange=1;
		skipTonextScene =0;
		
	}
	if (nextSean ==1) { sceneOrder =( sceneOrder+1)%4;sceneChange=1;nextSean =0;}
	//reordering seenes
/*
	if (sceneOrder ==0)sceneNum =4;
	if (sceneOrder ==1)sceneNum =1;
	if (sceneOrder ==2)sceneNum =2;
	if (sceneOrder ==3)sceneNum =0;
	if (sceneOrder ==4)sceneNum =3;

	if (sceneOrder ==0)sceneNum =4;
	if (sceneOrder ==1)sceneNum =2;
	if (sceneOrder ==2)sceneNum =0;
	if (sceneOrder ==3)sceneNum =3;
*/
	if (sceneOrder ==0)sceneNum =2;
	if (sceneOrder ==1)sceneNum =0;
	if (sceneOrder ==2)sceneNum =3;
	if (sceneOrder ==3)sceneNum =4;

	//if((sceneChange==1) && (witch_scene ==3)){scene_data_3_kill_audio();}
	//if((sceneChange==1) && (witch_scene ==0)){scene_data_0_kill_audio();}
	//if((sceneChange==1) && (witch_scene ==1)){scene_data_1_kill_audio();}
	//if((sceneChange==1) && (witch_scene ==2)){scene_data_2_kill_audio();}
	//if((sceneChange==1) && (witch_scene ==4)){scene_data_4_kill_audio();}

	if (sceneChange)
	{
		if (witch_scene == 0) 		scene_data_0_kill_audio();
		else if (witch_scene == 1)	scene_data_1_kill_audio();//not used
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
//		for (int i =0;i<128;i++){old_refl_hits[i] = h_debugData[i];}

}

bool AlgebraInMotion::processEvent(InteractionEvent * event)
{
	if(!_enable->getValue())
    {
        return false;
    }

    TrackedButtonInteractionEvent * tie = event->asTrackedButtonEvent();

//0 is leftmouce trigger
//1 is right mouse butt
//2 scrole weele
//
    if(tie)
    {
	if(tie->getHand() == hand_id)
	{
	    if((tie->getInteraction() == BUTTON_DOWN || tie->getInteraction() == BUTTON_DOUBLE_CLICK) && tie->getButton() <= 4)
	    { std::cout << " buttonPtresses " << (tie->getButton()) << std::endl;
		if(tie->getButton() == 0)
		{
			trigger = 1;
			std::cout << " trigger but2 " << trigger << " " << but2 << std::endl;
			if (but2 ==1){ skipTonextScene =1; ; return true;}
		}
		else if(tie->getButton() == 1)
		{
		    but1 = 1;
		}
		else if(tie->getButton() == 2)
		{
		    but2 = 1;
			return true;
			//captures but2 to prevent defalt navagation on but2
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
			return true;

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

    ValuatorInteractionEvent * vie = event->asValuatorEvent();
    if(vie && vie->getHand() == hand_id)
    {
        if(vie->getValuator() == 0)
        {
            _pointerHeading += vie->getValue() * 0.5;
            _pointerHeading = std::max(_pointerHeading,-90.0f);
            _pointerHeading = std::min(_pointerHeading,90.0f);
        }
    }

    return false;
}
//ContextChange one below os fr 1 screen two below is from 2 scr
#ifndef SCR2_PER_CARD
void AlgebraInMotion::perContextCallback(int contextid,PerContextCallback::PCCType type) const
#else
void AlgebraInMotion::perContextCallback(int contextid) const
#endif
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
// ContextChange if(1) is from 2 screen
	int cudaDevice = ScreenConfig::instance()->getCudaDevice(contextid);
	#ifdef SCR2_PER_CARD
	int scr2 =1;
	#else
	int scr2 =0;
	#endif
        if(scr2)
	{
            CUdevice device;
            cuDeviceGet(&device,cudaDevice);
            CUcontext cudaContext;
 	
           cuGLCtxCreate(&cudaContext, 0, device);
            cuGLInit();
 	
           
            cuCtxSetCurrent(cudaContext);
	}
        else
        {
	    cudaGLSetGLDevice(cudaDevice);
	    cudaSetDevice(cudaDevice);
        } 
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
    {//educational
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
	// zero debug array
/*
   if(contextid == 0)
    {
 
		for( int i=0;i<128;i++) {  h_debugData[i]=0;}	

		cuMemcpyHtoD(d_debugDataMap[contextid], h_debugData, 128 * sizeof(float));
		printCudaErr();
	}
*/
    CUdeviceptr d_vbo;
    GLuint vbo = _particleGeo->getOrCreateVertexBufferObject()->getOrCreateGLBufferObject(contextid)->getGLObjectID();

    checkMapBufferObj((void**)&d_vbo,vbo);

    float * d_colorptr = (float*)d_vbo;
    d_colorptr += 3*_positionArray->size();

    launchPoint1((float3*)d_vbo,(float4*)d_colorptr,(float*)d_particleDataMap[contextid],(float*)d_debugDataMap[contextid],CUDA_MESH_WIDTH,CUDA_MESH_HEIGHT,max_age,disappear_age,alphaControl,anim,gravity,colorFreq,0.0);


    printCudaErr();

    cudaThreadSynchronize();

    checkUnmapBufferObj(vbo);

    if(contextid == 0)
    {

		for (int i =0;i<128;i++){_old_refl_hits[i] = h_debugData[i];}
        cuMemcpyDtoH(h_debugData, d_debugDataMap[contextid], sizeDebug);
		for (int i =0;i<128;i++){_refl_hits[i] = h_debugData[i] - _old_refl_hits[i];}
 	
        printCudaErr();
		// update
    }

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

void AlgebraInMotion::initPart()
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
   skipTonextScene=0;skipTOnextSceneOld=0;

    // init seenes
    scene0Start =0;
    scene1Start =0;
    scene2Start =1;//// must be set to starting
    scene3Start =0;
    scene4Start =0;
    scene0ContextStart =0;
    scene1ContextStart =0;
    scene2ContextStart =0;
    scene3ContextStart =0;
    scene4ContextStart =0;
    sceneNum =0;
    sceneOrder = 0;
    nextSean =0;
    witch_scene =2;// must be set to starting
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
	_refl_hits = new float[sizeDebug];
    h_debugData = new float[sizeDebug];
	_old_refl_hits =  new float[sizeDebug];
    for (int i = 0; i < 128; ++i)
    {
	h_debugData[i]=0;
	_old_refl_hits[i] = 0;
	_refl_hits[i] = 0;
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

void AlgebraInMotion::initGeometry()
{
// init partical system
    _particleObject = new PDObject("Algebra In Motion",false,false,false,true,false);

    _particleObject->addMenuItem(_gravityRV);
    _particleObject->addMenuItem(_rotateInjCB);
	_particleObject->addMenuItem(_speedRV);
	_particleObject->addMenuItem(_reflectorCB);
 

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
   // osg::ref_ptr<osg::Image> image = osgDB::readImageFile(_dataDir + "glsl/sprite.png");
    osg::ref_ptr<osg::Image> image = osgDB::readImageFile(_dataDir + "images/sprite50_50.png");
 

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
//  PluginHelper::getScene()->addChild(...)
    _particleGeode->addDrawable(_particleGeo);
    _particleObject->addChild(_particleGeode);
    PluginHelper::registerSceneObject(_particleObject);
    _particleObject->attachToScene();
    _particleObject->setNavigationOn(true);
// init hand object injector reflector
    osg::Matrix m, ms, mt;
    m.makeRotate((90.0/180.0)*M_PI,osg::Vec3(1.0,0,0));
    ms.makeScale(osg::Vec3(1000.0,1000.0,1000.0));
    mt.makeTranslate(osg::Vec3(0,0,-Navigation::instance()->getFloorOffset()));
    _particleObject->setTransform(m*ms*mt);

    stateset = _particleGeode->getOrCreateStateSet();
    stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

    _handModelMT = new osg::MatrixTransform();
	
    PluginHelper::getScene()->addChild(_handModelMT);
/*
    osg::Geode * geode = new osg::Geode();
    
	//reflector Visual
	//_handModelMT->addChild(geode);

	_reflectorObjSwitch = new osg::Switch;
   _handModelMT->addChild(_reflectorObjSwitch);
	_reflectorObjSwitch->addChild(geode);
    osg::ShapeDrawable * sd = new osg::ShapeDrawable(new osg::Box(osg::Vec3(0,0,0),200,100,20));
    geode->addDrawable(sd);
    sd->setColor(osg::Vec4(0,1,0,1));
	_reflectorObjSwitch->setAllChildrenOff();
*/
// injector Vissual read

	osg::Node* particle_inj_face = NULL;
	osg::Node* particle_inj_line = NULL;
	particle_inj_face = osgDB::readNodeFile(_dataDir + "/models/particle-inj-face.obj");
    if(!particle_inj_face){std::cerr << "Error reading /models/particle-inj-face.obj" << std::endl;} 
	particle_inj_line = osgDB::readNodeFile(_dataDir + "/models/particle-inj-line.obj");
    if(!particle_inj_line){std::cerr << "Error reading /models/particle-inj-line.obj" << std::endl;}
// reflector Vissual read
	osg::Node* particle_ref_face = NULL;
	osg::Node* particle_ref_line = NULL;
	particle_ref_face = osgDB::readNodeFile(_dataDir + "/models/particle-ref-face.obj");
    if(!particle_ref_face){std::cerr << "Error reading /models/particle-ref-face.obj" << std::endl;} 
	particle_ref_line = osgDB::readNodeFile(_dataDir + "/models/particle-ref-line.obj");
    if(!particle_ref_line){std::cerr << "Error reading /models/particle-ref-line.obj" << std::endl;} 

/* defined in .h	
		osg::ref_ptr<osg::Switch> _refObjSwitchFace;
		osg::ref_ptr<osg::Switch> _injObjSwitchFace;
		osg::ref_ptr<osg::Switch> _refObjSwitchLine;
		osg::ref_ptr<osg::Switch> _injObjSwitchLine;
*/

// make switches
	_refObjSwitchFace = new osg::Switch;
	_refObjSwitchLine = new osg::Switch;
	_injObjSwitchFace = new osg::Switch;
	_injObjSwitchLine = new osg::Switch;
// atatch switches to hand position
	_handModelMT->addChild(_refObjSwitchFace);
	_handModelMT->addChild(_refObjSwitchLine);
	_handModelMT->addChild(_injObjSwitchFace);
	_handModelMT->addChild(_injObjSwitchLine);
// scale modeles
        osg::MatrixTransform * mt_ref_face = new osg::MatrixTransform();
        osg::MatrixTransform * mt_ref_line = new osg::MatrixTransform();
        osg::MatrixTransform * mt_inj_face = new osg::MatrixTransform();
        osg::MatrixTransform * mt_inj_line = new osg::MatrixTransform();
        osg::Matrix mm;
        osg::Matrix mmr;
		mm.makeScale(osg::Vec3(1000.0,1000.0,1000.0));
        mmr.makeScale(osg::Vec3(100.0,100.0,100.0));

        mt_ref_face->setMatrix(mmr);
        mt_ref_line->setMatrix(mmr);
        mt_inj_face->setMatrix(mm);
        mt_inj_line->setMatrix(mm);

// attatch modeles to matrix xforms
        mt_inj_face->addChild(particle_inj_face);
        mt_inj_line->addChild(particle_inj_line);
        mt_ref_face->addChild(particle_ref_face);
        mt_ref_line->addChild(particle_ref_line);

// atatch scaled models to switches
        _refObjSwitchFace->addChild(mt_ref_face);
		_refObjSwitchLine->addChild(mt_ref_line);
       	_injObjSwitchFace->addChild(mt_inj_face);
		_injObjSwitchLine->addChild(mt_inj_line);

// loadata for screen positions for placement of stuff on screenes.
		//int i =loadPhysicalScreensArrayTourCaveCalit2();
		//int i =loadPhysicalScreensArrayTourCaveCalit2_5lowerScr();
	std::string str2 = ("TourCaveSaudi"); 
	if (_DisplaySystem.compare(str2) == 0)
		{
			std::cout << "TourCaveSaudi" << std::endl;
			int i =loadPhysicalScreensArrayTourCaveSaudi();
			std::cout << "loadPhysicalScreens " << i <<" " << _PhScAr[i-1].index << " " << std::endl;
		}
	else
		{
			std::cout << "TourCavecalit2" << std::endl;
			int i =loadPhysicalScreensArrayTourCaveCalit2_5lowerScr();
			std::cout << "loadPhysicalScreens " << i <<" " << _PhScAr[i-1].index << " " << std::endl;


		}



	initGeoEdSection();

/*
		_quad1 = createQuad();
 
		SceneObject * so = new SceneObject("anything",false,false,false,false,false);
        osg::Geode * qgeode = new osg::Geode();
        qgeode->setCullingActive(false);
        qgeode->addDrawable(_quad1);
        so->addChild(qgeode);
        PluginHelper::registerSceneObject(so,"AlgebraInMotion");
        so->setPosition(osg::Vec3(0,1000,0));
        so->setScale(1000);
        so->attachToScene();
		so->setNavigationOn(true);

*/
/*
    _injecttorObjSwitch = new osg::Switch;

    _injFaceProgram = new osg::Program();

    if(!particle_inj_face)
    {
        std::cerr << "Error reading inj face obj" << std::endl;
    }
    else
    {
        osg::MatrixTransform * mt = new osg::MatrixTransform();
        osg::Matrix m;
        m.makeScale(osg::Vec3(1000.0,1000.0,1000.0));
        mt->setMatrix(m);
        mt->addChild(particle_inj_face);
        _injecttorObjSwitch->addChild(mt);

        _injFaceProgram->setName("InjFace");
        _injFaceProgram->addShader(osg::Shader::readShaderFile(osg::Shader::VERTEX, osgDB::findDataFile(_dataDir + "glsl/sprite.vert")));
        _injFaceProgram->addShader(osg::Shader::readShaderFile(osg::Shader::FRAGMENT, osgDB::findDataFile(_dataDir + "glsl/sprite.frag")));
        //mt->getOrCreateStateSet()->setAttribute(_injFaceProgram);
    }

    _injLineProgram = new osg::Program();

    if(!particle_inj_line)
    {
        std::cerr << "Error reading inj line obj" << std::endl;
    }
    else
    {
        osg::MatrixTransform * mt = new osg::MatrixTransform();
        osg::Matrix m;
        m.makeScale(osg::Vec3(1000.0,1000.0,1000.0));
        mt->setMatrix(m);
        mt->addChild(particle_inj_line);
        _injecttorObjSwitch->addChild(mt);

        _injLineProgram->setName("InjLine");
        _injLineProgram->addShader(osg::Shader::readShaderFile(osg::Shader::VERTEX, osgDB::findDataFile(_dataDir + "glsl/sprite.vert")));
        _injLineProgram->addShader(osg::Shader::readShaderFile(osg::Shader::FRAGMENT, osgDB::findDataFile(_dataDir + "glsl/sprite.frag")));
        //mt->getOrCreateStateSet()->setAttribute(_injLineProgram);
    }

	_injecttorObjSwitch->setAllChildrenOn();
  	_handModelMT->addChild(_injecttorObjSwitch);
*/
}

void AlgebraInMotion::initSound()
{
    std::string ipAddrStr;
    unsigned short port;
    std::string pathToAudioFiles;

    ipAddrStr =  ConfigManager::getEntry("ipAddr","Plugin.AlgebraInMotion.SoundServer","127.0.0.1");
;
    port = ConfigManager::getInt("port", "Plugin.AlgebraInMotion.SoundServer", 31231);

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
    short_sound_01a1.initialize(pathToAudioFiles, "dan_short_sound_01a.wav");
    short_sound_01a2.initialize(pathToAudioFiles, "dan_short_sound_01a.wav");
    short_sound_01a3.initialize(pathToAudioFiles, "dan_short_sound_01a.wav");
    short_sound_01a4.initialize(pathToAudioFiles, "dan_short_sound_01a.wav");
    short_sound_01a5.initialize(pathToAudioFiles, "dan_short_sound_01a.wav");
    short_sound_01a.setGain(10);
    short_sound_01a1.setGain(10);
    short_sound_01a2.setGain(10);
    short_sound_01a3.setGain(10);
    short_sound_01a4.setGain(10);
    short_sound_01a5.setGain(10);

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

void AlgebraInMotion::updateHand()
{
// set offsetfor injector and reflectormodeles
   osg::Vec3 offset(0.0,200,100.0);// tourcave
//	  osg::Vec3 offset(0.0,600,-100);// simulator
	
	std::string str2 = ("Simulator"); 
//	if (_DisplaySystem.compare(str2) == 0){ std::cout << "Simulator" << std::endl;}
	
	
    osg::Matrix hMat;
    hMat.makeRotate(_pointerHeading * M_PI / 180.0, osg::Vec3(0,0,1));
    osg::Matrix pMat;
    pMat.makeRotate(_pointerPitch * M_PI / 180.0, osg::Vec3(1,0,0));
    osg::Matrix rMat;
    rMat.makeRotate(_pointerRoll * M_PI / 180.0, osg::Vec3(0,1,0));
    osg::Matrix oMat = rMat * pMat * hMat;

    osg::Matrix m = PluginHelper::getHandMat(hand_id) * _particleObject->getWorldToObjectMatrix();
    osg::Vec3 handdir = osg::Vec3(0,1.0,0) * oMat * m;
    handdir = handdir - m.getTrans();
    handdir.normalize();

    osg::Vec3 handup = osg::Vec3(0,0,1.0) * oMat * m;
    handup = handup - m.getTrans();
    handup.normalize();

    osg::Vec3 handpos = offset * m;
    m.setTrans(handpos);

    wandPos[0] = handpos.x();
    wandPos[1] = handpos.y();
    wandPos[2] = handpos.z();
    wandVec[0] = handdir.x();
    wandVec[1] = handdir.y();
    wandVec[2] = handdir.z();
    wandMat[4] = handup.x();
    wandMat[5] = handup.y();
    wandMat[6] = handup.z();



    osg::Matrix handm;
    handm = PluginHelper::getHandMat(hand_id);
    osg::Vec3 modelPos = offset * handm;
    handm = oMat * handm;
    handm.setTrans(modelPos);
    _handModelMT->setMatrix(handm);
}

void AlgebraInMotion::pdata_init_age(int mAge)
{
    for (int i = 0; i < CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT; ++i)
    {
        // set age to random ages < max age to permit a respawn of the particle
        h_particleData[PDATA_ROW_SIZE*i] = rand() % mAge; // age
 
    }

}
void AlgebraInMotion::pdata_init_velocity(float vx,float vy,float vz)
{
    for (int i = 0; i < CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT; ++i)
    { 
        h_particleData[PDATA_ROW_SIZE * i + 1] = vx;
        h_particleData[PDATA_ROW_SIZE * i + 2] = vy;
        h_particleData[PDATA_ROW_SIZE * i + 3] = vz;
    }

}
void AlgebraInMotion::pdata_init_rand()
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

void AlgebraInMotion::copy_reflector( int sorce, int destination)
{
    for (int row =0;row < REFL_DATA_ROWS;row++)	
    {
        for (int ele =0;ele <3;ele++)
        {
            h_reflectorData[ destination][row][ele] = h_reflectorData[sorce ][row][ele];
        }
    }
}

void AlgebraInMotion::copy_injector( int sorce, int destination)
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



int AlgebraInMotion::load6wallcaveWalls(int firstRefNum)
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


int AlgebraInMotion::loadPhysicalScreensArrayTourCaveCalit2()
{
_PhScAr = new _PhSc [128];

float height, h, width, p, originX, originY,r,name,originZ, screen;
int i=0;

// _PhScAr[i].index  -1 is sentannel for end of list
     height= 805 ;h=  74.0; width= 1432 ;  p= 0.0    ;originX= -1949  ; originY= 577  ;  r= -90.0 ;   name= 0 ;  originZ= 0   ;   screen= 0; 
 	_PhScAr[i].index =i; 
  	_PhScAr[i].height =height;
  	_PhScAr[i].h=h;
  	_PhScAr[i].width = width;
 	_PhScAr[i].originX =originX;
  	_PhScAr[i].originY =originY;
  	_PhScAr[i].r=r;
  	_PhScAr[i].originZ = originZ;
  	_PhScAr[i].screen = screen; 
  	std::cout << " i , originX,originY, originZ _PhScAr[i].originX ,_PhScAr[i].originY ,  _PhScAr[i].originX " <<  i << " " << originX << " " << originY<< " " << originZ  << " " << _PhScAr[i].originX << " " <<_PhScAr[i].originY << " " <<  _PhScAr[i].originZ << std::endl; 
 	i++;

     height= 805 ;h=  74.0; width= 1432 ;  p= 0.0    ;originX= -1997  ; originY= 592  ;  r= -90.0 ;   name= 1 ;  originZ= 1490   ;   screen= 1;   
 	_PhScAr[i].index =i;
  	_PhScAr[i].height =height;
  	_PhScAr[i].h=h;
  	_PhScAr[i].width = width;
 	_PhScAr[i].originX =originX;
  	_PhScAr[i].originY =originY;
  	_PhScAr[i].r=r;
  	_PhScAr[i].originZ = originZ;
  	_PhScAr[i].screen = screen; 		
 	i++;

     height= 805 ;h=  50.0; width= 1432 ;  p= 0.0    ;originX= -1489  ; originY= 1236  ;  r= -90.0 ;   name= 0 ;  originZ= 0   ;   screen= 0; 
 	_PhScAr[i].index =i;
   	_PhScAr[i].height =height;
  	_PhScAr[i].h=h;
  	_PhScAr[i].width = width; 
 	_PhScAr[i].originX =originX;
  	_PhScAr[i].originY =originY;
  	_PhScAr[i].r=r;
  	_PhScAr[i].originZ = originZ;
  	_PhScAr[i].screen = screen; 		
 	i++;
 
     height= 805 ;h=  50.0; width= 1432 ;  p= 0.0    ;originX= -1527  ; originY= 1268  ;  r= -90.0 ;   name= 1 ;  originZ= 1490   ;   screen= 1;   
 	_PhScAr[i].index =i; 
  	_PhScAr[i].height =height;
  	_PhScAr[i].h=h;
  	_PhScAr[i].width = width;
 	_PhScAr[i].originX =originX;
  	_PhScAr[i].originY =originY;
  	_PhScAr[i].r=r;
  	_PhScAr[i].originZ = originZ;
  	_PhScAr[i].screen = screen; 		
 	i++;

     height= 805 ;h=  26.0; width= 1432 ;  p= 0.0    ;originX= -802  ; originY= 1657  ;  r= -90.0 ;   name= 0 ;  originZ= 0   ;   screen= 0;
  	_PhScAr[i].index =i;
   	_PhScAr[i].height =height;
  	_PhScAr[i].h=h;
  	_PhScAr[i].width = width;
  	_PhScAr[i].p = p;
 	_PhScAr[i].originX =originX;
  	_PhScAr[i].originY =originY;
  	_PhScAr[i].r=r;
  	_PhScAr[i].originZ = originZ;
  	_PhScAr[i].screen = screen; 		
 	i++;
 
     height= 805 ;h=  26.0; width= 1432 ;  p= 0.0    ;originX= -823  ; originY= 1702  ;  r= -90.0 ;   name= 1 ;  originZ= 1490   ;   screen= 1;   
  	_PhScAr[i].index =i; 
  	_PhScAr[i].height =height;
  	_PhScAr[i].h=h;
  	_PhScAr[i].width = width;
 	_PhScAr[i].p = p;
  	_PhScAr[i].originX =originX;
  	_PhScAr[i].originY =originY;
  	_PhScAr[i].r=r;
  	_PhScAr[i].originZ = originZ;
  	_PhScAr[i].screen = screen; 		
 	i++;

     height= 805 ;h=  0.0; width= 1432 ;  p= 0.0    ;originX= 0  ; originY= 1750  ;  r= -90.0 ;   name= 0 ;  originZ= 0   ;   screen= 0;  
 	_PhScAr[i].index =i; 
   	_PhScAr[i].height =height;
  	_PhScAr[i].h=h;
  	_PhScAr[i].width = width;
 	_PhScAr[i].p = p;
  	_PhScAr[i].originX =originX;
  	_PhScAr[i].originY =originY;
  	_PhScAr[i].r=r;
  	_PhScAr[i].originZ = originZ;
  	_PhScAr[i].screen = screen; 		
 	i++;

     height= 805 ;h=  0.0; width= 1432 ;  p= 0.0    ;originX= 0  ; originY= 1800  ;  r= -90.0 ;   name= 1 ;  originZ= 1490   ;   screen= 1;   
 	_PhScAr[i].index =i;
  	_PhScAr[i].height =height;
  	_PhScAr[i].h=h;
  	_PhScAr[i].width = width;
 	_PhScAr[i].p = p;
  	_PhScAr[i].originX =originX;
  	_PhScAr[i].originY =originY;
  	_PhScAr[i].r=r;
  	_PhScAr[i].originZ = originZ;
  	_PhScAr[i].screen = screen; 		
 	i++;

     height= 805 ;h=  -26.0; width= 1432 ;  p= 0.0    ;originX= 738  ; originY= 1481  ;  r= -90.0 ;   name= 0 ;  originZ= 0   ;   screen= 0; 
  	_PhScAr[i].index =i; 
   	_PhScAr[i].height =height;
  	_PhScAr[i].h=h;
  	_PhScAr[i].width = width;
 	_PhScAr[i].p = p;
  	_PhScAr[i].originX =originX;
  	_PhScAr[i].originY =originY;
  	_PhScAr[i].r=r;
  	_PhScAr[i].originZ = originZ;
  	_PhScAr[i].screen = screen; 		
 	i++;

     height= 805 ;h=  -26.0; width= 1432 ;  p= 0.0    ;originX= 760  ; originY= 1526  ;  r= -90.0 ;   name= 1 ;  originZ= 1490   ;   screen= 1;   
 	_PhScAr[i].index =i; 
  	_PhScAr[i].height =height;
  	_PhScAr[i].h=h;
  	_PhScAr[i].width = width;
 	_PhScAr[i].p = p;
  	_PhScAr[i].originX =originX;
  	_PhScAr[i].originY =originY;
  	_PhScAr[i].r=r;
  	_PhScAr[i].originZ = originZ;
  	_PhScAr[i].screen = screen; 		
 	i++;

     height= 805 ;h=  -55.0; width= 1432 ;  p= 0.0    ;originX= 1276  ; originY= 904  ;  r= -90.0 ;   name= 0 ;  originZ= 0   ;   screen= 0;
  	_PhScAr[i].index =i;
   	_PhScAr[i].height =height;
  	_PhScAr[i].h=h;
  	_PhScAr[i].width = width;
 	_PhScAr[i].p = p;
  	_PhScAr[i].originX =originX;
  	_PhScAr[i].originY =originY;
  	_PhScAr[i].r=r;
  	_PhScAr[i].originZ = originZ;
  	_PhScAr[i].screen = screen; 		
 	i++;
 
     height= 805 ;h=  -55.0; width= 1432 ;  p= 0.0    ;originX= 1317  ; originY= 932  ;  r= -90.0 ;   name= 1 ;  originZ= 1490   ;   screen= 1;   
 	_PhScAr[i].index =i; 
  	_PhScAr[i].height =height;
  	_PhScAr[i].h=h;
  	_PhScAr[i].width = width;
 	_PhScAr[i].p = p;
  	_PhScAr[i].originX =originX;
  	_PhScAr[i].originY =originY;
  	_PhScAr[i].r=r;
  	_PhScAr[i].originZ = originZ;
  	_PhScAr[i].screen = screen; 		
 	i++;

     height= 805 ;h=  -85.0; width= 1432 ;  p= 0.0    ;originX= 1471  ; originY= 135  ;  r= -90.0 ;   name= 0 ;  originZ= 0   ;   screen= 0; 
  	_PhScAr[i].index =i; 
   	_PhScAr[i].height =height;
  	_PhScAr[i].h=h;
  	_PhScAr[i].width = width;
 	_PhScAr[i].p = p;
  	_PhScAr[i].originX =originX;
  	_PhScAr[i].originY =originY;
  	_PhScAr[i].r=r;
  	_PhScAr[i].originZ = originZ;
  	_PhScAr[i].screen = screen; 		
 	i++;

     height= 805 ;h=  -85.0; width= 1432 ;  p= 0.0    ;originX= 1521  ; originY= 139  ;  r= -90.0 ;   name= 1 ;  originZ= 1490   ;   screen= 1;   
 	_PhScAr[i].index =i; 
  	_PhScAr[i].height =height;
  	_PhScAr[i].h=h;
  	_PhScAr[i].width = width;
 	_PhScAr[i].p = p;
  	_PhScAr[i].originX =originX;
  	_PhScAr[i].originY =originY;
  	_PhScAr[i].r=r;
  	_PhScAr[i].originZ = originZ;
  	_PhScAr[i].screen = screen; 		
 	i++;
 	
 	 _PhScAr[i].index =-1;
 	// clearly dont have this correct
 	// probibly need to xform positions and vectors from z up to z forward
 	for (int j=0;j<i;j++)
 		{
 			
 			//std::cout << " j,x,y,z pos " << j << " " << _PhScAr[j].originX << " " << _PhScAr[j].originY << " " << _PhScAr[j].originZ << std::endl;

	 		osg::Matrix hMat;
	   		hMat.makeRotate(_PhScAr[j].h * M_PI / 180.0, osg::Vec3(0,0,1));
	   	   	osg::Matrix pMat;
				pMat.makeRotate(_PhScAr[i].p* M_PI / 180.0, osg::Vec3(1,0,0));
				osg::Matrix rMat;
				rMat.makeRotate(_PhScAr[i].r * M_PI / 180.0, osg::Vec3(0,1,0));
	//    	osg::Matrix oMat = rMat* pMat * hMat;
				osg::Matrix oMat = hMat;

	   		osg::Vec3 test = oMat * osg::Vec3(0,-1,0);
	   		//std::cout << "test.x,y,z " << _PhScAr[j].h << " " << test[0] << " " << test[1] << " " << test[2] << std::endl;
	 		_PhScAr[j].vx = test[0] * -1;
	 		_PhScAr[j].vy = test[1];
			_PhScAr[j].vz = test[2];
			// rotatefrom z up ti z back
			// x stays same
	   		// y =z
	   		//z = -y
			_PhScAr[j].originZ = _PhScAr[j].originZ  + Navigation::instance()->getFloorOffset();
			float ytemp;
	   		if ( 1 == 1)
		   		{
	   				// process position
			   		ytemp = _PhScAr[j].originY;
			   		_PhScAr[j].originY = _PhScAr[j].originZ;
			   		_PhScAr[j].originZ = -ytemp;
			   		
			   		// vector
			   		ytemp = _PhScAr[j].vy;
			   		_PhScAr[j].vy = _PhScAr[j].vz;
			   		_PhScAr[j].vz = -ytemp;
					
		   		}
 			//std::cout << " j,x,y,z pos " << j << " " << _PhScAr[j].originX << " " << _PhScAr[j].originY << " " << _PhScAr[j].originZ << std::endl<< std::endl;
			
		 	}

 return i;
}


int AlgebraInMotion::loadPhysicalScreensArrayTourCaveCalit2_5lowerScr()
{
_PhScAr = new _PhSc [128];

float height, h, width, p, originX, originY,r,name,originZ, screen;
int i=0;

// _PhScAr[i].index  -1 is sentannel for end of list

     height= 805 ;h=  50.0; width= 1432 ;  p= 0.0    ;originX= -1489  ; originY= 1236  ;  r= -90.0 ;   name= 0 ;  originZ= 0   ;   screen= 0; 
 	_PhScAr[i].index =i;
   	_PhScAr[i].height =height;
  	_PhScAr[i].h=h;
  	_PhScAr[i].width = width;
 	_PhScAr[i].originX =originX;
  	_PhScAr[i].originY =originY;
  	_PhScAr[i].r=r;
  	_PhScAr[i].originZ = originZ;
  	_PhScAr[i].screen = screen; 		
 	i++;
 
     height= 805 ;h=  26.0; width= 1432 ;  p= 0.0    ;originX= -802  ; originY= 1657  ;  r= -90.0 ;   name= 0 ;  originZ= 0   ;   screen= 0;
  	_PhScAr[i].index =i;
   	_PhScAr[i].height =height;
  	_PhScAr[i].h=h;
  	_PhScAr[i].width = width;
  	_PhScAr[i].p = p;
 	_PhScAr[i].originX =originX;
  	_PhScAr[i].originY =originY;
  	_PhScAr[i].r=r;
  	_PhScAr[i].originZ = originZ;
  	_PhScAr[i].screen = screen; 		
 	i++;
 
 
     height= 805 ;h=  0.0; width= 1432 ;  p= 0.0    ;originX= 0  ; originY= 1750  ;  r= -90.0 ;   name= 0 ;  originZ= 0   ;   screen= 0;  
 	_PhScAr[i].index =i; 
   	_PhScAr[i].height =height;
  	_PhScAr[i].h=h;
  	_PhScAr[i].width = width;
 	_PhScAr[i].p = p;
  	_PhScAr[i].originX =originX;
  	_PhScAr[i].originY =originY;
  	_PhScAr[i].r=r;
  	_PhScAr[i].originZ = originZ;
  	_PhScAr[i].screen = screen; 		
 	i++;

 
     height= 805 ;h=  -26.0; width= 1432 ;  p= 0.0    ;originX= 738  ; originY= 1481  ;  r= -90.0 ;   name= 0 ;  originZ= 0   ;   screen= 0; 
  	_PhScAr[i].index =i; 
   	_PhScAr[i].height =height;
  	_PhScAr[i].h=h;
  	_PhScAr[i].width = width;
 	_PhScAr[i].p = p;
  	_PhScAr[i].originX =originX;
  	_PhScAr[i].originY =originY;
  	_PhScAr[i].r=r;
  	_PhScAr[i].originZ = originZ;
  	_PhScAr[i].screen = screen; 		
 	i++;

 
     height= 805 ;h=  -55.0; width= 1432 ;  p= 0.0    ;originX= 1276  ; originY= 904  ;  r= -90.0 ;   name= 0 ;  originZ= 0   ;   screen= 0;
  	_PhScAr[i].index =i;
   	_PhScAr[i].height =height;
  	_PhScAr[i].h=h;
  	_PhScAr[i].width = width;
 	_PhScAr[i].p = p;
  	_PhScAr[i].originX =originX;
  	_PhScAr[i].originY =originY;
  	_PhScAr[i].r=r;
  	_PhScAr[i].originZ = originZ;
  	_PhScAr[i].screen = screen; 		
 	i++;
 

 
 	 _PhScAr[i].index =-1;
 	// clearly dont have this correct
 	// probibly need to xform positions and vectors from z up to z forward
 	for (int j=0;j<i;j++)
 		{
 			
 			//std::cout << " j,x,y,z pos " << j << " " << _PhScAr[j].originX << " " << _PhScAr[j].originY << " " << _PhScAr[j].originZ << std::endl;

	 		osg::Matrix hMat;
	   		hMat.makeRotate(_PhScAr[j].h * M_PI / 180.0, osg::Vec3(0,0,1));
	   	   	osg::Matrix pMat;
				pMat.makeRotate(_PhScAr[i].p* M_PI / 180.0, osg::Vec3(1,0,0));
				osg::Matrix rMat;
				rMat.makeRotate(_PhScAr[i].r * M_PI / 180.0, osg::Vec3(0,1,0));
	//    	osg::Matrix oMat = rMat* pMat * hMat;
				osg::Matrix oMat = hMat;

	   		osg::Vec3 test = oMat * osg::Vec3(0,-1,0);
	   		//std::cout << "test.x,y,z " << _PhScAr[j].h << " " << test[0] << " " << test[1] << " " << test[2] << std::endl;
	 		_PhScAr[j].vx = test[0] * -1;
	 		_PhScAr[j].vy = test[1];
			_PhScAr[j].vz = test[2];
			// rotatefrom z up ti z back
			// x stays same
	   		// y =z
	   		//z = -y
			_PhScAr[j].originZ = _PhScAr[j].originZ  + Navigation::instance()->getFloorOffset();
			float ytemp;
	   		if ( 1 == 1)
		   		{
	   				// process position
			   		ytemp = _PhScAr[j].originY;
			   		_PhScAr[j].originY = _PhScAr[j].originZ;
			   		_PhScAr[j].originZ = -ytemp;
			   		
			   		// vector
			   		ytemp = _PhScAr[j].vy;
			   		_PhScAr[j].vy = _PhScAr[j].vz;
			   		_PhScAr[j].vz = -ytemp;
					
		   		}
 			//std::cout << " j,x,y,z pos " << j << " " << _PhScAr[j].originX << " " << _PhScAr[j].originY << " " << _PhScAr[j].originZ << std::endl<< std::endl;
			
		 	}

 return i;
}

int AlgebraInMotion::loadPhysicalScreensArrayTourCaveSaudi()
{
// not dune yet
//based on tourcaveLG-screens-5.xml
//   <Screen height="805" h="63.0" width="1432" p="0.0"   originX="-1415"  comment="S_A" originY="721"  r="-90.0" name="0" originZ="0"    screen="0" />
//    <Screen height="805" h="32.0" width="1432" p="0.0"   originX="-816"  comment="S_A" originY="1308"  r="-90.0" name="1" originZ="0"    screen="1" />
 //   <Screen height="805" h="0.0" width="1432" p="0.0"   originX="0"  comment="S_A" originY="1500"      r="-90.0" name="2" originZ="0"    screen="2" />
 //   <Screen height="805" h="-32.0" width="1432" p="0.0"   originX="774"  comment="S_A" originY="1235"  r="-90.0" name="3" originZ="0"    screen="3" />
//    <Screen height="805" h="-65.0" width="1432" p="0.0"   originX="1283"  comment="S_A" originY="598"  r="-90.0" name="4" originZ="0"    screen="4" />
 
_PhScAr = new _PhSc [128];

float height, h, width, p, originX, originY,r,name,originZ, screen;
int i=0;

// _PhScAr[i].index  -1 is sentannel for end of list

//   <Screen height="805" h="63.0" width="1432" p="0.0"   originX="-1415"  comment="S_A" originY="721"  r="-90.0" name="0" originZ="0"    screen="0" />
     height= 805 ;h=  63.0; width= 1432 ;  p= 0.0    ;originX= -1415  ; originY= 721  ;  r= -90.0 ;   name= 0 ;  originZ= 0   ;   screen= 0; 
 	_PhScAr[i].index =i;
   	_PhScAr[i].height =height;
  	_PhScAr[i].h=h;
  	_PhScAr[i].width = width;
 	_PhScAr[i].originX =originX;
  	_PhScAr[i].originY =originY;
  	_PhScAr[i].r=r;
  	_PhScAr[i].originZ = originZ;
  	_PhScAr[i].screen = screen; 		
 	i++;
//    <Screen height="805" h="32.0" width="1432" p="0.0"   originX="-816"  comment="S_A" originY="1308"  r="-90.0" name="1" originZ="0"    screen="1" />
 
     height= 805 ;h= 32.0; width= 1432 ;  p= 0.0    ;originX= -816  ; originY= 1308  ;  r= -90.0 ;   name= 1 ;  originZ= 0   ;   screen= 0;
  	_PhScAr[i].index =i;
   	_PhScAr[i].height =height;
  	_PhScAr[i].h=h;
  	_PhScAr[i].width = width;
  	_PhScAr[i].p = p;
 	_PhScAr[i].originX =originX;
  	_PhScAr[i].originY =originY;
  	_PhScAr[i].r=r;
  	_PhScAr[i].originZ = originZ;
  	_PhScAr[i].screen = screen; 		
 	i++;
 
  //   <Screen height="805" h="0.0" width="1432" p="0.0"   originX="0"  comment="S_A" originY="1500"      r="-90.0" name="2" originZ="0"    screen="2" />

     height= 805 ;h=  0.0; width= 1432 ;  p= 0.0    ;originX= 0  ; originY= 1500  ;  r= -90.0 ;   name= 2 ;  originZ= 0   ;   screen= 0;  
 	_PhScAr[i].index =i; 
   	_PhScAr[i].height =height;
  	_PhScAr[i].h=h;
  	_PhScAr[i].width = width;
 	_PhScAr[i].p = p;
  	_PhScAr[i].originX =originX;
  	_PhScAr[i].originY =originY;
  	_PhScAr[i].r=r;
  	_PhScAr[i].originZ = originZ;
  	_PhScAr[i].screen = screen; 		
 	i++;

 //   <Screen height="805" h="-32.0" width="1432" p="0.0"   originX="774"  comment="S_A" originY="1235"  r="-90.0" name="3" originZ="0"    screen="3" />
 
     height= 805 ;h=  -32.0; width= 1432 ;  p= 0.0    ;originX= 774  ; originY= 1235  ;  r= -90.0 ;   name= 3 ;  originZ= 0   ;   screen= 0; 
  	_PhScAr[i].index =i; 
   	_PhScAr[i].height =height;
  	_PhScAr[i].h=h;
  	_PhScAr[i].width = width;
 	_PhScAr[i].p = p;
  	_PhScAr[i].originX =originX;
  	_PhScAr[i].originY =originY;
  	_PhScAr[i].r=r;
  	_PhScAr[i].originZ = originZ;
  	_PhScAr[i].screen = screen; 		
 	i++;

 //    <Screen height="805" h="-65.0" width="1432" p="0.0"   originX="1283"  comment="S_A" originY="598"  r="-90.0" name="4" originZ="0"    screen="4" />
 
     height= 805 ;h=  -65.0; width= 1432 ;  p= 0.0    ;originX= 1283  ; originY= 598  ;  r= -90.0 ;   name= 4 ;  originZ= 0   ;   screen= 0;
  	_PhScAr[i].index =i;
   	_PhScAr[i].height =height;
  	_PhScAr[i].h=h;
  	_PhScAr[i].width = width;
 	_PhScAr[i].p = p;
  	_PhScAr[i].originX =originX;
  	_PhScAr[i].originY =originY;
  	_PhScAr[i].r=r;
  	_PhScAr[i].originZ = originZ;
  	_PhScAr[i].screen = screen; 		
 	i++;
 

 
 	 _PhScAr[i].index =-1;
 	// clearly dont have this correct
 	// probibly need to xform positions and vectors from z up to z forward
 	for (int j=0;j<i;j++)
 		{
 			
 			//std::cout << " j,x,y,z pos " << j << " " << _PhScAr[j].originX << " " << _PhScAr[j].originY << " " << _PhScAr[j].originZ << std::endl;

	 		osg::Matrix hMat;
	   		hMat.makeRotate(_PhScAr[j].h * M_PI / 180.0, osg::Vec3(0,0,1));
	   	   	osg::Matrix pMat;
				pMat.makeRotate(_PhScAr[i].p* M_PI / 180.0, osg::Vec3(1,0,0));
				osg::Matrix rMat;
				rMat.makeRotate(_PhScAr[i].r * M_PI / 180.0, osg::Vec3(0,1,0));
	//    	osg::Matrix oMat = rMat* pMat * hMat;
				osg::Matrix oMat = hMat;

	   		osg::Vec3 test = oMat * osg::Vec3(0,-1,0);
	   		//std::cout << "test.x,y,z " << _PhScAr[j].h << " " << test[0] << " " << test[1] << " " << test[2] << std::endl;
	 		_PhScAr[j].vx = test[0] * -1;
	 		_PhScAr[j].vy = test[1];
			_PhScAr[j].vz = test[2];
			// rotatefrom z up ti z back
			// x stays same
	   		// y =z
	   		//z = -y
			_PhScAr[j].originZ = _PhScAr[j].originZ  + Navigation::instance()->getFloorOffset();
			float ytemp;
	   		if ( 1 == 1)
		   		{
	   				// process position
			   		ytemp = _PhScAr[j].originY;
			   		_PhScAr[j].originY = _PhScAr[j].originZ;
			   		_PhScAr[j].originZ = -ytemp;
			   		
			   		// vector
			   		ytemp = _PhScAr[j].vy;
			   		_PhScAr[j].vy = _PhScAr[j].vz;
			   		_PhScAr[j].vz = -ytemp;
					
		   		}
 			//std::cout << " j,x,y,z pos " << j << " " << _PhScAr[j].originX << " " << _PhScAr[j].originY << " " << _PhScAr[j].originZ << std::endl<< std::endl;
			
		 	}

 return i;
}


int AlgebraInMotion::loadInjFountsFrScr(float dx,float dy,float dz,float speed)
	{
		// in z back space
		// in meters
		int injNum;
		
		
		//h_injectorData[0][0][0] =0;
		int firstInjNum = h_injectorData[0][0][0]+1 ;
		int i=0;
		while ( (_PhScAr[i].index) !=( -1)) 
			{
				injNum=firstInjNum +i;
				//std::cout << " injNum _PhScAr[i].originX /1000.0 .originY /1000.0 .originY /1000.0 " <<  injNum << " " << _PhScAr[i].originX /1000.0 << " " << _PhScAr[i].originY /1000.0 << " " << _PhScAr[i].originZ /1000.0 << std::endl;
				h_injectorData[injNum][1][0]=2;h_injectorData[injNum][1][1]=1.0;// type, injection ratio ie streem volume, ~
				h_injectorData[injNum][2][0]=_PhScAr[i].originX /1000.0 +dx;h_injectorData[injNum][2][1]=_PhScAr[i].originY / 1000.0+dy;h_injectorData[injNum][2][2]=_PhScAr[i].originZ /1000.0 + dz;//x,y,z position
				h_injectorData[injNum][3][0]=_PhScAr[i].vx * speed;h_injectorData[injNum][3][1]=_PhScAr[i].vy * speed;h_injectorData[injNum][3][2]=_PhScAr[i].vz * speed;//x,y,z velocity drection
				h_injectorData[injNum][4][0]=ftToM(0.25);h_injectorData[injNum][4][1]=ftToM(0.25);h_injectorData[injNum][4][2]=ftToM(0.25);//x,y,z size
				h_injectorData[injNum][5][0]=0.000;h_injectorData[injNum][5][1]=0.000;h_injectorData[injNum][5][2]=0.000;//t,u,v jiter v not implimented = speed 
				h_injectorData[injNum][6][0]=0.2000;h_injectorData[injNum][6][1]=0.0;h_injectorData[injNum][6][2]=0.0;//speed jiter ~ ~
				h_injectorData[injNum][7][0]=5;h_injectorData[injNum][7][1]=5;h_injectorData[injNum][7][2]=5;//centrality of rnd distribution speed dt tu ~
						
				i++;
			}  
			
		 h_injectorData[0][0][0]=injNum;
		return injNum;
	}

int AlgebraInMotion::loadReflFrScr()

{
		// in z back space
		// in meters

   float screenrad = ftToM(5.0);
    int reflNum;
    float damping =.5;
    float no_traping =0;
    float firstRefNum = h_reflectorData[0][0][0] + 1;
    reflNum = firstRefNum;
	int i=0;
	while ( (_PhScAr[i].index) !=( -1))
		{
			float xpos =  _PhScAr[i].originX /1000.0;
			float ypos =  _PhScAr[i].originY /1000.0;
			float zpos =  _PhScAr[i].originZ /1000.0;
			float vx = _PhScAr[i].vx;
			float vy = _PhScAr[i].vy;
			float vz = _PhScAr[i].vz;
			reflNum = firstRefNum +i;
  
			h_reflectorData[reflNum ][0][0]=1;h_reflectorData[reflNum ][0][1]=0;// type, age ie colormod, ~  0 is off 1 is plane reflector
			h_reflectorData[reflNum ][1][0]=xpos;    h_reflectorData[reflNum ][1][1]= ypos;h_reflectorData[reflNum ][1][2]=zpos;//x,y,z position
			h_reflectorData[reflNum ][2][0]= vx;  h_reflectorData[reflNum ][2][1]=vy;    h_reflectorData[reflNum ][2][2]=vz;//x,y,z normal
			h_reflectorData[reflNum ][3][0]=screenrad; h_reflectorData[reflNum ][3][1]=0.00; h_reflectorData[reflNum ][3][2]=0;//reflector radis ,~,~ 
			h_reflectorData[reflNum ][4][0]=0.000;h_reflectorData[reflNum ][4][1]=0.000;h_reflectorData[reflNum ][4][2]=0.000;//t,u,v jiter  not implimented = speed 
			h_reflectorData[reflNum ][5][0]= damping; h_reflectorData[reflNum ][5][1]=no_traping;  h_reflectorData[reflNum ][5][2]=0.0;//reflectiondamping , no_traping ~
			h_reflectorData[reflNum ][6][0]=0;    h_reflectorData[reflNum ][6][1]=0;    h_reflectorData[reflNum ][6][2]=0;// not implemented yet centrality of rnd distribution speed dt tu ~
			i++;

		}
  			 h_reflectorData[0][0][0]= reflNum;

		
	return reflNum ;
}

void AlgebraInMotion::scene_data_0_host()
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
	_refObjSwitchFace->setAllChildrenOff();
	_refObjSwitchLine->setAllChildrenOff();
	_injObjSwitchLine->setAllChildrenOn();
    h_reflectorData[0][0][0] =0;//turn off all reflectors
    h_injectorData[0][0][0] =0;// turn off all injectors ~ ~   ~ means dont care

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
    if(but2==1)	_injObjSwitchFace->setAllChildrenOn();
	
	if(but2==0)	_injObjSwitchFace->setAllChildrenOff();
	

    int injNum ;	
    h_injectorData[0][0][0] =1;// number of injectors ~ ~   ~ means dont care
    //injector 1
    injNum =1;

 
    if (soundEnabled)injSoundUpdate(injNum);

    injNum =1;
    h_injectorData[injNum][1][0]=1;h_injectorData[injNum][1][1]=but2;// type, injection ratio ie streem volume, ~
    h_injectorData[injNum][2][0]=wandPos[0];h_injectorData[injNum][2][1]=wandPos[1];h_injectorData[injNum][2][2]=wandPos[2];//x,y,z position
    h_injectorData[injNum][3][0]=wandVec[0];h_injectorData[injNum][3][1]=wandVec[1];h_injectorData[injNum][3][2]=wandVec[2];//x,y,z velocity direction
    h_injectorData[injNum][4][0]=0.00;h_injectorData[injNum][4][1]=0.00;h_injectorData[injNum][4][2]=.0;//x,y,z size
    h_injectorData[injNum][5][0]=0.010;h_injectorData[injNum][5][1]=0.010;h_injectorData[injNum][5][2]=0.000;//t,u,v jiter v not implimented = speed 
    h_injectorData[injNum][6][0]=.1;h_injectorData[injNum][6][1]=0.1;h_injectorData[injNum][6][2]=0.0;//speed jiter ~ ~
    h_injectorData[injNum][7][0]=5;h_injectorData[injNum][7][1]=5;h_injectorData[injNum][7][2]=5;//centrality of rnd distribution speed dt tu ~
    //if (but1){printf (" wandPos[0 ,1,2] wandVec[0,1,2] %f %f %f    %f %f %f \n", wandPos[0],wandPos[1],wandPos[2],wandVec[0],wandVec[1],wandVec[2]);}
    // load starcave wall reflectors
   // h_reflectorData[0][0][0] = loadStarcaveWalls(1);
    if (time_in_sean >5)
		{
			h_reflectorData[0][0][0] =load6wallcaveWalls(1);
			//h_reflectorData[0][0][0] =0;
 			//int numRef =loadReflFrScr();
			
		}

    scene0Start =0;
}

void AlgebraInMotion::scene_data_1_host()
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
		_injObjSwitchFace->setAllChildrenOff();
		_injObjSwitchLine->setAllChildrenOff();
		_refObjSwitchLine->setAllChildrenOn();
    h_reflectorData[0][0][0] =0;//turn off all reflectors
    h_injectorData[0][0][0] =0;// turn off all injectors ~ ~   ~ means dont care

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
//    if (time_in_sean >110)nextSean=1;
    if (time_in_sean >10)nextSean=1;// short time
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
// reflector obj switch
  if(but2==1) 
   if(but2==0)_refObjSwitchFace->setAllChildrenOff();
 
    h_reflectorData[reflNum ][0][0]=but2;h_reflectorData[reflNum ][0][1]=1;// type, age ie colormod, ~  0 is off 1 is plane reflector  0 is off 1 is plane reflector
    h_reflectorData[reflNum ][1][0]=x;    h_reflectorData[reflNum ][1][1]= y;h_reflectorData[reflNum ][1][2]=z;//x,y,z position
    h_reflectorData[reflNum ][2][0]=dx;  h_reflectorData[reflNum ][2][1]=dy;    h_reflectorData[reflNum ][2][2]=dz;//x,y,z normal
    h_reflectorData[reflNum ][3][0]=ftToM(0.5); h_reflectorData[reflNum ][3][1]=0.00; h_reflectorData[reflNum ][3][2]=0;//reflector radis ,~,~ 
    h_reflectorData[reflNum ][4][0]=0.000;h_reflectorData[reflNum ][4][1]=0.000;h_reflectorData[reflNum ][4][2]=0.000;//t,u,v jiter  not implimented = speed 
    h_reflectorData[reflNum ][5][0]= 1; h_reflectorData[reflNum ][5][1]= 1.00;  h_reflectorData[reflNum ][5][2]=0.0;//reflectiondamping , no_traping ~
    h_reflectorData[reflNum ][6][0]=0;    h_reflectorData[reflNum ][6][1]=0;    h_reflectorData[reflNum ][6][2]=0;// not implemented yet centrality of rnd distribution speed dt tu ~
     


    if (soundEnabled)reflSoundUpdate( reflNum);

    scene1Start =0;
}

void AlgebraInMotion::scene_data_2_host()
{
    //4 waterFalls
    // waterFallsFrom screenes

    draw_water_sky =0;
    // particalsysparamiteres--------------
    gravity = .005;	
    gravity = .001;	
    max_age = 2000;
    disappear_age =2000;
    colorFreq =128 *max_age/2000.0 ;
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
   std::cout << " init particals in seen2 " << std::endl;
	//cuMemcpyHtoD(d_particleData, h_particleData, size);
	_injObjSwitchFace->setAllChildrenOff();
	_injObjSwitchLine->setAllChildrenOff();
	_refObjSwitchLine->setAllChildrenOn();
    h_reflectorData[0][0][0] =0;//turn off all reflectors
    h_injectorData[0][0][0] =0;// turn off all injectors ~ ~   ~ means dont care



	if (DEBUG_PRINT >0)printf( "in start sean2 \n");
	time_in_sean =0 * TARGET_FR_RATE;
	//::user->home();
	if (DEBUG_PRINT >0)printf("scene0Start \n");

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
//     if (time_in_sean >8)nextSean=1;//short time
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


    h_injectorData[0][0][0] =0;// number of injectors ~ ~   ~ means dont care
//
	float speed = 0.2;
	loadInjFountsFrScr(0 ,.5,0,speed);
 
	

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

//   if(but2==1)_reflectorObjSwitch->setAllChildrenOn();
//   if(but2==0)_reflectorObjSwitch->setAllChildrenOff();
// paddel reflector
   if(but2==1)_refObjSwitchFace->setAllChildrenOn();
   if(but2==0)_refObjSwitchFace->setAllChildrenOff();


    h_reflectorData[reflNum ][0][0]=but2;h_reflectorData[reflNum ][0][1]=1;// type, age ie colormod, ~  0 is off 1 is plane reflector  0 is off 1 is plane reflector
    h_reflectorData[reflNum ][1][0]=x;    h_reflectorData[reflNum ][1][1]= y;h_reflectorData[reflNum ][1][2]=z;//x,y,z position
    h_reflectorData[reflNum ][2][0]=dx;  h_reflectorData[reflNum ][2][1]=dy;    h_reflectorData[reflNum ][2][2]=dz;//x,y,z normal
    h_reflectorData[reflNum ][3][0]=ftToM(0.5); h_reflectorData[reflNum ][3][1]=0.00; h_reflectorData[reflNum ][3][2]=0;//reflector radis ,~,~ 
    h_reflectorData[reflNum ][4][0]=0.000;h_reflectorData[reflNum ][4][1]=0.000;h_reflectorData[reflNum ][4][2]=0.000;//t,u,v jiter  not implimented = speed 
    h_reflectorData[reflNum ][5][0]= 1; h_reflectorData[reflNum ][5][1]= 1.00;  h_reflectorData[reflNum ][5][2]=0.0;//reflectiondamping , no_traping ~
    h_reflectorData[reflNum ][6][0]=0;    h_reflectorData[reflNum ][6][1]=0;    h_reflectorData[reflNum ][6][2]=0;// not implemented yet centrality of rnd distribution speed dt tu ~
	int reflectOn;
	if (time_in_sean > 15)
		{
			 reflectOn =1;
		}
	else
		{
			 reflectOn =0;

		}
   if (soundEnabled)reflSoundUpdate(reflNum);


	float dvx,dvy,dvz; dvx =0;dvy =0;dvx =-0;
	float rotSp = time_in_sean*2*M_PI /2;
		dvy = .1 * sin(rotSp);
		dvx =  .1 * cos(rotSp);

		//(sin(time_in_sean*2*M_PI))
// persone reflector
	  	h_reflectorData[0][0][0] =2;// number of reflectors ~ ~   ~ means dont care
		reflNum = 2;
		h_reflectorData[reflNum ][0][0]=reflectOn;h_reflectorData[reflNum ][0][1]=1;// type, age ie colormod, ~  0 is off 1 is plane reflector  0 is off 1 is plane reflector
		h_reflectorData[reflNum ][1][0]=0;    h_reflectorData[reflNum ][1][1]= 0;h_reflectorData[reflNum ][1][2]=0;//x,y,z position
		h_reflectorData[reflNum ][2][0]=0 + dvx;  h_reflectorData[reflNum ][2][1]=1 + dvy;    h_reflectorData[reflNum ][2][2]= -1 +dvz;//x,y,z normal
		h_reflectorData[reflNum ][3][0]=1; h_reflectorData[reflNum ][3][1]=0.00; h_reflectorData[reflNum ][3][2]=0;//reflector radis ,~,~ 
		h_reflectorData[reflNum ][4][0]=0.000;h_reflectorData[reflNum ][4][1]=0.000;h_reflectorData[reflNum ][4][2]=0.000;//t,u,v jiter  not implimented = speed 
		h_reflectorData[reflNum ][5][0]= 1; h_reflectorData[reflNum ][5][1]= 1.00;  h_reflectorData[reflNum ][5][2]=0.0;//reflectiondamping , no_traping ~
		h_reflectorData[reflNum ][6][0]=0;    h_reflectorData[reflNum ][6][1]=0;    h_reflectorData[reflNum ][6][2]=0;// not implemented yet centrality of rnd distribution speed dt tu ~

//large floor reflector
		  	h_reflectorData[0][0][0] =3;// number of reflectors ~ ~   ~ means dont care
		reflNum = 3;
		h_reflectorData[reflNum ][0][0]=reflectOn;h_reflectorData[reflNum ][0][1]=1;// type, age ie colormod, ~  0 is off 1 is plane reflector  0 is off 1 is plane reflector
		h_reflectorData[reflNum ][1][0]=0;    h_reflectorData[reflNum ][1][1]= 0;h_reflectorData[reflNum ][1][2]=0;//x,y,z position
		h_reflectorData[reflNum ][2][0]=0;  h_reflectorData[reflNum ][2][1]=1;    h_reflectorData[reflNum ][2][2]=0;//x,y,z normal
		h_reflectorData[reflNum ][3][0]=30; h_reflectorData[reflNum ][3][1]=0.00; h_reflectorData[reflNum ][3][2]=0;//reflector radis ,~,~ 
		h_reflectorData[reflNum ][4][0]=0.000;h_reflectorData[reflNum ][4][1]=0.000;h_reflectorData[reflNum ][4][2]=0.000;//t,u,v jiter  not implimented = speed 
		h_reflectorData[reflNum ][5][0]= 1; h_reflectorData[reflNum ][5][1]= 1.00;  h_reflectorData[reflNum ][5][2]=0.0;//reflectiondamping , no_traping ~
		h_reflectorData[reflNum ][6][0]=0;    h_reflectorData[reflNum ][6][1]=0;    h_reflectorData[reflNum ][6][2]=0;// not implemented yet centrality of rnd distribution speed dt tu ~



 
    scene2Start =0;
}

void AlgebraInMotion::scene_data_3_host()
{
    //painting skys

    //draw_water_sky =1;
    // particalsysparamiteres--------------
    gravity = .000005;	
    max_age = 2000;
    disappear_age =2000;
    colorFreq =64 *max_age/2000.0 ;
    alphaControl =0;//turns alph to transparent beloy y=0 1 transparenrt
    static float time_in_sean;
    static float rotRate;

    time_in_sean = time_in_sean + 1.0/TARGET_FR_RATE;

    // reload  rnd < max_age in to pdata

    if (scene3Start == 1)
    {
		size_t size = PDATA_ROW_SIZE * CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT * sizeof (float);
		//pdata_init_age( max_age);
		//pdata_init_velocity(-10000, -10000, -10000);
		pdata_init_rand();
		_refObjSwitchFace->setAllChildrenOff();
		_refObjSwitchLine->setAllChildrenOff();
		_injObjSwitchLine->setAllChildrenOn();
    h_reflectorData[0][0][0] =0;//turn off all reflectors
    h_injectorData[0][0][0] =0;// turn off all injectors ~ ~   ~ means dont care

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
    float maxTime_in_sean = 90;
 
    if (time_in_sean >maxTime_in_sean) nextSean=1;


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

	if(but2==1)	_injObjSwitchFace->setAllChildrenOn();

	if(but2==0)	_injObjSwitchFace->setAllChildrenOff();

    int injNum ;	
    h_injectorData[0][0][0] =2;// number of injectors ~ ~   ~ means dont care
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
    	texture_17_swirls3.setPosition(3 * h_injectorData[injNum][3][0], 0, -3 * h_injectorData[injNum][3][2]);
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

// floor reflector

/*
	int reflNum =1;
	h_reflectorData[0][0][0]=1;

    h_reflectorData[reflNum ][0][0]=1;h_reflectorData[reflNum ][0][1] =1;// type, age ie colormod, ~  0 is off 1 is plane reflector  0 is off 1 is plane reflector
    h_reflectorData[reflNum ][1][0]=0;    h_reflectorData[reflNum ][1][1]= 0;h_reflectorData[reflNum ][1][2]=0;//x,y,z position
    h_reflectorData[reflNum ][2][0]=0;  h_reflectorData[reflNum ][2][1]=1;    h_reflectorData[reflNum ][2][2]=0;//x,y,z normal
    h_reflectorData[reflNum ][3][0]=10; h_reflectorData[reflNum ][3][1]=0.00; h_reflectorData[reflNum ][3][2]=0;//reflector radis ,~,~ 
    h_reflectorData[reflNum ][4][0]=0.000;h_reflectorData[reflNum ][4][1]=0.000;h_reflectorData[reflNum ][4][2]=0.000;//t,u,v jiter  not implimented = speed 
    h_reflectorData[reflNum ][5][0]= 1; h_reflectorData[reflNum ][5][1]= 1.00;  h_reflectorData[reflNum ][5][2]=0.0;//reflectiondamping , no_traping ~
    h_reflectorData[reflNum ][6][0]=0;    h_reflectorData[reflNum ][6][1]=0;    h_reflectorData[reflNum ][6][2]=0;// not implemented yet centrality of rnd distribution speed dt tu ~


 

*/

    h_injectorData[injNum][1][0]=1;h_injectorData[injNum][1][1]=but2*4.0;// type, injection ratio ie streem volume, ~
    h_injectorData[injNum][2][0]=wandPos[0];h_injectorData[injNum][2][1]=wandPos[1];h_injectorData[injNum][2][2]=wandPos[2];//x,y,z position
    h_injectorData[injNum][3][0]=wandVec[0];h_injectorData[injNum][3][1]=wandVec[1];h_injectorData[injNum][3][2]=wandVec[2];//x,y,z velocity direction
    //h_injectorData[injNum][3][0]=0.02 * (sin(anim));h_injectorData[injNum][3][1]=0;h_injectorData[injNum][3][2]=0.02 * (cos(anim));//x,y,z velocity
    //h_injectorData[injNum][3][0]=0.02 *0.0;h_injectorData[injNum][3][1]=0;h_injectorData[injNum][3][2]=0.02 * -1;//x,y,z velocity
    h_injectorData[injNum][4][0]=0.00;h_injectorData[injNum][4][1]=0.00;h_injectorData[injNum][4][2]=.0;//x,y,z size
    h_injectorData[injNum][5][0]=0.010;h_injectorData[injNum][5][1]=0.010;h_injectorData[injNum][5][2]=0.000;//t,u,v jiter v not implimented = speed 
    h_injectorData[injNum][6][0]=0.1;h_injectorData[injNum][6][1]=0.0;h_injectorData[injNum][6][2]=0.0;//speed jiter ~ ~
    h_injectorData[injNum][7][0]=5;h_injectorData[injNum][7][1]=5;h_injectorData[injNum][7][2]=5;//centrality of rnd distribution speed dt tu ~
    //
    if (but2){if (DEBUG_PRINT >0) {printf(" wandPos[0 ,1,2] wandVec[0,1,2] %f %f %f    %f %f %f \n", wandPos[0],wandPos[1],wandPos[2],wandVec[0],wandVec[1],wandVec[2]);}}
    if (soundEnabled)injSoundUpdate(injNum);


    scene3Start =0;
}

void AlgebraInMotion::scene_data_4_host()
{
    //educational

    draw_water_sky =0;
    // particalsysparamiteres--------------
    //std::cerr << "Gravity: " << _gravityRV->getValue() << std::endl;
    gravity = .01;	
    gravity = .003;	
	  max_age = 2000;
    disappear_age =2000;
    colorFreq =64 *max_age/2000.0 ;
    alphaControl =0;//turns alph to transparent beloy y=0
    static float time_in_sean;
    time_in_sean = time_in_sean + 1.0/TARGET_FR_RATE;
	static float in_rot_time = 0;
		    static int subsceanStep;
    			static int suboldseanStep;
    			static int subseanCnangeTo;

    // reload  rnd < max_age in to pdata

    if (scene4Start == 1)
		{
		    subsceanStep =-1;
    			suboldseanStep  = -1;
    			subseanCnangeTo = -1;

		//size_t size = PDATA_ROW_SIZE * CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT * sizeof (float);
		//pdata_init_age( max_age);
		//pdata_init_velocity(-10000, -10000, -10000);
		//pdata_init_rand();
		//cuMemcpyHtoD(d_particleData, h_particleData, size);

		//::user->home();
		_injObjSwitchFace->setAllChildrenOff();
		_injObjSwitchLine->setAllChildrenOff();
		_refObjSwitchLine->setAllChildrenOn();
		h_reflectorData[0][0][0] =0;//turn off all reflectorsstatic oldGravity =0; 
		h_injectorData[0][0][0] =0;// turn off all injectors ~ ~   ~ means dont care
		texture_12.fade(0, 1);// cludge to stop shound from injector on skip to end

		_EdSecSwitchSlid1->setAllChildrenOn();
		in_rot_time = 10;

/*		_EdSecSwitchAxis->setAllChildrenOn();
*/	
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
		
    float maxTime_in_sean = 90;
 
    if (time_in_sean >maxTime_in_sean) nextSean=1;
    float fract_of_max_time_in_sean =time_in_sean/maxTime_in_sean;
    // creates seanCnangeTo varable 
    
    
    subsceanStep = fract_of_max_time_in_sean * 20;//10 steps	float rotTime =30;

    if (subsceanStep > suboldseanStep) { subseanCnangeTo = subsceanStep;suboldseanStep = subsceanStep;} else {subseanCnangeTo =-1;}
    if (subseanCnangeTo !=-1) std::cout << "subseanCnangeTo " << subseanCnangeTo << std::endl; 
    
    
    //std::cout << "time_in_sean , maxTime_in_sean , fract_of_max_time_in_sean , fract_of_max_time_in_sean " << time_in_sean << maxTime_in_sean << fract_of_max_time_in_sean << fract_of_max_time_in_sean << std::endl;
    //printf ("time_in_sean 4 %f\n",time_in_sean);
    if (DEBUG_PRINT >0)printf( "in sean4 \n");
    //printf( "in sean4 \n");
/*	static int oldGravity =0;
	int gravityChanged;
    gravity = _gravityRV->getValue();
	if (gravity != oldGravity) {gravityChanged =1;} else {gravityChanged =0;}

	if (gravityChanged){turnAllEduSlidsOff(); _EdSecSwitchSlid2->setAllChildrenOn();}
	oldGravity = gravity;

	int reflectorOnChanged;
	static int oldReflectorOn =0;
	int reflectorOn = (int) _reflectorCB->getValue();
	if (reflectorOn != oldReflectorOn ) {reflectorOnChanged =1;} else {reflectorOnChanged =0;}
	if (reflectorOnChanged &&  (reflectorOn ==1)){turnAllEduSlidsOff(); _EdSecSwitchSlid3->setAllChildrenOn();std::cout << "turning slide3 on \n";}
	oldReflectorOn = reflectorOn;
*/
    gravity = _gravityRV->getValue();
	std::cout << " sin refl grav " << _rotateInjCB->getValue() << " " <<  _reflectorCB->getValue() << " " << _gravityRV->getValue() << "\n";
	turnAllEduSlidsOff();
	if (_rotateInjCB->getValue())
		{
		if ((_gravityRV->getValue() >= .0001) && ( _reflectorCB->getValue() == 1)){ _EdSecSwitchSlid6->setAllChildrenOn()   ;std::cout << "rot on grav on reflector on \n";}
		if ((_gravityRV->getValue() >= .0001) && (! _reflectorCB->getValue())){_EdSecSwitchSlid5->setAllChildrenOn()    ;std::cout << "rot on grav on reflector off \n";}
		if ((_gravityRV->getValue() < .0001) && (! _reflectorCB->getValue())){_EdSecSwitchSlid4->setAllChildrenOn()    ;std::cout << "rot on grav off reflector off \n";}
		if ((_gravityRV->getValue() < .0001) && ( _reflectorCB->getValue())){_EdSecSwitchSlid8->setAllChildrenOn()    ;std::cout << "rot on grav off reflector on \n";}// need to change new slide

		}
	else
		{
		if ((_gravityRV->getValue() >= .0001) && ( _reflectorCB->getValue() == 1)){_EdSecSwitchSlid3->setAllChildrenOn()    ;std::cout << "rot off grav on reflector on \n";}
		if ((_gravityRV->getValue() >= .0001) && (! _reflectorCB->getValue())){ _EdSecSwitchSlid2->setAllChildrenOn()   ;std::cout << "rot off grav on reflector off \n";}
		if (( _gravityRV->getValue() < .0001) && (! _reflectorCB->getValue())){_EdSecSwitchSlid1->setAllChildrenOn()    ;std::cout << "rot off grav off reflector off \n";}
		if (( _gravityRV->getValue() < .0001) && (_reflectorCB->getValue())){_EdSecSwitchSlid7->setAllChildrenOn()    ;std::cout << "rot off grav off reflector on \n";}// need to make a new slide

		}



    // anim += 0.001;// .0001
    static float rotRate = .05;
    anim = showFrameNo * rotRate;
    rotRate += .000001;
/*
//   		xballPos[0]= 0;xballPos[1]= 0;xballPos[2]= 0;
//		yballPos[0]= 0;yballPos[1]= 0;yballPos[2]= 0;
//		zballPos[0]= 0;zballPos[1]= 0;zballPos[2]=0;

//		if (sceanStep >= 0){xballPos[0]= 300;yballPos[1]= 0;zballPos[2]= 0;}
 //		if (sceanStep > 1){xballPos[0]= 500;}
 //		if (sceanStep > 2){xballPos[0]= 900;}
 	if (subseanCnangeTo == 0) {xballPos[0]= 0;yballPos[1] = 0;zballPos[2] = 0;}

	if (subseanCnangeTo == 0) {_EdSecXballObjSwitch->setAllChildrenOn();_EdSecXboxObjSwitch->setAllChildrenOn();}
	if (subseanCnangeTo == 1) {xballPos[0]= 500;}
	if (subseanCnangeTo == 2) {xballPos[0]= 900;}
	
	if (subseanCnangeTo == 3) {_EdSecYballObjSwitch->setAllChildrenOn();_EdSecXYZballObjSwitch->setAllChildrenOn();}

*/
    //anim += 3.14159/4;
    //tracker data
    //printf("  devid %d \n",devid );
    // printf("pos  P %f %f %f\n", P[0], P[1], P[2]);
    //printf(" direc  V %f %f %f\n", V[0], V[1], V[2]);


    //	 injector data 
    h_reflectorData[0][0][0] =1;// turn off all reflectors

    //	 injector data 
    int injNum ;	
 
    h_injectorData[0][0][0] =1;// number of injectors 

    // injector 1
	float speed =.2;
	speed = _speedRV->getValue();
	float xvos,zvos;
	float rotTime = 15;
	if (_rotateInjCB->getValue()){in_rot_time = in_rot_time + 1.0/TARGET_FR_RATE;}
			xvos = (sin(in_rot_time*2*M_PI/rotTime))*speed;
			zvos = (cos(in_rot_time*2*M_PI/rotTime))*speed;

		
	//std::cout << " speed " << speed << std::endl;
    injNum =1;	
    h_injectorData[injNum][1][0]=2;h_injectorData[injNum][1][1]=1.0;// type, injection ratio ie streem volume, ~
    h_injectorData[injNum][2][0]=0;h_injectorData[injNum][2][1]=2;h_injectorData[injNum][2][2]= -1.5;//x,y,z position
    h_injectorData[injNum][3][0]=xvos;h_injectorData[injNum][3][1]=0.00;h_injectorData[injNum][3][2]=zvos;//x,y,z velocity drection
	h_injectorData[injNum][4][0]=ftToM(0.25);h_injectorData[injNum][4][1]=ftToM(0.25);h_injectorData[injNum][4][2]=ftToM(0.25);//x,y,z size
	h_injectorData[injNum][5][0]=0.000;h_injectorData[injNum][5][1]=0.000;h_injectorData[injNum][5][2]=0.000;//t,u,v jiter v not implimented = speed 
	h_injectorData[injNum][6][0]=0.2000;h_injectorData[injNum][6][1]=0.0;h_injectorData[injNum][6][2]=0.0;//speed jiter ~ ~
	h_injectorData[injNum][7][0]=2;h_injectorData[injNum][7][1]=2;h_injectorData[injNum][7][2]=2;//centrality of rnd distribution speed dt tu ~

 

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
//   if(but2==1)_reflectorObjSwitch->setAllChildrenOn();
//   if(but2==0)_reflectorObjSwitch->setAllChildrenOff();
   if(but2==1)_refObjSwitchFace->setAllChildrenOn();
   if(but2==0)_refObjSwitchFace->setAllChildrenOff();



    h_reflectorData[reflNum ][0][0]=but2;h_reflectorData[reflNum ][0][1] =1;// type, age ie colormod, ~  0 is off 1 is plane reflector  0 is off 1 is plane reflector
    h_reflectorData[reflNum ][1][0]=x;    h_reflectorData[reflNum ][1][1]= y;h_reflectorData[reflNum ][1][2]=z;//x,y,z position
    h_reflectorData[reflNum ][2][0]=dx;  h_reflectorData[reflNum ][2][1]=dy;    h_reflectorData[reflNum ][2][2]=dz;//x,y,z normal
    h_reflectorData[reflNum ][3][0]=ftToM(0.5); h_reflectorData[reflNum ][3][1]=0.00; h_reflectorData[reflNum ][3][2]=0;//reflector radis ,~,~ 
    h_reflectorData[reflNum ][4][0]=0.000;h_reflectorData[reflNum ][4][1]=0.000;h_reflectorData[reflNum ][4][2]=0.000;//t,u,v jiter  not implimented = speed 
    h_reflectorData[reflNum ][5][0]= 1; h_reflectorData[reflNum ][5][1]= 1.00;  h_reflectorData[reflNum ][5][2]=0.0;//reflectiondamping , no_traping ~
    h_reflectorData[reflNum ][6][0]=0;    h_reflectorData[reflNum ][6][1]=0;    h_reflectorData[reflNum ][6][2]=0;// not implemented yet centrality of rnd distribution speed dt tu ~

   

    if (soundEnabled) reflSoundUpdate(reflNum);


//florr reflector
    reflNum = 2;
	int floorReflectOn;
	
	if ((time_in_sean > 5) && _reflectorCB->getValue())	{floorReflectOn =1;}
	else					{floorReflectOn =0;}
    h_reflectorData[0][0][0] =reflNum;// number of reflectors ~ 1.0/TARGET_FR_RATE;~   ~ means dont care

    h_reflectorData[reflNum ][0][0]=floorReflectOn;h_reflectorData[reflNum ][0][1] =1;// type, age ie colormod, ~  0 is off 1 is plane reflector  0 is off 1 is plane reflector
    h_reflectorData[reflNum ][1][0]=0;    h_reflectorData[reflNum ][1][1]= 0;h_reflectorData[reflNum ][1][2]=0;//x,y,z position
    h_reflectorData[reflNum ][2][0]=0;  h_reflectorData[reflNum ][2][1]=1;    h_reflectorData[reflNum ][2][2]=0;//x,y,z normal
    h_reflectorData[reflNum ][3][0]=10; h_reflectorData[reflNum ][3][1]=0.00; h_reflectorData[reflNum ][3][2]=0;//reflector radis ,~,~ 
    h_reflectorData[reflNum ][4][0]=0.000;h_reflectorData[reflNum ][4][1]=0.000;h_reflectorData[reflNum ][4][2]=0.000;//t,u,v jiter  not implimented = speed 
    h_reflectorData[reflNum ][5][0]= 1; h_reflectorData[reflNum ][5][1]= 1.00;  h_reflectorData[reflNum ][5][2]=0.0;//reflectiondamping , no_traping ~
    h_reflectorData[reflNum ][6][0]=0;    h_reflectorData[reflNum ][6][1]=0;    h_reflectorData[reflNum ][6][2]=0;// not implemented yet centrality of rnd distribution speed dt tu ~


    scene4Start =0;
}

void AlgebraInMotion::scene_data_0_context(int contextid) const
{
    if(scene0ContextStart)
    {
	size_t size = PDATA_ROW_SIZE * CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT * sizeof (float);
	cuMemcpyHtoD(d_particleDataMap[contextid], h_particleData, size);
    }

}

void AlgebraInMotion::scene_data_1_context(int contextid) const
{
    if(scene1ContextStart)
    {
    }
}

void AlgebraInMotion::scene_data_2_context(int contextid) const
{
    if(scene2ContextStart)
    {
    }
}

void AlgebraInMotion::scene_data_3_context(int contextid) const
{
    if(scene3ContextStart)
    {
	size_t size = PDATA_ROW_SIZE * CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT * sizeof (float);
	cuMemcpyHtoD(d_particleDataMap[contextid], h_particleData, size);

    }
}

void AlgebraInMotion::scene_data_4_context(int contextid) const
{
    if(scene4ContextStart)
    {
    }
}

void AlgebraInMotion::scene_data_0_kill_audio()
{
	h_reflectorData[0][0][0] = 0;
	h_injectorData[0][0][0] = 0;

	if (soundEnabled)
	{
		dan_ambiance_2.fade(0, 1.5);
		dan_10122606_sound_spray.setGain(0);
	}
}

void AlgebraInMotion::scene_data_1_kill_audio()
{
	h_reflectorData[0][0][0] = 0;
	h_injectorData[0][0][0] = 0;
	std::cout << "scene_data_1_kill_audio() " << std::endl;
	if (soundEnabled)
	{
		dan_5min_ostinato.fade(0, 1.50);
		dan_10122606_sound_spray.setGain(0);
	}
}


void AlgebraInMotion::scene_data_2_kill_audio()
{
	h_reflectorData[0][0][0] = 0;
	h_injectorData[0][0][0] = 0;
	std::cout << "scene_data_2_kill_audio() " << std::endl;

	if (soundEnabled)
	{
		harmonicAlgorithm.fade(0, 1.50);
		dan_10122606_sound_spray.setGain(0);
	}
}

void AlgebraInMotion::scene_data_3_kill_audio()
{
	h_reflectorData[0][0][0] = 0;
	h_injectorData[0][0][0] = 0;
	std::cout << "scene_data_3_kill_audio() " << std::endl;
	if (soundEnabled)
	{
		rain_at_sea.fade(0, 1.50);
		texture_17_swirls3.fade(0, 1.50);
		dan_10122606_sound_spray.setGain(0);

		dan_10120600_rezS3_rez2.setGain(0);
	}
}

void AlgebraInMotion::scene_data_4_kill_audio()
{
	h_reflectorData[0][0][0] = 0;
	h_injectorData[0][0][0] = 0;

	turnAllEduSlidsOff();
	std::cout << "scene_data_4_kill_audio() " << std::endl;

	if (soundEnabled)
	{
		dan_rain_at_sea_loop.fade(0, 1.50);
		dan_10122606_sound_spray.setGain(0);
	}
}


float AlgebraInMotion::reflSoundUpdate(int reflNum)
	{
		static float OldReflectOn =0;
		float ReflectOn = h_reflectorData[reflNum ][0][0];

		float x,y,z;
	    	if (ReflectOn >0)
			{
				float newGain = _refl_hits[reflNum]/500.0;
				//if (newGain >0) printf ("reflHits %f\n",newGain);;
				if (newGain > 0.5) newGain = 0.5;

				dan_10122606_sound_spray.setGain(newGain);
				x = h_reflectorData[reflNum ][1][0];   y= h_reflectorData[reflNum ][1][1];z = h_reflectorData[reflNum ][1][2];
				if (ENABLE_SOUND_POS_UPDATES) dan_10122606_sound_spray.setPosition(x*4, -z*0, y*0);//convert from z bzck to z up

			}

    		if ((ReflectOn == 0) && (OldReflectOn > 0))
			{
		
 		   		dan_10122606_sound_spray.setGain(0);
    		}
	OldReflectOn = ReflectOn;
	return 	_refl_hits[reflNum];

    }
	


//    h_injectorData[injNum][1][0]=2;h_injectorData[injNum][1][1]=1.0;// type, injection ratio ie streem volume, ~
//    h_injectorData[injNum][2][0]=0;h_injectorData[injNum][2][1]=ftToM(10);h_injectorData[injNum][2][2]=ftToM(-2);//x,y,z position


void	AlgebraInMotion::injSoundUpdate(int injNum)

    {
		static float OldInjOn =0;
		float injOn = h_injectorData[injNum][1][1];
		float x,y,z;
			x = h_injectorData[injNum][2][0];
			y = h_injectorData[injNum][2][1];

			z = h_injectorData[injNum][2][2];
			y =0;z =0;
			x = x*4;
 
		static int roundRobenSound = 0;
		int roundRobenMod =5;

    	if (OldInjOn == 0 && (injOn > 0))
    	{
			roundRobenSound = roundRobenSound +1;
			roundRobenSound = roundRobenSound%roundRobenMod;
			if (ENABLE_SOUND_POS_UPDATES)
				{
					if (roundRobenSound ==0 ){short_sound_01a.play();short_sound_01a.setPosition(x, -z, y);}
					if (roundRobenSound ==1 ){short_sound_01a1.play();short_sound_01a1.setPosition(x, -z, y);}
					if (roundRobenSound ==2 ){short_sound_01a2.play();short_sound_01a2.setPosition(x, -z, y);}
					if (roundRobenSound ==3 ){short_sound_01a3.play();short_sound_01a3.setPosition(x, -z, y);}
					if (roundRobenSound ==4 ){short_sound_01a4.play();short_sound_01a4.setPosition(x, -z, y);}
				}

			if (! ENABLE_SOUND_POS_UPDATES)
				{
					if (roundRobenSound ==0 ){short_sound_01a.play();}
					if (roundRobenSound ==1 ){short_sound_01a1.play();}
					if (roundRobenSound ==2 ){short_sound_01a2.play();}
					if (roundRobenSound ==3 ){short_sound_01a3.play();}
					if (roundRobenSound ==4 ){short_sound_01a4.play();}
				}


    		texture_12.fade(4, 1);

    	}
		
    	if (OldInjOn > 0 && injOn == 0)
    	{
    		texture_12.fade(0, 1);
    	}
		if 	((injOn > 0) && (ENABLE_SOUND_POS_UPDATES))

		{
			x = h_injectorData[injNum][2][0];
			y = h_injectorData[injNum][2][1];
			z = h_injectorData[injNum][2][2];
    		texture_12.setPosition(x, -z, y);
    		short_sound_01a.setPosition(x, -z, y);

		}
	OldInjOn = injOn;
	}

osg::Geometry* AlgebraInMotion::createQuad()
	{
	osg::ref_ptr<osg::Texture2D> texture = new osg::Texture2D;
	osg::ref_ptr<osg::Image> image = osgDB::readImageFile("/data-nfs/ParticleDreams/images/8lineblonwhitecopy.png");
    if(!image)
    {
       std::cerr << "Failed to read quad texture." << std::endl;
    }
	//texture->setImage(image.get());
	osg::ref_ptr<osg::Geometry> quad =
			osg::createTexturedQuadGeometry(
											osg::Vec3(-0.5f,0.0f,-0.5f),
											osg::Vec3(1.0f,0.0f,0.0f),
											osg::Vec3(0.0f,0.0f,1.0f) );
	osg::StateSet* ss = quad->getOrCreateStateSet() ;
	//ss->setTextureAttributeAndModes(0 , texture.get());
    ss->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
	return quad.get();
	}

void AlgebraInMotion::turnAllEduSlidsOff()
	{
       _EdSecSwitchSlid1->setAllChildrenOff();
       _EdSecSwitchSlid2->setAllChildrenOff();
       _EdSecSwitchSlid3->setAllChildrenOff();
       _EdSecSwitchSlid4->setAllChildrenOff();
       _EdSecSwitchSlid5->setAllChildrenOff();
       _EdSecSwitchSlid6->setAllChildrenOff();
       _EdSecSwitchSlid7->setAllChildrenOff();
       _EdSecSwitchSlid8->setAllChildrenOff();
	}

void	AlgebraInMotion::initGeoEdSection()
	{

osg::Node* tidleSlide1 = NULL;
	tidleSlide1 = osgDB::readNodeFile(_dataDir + "/models/frameS1.obj");//xyz
    if(!tidleSlide1){std::cerr << "Error reading /models/frameS1.obj" << std::endl;}
osg::Node* tidleSlide2 = NULL;
	tidleSlide2 = osgDB::readNodeFile(_dataDir + "/models/frameS2.obj");//;xyza
    if(!tidleSlide2){std::cerr << "Error reading /models/frameS2.obj" << std::endl;}
osg::Node* tidleSlide3 = NULL;
	tidleSlide3 = osgDB::readNodeFile(_dataDir + "/models/frameS3.obj");//xyzar
    if(!tidleSlide3){std::cerr << "Error reading /models/frameS3.obj" << std::endl;}
osg::Node* tidleSlide4 = NULL;
	tidleSlide4 = osgDB::readNodeFile(_dataDir + "/models/frameS4.obj");//sinxyz
    if(!tidleSlide4){std::cerr << "Error reading /models/frameS4.obj" << std::endl;}
osg::Node* tidleSlide5 = NULL;
	tidleSlide5 = osgDB::readNodeFile(_dataDir + "/models/frameS5.obj");//sinxyza
    if(!tidleSlide5){std::cerr << "Error reading /models/frameS5.obj" << std::endl;}
osg::Node* tidleSlide6 = NULL;
	tidleSlide6 = osgDB::readNodeFile(_dataDir + "/models/frameS6.obj");//sin xyzar
    if(!tidleSlide6){std::cerr << "Error reading /models/frameS6.obj" << std::endl;}

osg::Node* tidleSlide7 = NULL;
	tidleSlide7 = osgDB::readNodeFile(_dataDir + "/models/frameS7.obj");//sin xyzr
    if(!tidleSlide7){std::cerr << "Error reading /models/frameS7.obj" << std::endl;}
osg::Node* tidleSlide8 = NULL;
	tidleSlide8 = osgDB::readNodeFile(_dataDir + "/models/frameS8.obj");// xyz
    if(!tidleSlide8){std::cerr << "Error reading /models/frameS8.obj" << std::endl;}

std::cout << " loading slides " << "\n" ;
//creat switchnode
	_EdSecSwitchSlid1 = new osg::Switch;
	_EdSecSwitchSlid2 = new osg::Switch;
	_EdSecSwitchSlid3 = new osg::Switch;
	_EdSecSwitchSlid4 = new osg::Switch;
	_EdSecSwitchSlid5 = new osg::Switch;
	_EdSecSwitchSlid6 = new osg::Switch;
	_EdSecSwitchSlid7 = new osg::Switch;
	_EdSecSwitchSlid8 = new osg::Switch;
	AlgebraInMotion::turnAllEduSlidsOff();

// creat fixes xform on slides
        osg::Matrix ms;
        osg::MatrixTransform * mtSlide1 = new osg::MatrixTransform();
        osg::MatrixTransform * mtSlide2 = new osg::MatrixTransform();
        osg::MatrixTransform * mtSlide3 = new osg::MatrixTransform();
        osg::MatrixTransform * mtSlide4 = new osg::MatrixTransform();
        osg::MatrixTransform * mtSlide5 = new osg::MatrixTransform();
        osg::MatrixTransform * mtSlide6 = new osg::MatrixTransform();
        osg::MatrixTransform * mtSlide7 = new osg::MatrixTransform();
        osg::MatrixTransform * mtSlide8 = new osg::MatrixTransform();
//matrices for interdediat computations
        osg::Matrix mss;
        osg::Matrix msr;

        osg::Matrix msrh;
        osg::Matrix mst;

        osg::Matrix mresult;


 
//    height= 805 ;h=  26.0; width= 1432 ;  p= 0.0    ;originX= -802  ; originY= 1657  ;  r= -90.0 ;   name= 0 ;  originZ= 0   ;   screen= 0;
		int i =1;
		std::cout << " i, _PhScAr[i].originX  ,_PhScAr[i].originY _PhScAr[i].h "<< i<< " " << _PhScAr[i].originX  << " " << _PhScAr[i].originY << " " << _PhScAr[i].h << "\n";
 
      mst.makeTranslate(osg::Vec3(_PhScAr[i].originX,_PhScAr[i].originY +150,450));
     // mst.makeTranslate(osg::Vec3(-802,1657,450));
		ms.makeScale(osg::Vec3(40.0,25,1.0));
       msr.makeRotate(osg::DegreesToRadians(90.0), osg::Vec3(1,0,0));
       msrh.makeRotate(osg::DegreesToRadians(25.0), osg::Vec3(0,0,1));
      msrh.makeRotate(osg::DegreesToRadians(_PhScAr[i].h), osg::Vec3(0,0,1));


 
	mresult.set ( ms*msr*msrh * mst);
	mtSlide1->setMatrix(mresult);
	mtSlide2->setMatrix(mresult);
	mtSlide3->setMatrix(mresult);
	mtSlide4->setMatrix(mresult);
	mtSlide5->setMatrix(mresult);
	mtSlide6->setMatrix(mresult);
	mtSlide7->setMatrix(mresult);
	mtSlide8->setMatrix(mresult);


//atatch tidleSlide to matrix transform
		mtSlide1->addChild(tidleSlide1);
		mtSlide2->addChild(tidleSlide2);
		mtSlide3->addChild(tidleSlide3);
		mtSlide4->addChild(tidleSlide4);
		mtSlide5->addChild(tidleSlide5);
		mtSlide6->addChild(tidleSlide6);
		mtSlide7->addChild(tidleSlide7);
		mtSlide8->addChild(tidleSlide8);


// atatch scaled model to switch 
        _EdSecSwitchSlid1->addChild(mtSlide1);
        _EdSecSwitchSlid2->addChild(mtSlide2);
        _EdSecSwitchSlid3->addChild(mtSlide3);
        _EdSecSwitchSlid4->addChild(mtSlide4);
        _EdSecSwitchSlid5->addChild(mtSlide5);
        _EdSecSwitchSlid6->addChild(mtSlide6);
        _EdSecSwitchSlid7->addChild(mtSlide7);
        _EdSecSwitchSlid8->addChild(mtSlide8);

		turnAllEduSlidsOff(); 
       
// atatch switch to scene
 

// addTidle screene.
		SceneObject * so = new SceneObject("EdSlide1",false,false,false,false,false);
      so->addChild(_EdSecSwitchSlid1);
      so->addChild(_EdSecSwitchSlid2);
      so->addChild(_EdSecSwitchSlid3);
      so->addChild(_EdSecSwitchSlid4);
      so->addChild(_EdSecSwitchSlid5);
      so->addChild(_EdSecSwitchSlid6);
      so->addChild(_EdSecSwitchSlid7);
      so->addChild(_EdSecSwitchSlid8);

        PluginHelper::registerSceneObject(so,"AlgebraInMotion");

        so->setPosition(osg::Vec3(0,0,0));

        so->setScale(1);
        so->attachToScene();
		so->setNavigationOn(true);

	}



