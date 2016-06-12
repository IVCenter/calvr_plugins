#include "ParticleDreams.h"
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

float lerp(float in,float beginIN,float endIn,float beginOut,float endOut)
{
    float pers;
    float out;
    pers =   (in - beginIN)/( endIn - beginIN);
    out = beginOut + pers * (endOut-beginOut);
    return out;
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
    _pointerHeading = 0.0;
    _pointerPitch = 0.0;
    _pointerRoll = 0.0;
    _callbackAdded = false;
    _callbackActive = false;
    _skyTime = 80000.0;
}

ParticleDreams::~ParticleDreams()
{
#ifdef OAS_SOUND

	oasclient::ClientInterface::shutdown();
#endif
#ifdef OMICRON_SOUND

	std::cout << " in ~ParticleDreams() \n";
	cleanupSound();
#endif
}

bool ParticleDreams::init()
{
    _myMenu = new SubMenu("Partical Dreams");

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

    _dataDir = ConfigManager::getEntry("value","Plugin.ParticleDreams.DataDir","") + "/";

    PluginHelper::addRootMenuItem(_myMenu);

    hand_id = ConfigManager::getInt("value","Plugin.ParticleDreams.HandID",0);
    head_id = ConfigManager::getInt("value","Plugin.ParticleDreams.HeadID",1);

	_TargetSystem = ConfigManager::getEntry("value","Plugin.ParticleDreams.TargetSystem","TourCaveCalit2");
	std::cout << "targetSystem " << _TargetSystem << "\n";
	// TourCaveCalit2 TourCaveSaudi NexCaveCalit2 StarCave Cave2 Wave

//	ContextChange block comented out id fr 1 screen version
	
	_DisplaySystem = ConfigManager::getEntry("value","Plugin.ParticleDreams.DisplaySystem","Simulator");
	std::cout << "DisplaySystem " << _DisplaySystem << "\n";
//	_DisplaySystem = ConfigManager::getEntry("value","Plugin.ParticleDreams.DisplaySystem","TourCaveCalit2");
	// TourCaveCalit2 TourCaveSaudi NexCaveCalit2 StarCave Cave2 Wave Simulator

	tidleSlideTime = ConfigManager::getFloat("value","Plugin.ParticleDreams.TidleSlideTime",10.0);
	n_waterfalls_timeInScene = ConfigManager::getFloat("value","Plugin.ParticleDreams.N_waterfalls_timeInScene",30.0);
	sprial_fountens_timeInScene = ConfigManager::getFloat("value","Plugin.ParticleDreams.Sprial_fountens_timeInScene",30.0);
	paint_on_walls_timeInScene = ConfigManager::getFloat("value","Plugin.ParticleDreams.Paint_on_walls_timeInScene",30.0);
	painting_skys_timeInScene = ConfigManager::getFloat("value","Plugin.ParticleDreams.Painting_skys_timeInScene",30.0);
	rain_timeInScene = ConfigManager::getInt("value","Plugin.ParticleDreams.Rain_timeInScene",30.0);
	std::cout <<"tidleSlideTime " << tidleSlideTime << "\n";
	std::cout <<"n_waterfalls_timeInScene " << n_waterfalls_timeInScene << "\n";
	std::cout <<"sprial_fountens_timeInScene " << sprial_fountens_timeInScene << "\n";
	std::cout <<"paint_on_walls_timeInScene  " << paint_on_walls_timeInScene  << "\n";
	std::cout <<"painting_skys_timeInScene " << painting_skys_timeInScene  << "\n";
	std::cout <<"rain_timeInScene " << rain_timeInScene << "\n";

	
	targetFrameRate = ConfigManager::getInt("value","Plugin.ParticleDreams.TargetFrameRate",30);
	std::cout <<"targetFrameRate " << targetFrameRate << "\n";
   return true;
}

void ParticleDreams::menuCallback(MenuItem * item)
{
    if(item == _enable)
    {
	if(_enable->getValue())
	{

	    CVRViewer::instance()->getStatsHandler()->addStatTimeBar(CVRStatsHandler::CAMERA_STAT,"AIMCuda Time:","PD Cuda duration","PD Cuda start","PD Cuda end",osg::Vec3(0,1,0),"PD stats");
	    //CVRViewer::instance()->getStatsHandler()->addStatTimeBar(CVRStatsHandler::CAMERA_STAT,"PDCuda Copy:","PD Cuda Copy duration","PD Cuda Copy start","PD Cuda Copy end",osg::Vec3(0,0,1),"PD stats");
// change back
	    SceneManager::instance()->setHidePointer(true);

	    initPart();
	    initGeometry();
	    initSound();
	    initWater();
	    setWater(true);
	}
	else
	{
	    cleanupSound();
	    cleanupGeometry();
	    cleanupPart();
	    
	    SceneManager::instance()->setHidePointer(false);

	    CVRViewer::instance()->getStatsHandler()->removeStatTimeBar("PD Cuda duration");
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

void ParticleDreams::preFrame()
{

    if(_enable->getValue())
    {
	// set water shader time
	double times = PluginHelper::getProgramDuration();
	if(_waterTimeUni)
	{
	    _waterTimeUni->set((float)times);
	}

	// set water shader light position
	if(_waterLightUni)
	{
	    // TODO: get this from somewhere and advance time
	    // 11am
	    osg::Matrix spin, tilt;
	    tilt.makeRotate(120.0*M_PI/180.0,osg::Vec3(1.0,0,0));
	    spin.makeRotate((360.0*_skyTime/(24.0*60.0*60.0))*(M_PI/180.0),osg::Vec3(0,1.0,0));
	    
	    osg::Vec3 lightPoint(0,0,8192.0);
	    lightPoint = lightPoint * spin * tilt;// * _particleObject->getWorldToObjectMatrix();
	    _waterLightUni->set(lightPoint);
	}

	//do driver thread part of step

	if(SceneManager::instance()->getMenuOpenObject() == _particleObject)
	{
	    SceneManager::instance()->setHidePointer(true);
	}
	else
	{//change back
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
	updateHead();


	sceneChange =0;
	
	
	
	//STATE GENERATOR
	// ADD and Change SCENES HERE
	// check for above comment elsewhere
	name_paint_on_walls = 0;
	name_sprial_fountens =1;
	name_N_waterfalls =2;
	name_painting_skys =3;
	name_rain =4;
	if (sceneOrder ==-1) // program start
		{
			nextSean =1;
		    paint_on_walls_start =0;
			sprial_fountens_start =0;
			N_waterfalls_start=0;
			painting_skys_start =0;
			rain_start =0;
			paint_on_wallsContextStart =0;
			sprial_fountensContextStart =0;
			N_waterfallsContextStart =0;
			painting_skysContextStart =0;
			rainContextStart =0;	
		}
	
	// ADD and Change SCENES HERE

	int numberOfScenes = 6;
	//if ((_TargetSystem.compare("TourCaveCalit2") == 0) ||(_TargetSystem.compare("Cave2") == 0)||(_TargetSystem.compare("Wave") == 0) )numberOfScenes = 4;
	//if (_TargetSystem.compare("Cave2") == 0)numberOfScenes = 4;
	if (skipTonextScene ==1)
	{	std::cout << "skipTonextScene ==1 " << std::endl;
	    sceneOrder =( sceneOrder+1)%(numberOfScenes -1);sceneChange=1;
	    skipTonextScene =0;

	}
if (nextSean ==1) { sceneOrder =( sceneOrder+1)%(numberOfScenes -1);sceneChange=1;nextSean =0;}
   
		if (sceneChange)
	{
		//previous state
	    if 		(sceneName == name_N_waterfalls) 	N_waterfalls__clean_up();
	    else if (sceneName == name_paint_on_walls)	paint_on_walls__clean_up();//not used??
	    else if (sceneName == name_painting_skys)	painting_skys__clean_up();
	    else if (sceneName == name_rain)			rain__clean_up();
	    else if (sceneName == name_sprial_fountens)	sprial_fountens__clean_up();
	}   
	 // neststate 
 
	if (sceneOrder ==0)sceneName =name_N_waterfalls;
	if (sceneOrder ==1)sceneName =name_sprial_fountens;
	if (sceneOrder ==2)sceneName =name_paint_on_walls;
	if (sceneOrder ==3)sceneName =name_painting_skys;
	if (sceneOrder ==4)sceneName =name_rain;
/*
	if ((_TargetSystem.compare("TourCaveCalit2") == 0) || (_TargetSystem.compare("Cave2") == 0)|| (_TargetSystem.compare("Wave") == 0))
		{
		//std::cout << "_TargetSystem.compare(TourCaveCalit2) == 0\n ";   mtSlide1->setMatrix(SlideMatrix);
    mtSlide2->setMatrix(SlideMatrix);
    mtSlide3->setMatrix(SlideMatrix);
	resetSlidePosition();
 
			if (sceneOrder ==0)sceneName =name_N_waterfalls;
			if (sceneOrder ==1)sceneName =name_paint_on_walls;
			if (sceneOrder ==2)sceneName =name_painting_skys;
			if (sceneOrder ==3)sceneName =name_rain;

		}

*/

	if (sceneName ==name_paint_on_walls)
	{//paint_on_walls
		if (paint_on_walls_timeInScene == 0)
		{
			nextSean =1;
		}
		else
		{
	    if (sceneChange ==1){paint_on_walls_start =1;sceneChange =0;}
	    paint_on_walls_host();
	    }
	}
	if (sceneName ==name_sprial_fountens)  
	{//sprial_fountens
	if ( sprial_fountens_timeInScene == 0)
		{
			nextSean =1;
		}
		else
		{
	    if (sceneChange ==1){sprial_fountens_start =1;sceneChange =0;}
	    sprial_fountens_host();
	    }
	}
	if (sceneName ==name_N_waterfalls)  
	{//N_waterfalls
	if ( n_waterfalls_timeInScene == 0)
		{
			nextSean =1;
		}
		else
		{
	    if (sceneChange ==1){N_waterfalls_start=1;sceneChange =0;}
	    N_waterfalls_host();
		}
	}
	if (sceneName ==name_painting_skys) 
	{//painting_skys
	if (painting_skys_timeInScene == 0) 
		{
			nextSean =1;
		}
		else
		{
	    if (sceneChange ==1){painting_skys_start =1;sceneChange =0;}
	    painting_skys_host();
	    }
	}

	if (sceneName ==name_rain)
	{//rain
	if (rain_timeInScene == 0)
		{
			nextSean =1;
		}
		else
		{
	    if (sceneChange ==1){rain_start =1;sceneChange =0;}
	    rain_host();
	    }
	}




	for ( int n =1;n < h_injectorData[0][0][0] +1;n++)
	{
	    // kludge to handel gimbel lock for velociys straight up			
	    if (h_injectorData[n][3][0] ==0 && h_injectorData[n][3][2] == 0){ h_injectorData[n][3][0] += .0001;}
	}

	but4old = but4;
	but3old = but3;
	hand_inject_or_reflectold = hand_inject_or_reflect;
	but1old = but1;
	triggerold = trigger;
    }
//		for (int i =0;i<128;i++){old_refl_hits[i] = h_debugData[i];}

}

bool ParticleDreams::processEvent(InteractionEvent * event)
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

//StarCave
// triger ==0
//ll 4
//lc 3
//rc  2
//rr is 1 traped above

//Wave
//triger = 0
//ll 4
//lc 3
//rc  2  becomes navigation
//rr is 1 traped above


  if ((_DisplaySystem.compare("StarCave") == 0) ||(_DisplaySystem.compare("Wave") == 0) )// esentualy ART tracker
  {
      if(tie)
      {
	      if(tie->getHand() == hand_id)
	      {
	         
	              if((tie->getInteraction() == BUTTON_DOWN || tie->getInteraction() == BUTTON_DOUBLE_CLICK) && tie->getButton() <= 4)
	              {// std::cout << " buttonPtresses " << (tie->getButton()) << std::endl;
		              if(tie->getButton() == 0)
		              {
			              trigger = 1;
			              hand_inject_or_reflect =1;
			              //std::cout << " trigger hand_inject_or_reflect " << trigger << " " << hand_inject_or_reflect << std::endl;
			              //if (hand_inject_or_reflect ==1){ skipTonextScene =1; ; return true;}
			              return true;
		              }
		              else if(tie->getButton() == 1)
		              {
		                  but1 = 1;
		                  //return true;
		              }
		              else if(tie->getButton() == 2)
		              {
		               // return true;
		              }
		              else if(tie->getButton() == 3)
		              {
		                if (but4) skipTonextScene =1;
			              //return true;
	
		                  but3 = 1;
		              }
		              else if(tie->getButton() == 4)
		              {
		                  but4 = 1;
		                //return true;  
		              }
	              }
	              else if(tie->getInteraction() == BUTTON_UP && tie->getButton() <= 4)
	              {
		              if(tie->getButton() == 0)
		              {
		                  trigger = 0;
		                  hand_inject_or_reflect = 0;
		              }
		              else if(tie->getButton() == 1)
		              {
		                  but1 = 0;
		              }
		              else if(tie->getButton() == 2)
		              {
		                
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
  }
  else // trackers whos buttons that act like mice
  {
      if(tie)
      {
	      if(tie->getHand() == hand_id)
	      {
	         
	              if((tie->getInteraction() == BUTTON_DOWN || tie->getInteraction() == BUTTON_DOUBLE_CLICK) && tie->getButton() <= 4)
	              {// std::cout << " buttonPtresses " << (tie->getButton()) << std::endl;
		              if(tie->getButton() == 0)
		              {
			              trigger = 1;
			              //std::cout << " trigger hand_inject_or_reflect " << trigger << " " << hand_inject_or_reflect << std::endl;
			              if (hand_inject_or_reflect ==1){ skipTonextScene =1; ; return true;}
		              }
		              else if(tie->getButton() == 1)
		              {
		                  but1 = 1;
		              }
		              else if(tie->getButton() == 2)
		              {
		                  hand_inject_or_reflect = 1;
			              return true;
			              //captures hand_inject_or_reflect to prevent defalt navagation on hand_inject_or_reflect
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
		                  hand_inject_or_reflect = 0;
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
   }
    ValuatorInteractionEvent * vie = event->asValuatorEvent();
    if(vie && vie->getHand() == hand_id)
    {
        if(vie->getValuator() == 0)
        {    // Ask Andrew or Bob kludge
           // _pointerHeading += vie->getValue() * 0.5;
           // _pointerHeading = std::max(_pointerHeading,-90.0f);
           // _pointerHeading = std::min(_pointerHeading,90.0f);
        }
    }

    return false;
}

void ParticleDreams::perContextCallback(int contextid,PerContextCallback::PCCType type) const
{
    if(CVRViewer::instance()->done() || !_callbackActive)
    {
	_callbackLock.lock();

	if(_callbackInit[contextid])
	{
	    cuMemFree(d_particleDataMap[contextid]);
	    cuMemFree(d_debugDataMap[contextid]);

#ifdef SCR2_PER_CARD
	    //cuCtxSetCurrent(NULL);
#endif

	    _callbackInit[contextid] = false;
	}

	_callbackLock.unlock();
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
	    if(!_cudaContextSet[contextid])
	    {
		CUdevice device;
		cuDeviceGet(&device,cudaDevice);
		CUcontext cudaContext;

		cuGLCtxCreate(&cudaContext, 0, device);
		cuGLInit();
		cuCtxSetCurrent(cudaContext);
	    }
           
            //cuCtxSetCurrent(_cudaContextMap[contextid]);
	}
        else
        {
	    if(!_cudaContextSet[contextid])
	    {
		cudaGLSetGLDevice(cudaDevice);
		cudaSetDevice(cudaDevice);
	    }
        } 
	//std::cerr << "CudaDevice: " << cudaDevice << std::endl;


	if(!_cudaContextSet[contextid])
	{
	    printCudaErr();
	    osg::VertexBufferObject * vbo = _particleGeo->getOrCreateVertexBufferObject();
	    vbo->setUsage(GL_DYNAMIC_DRAW);
	    osg::GLBufferObject * glbo = vbo->getOrCreateGLBufferObject(contextid);
	    //std::cerr << "Context: " << contextid << " VBO id: " << glbo->getGLObjectID() << " size: " << vbo->computeRequiredBufferSize() << std::endl;
	    checkRegBufferObj(glbo->getGLObjectID());
	    _cudaContextSet[contextid] = true;
	}
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



// ADD NEW SCENES HERE

    if (sceneName ==name_paint_on_walls)
    {//paint on walls
	paint_on_walls_context(contextid);
    }
    if (sceneName ==name_sprial_fountens)
    {//sprial fountens
	sprial_fountens_context(contextid);
    }
    if (sceneName ==name_N_waterfalls)
    {//4 waterfalls
	N_waterfalls_context(contextid);
    }
    if (sceneName ==name_painting_skys)
    {//painting skys
	painting_skys_context(contextid);
    }

    if (sceneName ==name_rain)
    {
	rain_context(contextid);
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
    //draw_water_sky = 1;
    //TODO: get from config file
    state =0;
    trigger =0;
    but4 =0;
    but3 =0;
    hand_inject_or_reflect =0;
    but1 =0;
   skipTonextScene=0;skipTOnextSceneOld=0;

    // init_scenes
    sceneOrder = -1;// tels state mechene program start
    nextSean =0;
    
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

    if(!_callbackAdded)
    {
	CVRViewer::instance()->addPerContextPostFinishCallback(this);
	_callbackAdded = true;
    }

    _callbackActive = true;
}

void ParticleDreams::initGeometry()
{
// init partical system
    _particleObject = new PDObject("Algebra In Motion",false,false,false,true,false);

    _particleObject->addMenuItem(_gravityRV);
    _particleObject->addMenuItem(_rotateInjCB);
	_particleObject->addMenuItem(_speedRV);
	_particleObject->addMenuItem(_reflectorCB);
 

    _particleGeode = new osg::Geode();

    if(!_particleGeo)
    {
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
    }
    else
    {
	for(int i = 0; i < _positionArray->size(); ++i)
	{
	    _positionArray->at(i) = osg::Vec3(0,0,0);
	}
	_positionArray->dirty();
	for(int i = 0; i < _colorArray->size(); ++i)
	{
	    _colorArray->at(i) = osg::Vec4(0,0,0,0);
	}
	_colorArray->dirty();
    }

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
   // osg::ref_ptr<osg::Image> image = osgDB::readImageFile(_dataDir + "images/sprite50_50.png");
    // osg::ref_ptr<osg::Image> image = osgDB::readImageFile(_dataDir + "images/sprite50_50.png");
    osg::ref_ptr<osg::Image> image = osgDB::readImageFile(_dataDir + "images/sprite100_100.png");
 

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
	std::cerr << "Unable to read sprite texture: " << _dataDir + "images/sprite50_50.png" << std::endl;
    }
//  PluginHelper::getScene()->addChild(...)
    _particleGeode->addDrawable(_particleGeo);
    _particleObject->addChild(_particleGeode);
    PluginHelper::registerSceneObject(_particleObject);
    _particleObject->attachToScene();
    _particleObject->resetPosition();
// init hand object injector reflector

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
	//std::string str2 = ("TourCaveSaudi"); 
	std::cout << "_TargetSystem " << _TargetSystem << "\n";
	if (_TargetSystem.compare("TourCaveSaudi" ) == 0)
		{
			std::cout << "TourCaveSaudi" << std::endl;
			int i =loadPhysicalScreensArrayTourCaveSaudi();
			std::cout << "loadPhysicalScreens " << i <<" " << _PhScAr[i-1].index << " " << std::endl;
		}
	
	else if (_TargetSystem.compare("TourCaveCalit2") == 0)

		{
			std::cout << "TourCaveCalit2" << std::endl;
			int i =loadPhysicalScreensArrayTourCaveCalit2_5lowerScr();
			std::cout << "loadPhysicalScreens " << i <<" " << _PhScAr[i-1].index << " " << std::endl;


		}
	
	else if (_TargetSystem.compare("Cave2") == 0)

		{
			std::cout << "Cave2  loaded" << std::endl;
			int i =loadOneHalfPhysicalScreensArrayCave2();
			std::cout << "loadPhysicalScreens " << i <<" " << _PhScAr[i-1].index << " " << std::endl;


		}

	else if (_TargetSystem.compare("StarCave") == 0)

		{
			std::cout << "StarCave  loaded" << std::endl;
			//int i =loadPhysicalScreensArrayStarCave();
			int i =loadPhysicalScreensArrayStarCaveCenterRow();
			std::cout << "loadPhysicalScreens " << i <<" " << _PhScAr[i-1].index << " " << std::endl;
		

		}
	else if (_TargetSystem.compare("Wave") == 0)
        // note to update with Wave Screene info
		{
			std::cout << "Wave screens  loaded" << std::endl;
			//int i =loadPhysicalScreensArrayWave();
			int i =loadPhysicalScreensArrayStarCaveCenterRow();
			std::cout << "loadPhysicalScreens " << i <<" " << _PhScAr[i-1].index << " " << std::endl;
		

		}

	else  

		{
			std::cout << " else TourCaveCalit2" << std::endl;
			int i =loadPhysicalScreensArrayTourCaveCalit2_5lowerScr();
			std::cout << "loadPhysicalScreens " << i <<" " << _PhScAr[i-1].index << " " << std::endl;


		}

	initGeoEdSection();//SlideStuff



}

void ParticleDreams::initSound()
{
	soundEnabled = false;
	#ifdef OMICRON_SOUND
	soundEnabled = true;
	float SoundWidth =2;
	//ToDoSound
	_SoundMng = new omicron::SoundManager();

	if (_DisplaySystem.compare("Cave2") == 0){	_SoundMng->connectToServer("xenakis.evl.uic.edu", 57120);}
	else {	_SoundMng->connectToServer("131.193.76.39", 57120);}
	_SoundMng->startSoundServer();
	_SoundEnv = _SoundMng->getSoundEnvironment();
	
	if (_DisplaySystem.compare("Cave2") == 0){_SoundEnv->setAssetDirectory("/dan/particle_dreams/");}
	else {_SoundEnv->setAssetDirectory("/Users/dan/data/OASCache/");}
	
	
	if(cvr::ComController::instance()->isMaster())
	{ std::cout << "_SoundMng->isSoundServerRunning \n";
		if (_DisplaySystem.compare("Cave2") == 0)while( !_SoundMng->isSoundServerRunning() ){}
		std::cout << "_SoundMng->isSoundServerRunning end \n";
	}
	
	_harmonicAlgorithm = _SoundEnv->loadSoundFromFile("harmonicAlgorithm", "harmonicAlgorithm.wav");
	_dan_10122606_sound_spray = _SoundEnv->loadSoundFromFile("soundSpray", "dan_10122606_sound_spray.wav");
	
    _pinkNoise = _SoundEnv->loadSoundFromFile("dummyName", "cdtds.31.pinkNoise.wav");
    _texture_12 = _SoundEnv->loadSoundFromFile("dummyName", "dan_texture_12.wav");
    _short_sound_01a = _SoundEnv->loadSoundFromFile("dummyName", "dan_short_sound_01a.wav");
    _short_sound_01a1 = _SoundEnv->loadSoundFromFile("dummyName", "dan_short_sound_01a.wav");
    _short_sound_01a2 = _SoundEnv->loadSoundFromFile("dummyName", "dan_short_sound_01a.wav");
    _short_sound_01a3 = _SoundEnv->loadSoundFromFile("dummyName", "dan_short_sound_01a.wav");
    _short_sound_01a4 = _SoundEnv->loadSoundFromFile("dummyName", "dan_short_sound_01a.wav");
    _short_sound_01a5 = _SoundEnv->loadSoundFromFile("dummyName", "dan_short_sound_01a.wav");
    _texture_17_swirls3 = _SoundEnv->loadSoundFromFile("dummyName", "dan_texture_17_swirls3.wav");
    _rain_at_sea = _SoundEnv->loadSoundFromFile("dummyName", "dan_texture_18_rain_at_sea.wav");
    _dan_texture_13 = _SoundEnv->loadSoundFromFile("dummyName", "dan_texture_13.wav");
    _dan_texture_05 = _SoundEnv->loadSoundFromFile("dummyName", "dan_texture_05.wav");
    _dan_short_sound_04 = _SoundEnv->loadSoundFromFile("dummyName", "dan_short_sound_04.wav");
    _dan_ambiance_2 = _SoundEnv->loadSoundFromFile("dummyName", "dan_ambiance_2.wav");
    _dan_ambiance_1 = _SoundEnv->loadSoundFromFile("dummyName", "dan_ambiance_1.wav");
    _dan_5min_ostinato = _SoundEnv->loadSoundFromFile("dummyName", "dan_10120607_5_min_ostinato.WAV");
    _dan_10120603_Rez1 = _SoundEnv->loadSoundFromFile("dummyName", "dan_10120603_Rez.1.wav");
    _dan_mel_amb_slower = _SoundEnv->loadSoundFromFile("dummyName", "dan_10122604_mel_amb_slower.wav");
    _dan_rain_at_sea_loop = _SoundEnv->loadSoundFromFile("dummyName", "dan_rain_at_sea_loop.wav");
    _dan_10122608_sound_spray_low = _SoundEnv->loadSoundFromFile("dummyName", "dan_10122608_sound_spray_low.wav");
    _dan_10120600_rezS3_rez2 = _SoundEnv->loadSoundFromFile("dummyName", "dan_10120600_RezS.3_Rez.2.wav");
    

	
	for (int i=1;i<10000000;i++)
		{
			float b =25*25;
		}

	_harmonicAlgorithmInstance = new omicron::SoundInstance(_harmonicAlgorithm);
	_harmonicAlgorithmInstance->setLoop(true);
	//_harmonicAlgorithmInstance->playStereo();

	
	_dan_10122606_sound_sprayInstance = new omicron::SoundInstance(_dan_10122606_sound_spray);
	_dan_10122606_sound_sprayInstance->setVolume(0);
	_dan_10122606_sound_sprayInstance->setLoop(true);
	//_dan_10122606_sound_sprayInstance->setWetness(1); 
	//_dan_10122606_sound_sprayInstance->setRoomSize(.1);
	//_dan_10122606_sound_sprayInstance->setReverbMix(1) 
	//_dan_10122606_sound_sprayInstance->setReverb(.1);
	
	_dan_10122606_sound_sprayInstance->play();
	

	 _pinkNoiseInstance =new omicron::SoundInstance(_pinkNoise);
    _texture_12Instance =new omicron::SoundInstance(_texture_12);
    _short_sound_01aInstance =new omicron::SoundInstance(_short_sound_01a);
    _short_sound_01a1Instance =new omicron::SoundInstance(_short_sound_01a1);
    _short_sound_01a2Instance =new omicron::SoundInstance(_short_sound_01a2);
    _short_sound_01a3Instance =new omicron::SoundInstance(_short_sound_01a3);
    _short_sound_01a4Instance =new omicron::SoundInstance(_short_sound_01a4);
    _short_sound_01a5Instance =new omicron::SoundInstance(_short_sound_01a5);
    _texture_17_swirls3Instance =new omicron::SoundInstance(_texture_17_swirls3);
    _rain_at_seaInstance =new omicron::SoundInstance(_rain_at_sea);
    _dan_texture_13Instance =new omicron::SoundInstance(_dan_texture_13);
    _dan_texture_05Instance =new omicron::SoundInstance(_dan_texture_05);
    _dan_short_sound_04Instance =new omicron::SoundInstance(_dan_short_sound_04);
    _dan_ambiance_2Instance =new omicron::SoundInstance(_dan_ambiance_2);
    _dan_ambiance_1Instance =new omicron::SoundInstance(_dan_ambiance_1);
    _dan_5min_ostinatoInstance =new omicron::SoundInstance(_dan_5min_ostinato);
    _dan_10120603_Rez1Instance =new omicron::SoundInstance(_dan_10120603_Rez1);
    _dan_mel_amb_slowerInstance =new omicron::SoundInstance(_dan_mel_amb_slower);
    _dan_rain_at_sea_loopInstance =new omicron::SoundInstance(_dan_rain_at_sea_loop);
    _dan_10122608_sound_spray_lowInstance =new omicron::SoundInstance(_dan_10122608_sound_spray_low);
    _dan_10120600_rezS3_rez2Instance =new omicron::SoundInstance(_dan_10120600_rezS3_rez2);
       _pinkNoiseInstance->setVolume(0);
    _pinkNoiseInstance->setLoop(true);
    _pinkNoiseInstance->play(); 
   // _dan_texture_09Instance->setVolume(0);
   // _dan_texture_09Instance->setLoop(true);
   // _dan_texture_09Instance->play();
    _texture_12Instance->setVolume(0);
    _texture_12Instance->setWidth(SoundWidth);
      _texture_12Instance->setLoop(true);
    _texture_12Instance->play();
    _short_sound_01aInstance->play();
    _short_sound_01aInstance->setVolume(1);
    _short_sound_01a1Instance->setVolume(1);
    _short_sound_01a2Instance->setVolume(1);
    _short_sound_01a3Instance->setVolume(1);
    _short_sound_01a4Instance->setVolume(1);
    _short_sound_01a5Instance->setVolume(1);
    _short_sound_01aInstance->setVolume(1);
    _short_sound_01a1Instance->setWidth(SoundWidth);
    _short_sound_01a2Instance->setWidth(SoundWidth);
    _short_sound_01a3Instance->setWidth(SoundWidth);
    _short_sound_01a4Instance->setWidth(SoundWidth);
    _short_sound_01a5Instance->setWidth(SoundWidth);

 

#endif


#ifdef OAS_SOUND
    std::string ipAddrStr;
    unsigned short port;
    std::string pathToAudioFiles;

    ipAddrStr =  ConfigManager::getEntry("ipAddr","Plugin.ParticleDreams.SoundServer","127.0.0.1");

    port = ConfigManager::getInt("port", "Plugin.ParticleDreams.SoundServer", 31231);
	std::cout <<  "OAS_SOUND enabled \n";
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
#endif

}

void ParticleDreams::initWater()
{
    // find max window size, i will assume all windows are the same for the moment
    cvr::WindowInfo * wi = ScreenConfig::instance()->getWindowInfo(0);
    if(!wi)
    {
	std::cerr << "No Valid WindowInfo." << std::endl;
	return;
    }

    // init fbo textures
    _waterColorTexture = new osg::TextureRectangle();
    _waterColorTexture->setTextureSize(wi->width,wi->height);
    _waterColorTexture->setInternalFormat(GL_RGBA);
    _waterColorTexture->setFilter(osg::Texture2D::MIN_FILTER,osg::Texture2D::NEAREST);
    _waterColorTexture->setFilter(osg::Texture2D::MAG_FILTER,osg::Texture2D::NEAREST);
    _waterColorTexture->setResizeNonPowerOfTwoHint(false);
    _waterColorTexture->setUseHardwareMipMapGeneration(false);

    _waterDepthTexture = new osg::Texture2D();
    _waterDepthTexture->setTextureSize(wi->width,wi->height);
    _waterDepthTexture->setInternalFormat(GL_DEPTH_COMPONENT);
    _waterDepthTexture->setFilter(osg::Texture2D::MIN_FILTER,osg::Texture2D::NEAREST);
    _waterDepthTexture->setFilter(osg::Texture2D::MAG_FILTER,osg::Texture2D::NEAREST);
    _waterDepthTexture->setResizeNonPowerOfTwoHint(false);
    _waterDepthTexture->setUseHardwareMipMapGeneration(false);

    // init cameras
    _waterPreCamera = new osg::Camera();
    _waterPreCamera->setAllowEventFocus(false);
    _waterPreCamera->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //_waterPreCamera->setClearColor(osg::Vec4(1.0,0,0,1.0));
    _waterPreCamera->setRenderOrder(osg::Camera::PRE_RENDER);
    _waterPreCamera->setRenderTargetImplementation(osg::Camera::FRAME_BUFFER_OBJECT);
    _waterPreCamera->attach(osg::Camera::COLOR_BUFFER0,_waterColorTexture);
    _waterPreCamera->attach(osg::Camera::DEPTH_BUFFER,_waterDepthTexture);
    _waterPreCamera->setReferenceFrame(osg::Transform::RELATIVE_RF);

    _waterMirrorXform = new osg::MatrixTransform();
    osg::Matrix m;
    m.makeScale(osg::Vec3(1.0,-1.0,1.0));
    _waterMirrorXform->setMatrix(m);
    _waterPreCamera->addChild(_waterMirrorXform);
    _waterMirrorXform->addChild(_particleGeode);

    _waterPostCamera = new osg::Camera();
    _waterPostCamera->setAllowEventFocus(false);
    _waterPostCamera->setClearMask(0);
    _waterPostCamera->setRenderOrder(osg::Camera::POST_RENDER);
    _waterPostCamera->setReferenceFrame(osg::Transform::RELATIVE_RF);
    _waterPostCamera->addChild(_particleGeode);

    _waterNestedCamera = new osg::Camera();
    _waterNestedCamera->setAllowEventFocus(false);
    _waterNestedCamera->setClearMask(0);
    _waterNestedCamera->setRenderOrder(osg::Camera::NESTED_RENDER);
    //_waterNestedCamera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);

    _waterSkyGeode = new osg::Geode();
    _waterSkyGeometry = new osg::Geometry();
    _waterSkyGeometry->setUseDisplayList(false);
    _waterSkyGeometry->setUseVertexBufferObjects(true);
   
    MyComputeBounds * mcb = new MyComputeBounds();
    _waterSkyGeometry->setComputeBoundingBoxCallback(mcb);
    mcb->_bound = osg::BoundingBox(osg::Vec3(-100000,-100000,-100000),osg::Vec3(100000,100000,100000));
    
    osg::Vec2Array * verts = new osg::Vec2Array(4);
    osg::Vec4Array * colors = new osg::Vec4Array(1);
    osg::Vec2Array * tcoords = new osg::Vec2Array(4);
    
    verts->at(0) = osg::Vec2(-1.0,1.0);
    verts->at(1) = osg::Vec2(-1.0,-1.0);
    verts->at(2) = osg::Vec2(1.0,-1.0);
    verts->at(3) = osg::Vec2(1.0,1.0);

    colors->at(0) = osg::Vec4(1.0,1.0,1.0,1.0);

    tcoords->at(0) = osg::Vec2(0.0,750.0);
    tcoords->at(1) = osg::Vec2(0.0,0.0);
    tcoords->at(2) = osg::Vec2(1500.0,0.0);
    tcoords->at(3) = osg::Vec2(1500.0,750.0);

    _waterSkyGeometry->setVertexArray(verts);
    _waterSkyGeometry->setColorArray(colors);
    _waterSkyGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);
    //TODO add version ifdef
    //_waterSkyGeometry->setTexCoordArray(0,tcoords,osg::Array::BIND_PER_VERTEX);
    _waterSkyGeometry->setTexCoordArray(0,tcoords);
    _waterSkyGeometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,4));
    _waterSkyGeode->addDrawable(_waterSkyGeometry);
    _waterNestedCamera->addChild(_waterSkyGeode);
    _waterNestedCamera->getOrCreateStateSet()->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
    _waterNestedCamera->getOrCreateStateSet()->setMode(GL_BLEND,osg::StateAttribute::ON);
    _waterNestedCamera->getOrCreateStateSet()->setTextureAttributeAndModes(3,_waterColorTexture,osg::StateAttribute::ON);

    osg::BlendFunc * bf = new osg::BlendFunc(osg::BlendFunc::SRC_ALPHA,osg::BlendFunc::ONE_MINUS_SRC_ALPHA);
    _waterNestedCamera->getOrCreateStateSet()->setAttributeAndModes(bf,osg::StateAttribute::ON);
        

    _waterVert = osg::Shader::readShaderFile(osg::Shader::VERTEX, osgDB::findDataFile(_dataDir + "/mirror-water/mirror-water.vert"));
    _waterFrag = osg::Shader::readShaderFile(osg::Shader::FRAGMENT, osgDB::findDataFile(_dataDir + "/mirror-water/mirror-water.frag"));
    _waterProgram = new osg::Program();
    _waterProgram->setName("mirror-water");
    _waterProgram->addShader(_waterVert);
    _waterProgram->addShader(_waterFrag);

    _waterMirrorVert = osg::Shader::readShaderFile(osg::Shader::VERTEX, osgDB::findDataFile(_dataDir + "/mirror-water/sprite-mirror.vert"));
    _waterMirrorFrag = osg::Shader::readShaderFile(osg::Shader::FRAGMENT, osgDB::findDataFile(_dataDir + "/mirror-water/sprite-mirror.frag"));
    _waterMirrorProgram = new osg::Program();
    _waterMirrorProgram->setName("mirror-render");
    _waterMirrorProgram->addShader(_waterMirrorVert);
    _waterMirrorProgram->addShader(_waterMirrorFrag);

    _waterNestedCamera->getOrCreateStateSet()->setAttribute(_waterProgram);
    _waterPreCamera->getOrCreateStateSet()->setAttribute(_waterMirrorProgram);
    _waterPostCamera->getOrCreateStateSet()->setAttribute(_waterMirrorProgram);

    _waterTimeUni = new osg::Uniform(osg::Uniform::FLOAT,"time");
    _waterLightUni = new osg::Uniform(osg::Uniform::FLOAT_VEC3,"light_position");
    _waterGlowUni = new osg::Uniform(osg::Uniform::SAMPLER_2D,"glow");
    _waterGlowUni->set(1);
    _waterFillUni = new osg::Uniform(osg::Uniform::SAMPLER_2D,"fill");
    _waterFillUni->set(0);
    _waterNormUni = new osg::Uniform(osg::Uniform::SAMPLER_2D,"norm");
    _waterNormUni->set(2);
    //TODO add version ifdef
    //_waterReflUni = new osg::Uniform(osg::Uniform::SAMPLER_2D_RECT,"refl");
    _waterReflUni = new osg::Uniform(osg::Uniform::SAMPLER_2D,"refl");
    _waterReflUni->set(3);

    _waterNestedCamera->getOrCreateStateSet()->addUniform(_waterTimeUni);
    _waterNestedCamera->getOrCreateStateSet()->addUniform(_waterLightUni);
    _waterNestedCamera->getOrCreateStateSet()->addUniform(_waterGlowUni);
    _waterNestedCamera->getOrCreateStateSet()->addUniform(_waterFillUni);
    _waterNestedCamera->getOrCreateStateSet()->addUniform(_waterNormUni);
    _waterNestedCamera->getOrCreateStateSet()->addUniform(_waterReflUni);

    _waterNormalTexture = new osg::Texture2D();
    _waterSkyFillTexture = new osg::Texture2D();
    _waterSkyGlowTexture = new osg::Texture2D();

    osg::ref_ptr<osg::Image> image = osgDB::readImageFile(_dataDir + "/mirror-water/water-normal.png");

    if(image)
    {
	_waterNormalTexture->setImage(image);
	_waterNormalTexture->setWrap(osg::Texture::WRAP_S,osg::Texture::REPEAT);
	_waterNormalTexture->setWrap(osg::Texture::WRAP_T,osg::Texture::REPEAT);
	_waterNormalTexture->setFilter(osg::Texture::MIN_FILTER,osg::Texture::LINEAR_MIPMAP_LINEAR);
	_waterNormalTexture->setFilter(osg::Texture::MAG_FILTER,osg::Texture::LINEAR);
	_waterNormalTexture->setResizeNonPowerOfTwoHint(false);
	_waterNestedCamera->getOrCreateStateSet()->setTextureAttributeAndModes(2,_waterNormalTexture,osg::StateAttribute::ON);
    }
    else
    {
	std::cerr << "Unable to read texture: " << _dataDir + "/mirror-water/water-normal.png" << std::endl;
    }
    
    image = osgDB::readImageFile(_dataDir + "/mirror-water/sky-fill.png");

    if(image)
    {
	_waterSkyFillTexture->setImage(image);
	_waterSkyFillTexture->setWrap(osg::Texture::WRAP_S,osg::Texture::CLAMP_TO_EDGE);
	_waterSkyFillTexture->setWrap(osg::Texture::WRAP_T,osg::Texture::CLAMP_TO_EDGE);
	_waterSkyFillTexture->setFilter(osg::Texture::MIN_FILTER,osg::Texture::NEAREST_MIPMAP_LINEAR);
	_waterSkyFillTexture->setFilter(osg::Texture::MAG_FILTER,osg::Texture::LINEAR);
	_waterSkyFillTexture->setResizeNonPowerOfTwoHint(false);
	_waterNestedCamera->getOrCreateStateSet()->setTextureAttributeAndModes(0,_waterSkyFillTexture,osg::StateAttribute::ON);
    }
    else
    {
	std::cerr << "Unable to read texture: " << _dataDir + "/mirror-water/sky-fill.png" << std::endl;
    }

    image = osgDB::readImageFile(_dataDir + "/mirror-water/sky-glow.png");

    if(image)
    {
	_waterSkyGlowTexture->setImage(image);
	_waterSkyGlowTexture->setWrap(osg::Texture::WRAP_S,osg::Texture::CLAMP_TO_EDGE);
	_waterSkyGlowTexture->setWrap(osg::Texture::WRAP_T,osg::Texture::CLAMP_TO_EDGE);
	_waterSkyGlowTexture->setFilter(osg::Texture::MIN_FILTER,osg::Texture::NEAREST_MIPMAP_LINEAR);
	_waterSkyGlowTexture->setFilter(osg::Texture::MAG_FILTER,osg::Texture::LINEAR);
	_waterSkyGlowTexture->setResizeNonPowerOfTwoHint(false);
	_waterNestedCamera->getOrCreateStateSet()->setTextureAttributeAndModes(1,_waterSkyGlowTexture,osg::StateAttribute::ON);
    }
    else
    {
	std::cerr << "Unable to read texture: " << _dataDir + "/mirror-water/sky-glow.png" << std::endl;
    }
}

void ParticleDreams::cleanupPart()
{
    delete[] h_particleData;
    delete[] _old_refl_hits;
    delete[] h_debugData;
    delete[] _refl_hits;

    _callbackActive = false;
}

void ParticleDreams::cleanupGeometry()
{
 

    delete[] _PhScAr;

    _refObjSwitchFace = NULL;
    _refObjSwitchLine = NULL;
    _injObjSwitchFace = NULL;
    _injObjSwitchLine = NULL;

    PluginHelper::getScene()->removeChild(_handModelMT);

    _handModelMT = NULL;

    _spriteTexture = NULL;

    _spriteVert = NULL;
    _spriteFrag = NULL;
    _spriteProgram = NULL;
    _primitive = NULL;

    _particleGeode = NULL;

    delete _particleObject;
}

void ParticleDreams::cleanupSound()
{
    if(!soundEnabled)
    {
	return;
    }

    soundEnabled = false;
#ifdef OAS_SOUND
    chimes.release();

    pinkNoise.release();

    dan_texture_09.release();

    texture_12.release();

    short_sound_01a.release();
    short_sound_01a1.release();
    short_sound_01a2.release();
    short_sound_01a3.release();
    short_sound_01a4.release();
    short_sound_01a5.release();

    texture_17_swirls3.release();
    rain_at_sea.release();
    dan_texture_13.release();
    dan_texture_05.release();
    dan_short_sound_04.release();
    dan_ambiance_2.release();
    dan_ambiance_1.release();
    dan_5min_ostinato.release();
    dan_10120603_Rez1.release();
    dan_mel_amb_slower.release();
    harmonicAlgorithm.release();
    dan_rain_at_sea_loop.release();

    dan_10122606_sound_spray.release();

    dan_10122608_sound_spray_low.release();
    dan_10120600_rezS3_rez2.release();

    oasclient::ClientInterface::shutdown();
#endif
	#ifdef OMICRON_SOUND
std::cout << " _SoundMng distructorCaled \n";
	_SoundMng->~SoundManager();
	
	//sound
	//harmonicAlgorithm.release();
	//dan_10122606_sound_spray.release();
	 #endif

}

void ParticleDreams::updateHead()
{
 //   osg::Matrix m = PluginHelper::getHeadMat(head_id) * _particleObject->getWorldToObjectMatrix();
   osg::Matrix hm = PluginHelper::getHeadMat(head_id) ;
   headPos = osg::Vec3(0,0.0,0) * hm;
   navHeadPos=osg::Vec3(0,0.0,0) * hm*_particleObject->getWorldToObjectMatrix();
 
    osg::Matrix hhMat;
    hhMat.makeRotate(_headHeading * M_PI / 180.0, osg::Vec3(0,0,1));
    osg::Matrix hpMat;
    hpMat.makeRotate(_headPitch * M_PI / 180.0, osg::Vec3(1,0,0));
    osg::Matrix hrMat;
    hrMat.makeRotate(_headRoll * M_PI / 180.0, osg::Vec3(0,1,0));
    osg::Matrix hoMat = hrMat * hpMat * hhMat;

    osg::Matrix m = PluginHelper::getHeadMat(head_id) * _particleObject->getWorldToObjectMatrix();
    osg::Vec3 headdir = osg::Vec3(0,1.0,0) * hoMat *hm;
    headdir = headdir - hm.getTrans();
    headdir.normalize();

    osg::Vec3 headup = osg::Vec3(0,0,1.0) * hoMat * hm;
    headup = headup - hm.getTrans();
    headup.normalize();
	osg::Vec3 headoffset(0.0,0,0.0);
	std::string str2 = ("Simulator"); 
	if (_DisplaySystem.compare(str2) == 0)
	{ 
		//std::cout << "Simulator Head Offset" << std::endl;
		//headoffset[0]= 0;headoffset[1]= 1.5;headoffset[0]= 0.17;
	}

    osg::Vec3 headpos = headoffset * hm;
    hm.setTrans(headpos);

 	headPos[0] = headpos.x();
    headPos[1] = headpos.y();
    headPos[2] = headpos.z();
    headVec[0] = headdir.x();
    headVec[1] = headdir.y();
    headVec[2] = headdir.z();
    headMat[4] = headup.x();
    headMat[5] = headup.y();
    headMat[6] = headup.z();

}

void ParticleDreams::setWater(bool b)
{
    if(_particleObject)
    {
	while(_particleObject->getNumChildNodes())
	{
	    _particleObject->removeChild(_particleObject->getChildNode(0));
	}

	if(b)
	{
	    //_particleObject->addChild(_particleGeode);
	    _particleObject->addChild(_waterNestedCamera);
	    _particleObject->addChild(_waterPreCamera);
	    _particleObject->addChild(_waterPostCamera);

	    _particleGeode->getDrawable(0)->getOrCreateStateSet()->removeAttribute(_spriteProgram);
	}
	else
	{
	    _particleObject->addChild(_particleGeode);
	    _particleGeode->getDrawable(0)->getOrCreateStateSet()->setAttribute(_spriteProgram);
	}
    }
}

void ParticleDreams::updateHand()
{
// set offsetfor injector and reflectormodeles
   osg::Vec3 offset(0.0,200,100.0);// tourcave
//	  osg::Vec3 offset(0.0,600,-100);// simulator
	
	std::string str2 = ("Simulator"); 
	if (_DisplaySystem.compare(str2) == 0)
	{ 
		//std::cout << "Simulator" << std::endl;
		offset[0]= 0;offset[1]= 600;offset[0]= -100;
	}
	
	
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
    
    		 //reflectorGetMaxnumNum();
		 //reflectorSetMaxnumNum( maxNumber);
		 reflectorSetType ( reflNum , 1);// 0 os off, 1 is plain
		 //reflectorSetDifaults( reflNum );
		 reflectorSetPosition ( reflNum , ftToM(5), 0,  0, axisUpZ);
		 reflectorSetNormal( reflNum , -1, 0,  0, axisUpZ);
		 reflectorSetSize ( reflNum , caverad, axisUpZ);
		 reflectorSetDamping ( reflNum , damping);
		 reflectorSetNoTraping ( reflNum , no_traping);

    reflNum = firstRefNum;
    //front
    		 reflectorSetPosition ( reflNum , 0, caverad,  -caverad, axisUpZ);
		 reflectorSetNormal( reflNum , 0, 0,  1, axisUpZ);

    copy_reflector( reflNum, reflNum +1);
    reflNum++;//back
    
     	reflectorSetPosition ( reflNum , 0, caverad,  caverad, axisUpZ);
		reflectorSetNormal( reflNum , 0, 0,  -1, axisUpZ);

    copy_reflector( reflNum, reflNum +1);
    reflNum++;//right
    
      	reflectorSetPosition ( reflNum , caverad, caverad,  0, axisUpZ);
		reflectorSetNormal( reflNum , -1, 0,  0, axisUpZ);
 
    copy_reflector( reflNum, reflNum +1);

    reflNum++;//left
        reflectorSetPosition ( reflNum , -caverad, caverad,  0, axisUpZ);
		reflectorSetNormal( reflNum , 1, 0,  0, axisUpZ);

    copy_reflector( reflNum, reflNum +1);
    reflNum++;//top
        reflectorSetPosition ( reflNum , 0, 2 * caverad,  0, axisUpZ);
		reflectorSetNormal( reflNum , 0, -1,  0, axisUpZ);
  
 
    copy_reflector( reflNum, reflNum +1);
    reflNum++;//bottom
    
        reflectorSetPosition ( reflNum , 0, 0,  0, axisUpZ);
		reflectorSetNormal( reflNum , 0, 1,  0, axisUpZ);

    
    return reflNum;

}


int ParticleDreams::loadPhysicalScreensArrayTourCaveCalit2()
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


int ParticleDreams::loadPhysicalScreensArrayTourCaveCalit2_5lowerScr()
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
 int ParticleDreams::loadPhysicalScreensArrayWave()
{
    _PhScAr = new _PhSc [128];

    float height, h, width, p, originX, originY,r,name,originZ, screen;
    int i=0;

    // _PhScAr[i].index  -1 is sentannel for end of list

         height=681; h=0.0; width=1209; p=22.5 ;  originX=-2428;   originY=1250;  r=0.0; name=0; originZ=660;
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

     height=681; h=0.0; width=1209; p=22.5 ;  originX=-2428;  ; originY=1250;  r=0.0; name=0; originZ=660;
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


       height=681; h=0.0; width=1209; p=22.5 ;  originX=0;  ; originY=1250;  r=0.0; name=0; originZ=660 ;   screen=0 ;
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


       height=681; h=0.0; width=1209; p=22.5;   originX=1214;  ; originY=1250;  r=0.0; name=1; originZ=660;    screen=1 ;
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


     height=681; h=0.0; width=1209; p=22.5;   originX=2428;  ;originY=1250;  r=0.0; name=0; originZ=660 ;   screen=0 ;
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


int ParticleDreams::loadPhysicalScreensArrayTourCaveSaudi()
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

int ParticleDreams::loadPhysicalScreensArrayCave2()

{
_PhScAr = new _PhSc [128];
float height, h, width, p, originX, originY,r,name,originZ, screen;
int i=0;
std::cout << "in loadPhysicalScreensArrayCave2 \n";
//			<Screen comment=RelColumn -9 row 0 column 18 width= 1018.3 height= 572.5 originX= -1008.81 originY= -3104.81 originZ= 869.7 h= 162 p= -0 r= 0 name= 0 />
			 width= 1018.3; height= 572.5; originX= -1008.81; originY= -3104.81; originZ= 869.7 ;h= 162; p= -0; r= 0; name= 0 ;
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
			
			
//			<Screen comment=RelColumn -8 row 0 column 17 width= 1018.3 height= 572.5 originX= -1918.88 originY= -2641.11 originZ= 869.7 h= 144 p= -0 r= 0 name= 0 />
			 width= 1018.3; height= 572.5 ;originX= -1918.88 ;originY= -2641.11 ;originZ= 869.7 ;h= 144; p= -0 ;r= 0; name= 0 ;
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
			
			
//			<Screen comment=RelColumn -7 row 0 column 16 width= 1018.3 height= 572.5 originX= -2641.11 originY= -1918.88 originZ= 869.7 h= 126 p= -0 r= 0 name= 0 />
			 width= 1018.3 ;height= 572.5; originX= -2641.11; originY= -1918.88; originZ= 869.7; h= 126; p= -0 ;r= 0 ;name= 0 ;
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


//			<Screen comment=RelColumn -6 row 0 column 15 width= 1018.3 height= 572.5 originX= -3104.81 originY= -1008.81 originZ= 869.7 h= 108 p= -0 r= 0 name= 0 />
			 width= 1018.3; height= 572.5; originX= -3104.81 ;originY= -1008.81 ;originZ= 869.7; h= 108; p= -0 ;r= 0 ;name= 0 ;
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


//			<Screen comment=RelColumn -5 row 0 column 14 width= 1018.3 height= 572.5 originX= -3264.59 originY= -0.0001427 originZ= 869.7 h= 90 p= -0 r= 0 name= 0 />
			width= 1018.3; height= 572.5; originX= -3264.59; originY= -0.0001427; originZ= 869.7; h= 90 ;p= -0; r= 0 ;name= 0 ;
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


//			<Screen comment=RelColumn -4 row 0 column 13 width= 1018.3 height= 572.5 originX= -3104.81 originY= 1008.81 originZ= 869.7 h= 72 p= -0 r= 0 name= 0 />
			 width= 1018.3; height= 572.5; originX= -3264.59 ;originY= -0.0001427 ;originZ= 869.7 ;h= 90 ;p= -0; r= 0; name= 0 ;
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


//			<Screen comment=RelColumn -3 row 0 column 12 width= 1018.3 height= 572.5 originX= -2641.11 originY= 1918.88 originZ= 869.7 h= 54 p= -0 r= 0 name= 0 />
			 width= 1018.3; height= 572.5; originX= -2641.11; originY= 1918.88; originZ= 869.7; h= 54; p= -0; r= 0; name= 0 ;
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


//			<Screen comment=RelColumn -2 row 0 column 11 width= 1018.3 height= 572.5 originX= -1918.88 originY= 2641.11 originZ= 869.7 h= 36 p= -0 r= 0 name= 0 />
			width= 1018.3; height= 572.5; originX= -1918.88; originY= 2641.11; originZ= 869.7; h= 36; p= -0 ;r= 0; name= 0 ;
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


//			<Screen comment=RelColumn -1 row 0 column 10 width= 1018.3 height= 572.5 originX= -1008.81 originY= 3104.81 originZ= 869.7 h= 18 p= -0 r= 0 name= 0 />
			 width= 1018.3; height= 572.5; originX= -1008.81; originY= 3104.81; originZ= 869.7; h= 18; p= -0; r= 0; name= 0 ;
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


//			<Screen comment=RelColumn 0 row 0 column 9 width= 1018.3 height= 572.5 originX= 0 originY= 3264.59 originZ= 869.7 h= 0 p= -0 r= 0 name= 0 />
			 width= 1018.3 ;height= 572.5; originX= 0; originY= 3264.59; originZ= 869.7; h= 0; p= -0; r= 0; name= 0 ;
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


//			<Screen comment=RelColumn 1 row 0 column 8 width= 1018.3 height= 572.5 originX= 1008.81 originY= 3104.81 originZ= 869.7 h= -18 p= -0 r= 0 name= 0 />
			 width= 1018.3; height= 572.5 ;originX= 1008.81 ;originY= 3104.81; originZ= 869.7; h= -18; p= -0; r= 0; name= 0 ;
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


//			<Screen comment=RelColumn 2 row 0 column 7 width= 1018.3 height= 572.5 originX= 1918.88 originY= 2641.11 originZ= 869.7 h= -36 p= -0 r= 0 name= 0 />
			 width= 1018.3; height= 572.5; originX= 1918.88; originY= 2641.11; originZ= 869.7; h= -36; p= -0; r= 0 ;name= 0 ;
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


//			<Screen comment=RelColumn 3 row 0 column 6 width= 1018.3 height= 572.5 originX= 2641.11 originY= 1918.88 originZ= 869.7 h= -54 p= -0 r= 0 name= 0 />
			width= 1018.3 ;height= 572.5 ;originX= 2641.11 ;originY= 1918.88 ;originZ= 869.7; h= -54; p= -0; r= 0; name= 0 ;
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


//			<Screen comment=RelColumn 4 row 0 column 5 width= 1018.3 height= 572.5 originX= 3104.81 originY= 1008.81 originZ= 869.7 h= -72 p= -0 r= 0 name= 0 />
			width= 1018.3 ;height= 572.5 ;originX= 3104.81; originY= 1008.81; originZ= 869.7; h= -72; p= -0; r= 0; name= 0 ;
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


//			<Screen comment=RelColumn 5 row 0 column 4 width= 1018.3 height= 572.5 originX= 3264.59 originY= -0.0001427 originZ= 869.7 h= -90 p= -0 r= 0 name= 0 />
			 width= 1018.3 ;height= 572.5; originX= 3264.59 ;originY= -0.0001427; originZ= 869.7; h= -90; p= -0; r= 0; name= 0;
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


//			<Screen comment=RelColumn 6 row 0 column 3 width= 1018.3 height= 572.5 originX= 3104.81 originY= -1008.81 originZ= 869.7 h= -108 p= -0 r= 0 name= 0 />
			width= 1018.3 ;height= 572.5; originX= 3104.81 ;originY= -1008.81; originZ= 869.7; h= -108; p= -0; r= 0 ;name= 0 ;
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


//			<Screen comment=RelColumn 7 row 0 column 2 width= 1018.3 height= 572.5 originX= 2641.11 originY= -1918.88 originZ= 869.7 h= -126 p= -0 r= 0 name= 0 />
			width= 1018.3 ;height= 572.5 ;originX= 2641.11 ;originY= -1918.88; originZ= 869.7; h= -126 ;p= -0 ;r= 0 ;name= 0 ;
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


//			<Screen comment=RelColumn 8 row 0 column 1 width= 1018.3 height= 572.5 originX= 1918.88 originY= -2641.11 originZ= 869.7 h= -144 p= -0 r= 0 name= 0 />
			 width= 1018.3 ;height= 572.5 ;originX= 1918.88 ;originY= -2641.11 ;originZ= 869.7 ;h= -144; p= -0; r= 0 ;name= 0 ;
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




int ParticleDreams::loadPhysicalScreensArrayStarCave()

{
_PhScAr = new _PhSc [128];
float height, h, width, p, originX, originY,r,name,originZ, screen;
int i=0;
std::cout << "in loadPhysicalScreensArrayStarCave \n";


originX= 1249   ;originY= -406   ;originZ= -1180   ;width= 2134   ;height= 1200  ; h= -108.0  ; p= -14.9  ; r= 0.0  ;name= 0;  
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

originX= 1397   ;originY= -454   ;originZ= 0   ;width= 2134   ;height= 1200  ; h= -108.0  ; p= 0.0  ; r= 0.0  ;name= 0 ; 
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

originX= 1249   ;originY= -406   ;originZ= 1180   ;width= 2134   ;height= 1200  ; h= -108.0  ; p= 14.9  ; r= 0.0  ;name= 0; 
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

originX= 772   ;originY= 1062   ;originZ= -1180   ;width= 2134   ;height= 1200  ; h= -36.0  ; p= -14.9  ; r= 0.0  ;name= 0; 
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

originX= 863   ;originY= 1188   ;originZ= 0   ;width= 2134   ;height= 1200  ; h= -36.0  ; p= 0.0  ; r= 0.0  ;name= 0 ; 
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

originX= 772   ;originY= 1062   ;originZ= 1180   ;width= 2134   ;height= 1200  ; h= -36.0  ; p= 14.9  ; r= 0.0  ;name= 0; 
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

originX= -772   ;originY= 1062   ;originZ= -1180   ;width= 2134   ;height= 1200  ; h= 36.0  ; p= -14.9  ; r= 0.0  ;name= 0 ;
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

originX= -863   ;originY= 1188   ;originZ= 0   ;width= 2134   ;height= 1200  ; h= 36.0  ; p= 0.0  ; r= 0.0  ;name= 0 ; 
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

originX= -772   ;originY= 1062   ;originZ= 1180   ;width= 2134   ;height= 1200  ; h= 36.0  ; p= 14.9  ; r= 0.0  ;name= 0 ;
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

originX= -1249   ;originY= -406   ;originZ= -1180   ;width= 2134   ;height= 1200  ; h= 108.0  ; p= -14.9  ; r= 0.0  ;name= 0 ;
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
	
originX= -1397   ;originY= -454   ;originZ= 0   ;width= 2134   ;height= 1175  ; h= 108.0  ; p= 0.0  ; r= 0.0  ;name= 0 ;
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
	
originX= -1249   ;originY= -406   ;originZ= 1180   ;width= 2134   ;height= 1200  ; h= 108.0  ; p= 14.9  ; r= 0.0  ;name= 0 ;
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
	
originX= 0   ;originY= -1313   ;originZ= -1180   ;width= 2134   ;height= 1200  ; h= 180.0  ; p= -14.9  ; r= 0.0  ;name= 0 ;
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
	
originX= 0   ;originY= -1468   ;originZ= 0   ;width= 2134   ;height= 1200  ; h= 180.0  ; p= 0.0  ; r= 0.0  ;name= 0  ;
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
	
originX= 0   ;originY= -1313   ;originZ= 1180   ;width= 2134   ;height= 1200  ; h= 180.0  ; p= 14.9  ; r= 0.0  ;name= 0 ;
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
	
originX= 0   ;originY= -766   ;originZ= -1759   ;width= 2723   ;height= 1532  ; h= 0.0  ; p= -90.0  ; r= 0.0  ;name= 0 ; 
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
	
originX= 0   ;originY= 766   ;originZ= -1759   ;width= 2723   ;height= 1532  ; h= 0.0  ; p= -90.0  ; r= 0.0  ;name= 0 ;
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

int ParticleDreams::loadPhysicalScreensArrayStarCaveCenterRow()

{
_PhScAr = new _PhSc [128];
float height, h, width, p, originX, originY,r,name,originZ, screen;
int i=0;
std::cout << "in loadPhysicalScreensArrayStarCaveCenterRow \n";

/*
originX= 1249   ;originY= -406   ;originZ= -1180   ;width= 2134   ;height= 1200  ; h= -108.0  ; p= -14.9  ; r= 0.0  ;name= 0;  
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
*/
originX= 1397   ;originY= -454   ;originZ= 0   ;width= 2134   ;height= 1200  ; h= -108.0  ; p= 0.0  ; r= 0.0  ;name= 0 ; 
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
/*
originX= 1249   ;originY= -406   ;originZ= 1180   ;width= 2134   ;height= 1200  ; h= -108.0  ; p= 14.9  ; r= 0.0  ;name= 0; 
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
*/
/*
originX= 772   ;originY= 1062   ;originZ= -1180   ;width= 2134   ;height= 1200  ; h= -36.0  ; p= -14.9  ; r= 0.0  ;name= 0; 
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
*/
originX= 863   ;originY= 1188   ;originZ= 0   ;width= 2134   ;height= 1200  ; h= -36.0  ; p= 0.0  ; r= 0.0  ;name= 0 ; 
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
/*
originX= 772   ;originY= 1062   ;originZ= 1180   ;width= 2134   ;height= 1200  ; h= -36.0  ; p= 14.9  ; r= 0.0  ;name= 0; 
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
*/
/*
originX= -772   ;originY= 1062   ;originZ= -1180   ;width= 2134   ;height= 1200  ; h= 36.0  ; p= -14.9  ; r= 0.0  ;name= 0 ;
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
*/
originX= -863   ;originY= 1188   ;originZ= 0   ;width= 2134   ;height= 1200  ; h= 36.0  ; p= 0.0  ; r= 0.0  ;name= 0 ; 
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
/*
originX= -772   ;originY= 1062   ;originZ= 1180   ;width= 2134   ;height= 1200  ; h= 36.0  ; p= 14.9  ; r= 0.0  ;name= 0 ;
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
*/
/*
originX= -1249   ;originY= -406   ;originZ= -1180   ;width= 2134   ;height= 1200  ; h= 108.0  ; p= -14.9  ; r= 0.0  ;name= 0 ;
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
*/	
originX= -1397   ;originY= -454   ;originZ= 0   ;width= 2134   ;height= 1175  ; h= 108.0  ; p= 0.0  ; r= 0.0  ;name= 0 ;
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
/*	
originX= -1249   ;originY= -406   ;originZ= 1180   ;width= 2134   ;height= 1200  ; h= 108.0  ; p= 14.9  ; r= 0.0  ;name= 0 ;
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
	*/
	/*
originX= 0   ;originY= -1313   ;originZ= -1180   ;width= 2134   ;height= 1200  ; h= 180.0  ; p= -14.9  ; r= 0.0  ;name= 0 ;
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
*/	
originX= 0   ;originY= -1468   ;originZ= 0   ;width= 2134   ;height= 1200  ; h= 180.0  ; p= 0.0  ; r= 0.0  ;name= 0  ;
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
/*	
originX= 0   ;originY= -1313   ;originZ= 1180   ;width= 2134   ;height= 1200  ; h= 180.0  ; p= 14.9  ; r= 0.0  ;name= 0 ;
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
*/	
/*
originX= 0   ;originY= -766   ;originZ= -1759   ;width= 2723   ;height= 1532  ; h= 0.0  ; p= -90.0  ; r= 0.0  ;name= 0 ; 
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
*/	
/*
originX= 0   ;originY= 766   ;originZ= -1759   ;width= 2723   ;height= 1532  ; h= 0.0  ; p= -90.0  ; r= 0.0  ;name= 0 ;
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
	
*/
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


int ParticleDreams::loadOneHalfPhysicalScreensArrayCave2()

{
_PhScAr = new _PhSc [128];
float height, h, width, p, originX, originY,r,name,originZ, screen;
int i=0;
std::cout << "in loadOneHalPhysicalScreensArrayCave2 \n";
		
			
//			<Screen comment=RelColumn -8 row 0 column 17 width= 1018.3 height= 572.5 originX= -1918.88 originY= -2641.11 originZ= 869.7 h= 144 p= -0 r= 0 name= 0 />
			 width= 1018.3; height= 572.5 ;originX= -1918.88 ;originY= -2641.11 ;originZ= 869.7 ;h= 144; p= -0 ;r= 0; name= 0 ;
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
			
			


//			<Screen comment=RelColumn -6 row 0 column 15 width= 1018.3 height= 572.5 originX= -3104.81 originY= -1008.81 originZ= 869.7 h= 108 p= -0 r= 0 name= 0 />
			 width= 1018.3; height= 572.5; originX= -3104.81 ;originY= -1008.81 ;originZ= 869.7; h= 108; p= -0 ;r= 0 ;name= 0 ;
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




//			<Screen comment=RelColumn -4 row 0 column 13 width= 1018.3 height= 572.5 originX= -3104.81 originY= 1008.81 originZ= 869.7 h= 72 p= -0 r= 0 name= 0 />
			 width= 1018.3; height= 572.5; originX= -3264.59 ;originY= -0.0001427 ;originZ= 869.7 ;h= 90 ;p= -0; r= 0; name= 0 ;
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




//			<Screen comment=RelColumn -2 row 0 column 11 width= 1018.3 height= 572.5 originX= -1918.88 originY= 2641.11 originZ= 869.7 h= 36 p= -0 r= 0 name= 0 />
			width= 1018.3; height= 572.5; originX= -1918.88; originY= 2641.11; originZ= 869.7; h= 36; p= -0 ;r= 0; name= 0 ;
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




//			<Screen comment=RelColumn 0 row 0 column 9 width= 1018.3 height= 572.5 originX= 0 originY= 3264.59 originZ= 869.7 h= 0 p= -0 r= 0 name= 0 />
			 width= 1018.3 ;height= 572.5; originX= 0; originY= 3264.59; originZ= 869.7; h= 0; p= -0; r= 0; name= 0 ;
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




//			<Screen comment=RelColumn 2 row 0 column 7 width= 1018.3 height= 572.5 originX= 1918.88 originY= 2641.11 originZ= 869.7 h= -36 p= -0 r= 0 name= 0 />
			 width= 1018.3; height= 572.5; originX= 1918.88; originY= 2641.11; originZ= 869.7; h= -36; p= -0; r= 0 ;name= 0 ;
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




//			<Screen comment=RelColumn 4 row 0 column 5 width= 1018.3 height= 572.5 originX= 3104.81 originY= 1008.81 originZ= 869.7 h= -72 p= -0 r= 0 name= 0 />
			width= 1018.3 ;height= 572.5 ;originX= 3104.81; originY= 1008.81; originZ= 869.7; h= -72; p= -0; r= 0; name= 0 ;
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




//			<Screen comment=RelColumn 6 row 0 column 3 width= 1018.3 height= 572.5 originX= 3104.81 originY= -1008.81 originZ= 869.7 h= -108 p= -0 r= 0 name= 0 />
			width= 1018.3 ;height= 572.5; originX= 3104.81 ;originY= -1008.81; originZ= 869.7; h= -108; p= -0; r= 0 ;name= 0 ;
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




//			<Screen comment=RelColumn 8 row 0 column 1 width= 1018.3 height= 572.5 originX= 1918.88 originY= -2641.11 originZ= 869.7 h= -144 p= -0 r= 0 name= 0 />
			 width= 1018.3 ;height= 572.5 ;originX= 1918.88 ;originY= -2641.11 ;originZ= 869.7 ;h= -144; p= -0; r= 0 ;name= 0 ;
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
int ParticleDreams::loadInjFountsFrScr(float dx,float dy,float dz,float speed)
	{
		// in z back space
		// in meters
		int injNum;
		
		
		//h_injectorData[0][0][0] =0;
		//int firstInjNum = h_injectorData[0][0][0]+1 ;
		int firstInjNum = injectorGetMaxnumNum() +1;
		int i=0;
		while ( (_PhScAr[i].index) !=( -1)) 
			{
				injNum=firstInjNum +i;
				 //const char up = "z";
				//injectorSetDifaults(injNum);
				injectorSetType (injNum, 2);		
				injectorSetInjtRatio (injNum, 1);
				injectorSetPosition (injNum, _PhScAr[i].originX /1000.0 +dx, _PhScAr[i].originY /1000.0 +dy,  _PhScAr[i].originZ /1000.0 +dz,  axisUpZ);
				injectorSetVelocity (injNum, _PhScAr[i].vx * speed, _PhScAr[i].vy * speed,  _PhScAr[i].vz * speed,  axisUpZ);
				injectorSetSize (injNum, ftToM(0.25), ftToM(0.25), ftToM(0.25),  axisUpZ);
				injectorSetSpeedDist (injNum, 0, 0,  0,  axisUpZ);
				injectorSetSpeedJitter (injNum, 0.2, 0,  0,  axisUpZ);
				injectorSetSpeedCentrality (injNum, 5, 5,  5,  axisUpZ);

				
				i++;
			}  
			
		 injectorSetMaxnumNum(injNum);
		return injNum;
	}

int ParticleDreams::loadReflFrScr()

{
		// in z back space
		// in meters

   float screenrad = ftToM(5.0);
    int reflNum;
    float damping =.5;
    float no_traping =0;
    float firstRefNum = reflectorGetMaxnumNum(); + 1;
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
			  		 //reflectorGetMaxnumNum();
		 //reflectorSetMaxnumNum( maxNumber);
		 reflectorSetType ( reflNum , 1);// 0 os off, 1 is plain
		 //reflectorSetDifaults( reflNum );
		 reflectorSetPosition ( reflNum , xpos, ypos,  zpos, axisUpZ);
		 reflectorSetNormal( reflNum , vx, vy,  vz, axisUpZ);
		 reflectorSetSize ( reflNum , screenrad, axisUpZ);
		 reflectorSetDamping ( reflNum , damping);
		 reflectorSetNoTraping ( reflNum , no_traping);
	
	
			h_reflectorData[reflNum ][0][0]=1;h_reflectorData[reflNum ][0][1]=0;// type, age ie colormod, ~  0 is off 1 is plane reflector
			h_reflectorData[reflNum ][1][0]=xpos;    h_reflectorData[reflNum ][1][1]= ypos;h_reflectorData[reflNum ][1][2]=zpos;//x,y,z position
			h_reflectorData[reflNum ][2][0]= vx;  h_reflectorData[reflNum ][2][1]=vy;    h_reflectorData[reflNum ][2][2]=vz;//x,y,z normal
			h_reflectorData[reflNum ][3][0]=screenrad; h_reflectorData[reflNum ][3][1]=0.00; h_reflectorData[reflNum ][3][2]=0;//reflector radis ,~,~ 
			h_reflectorData[reflNum ][4][0]=0.000;h_reflectorData[reflNum ][4][1]=0.000;h_reflectorData[reflNum ][4][2]=0.000;//t,u,v jiter  not implemented = speed 
			h_reflectorData[reflNum ][5][0]= damping; h_reflectorData[reflNum ][5][1]=no_traping;  h_reflectorData[reflNum ][5][2]=0.0;//reflectiondamping , no_traping ~
			h_reflectorData[reflNum ][6][0]=0;    h_reflectorData[reflNum ][6][1]=0;    h_reflectorData[reflNum ][6][2]=0;// not implemented yet centrality of rnd distribution speed dt tu ~

			i++;

		}
  			 reflectorSetMaxnumNum( reflNum);

		
	return reflNum ;
}

void ParticleDreams::paint_on_walls_host()
{
    gravity = .1;
    gravity = 0.000001;
    //gravity = 0.0001;
    colorFreq =16;
    max_age = 2000;
    disappear_age =2000;
    alphaControl =0;
    static float time_in_sean;
    time_in_sean = time_in_sean + 1.0/targetFrameRate;

    if (paint_on_walls_start == 1)
    {
	size_t size = PDATA_ROW_SIZE * CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT * sizeof (float);
       _particleObject->resetPosition();
	//pdata_init_age( max_age);
	pdata_init_velocity(0, -0.005, 0);
	pdata_init_rand();
	//cuMemcpyHtoD(d_particleData, h_particleData, size);
	time_in_sean =0 * targetFrameRate;
	//::user->home();
	_refObjSwitchFace->setAllChildrenOff();
	_refObjSwitchLine->setAllChildrenOff();
	_injObjSwitchLine->setAllChildrenOn();
    reflectorSetMaxnumNum(0);//turn off all reflectors
    //h_injectorData[0][0][0] =0;// turn off all injectors ~ ~   ~ means dont care
	injectorSetMaxnumNum(0);// turn off all injectors
	if (DEBUG_PRINT >0);;
	printf("paint_on_walls_start \n");
	if (soundEnabled)
	{ambient_sound_start_paint_on_walls();
	//	dan_ambiance_2.setGain(1);
	//	dan_ambiance_2.play(1);
	}

	paint_on_wallsContextStart = 1;
    }
    else
    {
	paint_on_wallsContextStart = 0;
    }


    if (time_in_sean > paint_on_walls_timeInScene)nextSean=1;          
   // if (time_in_sean >5)nextSean=1;  // temp shorttime        

    anim = showFrameNo * .001;

    //draw_water_sky =0;
    setWater(false);
    if(hand_inject_or_reflect==1)	_injObjSwitchFace->setAllChildrenOn();
	
	if(hand_inject_or_reflect==0)	_injObjSwitchFace->setAllChildrenOff();
	

    int injNum ;	
    //h_injectorData[0][0][0] =1;// number of injectors ~ ~   ~ means dont care
    injectorSetMaxnumNum(1);
    //injector 1
    injNum =1;

 
    if (soundEnabled)injSoundUpdate(injNum);

    injNum =1;
	injectorSetType (injNum, 1);		
	injectorSetInjtRatio (injNum, hand_inject_or_reflect);
	injectorSetPosition (injNum, wandPos[0], wandPos[1],  wandPos[2],  axisUpZ);
	injectorSetVelocity (injNum, wandVec[0], wandVec[1], wandVec[2],  axisUpZ);
	injectorSetSize (injNum, 0, 0, 0,  axisUpZ);
	injectorSetSpeedDist (injNum, 0.01, 0.01,  0,  axisUpZ);
	injectorSetSpeedJitter (injNum, 0.1, 0.1,  0,  axisUpZ);
	injectorSetSpeedCentrality (injNum, 5, 5,  5,  axisUpZ);


    //if (but1){printf (" wandPos[0 ,1,2] wandVec[0,1,2] %f %f %f    %f %f %f \n", wandPos[0],wandPos[1],wandPos[2],wandVec[0],wandVec[1],wandVec[2]);}
    // load starcave wall reflectors
   // h_reflectorData[0][0][0] = loadStarcaveWalls(1);
    if (time_in_sean >5)
		{
			int tmp =load6wallcaveWalls(1);
			reflectorSetMaxnumNum(tmp);
			//h_reflectorData[0][0][0] =0;
 			//int numRef =loadReflFrScr();
			
		}

    paint_on_walls_start =0;
}

void ParticleDreams::sprial_fountens_host()
{
    //draw_water_sky =0;
    setWater(false);
// particalsysparamiteres--------------
    gravity = .00005;	
    max_age = 2000;
    disappear_age =2000;
    colorFreq =64 *max_age/2000.0 ;
    alphaControl =0;//turns alph to transparent beloy y=0
// reload  rnd < max_age in to pdata
    static float time_in_sean;
    time_in_sean = time_in_sean + 1.0/targetFrameRate;


    if (sprial_fountens_start == 1)
    {
        //size_t size = PDATA_ROW_SIZE * CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT * sizeof (float);
        //pdata_init_age( max_age);
        //pdata_init_velocity(-10000, -10000, -10000);
        //pdata_init_rand();
        //cuMemcpyHtoD(d_particleData, h_particleData, size);
        _particleObject->resetPosition();
		_injObjSwitchFace->setAllChildrenOff();
		_injObjSwitchLine->setAllChildrenOff();
		_refObjSwitchLine->setAllChildrenOn();
    reflectorSetMaxnumNum(0);//turn off all reflectors
   // h_injectorData[0][0][0] =0;// turn off all injectors ~ ~   ~ means dont care
		injectorSetMaxnumNum(0);// turn off all injectors
        time_in_sean =0 * targetFrameRate;
        //::user->home();
        //printf( "in start sean3 \n");
        if (DEBUG_PRINT >0);;
        printf("sprial_fountens__start \n");
        if (soundEnabled)
        {ambient_sound_start_sprial_fountens();
        //	dan_5min_ostinato.setGain(0.5);
        //	dan_5min_ostinato.play(1);
        }
            
        //printf( " time %f \n", ::user->get_t());
	sprial_fountensContextStart = 1;
    }
    else
    {
	sprial_fountensContextStart = 0;
    }

//printf ("time_in_sean 1 %f\n",time_in_sean);
    if (time_in_sean >sprial_fountens_timeInScene)nextSean=1;
    //if (time_in_sean >10)nextSean=1;// temp shorttime
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
    
    				injectorSetType (injNum, 2);		
				injectorSetInjtRatio (injNum, 1);
				injectorSetPosition (injNum, 0 , ftToM(0.1),  0,  axisUpZ);
				injectorSetVelocity (injNum, 0.02 * (sin(time_in_sean*2*M_PI)), 0.5,  0.02 * (cos(time_in_sean*2*M_PI)),  axisUpZ);
				injectorSetSize (injNum, 0.03, 0.03, 0.03,  axisUpZ);
				injectorSetSpeedDist (injNum, 0, 0,  0,  axisUpZ);
				injectorSetSpeedJitter (injNum, 0.03, 0,  0,  axisUpZ);
				injectorSetSpeedCentrality (injNum, 5, 5,  5,  axisUpZ);


    int reflect_on;
    reflect_on =0;
    float speed= 1.0/30; //one rotation every 
    speed = 1.0/45;	
    float stime =0 ;
    float t,rs,as,mr=9.0;
	
    int numobj =5;
 //   h_injectorData[0][0][0] =numobj;
	injectorSetMaxnumNum(numobj);



    for(int i=1;i<=numobj;i++)
    {
        if (time_in_sean > 1.0/speed)reflect_on =1;	
        if (time_in_sean > 1.5/speed)
        {
            reflect_on =1;
			injectorSetVelocity (injNum,0, 0.5,  0,  axisUpZ);
            //h_injectorData[injNum][3][0]= 0;h_injectorData[injNum][3][1]=.5;h_injectorData[injNum][3][2]=0.0;//x,y,z velocity
            if (time_in_sean > 2.0/speed)
            {
                //h_injectorData[injNum][3][0]= -ftToM(rs*sin(as))/8 ;h_injectorData[injNum][3][1]=.5;h_injectorData[injNum][3][2]=-ftToM(rs*sin(as))/8;//x,y,z velocity
                //h_injectorData[injNum][6][0]= .026;h_injectorData[injNum][6][1]=0.0;h_injectorData[injNum][6][2]=0.0;//speed jiter ~ ~
            		injectorSetVelocity (injNum,-ftToM(rs*sin(as))/8, 0.5,  -ftToM(rs*sin(as))/8,  axisUpZ);
    				injectorSetSpeedJitter (injNum, 0.026, 0,  0,  axisUpZ);
                
                           
            }
        }	 	
        copy_injector(1, i);
        stime= i*(1/speed/numobj);
        if (time_in_sean >stime)
        {
            injNum =i; t = time_in_sean - stime;rs =2*speed*(2*M_PI)*t;as =speed*(2*M_PI)*t;if (rs >mr)rs=mr;if (rs < 2)rs = 2;
            //rs =mr;
            //h_injectorData[injNum][2][0]=ftToM(rs*sin(as));h_injectorData[injNum][2][2]=-ftToM(rs*cos(as))  ;
			injectorSetPosition (injNum, ftToM(rs*sin(as)),ftToM(0.1),  -ftToM(rs*cos(as)),  axisUpZ);
					
        }		
    }	

   // h_reflectorData[0][0][0] =2;// number of reflectors ~ ~   ~ means dont care
	reflectorSetMaxnumNum(2);
	
    int reflNum = 1;
    
       		 //reflectorGetMaxnumNum();
		 //reflectorSetMaxnumNum( maxNumber);
		 reflectorSetType ( reflNum , reflect_on);// 0 os off, 1 is plain
		 //reflectorSetDifaults( reflNum );
		 reflectorSetPosition ( reflNum , 0, 0,  0, axisUpZ);
		 reflectorSetNormal( reflNum , 0, 1,  0, axisUpZ);
		 reflectorSetSize ( reflNum , ftToM(10.00), axisUpZ);
		 reflectorSetDamping ( reflNum ,  0.4);
		 reflectorSetNoTraping ( reflNum , 0);
	
 
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
  if(hand_inject_or_reflect==1) _refObjSwitchFace->setAllChildrenOn();
   if(hand_inject_or_reflect==0)_refObjSwitchFace->setAllChildrenOff();
 //  std::cout << " Wand x,y,z,dx,dy,dz " << " " <<x<< " " <<y<< " " <<z<< " " <<dx<< " " <<dy<< " " <<dz << std::endl;

      		 //reflectorGetMaxnumNum();
		 //reflectorSetMaxnumNum( maxNumber);
		 reflectorSetDifaults( reflNum );

		 reflectorSetType ( reflNum , hand_inject_or_reflect);// 0 os off, 1 is plain
		// std::cout << " hand_inject_or_reflect " << hand_inject_or_reflect << std::endl;
		 reflectorSetPosition ( reflNum , x,y,x, axisUpZ);
		 reflectorSetNormal( reflNum , dx,dy,dz, axisUpZ);
		 reflectorSetSize ( reflNum , ftToM(1.5), axisUpZ);
		 reflectorSetDamping ( reflNum , 1);
		 reflectorSetNoTraping ( reflNum , 1);
	
    if (soundEnabled)reflSoundUpdate( reflNum);
 
 //   HeadReflectorsMake();

    sprial_fountens_start =0;
}

void ParticleDreams::N_waterfalls_host()
{
    //4 waterFalls
    // waterFallsFrom screenes

    //draw_water_sky =0;
    setWater(false);
    // particalsysparamiteres--------------
    gravity = .005;	
    gravity = .001;	
    max_age = 2000;
    disappear_age =2000;
    colorFreq =128 *max_age/2000.0 ;
    alphaControl =0;//turns alph to transparent beloy y=0
    static float time_in_sean;
    time_in_sean = time_in_sean + 1.0/targetFrameRate;

    // reload  rnd < max_age in to pdata

    if (N_waterfalls_start== 1)
    {
	   resetSlidePosition();

    _particleObject->resetPosition();
  // resetSlidePosition();
     
	//size_t size = PDATA_ROW_SIZE * CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT * sizeof (float);
	//pdata_init_age( max_age);
	//pdata_init_velocity(-10000, -10000, -10000);
	//pdata_init_rand();
   std::cout << " N_waterfalls_start " << std::endl;
	//cuMemcpyHtoD(d_particleData, h_particleData, size);
	_injObjSwitchFace->setAllChildrenOff();
	_injObjSwitchLine->setAllChildrenOff();
	_refObjSwitchLine->setAllChildrenOn();
    reflectorSetMaxnumNum(0);;//turn off all reflectors
   // h_injectorData[0][0][0] =0;// turn off all injectors ~ ~   ~ means dont care
	injectorSetMaxnumNum(0);// turn off all injectors



	if (DEBUG_PRINT >0)printf( "in start sean2 \n");
	time_in_sean =0 * targetFrameRate;
	//::user->home();
	if (DEBUG_PRINT >0)printf("paint_on_walls_start \n");

	if (soundEnabled)
	{ambient_sound_start_N_waterfalls();
	//	harmonicAlgorithm.setGain(1);
	//	harmonicAlgorithm.play(1);
	}

	//printf( " time %f \n", ::user->get_t());

	N_waterfallsContextStart = 1;
    }
    else
    {
	N_waterfallsContextStart = 0;
    }
_refObjSwitchFace->setAllChildrenOff();


    if (_TargetSystem.compare("Wave") != 0)// dont display tidle screen in Wave
    {
	    if ((time_in_sean > 0) && (time_in_sean < tidleSlideTime/2)){ _EdSecSwitchSlid1->setAllChildrenOn();}//slideStuff
	    else { _EdSecSwitchSlid1->setAllChildrenOff();}
	
	    if ((time_in_sean > tidleSlideTime/2) && (time_in_sean <tidleSlideTime)) _EdSecSwitchSlid2->setAllChildrenOn();//slideStuff
	    else _EdSecSwitchSlid2->setAllChildrenOff();
    }
	//if ((time_in_sean >10) && (time_in_sean <15)) _EdSecSwitchSlid3->setAllChildrenOn();//slideStuff
	//else _EdSecSwitchSlid3->setAllChildrenOff();

    if (time_in_sean > n_waterfalls_timeInScene)nextSean=1;
 //    if ((_TargetSystem.compare("Wave") == 0) || (_TargetSystem.compare("Cave2") == 0)) {   if(time_in_sean >110)nextSean=1;}
    //if (time_in_sean >8)nextSean=1;//short time
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


   // h_injectorData[0][0][0] =0;// number of injectors ~ ~   ~ means dont care
    injectorSetMaxnumNum(0); //turn all injt off
//
	float speed = 0.2;
	float YdispFonts = .5;
	if (_TargetSystem.compare("StarCave") == 0){speed = 0.1;YdispFonts = 0;}
	if (_TargetSystem.compare("Cave2") == 0)YdispFonts = 0;
	//if (_TargetSystem.compare("Wave") == 0){YdispFonts = 0;speed = 0.1;}
	loadInjFountsFrScr(0 ,YdispFonts,0,speed);// axisyup
 
	
	
    reflectorSetMaxnumNum(1);// number of reflectors ~ ~   ~ means dont care
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

//   if(hand_inject_or_reflect==1)_reflectorObjSwitch->setAllChildrenOn();
//   if(hand_inject_or_reflect==0)_reflectorObjSwitch->setAllChildrenOff();
// paddel reflector
	
   if(hand_inject_or_reflect==1)_refObjSwitchFace->setAllChildrenOn();
   if(hand_inject_or_reflect==0)_refObjSwitchFace->setAllChildrenOff();

  		 //reflectorGetMaxnumNum();
		 //reflectorSetMaxnumNum( maxNumber);
		 reflectorSetType ( reflNum , hand_inject_or_reflect);// 0 os off, 1 is plain
		 //reflectorSetDifaults( reflNum );
		 reflectorSetPosition ( reflNum ,x,y,z, axisUpZ);
		// std::cout << " Wand x,y,z,dx,dy,dz " << " " <<x<< " " <<y<< " " <<z<< " " <<dx<< " " <<dy<< " " <<dz << std::endl;
		 reflectorSetNormal( reflNum , dx,dy,dz, axisUpZ);
		 reflectorSetSize ( reflNum , ftToM(0.5), axisUpZ);
		 reflectorSetDamping ( reflNum , 1);
		 reflectorSetNoTraping ( reflNum , 1);
	

	int reflectOn;
	if (time_in_sean > n_waterfalls_timeInScene/10)
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
// rotating floor reflector
	  	reflectorSetMaxnumNum(reflectorGetMaxnumNum()+ 1);;// number of reflectors ~ ~   ~ means dont care
		reflNum = reflectorGetMaxnumNum();
		
		  		 //reflectorGetMaxnumNum();
		 //reflectorSetMaxnumNum( maxNumber);
		 reflectorSetType ( reflNum , reflectOn);// 0 os off, 1 is plain
		 //reflectorSetDifaults( reflNum );
		 reflectorSetPosition ( reflNum , 0, 0,  0, axisUpZ);
		 reflectorSetNormal( reflNum , dvx, 1 + dvy,   -1 +dvz, axisUpZ);
		 reflectorSetSize ( reflNum , 1, axisUpZ);
		 reflectorSetDamping ( reflNum , 1);
		 reflectorSetNoTraping ( reflNum , 1);
	

//large floor reflector
		  	reflectorSetMaxnumNum( reflectorGetMaxnumNum()+ 1);// number of reflectors ~ ~   ~ means dont care
		reflNum = reflectorGetMaxnumNum();
		
	  		 //reflectorGetMaxnumNum();
		 //reflectorSetMaxnumNum( maxNumber);
		 reflectorSetType ( reflNum , reflectOn);// 0 os off, 1 is plain
		 //reflectorSetDifaults( reflNum );
		 reflectorSetPosition ( reflNum ,0, 0,  0, axisUpZ);
		 reflectorSetNormal( reflNum , 0, 1.0,  0, axisUpZ);
		 reflectorSetSize ( reflNum , 30, axisUpZ);
		 reflectorSetDamping ( reflNum , 1.0);
		 reflectorSetNoTraping ( reflNum , 1);
		
//persone reflectorA
//		HeadReflectorsMake();


 
    N_waterfalls_start=0;
}

void ParticleDreams::painting_skys_host()
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
    static float time_in_Watersean;
 
    static float rotRate;
	float last_time_in_scene ;
    //time_in_sean = time_in_sean + 1.0/targetFrameRate;
    float maxTime_in_sean = painting_skys_timeInScene;//was180
    float begin_Time_in_Watersean = painting_skys_timeInScene/3.0;
     

      
  

    // reload  rnd < max_age in to pdata

    if (painting_skys_start == 1)
    {
		size_t size = PDATA_ROW_SIZE * CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT * sizeof (float);
		//pdata_init_age( max_age);
		//pdata_init_velocity(-10000, -10000, -10000);
		pdata_init_rand();
		_particleObject->resetPosition();

		_refObjSwitchFace->setAllChildrenOff();
		_refObjSwitchLine->setAllChildrenOff();
		_injObjSwitchLine->setAllChildrenOn();
    reflectorSetMaxnumNum( 0);;//turn off all reflectors
    //h_injectorData[0][0][0] =0;// turn off all injectors ~ ~   ~ means dont care
	injectorSetMaxnumNum(0);// turn off all injectors
		///cuMemcpyHtoD(d_particleData, h_particleData, size);
		if (DEBUG_PRINT >0);;
		printf( "painting_skys_start \n");
		time_in_sean =0 * targetFrameRate;
		time_in_Watersean = 0 * targetFrameRate;
		//::user->home();
		rotRate = .05;
		///::user->set_t(86400);// set to 00:00 monday  old
		///::user->set_t(1);// set to 0 days 1 sec
		///::user->set_t(63711);// set to 0 days 1 sec
		_skyTime = 63711.0;
		_skyTime = 80000;

		//::user->time(43200/2.0);// set to dawn  old
		//::user->set_t(43200/2.0);

		//::user->pass(43200/2.0);
		//printf( " time %f \n", ::user->get_t());
        



		if (soundEnabled)
		{ambient_sound_start_painting_skys();
		//	rain_at_sea.setLoop(true);
		#ifdef OAS_SOUND
		//TODOSound		
			texture_12.fade(0,1);// kludge to stop sound from injector on skip to end
		#endif
		#ifdef OMICRON_SOUND
			_texture_12Instance->fade(0,1);// kludge to stop sound from injector on skip to end
		#endif
		//	rain_at_sea.setGain(1);
		//	rain_at_sea.play(1);

		//	texture_17_swirls3.setLoop(true);
		//	texture_17_swirls3.setGain(1);
		//	texture_17_swirls3.play(1);
		}
		painting_skysContextStart = 1;
    }
    else
    {
	painting_skysContextStart = 0;
    }
    //printf ("time_in_sean 3 %f\n",time_in_sean);

    if (time_in_sean >maxTime_in_sean) nextSean=1;


    if (DEBUG_PRINT >0)printf( "in sean3 \n");
    // printf ( "seantime time %f %f\n",time_in_sean,::user->get_t());
    //lerp(in, beginIN, endIn, beginOut, endOut)
	last_time_in_scene = time_in_sean;
    time_in_sean = time_in_sean + 1.0/targetFrameRate;
    time_in_Watersean = time_in_sean -begin_Time_in_Watersean;
	if ((last_time_in_scene <= begin_Time_in_Watersean) && (time_in_sean >= begin_Time_in_Watersean)){setWater(true);}
    if((time_in_Watersean >= 0)&& (time_in_Watersean <= 10)) _skyTime = lerp(time_in_Watersean, 0, 10, 80000, 70000);
    if((time_in_Watersean > 30)&& (time_in_Watersean <= 120)) _skyTime = lerp(time_in_Watersean, 30, 90, 70000, 18000);//was 17000
    //if((time_in_sean > 30)&& (time_in_sean <= 40)) user->set_t(lerp(time_in_sean, 30, 40, 74000, 110000));
    //     ::user->set_t(107000);// set to 0 days 1 sec

//if(time_in_Watersean > 0) { std::cout << "_skyTime = " << _skyTime << " " <<time_in_Watersean <<"\n";}
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

	if(hand_inject_or_reflect==1)	_injObjSwitchFace->setAllChildrenOn();

	if(hand_inject_or_reflect==0)	_injObjSwitchFace->setAllChildrenOff();

    int injNum ;	
   // h_injectorData[0][0][0] =2;// number of injectors ~ ~   ~ means dont care
    injectorSetMaxnumNum(2);
    //injector 1
    injNum =1;
    		
				 //const char up = "z";
				//injectorSetDifaults(injNum);
				injectorSetType (injNum, 1);		
				injectorSetInjtRatio (injNum, 1);
				injectorSetPosition (injNum, 0, ftToM(8.00),  0,  axisUpZ);
				injectorSetVelocity (injNum, 0.02 * (sin(anim)), 0,  0.02 * (cos(anim)),  axisUpZ);
				injectorSetSize (injNum, 0, 0, 0,  axisUpZ);
				injectorSetSpeedDist (injNum, 0.001, 0.001,  0,  axisUpZ);
				injectorSetSpeedJitter (injNum, 1.1, 0,  0,  axisUpZ);
				injectorSetSpeedCentrality (injNum, 5, 5,  5,  axisUpZ);

    if (soundEnabled && ENABLE_SOUND_POS_UPDATES)
    {
    texture_swirls_setPosition(injNum);
    //	texture_17_swirls3.setPosition(3 * h_injectorData[injNum][3][0], 0, -3 * h_injectorData[injNum][3][2]);
    }

    //
    // injector 2
    injNum =2; //over written below
    				injectorSetType (injNum, 1);		
				injectorSetInjtRatio (injNum, 1);
				injectorSetPosition (injNum, 0, 2.5,  0,  axisUpZ);
				injectorSetVelocity (injNum, 0, 1,  0,  axisUpZ);
				injectorSetSize (injNum, 0.25, 0.25, 0,  axisUpZ);
				injectorSetSpeedDist (injNum, 0.0, 0.0,  0,  axisUpZ);
				injectorSetSpeedJitter (injNum, 0, 0.5,  0,  axisUpZ);
				injectorSetSpeedCentrality (injNum, 5, 5,  5,  axisUpZ);

// floor reflector


				injectorSetType (injNum, 1);		
				injectorSetInjtRatio (injNum, hand_inject_or_reflect*4.0);
				injectorSetPosition (injNum, wandPos[0], wandPos[1],  wandPos[2],  axisUpZ);
				injectorSetVelocity (injNum, wandVec[0], wandVec[1],  wandVec[2],  axisUpZ);
				injectorSetSize (injNum, 0, 0, 0,  axisUpZ);
				injectorSetSpeedDist (injNum, 0.01, 0.01,  0,  axisUpZ);
				injectorSetSpeedJitter (injNum, 0.1, 0,  0,  axisUpZ);
				injectorSetSpeedCentrality (injNum, 5, 5,  5,  axisUpZ);

    //
    if (hand_inject_or_reflect){if (DEBUG_PRINT >0) {printf(" wandPos[0 ,1,2] wandVec[0,1,2] %f %f %f    %f %f %f \n", wandPos[0],wandPos[1],wandPos[2],wandVec[0],wandVec[1],wandVec[2]);}}
    if (soundEnabled)injSoundUpdate(injNum);


    painting_skys_start =0;
}

void ParticleDreams::rain_host()
{
    //educational stub fro firtherDev

    //draw_water_sky =0;
    setWater(false);
    // particalsysparamiteres--------------
    //std::cerr << "Gravity: " << _gravityRV->getValue() << std::endl;
    gravity = .01;	
    gravity = .003;	
	  max_age = 2000;
    disappear_age =2000;
    colorFreq =64 *max_age/2000.0 ;
    alphaControl =0;//turns alph to transparent beloy y=0
    static float time_in_sean;
    time_in_sean = time_in_sean + 1.0/targetFrameRate;
	static float in_rot_time = 0;

    // reload  rnd < max_age in to pdata

    if (rain_start == 1)
		{
         _particleObject->resetPosition();
		_injObjSwitchFace->setAllChildrenOff();
		_injObjSwitchLine->setAllChildrenOff();
		_refObjSwitchLine->setAllChildrenOn();
		reflectorSetMaxnumNum( 0);//turn off all reflectorsstatic oldGravity =0; 
		//h_injectorData[0][0][0] =0;// turn off all injectors ~ ~   ~ means dont care
		injectorSetMaxnumNum(0);// turn off all injectors
#ifdef OAS_SOUND
		//TODOSound		
		texture_12.fade(0, 1);// cludge to stop sound from injector on skip to end
#endif
#ifdef OMICRON_SOUND
		_texture_12Instance->fade(0, 1);// cludge to stop sound from injector on skip to end


#endif
		in_rot_time = 10;

/*		_EdSecSwitchAxis->setAllChildrenOn();
*/	
		if (DEBUG_PRINT >0);;
		printf( "rain_start \n");
		time_in_sean =0 * targetFrameRate;

		if (soundEnabled)
		{ambient_sound_start_rain();
		//	dan_rain_at_sea_loop.setLoop(1);
		//	dan_rain_at_sea_loop.setGain(1);

		//	dan_rain_at_sea_loop.play();
		}
		//printf( " time %f \n", ::user->get_t());
		rainContextStart = 1;
		}
		else
		{
		rainContextStart = 0;
		}
		
    float maxTime_in_sean = 100;
 
    if (time_in_sean >maxTime_in_sean) nextSean=1;
    float fract_of_max_time_in_sean =time_in_sean/maxTime_in_sean;
    // creates seanCnangeTo varable 
    
    
   
    
   if (DEBUG_PRINT >0)printf( "in sean4 \n");

    gravity = _gravityRV->getValue();
	//std::cout << " sin refl grav " << _rotateInjCB->getValue() << " " <<  _reflectorCB->getValue() << " " << _gravityRV->getValue() << "\n";
	



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
    //h_reflectorData[0][0][0] =1;// turn off all reflectors ???????

    //	 injector data 
    int injNum ;	
 
    //h_injectorData[0][0][0] =1;// number of injectors 
	injectorSetMaxnumNum(1);
    // injector 1
	float speed =.2;
	speed = _speedRV->getValue();
	float xvos,zvos;
	float rotTime = 15;
	if (_rotateInjCB->getValue()){in_rot_time = in_rot_time + 1.0/targetFrameRate;}
			xvos = (sin(in_rot_time*2*M_PI/rotTime))*speed;
			zvos = (cos(in_rot_time*2*M_PI/rotTime))*speed;

		
	//std::cout << " speed " << speed << std::endl;
    injNum =1;	
    				injectorSetType (injNum, 2);		
				injectorSetInjtRatio (injNum, 1);
				injectorSetPosition (injNum, 0, 2,  -1.5,  axisUpZ);
				injectorSetVelocity (injNum, xvos, 0, zvos,  axisUpZ);
				injectorSetSize (injNum, ftToM(0.25), ftToM(0.25), ftToM(0.25),  axisUpZ);
				injectorSetSpeedDist (injNum, 0, 0,  0,  axisUpZ);
				injectorSetSpeedJitter (injNum, 0.2, 0,  0,  axisUpZ);
				injectorSetSpeedCentrality (injNum, 2, 2,  2,  axisUpZ);


 

    //h_reflectorData[0][0][0] =1;// number of reflectors ~ ~   ~ means dont care
    reflectorSetMaxnumNum( 1);
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
//   if(hand_inject_or_reflect==1)_reflectorObjSwitch->setAllChildrenOn();
//   if(hand_inject_or_reflect==0)_reflectorObjSwitch->setAllChildrenOff();
   if(hand_inject_or_reflect==1)_refObjSwitchFace->setAllChildrenOn();
   if(hand_inject_or_reflect==0)_refObjSwitchFace->setAllChildrenOff();

   		 //reflectorGetMaxnumNum();
		 //reflectorSetMaxnumNum( maxNumber);
		 reflectorSetType ( reflNum , hand_inject_or_reflect);// 0 os off, 1 is plain
		 //reflectorSetDifaults( reflNum );
		 reflectorSetPosition ( reflNum , x,y,z, axisUpZ);
		 reflectorSetNormal( reflNum , dx,dy,dz, axisUpZ);
		 reflectorSetSize ( reflNum , ftToM(0.5), axisUpZ);
		 reflectorSetDamping ( reflNum , 1);
		 reflectorSetNoTraping ( reflNum , 1);
	


    if (soundEnabled) reflSoundUpdate(reflNum);


//florr reflector
    reflNum = 2;
	int floorReflectOn;
	
	if ((time_in_sean > 5) && _reflectorCB->getValue())	{floorReflectOn =1;}
	else					{floorReflectOn =0;}
    //h_reflectorData[0][0][0] =reflNum;// number of reflectors ~ 1.0/targetFrameRate;~   ~ means dont care
    reflectorSetMaxnumNum( reflNum);
       		 //reflectorGetMaxnumNum();
		 //reflectorSetMaxnumNum( maxNumber);
		 reflectorSetType ( reflNum , floorReflectOn);// 0 os off, 1 is plain
		 //reflectorSetDifaults( reflNum );
		 reflectorSetPosition ( reflNum , 0, 0,  0, axisUpZ);
		 reflectorSetNormal( reflNum , 0, 1,  0, axisUpZ);
		 reflectorSetSize ( reflNum , 10, axisUpZ);
		 reflectorSetDamping ( reflNum , 1);
		 reflectorSetNoTraping ( reflNum , 1);
	
    



    rain_start =0;
}

void ParticleDreams::paint_on_walls_context(int contextid) const
{
    if(paint_on_wallsContextStart)
    {
	size_t size = PDATA_ROW_SIZE * CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT * sizeof (float);
	cuMemcpyHtoD(d_particleDataMap[contextid], h_particleData, size);
    }

}

void ParticleDreams::sprial_fountens_context(int contextid) const
{
    if(sprial_fountensContextStart)
    {
    }
}

void ParticleDreams::N_waterfalls_context(int contextid) const
{
    if(N_waterfallsContextStart)
    {
    }
}

void ParticleDreams::painting_skys_context(int contextid) const
{
    if(painting_skysContextStart)
    {
	size_t size = PDATA_ROW_SIZE * CUDA_MESH_WIDTH * CUDA_MESH_HEIGHT * sizeof (float);
	cuMemcpyHtoD(d_particleDataMap[contextid], h_particleData, size);

    }
}

void ParticleDreams::rain_context(int contextid) const
{
    if(rainContextStart)
    {
    }
}

void ParticleDreams::paint_on_walls__clean_up()
{
	//h_reflectorData[0][0][0] = 0;
	reflectorSetMaxnumNum( 0);
	//h_injectorData[0][0][0] = 0;
	injectorSetMaxnumNum(0);
	if (soundEnabled)
	{sound_stop_paint_on_walls();
	//	dan_ambiance_2.fade(0, 1.5);
	//	dan_10122606_sound_spray.setGain(0);
	}
}

void ParticleDreams::sprial_fountens__clean_up()
{
	//h_reflectorData[0][0][0] = 0;
	reflectorSetMaxnumNum( 0);
	
	injectorSetMaxnumNum(0);
	std::cout << "sprial_fountens__clean_up() " << std::endl;
	if (soundEnabled)
	{sound_stop_sprial_fountens();
	//	dan_5min_ostinato.fade(0, 1.50);
	//	dan_10122606_sound_spray.setGain(0);
	}
}


void ParticleDreams::N_waterfalls__clean_up()
{
	//h_reflectorData[0][0][0] = 0;
		reflectorSetMaxnumNum( 0);

	injectorSetMaxnumNum(0);
	std::cout << "N_waterfalls__clean_up() " << std::endl;
	_EdSecSwitchSlid1->setAllChildrenOff();
	_EdSecSwitchSlid2->setAllChildrenOff();
	_EdSecSwitchSlid3->setAllChildrenOff();

	if (soundEnabled)
	{sound_stop_N_waterfalls();
	//	harmonicAlgorithm.fade(0, 1.50);
	//	dan_10122606_sound_spray.setGain(0);
	}
}

void ParticleDreams::painting_skys__clean_up()
{
	//h_reflectorData[0][0][0] = 0;
	//h_injectorData[0][0][0] = 0;
	reflectorSetMaxnumNum( 0);
	injectorSetMaxnumNum(0);

	std::cout << "painting_skys__clean_up() " << std::endl;
	if (soundEnabled)
	{sound_stop_painting_skys();
	//	rain_at_sea.fade(0, 1.50);
	//	texture_17_swirls3.fade(0, 1.50);
	//	dan_10122606_sound_spray.setGain(0);

	//	dan_10120600_rezS3_rez2.setGain(0);
	}
}

void ParticleDreams::rain__clean_up()
{
	//h_reflectorData[0][0][0] = 0;
	reflectorSetMaxnumNum( 0);
	injectorSetMaxnumNum(0);

	
	std::cout << "rain__clean_up() " << std::endl;

	if (soundEnabled)
	{sound_stop_rain();
	//	dan_rain_at_sea_loop.fade(0, 1.50);
	//	dan_10122606_sound_spray.setGain(0);
	}
}

void ParticleDreams::HeadReflectorsMake()

	{
//persone reflector
	int reflNum;
			//std::cout << "reflectorA reflectorGetMaxnumNum " << reflectorGetMaxnumNum() << std::endl;
		  	reflectorSetMaxnumNum( reflectorGetMaxnumNum() + 1);// number of reflectors ~ ~   ~ means dont care
		reflNum = reflectorGetMaxnumNum();
    float head_x = headPos[0];
    float head_y = headPos[1];
    float head_z = headPos[2];
    float head_dx = headVec[0]/2;
    float head_dy = headVec[1]/2;
    float head_dz = headVec[2]/2;
    
    float dx = headMat[4];
    float dy = headMat[5];
    float dz = headMat[6];
		
	  		 //reflectorGetMaxnumNum();
		 //reflectorSetMaxnumNum( maxNumber);
		 reflectorSetType ( reflNum , 1);// 0 os off, 1 is plain
		 //reflectorSetDifaults( reflNum );
		 reflectorSetPosition ( reflNum ,head_x, head_y,head_z, axisUpZ);
		 reflectorSetNormal( reflNum , 0, 1.0,  0, axisUpZ);
		//std::cout << "head_x, head_y,head_z " << head_x << " " << head_y<< " " << head_z << std::endl;
		 reflectorSetSize ( reflNum , 0.5, axisUpZ);
		 reflectorSetDamping ( reflNum , 1.0);
		 reflectorSetNoTraping ( reflNum , 1);
	   if (soundEnabled)reflSoundUpdate(reflNum);


	
	
	
	
	}
float ParticleDreams::reflSoundUpdate(int reflNum)
	{
		static float OldReflectOn =0;
		float ReflectOn = h_reflectorData[reflNum ][0][0];

		float x,y,z;
		
	    	if (ReflectOn >0)
			{
				float newGain = _refl_hits[reflNum]/500.0;
				//if (newGain >0) printf ("reflHits %f\n",newGain);;
				if (newGain > 0.5) newGain = 0.5;
#ifdef OAS_SOUND
				dan_10122606_sound_spray.setGain(newGain);
				x = h_reflectorData[reflNum ][1][0];   y= h_reflectorData[reflNum ][1][1];z = h_reflectorData[reflNum ][1][2];
				if (ENABLE_SOUND_POS_UPDATES) dan_10122606_sound_spray.setPosition(x*4, -z*0, y*0);//convert from z bzck to z up
#endif
#ifdef OMICRON_SOUND
				//ToDoSound
				//std:cout << " newGain " << newGain << " \n";
				_dan_10122606_sound_sprayInstance->setVolume(newGain);
				x = h_reflectorData[reflNum ][1][0];   y= h_reflectorData[reflNum ][1][1];z = h_reflectorData[reflNum ][1][2];
				omicron::Vector3f pos;
				pos[0] =x;pos[1] =y;pos[2]=z;
				if (ENABLE_SOUND_POS_UPDATES) _dan_10122606_sound_sprayInstance->setPosition(pos);

#endif


			}

    		if ((ReflectOn == 0) && (OldReflectOn > 0))
			{
#ifdef OAS_SOUND		
 		   		dan_10122606_sound_spray.setGain(0);
#endif
 	#ifdef OMICRON_SOUND
 	//ToDoSound
 	 _dan_10122606_sound_sprayInstance->setVolume(0);

	 #endif

    		}
	OldReflectOn = ReflectOn;
	return 	_refl_hits[reflNum];

    }
	


//    h_injectorData[injNum][1][0]=2;h_injectorData[injNum][1][1]=1.0;// type, injection ratio ie streem volume, ~
//    h_injectorData[injNum][2][0]=0;h_injectorData[injNum][2][1]=ftToM(10);h_injectorData[injNum][2][2]=ftToM(-2);//x,y,z position


void	ParticleDreams::injSoundUpdate(int injNum)

    {
        	#ifdef OAS_SOUND

		static float OldInjOn =0;
		float injOn = h_injectorData[injNum][1][1];
		float x,y,z;
			x = h_injectorData[injNum][2][0];
			y = h_injectorData[injNum][2][1];

			z = h_injectorData[injNum][2][2];
			y =0;z =0;
			x = x*4;
 		osg::Vec3f pos;
		pos[0] = x;
		pos[1] = y;
		pos[2] = z;
		pos = pos - headPos;
		//if (hand_inject_or_reflect) std::cout << "pos - headPos " << pos <<"\n";
	
		static int roundRobenSound = 0;
		int roundRobenMod =1;

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
#endif
	
	#ifdef OMICRON_SOUND
		static float OldInjOn =0;
		float injOn = h_injectorData[injNum][1][1];
		float x,y,z;
		x = h_injectorData[injNum][2][0];
		y = h_injectorData[injNum][2][1];

		z = h_injectorData[injNum][2][2];
			//y =0;z =0;
			//x = x*4;
 		omicron::Vector3f pos;
		//pos[0] = x - headPos[0];
		//pos[1] = y - headPos[1];
		//pos[2] = z - headPos[2];
		//pos[0] = x ;
		//pos[1] = y ;
		//pos[2] = z ;
		
		
		//if(hand_inject_or_reflect) {std::cout << "pos - headPos " << pos <<"\n";}
		static int roundRobenSound = 0;
		int roundRobenMod =5;

	    	if (OldInjOn == 0 && (injOn > 0))
	    	{
				roundRobenSound = roundRobenSound +1;
				roundRobenSound = roundRobenSound%roundRobenMod;
				if (ENABLE_SOUND_POS_UPDATES)
					{
						if (roundRobenSound ==0 ){_short_sound_01aInstance->setPosition(pos);_short_sound_01aInstance->play();}
						if (roundRobenSound ==1 ){_short_sound_01a1Instance->setPosition(pos);_short_sound_01a1Instance->play();}
						if (roundRobenSound ==2 ){_short_sound_01a2Instance->setPosition(pos);_short_sound_01a2Instance->play();}
						if (roundRobenSound ==3 ){_short_sound_01a3Instance->setPosition(pos);_short_sound_01a3Instance->play();}
						if (roundRobenSound ==4 ){_short_sound_01a4Instance->setPosition(pos);_short_sound_01a4Instance->play();}
					}

				if (! ENABLE_SOUND_POS_UPDATES)
					{
						if (roundRobenSound ==0 ){_short_sound_01aInstance->play();}
						if (roundRobenSound ==1 ){_short_sound_01a1Instance->play();}
						if (roundRobenSound ==2 ){_short_sound_01a2Instance->play();}
						if (roundRobenSound ==3 ){_short_sound_01a3Instance->play();}
						if (roundRobenSound ==4 ){_short_sound_01a4Instance->play();}
					}


	    		_texture_12Instance->fade(4, 1);
		
	    	}
		
    	if (OldInjOn > 0 && injOn == 0)
    	{
    		_texture_12Instance->fade(0, 1);
    	}
		if 	((injOn > 0) && (ENABLE_SOUND_POS_UPDATES))

		{
   		_texture_12Instance->setPosition(pos);
    		//short_sound_01a.setPosition(x, -z, y);

		}
	OldInjOn = injOn;
		
	 #endif


	}
	

void ParticleDreams::ambient_sound_start_paint_on_walls()
	{
	#ifdef OAS_SOUND
		dan_ambiance_2.setGain(1);
		dan_ambiance_2.play(1);
	#endif	
	#ifdef OMICRON_SOUND
		_dan_ambiance_2Instance->setVolume(1);
		_dan_ambiance_2Instance->play();
		
	 #endif

	}
void ParticleDreams::ambient_sound_start_sprial_fountens()
	{
	#ifdef OAS_SOUND

        dan_5min_ostinato.setGain(0.5);
        dan_5min_ostinato.play(1);
	 #endif
	#ifdef OMICRON_SOUND
        _dan_5min_ostinatoInstance->setVolume(0.5);
        _dan_5min_ostinatoInstance->play();
		
	 #endif

 	
	}
void ParticleDreams::ambient_sound_start_N_waterfalls()
	{
	
	#ifdef OAS_SOUND
		//TODOSound		
			texture_12.fade(0,1);// kludge to stop sound from injector on skip to end
	#endif
	#ifdef OMICRON_SOUND
			_texture_12Instance->fade(0,1);// kludge to stop sound from injector on skip to end
	#endif

	#ifdef OAS_SOUND
	
		harmonicAlgorithm.setGain(1);
		harmonicAlgorithm.setLoop(1);
		harmonicAlgorithm.play(1);
		
	 #endif
	#ifdef OMICRON_SOUND
	//ToDoSound
		_harmonicAlgorithmInstance->setVolume(1);
		_harmonicAlgorithmInstance->play();
	
	 #endif

	
	}
void ParticleDreams::ambient_sound_start_painting_skys()
	{
	#ifdef OAS_SOUND
		rain_at_sea.setLoop(true);
		rain_at_sea.setGain(1);
		rain_at_sea.play(1);

		texture_17_swirls3.setLoop(true);
		texture_17_swirls3.setGain(1);
		texture_17_swirls3.play(1);
	 #endif
	#ifdef OMICRON_SOUND

		_rain_at_seaInstance->setLoop(true);
		_rain_at_seaInstance->setVolume(.5);
		_rain_at_seaInstance->play();

		_texture_17_swirls3Instance->setLoop(true);
		_texture_17_swirls3Instance->setVolume(1);
		_texture_17_swirls3Instance->play();

	 #endif

	
	}
void ParticleDreams::ambient_sound_start_rain()
	{
	#ifdef OAS_SOUND
		dan_rain_at_sea_loop.setLoop(1);
		dan_rain_at_sea_loop.setGain(1);

		dan_rain_at_sea_loop.play();
	 #endif
	#ifdef OMICRON_SOUND
		_dan_rain_at_sea_loopInstance->setLoop(1);
		_dan_rain_at_sea_loopInstance->setVolume(1);

		_dan_rain_at_sea_loopInstance->play();
	 #endif

	
	}

void ParticleDreams::sound_stop_paint_on_walls()
	{
	#ifdef OAS_SOUND
		dan_ambiance_2.fade(0, 1.5);
		dan_10122606_sound_spray.setGain(0);
	 #endif
	#ifdef OMICRON_SOUND
		_dan_ambiance_2Instance->fade(0, 1.5);
		_dan_10122606_sound_sprayInstance->setVolume(0);

	 #endif
	
	}
void ParticleDreams::sound_stop_sprial_fountens()
	{
	#ifdef OAS_SOUND
		dan_5min_ostinato.fade(0, 1.50);
		dan_10122606_sound_spray.setGain(0);
	 #endif
	#ifdef OMICRON_SOUND
		_dan_5min_ostinatoInstance->fade(0, 1.50);
		_dan_10122606_sound_sprayInstance->setVolume(0);

	 #endif

	
	}
void ParticleDreams::sound_stop_N_waterfalls() 
	{
	#ifdef OAS_SOUND
		harmonicAlgorithm.fade(0, 1.50);
		//harmonicAlgorithm.setLoop(0);
		dan_10122606_sound_spray.setGain(0);
	 #endif
	#ifdef OMICRON_SOUND
	//ToDoSound
		_harmonicAlgorithmInstance->fade(0, 1.50);
		_dan_10122606_sound_sprayInstance->setVolume(0);

	
	 #endif

	
	}
	
void ParticleDreams::sound_stop_painting_skys()
	{
	#ifdef OAS_SOUND
		rain_at_sea.fade(0, 1.50);
		texture_17_swirls3.fade(0, 1.50);
		dan_10122606_sound_spray.setGain(0);
		dan_10120600_rezS3_rez2.setGain(0);
	 #endif
	#ifdef OMICRON_SOUND

		_rain_at_seaInstance->fade(0, 1.0);
		_texture_17_swirls3Instance->fade(0, 1.50);
		_dan_10122606_sound_sprayInstance->setVolume(0);
		_dan_10120600_rezS3_rez2Instance->setVolume(0);

	 #endif

	
	}
void ParticleDreams::sound_stop_rain()
	{
	#ifdef OAS_SOUND
		dan_rain_at_sea_loop.fade(0, 1.50);
		dan_10122606_sound_spray.setGain(0);
	 #endif
	#ifdef OMICRON_SOUND
		_dan_rain_at_sea_loopInstance->fade(0, 1.50);
		_dan_10122606_sound_sprayInstance->setVolume(0);
	 #endif

	
	}
	
void ParticleDreams::texture_swirls_setPosition(int injNum)
	{
	float x = h_injectorData[injNum][3][0];//gets velocity
	float y = h_injectorData[injNum][2][1];//get position info
	float z = h_injectorData[injNum][3][2];// gets velocity
	osg::Vec3f injPos ;
	injPos[0]=x/(.02) *3;
	injPos[1]=y;
	injPos[2]=z/(.02) *3;
	injPos = injPos + navHeadPos;
	///std::cout <<" injPos " << injPos << "\n";
	x= injPos[0];
	y= injPos[1];
	z= injPos[2];

	#ifdef OAS_SOUND
	
	texture_17_swirls3.setPosition( x, y,  z);
	    
	 #endif
	 
	#ifdef OMICRON_SOUND
	omicron::Vector3f injPos2(x,y,z);
	//std::cout <<" injPos2 " << injPos2 << "\n";
	

	_texture_17_swirls3Instance->setPosition(injPos2);

	 #endif


	}
	
osg::Geometry* ParticleDreams::createQuad()
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

void  ParticleDreams::resetSlidePosition(){
 	mtSlide1->setMatrix(SlideMatrix);
    mtSlide2->setMatrix(SlideMatrix);
    mtSlide3->setMatrix(SlideMatrix);
	}
 
void	ParticleDreams::initGeoEdSection()
{

    osg::Node* tidleSlide1 = NULL;
    tidleSlide1 = osgDB::readNodeFile(_dataDir + "/models/Dan-Spherical_Credits-001.obj");//Tidle
    if(!tidleSlide1){std::cerr << "Error reading /models/Dan-Spherical_Credits-001.obj" << std::endl;}
    osg::Node* tidleSlide2 = NULL;
    tidleSlide2 = osgDB::readNodeFile(_dataDir + "/models/Dan-Spherical_Credits-002.obj");//Credit1
    if(!tidleSlide2){std::cerr << "Error reading /models/Dan-Spherical_Credits-002.obj" << std::endl;}
    osg::Node* tidleSlide3 = NULL;
    tidleSlide3 = osgDB::readNodeFile(_dataDir + "/models/frameS3.obj");//Credit2
    if(!tidleSlide3){std::cerr << "Error reading /models/frameS3.obj" << std::endl;}
    std::cout << "slidesLoaded \n";

    std::cout << " loading slides " << "\n" ;
    //creat switchnode
    _EdSecSwitchSlid1 = new osg::Switch;
    _EdSecSwitchSlid2 = new osg::Switch;
    _EdSecSwitchSlid3 = new osg::Switch;

    ParticleDreams::turnAllEduSlidsOff();

    // creat fixes xform on slides
    osg::Matrix ms;
    mtSlide1 = new osg::MatrixTransform();
    mtSlide2 = new osg::MatrixTransform();
    mtSlide3 = new osg::MatrixTransform();

    //matrices for interdediat computations
    osg::Matrix mss;
    osg::Matrix msr;

    osg::Matrix msrh;
    osg::Matrix mst;

    


if (_TargetSystem.compare("TourCaveCalit2") == 0)
    {
         //    height= 805 ;h=  26.0; width= 1432 ;  p= 0.0    ;originX= -802  ; originY= 1657  ;  r= -90.0 ;   name= 0 ;  originZ= 0   ;   screen= 0;
    int i =1;
   // std::cout << " i, _PhScAr[i].originX  ,_PhScAr[i].originY _PhScAr[i].h "<< i<< " " << _PhScAr[i].originX  << " " << _PhScAr[i].originY << " " << _PhScAr[i].h << "\n";

    mst.makeTranslate(osg::Vec3(_PhScAr[i].originX,_PhScAr[i].originY +150,1000));//was450
    // mst.makeTranslate(osg::Vec3(-802,1657,450));
    ms.makeScale(osg::Vec3(40.0,30,1.0));
    msr.makeRotate(osg::DegreesToRadians(90.0), osg::Vec3(1,0,0));
    msrh.makeRotate(osg::DegreesToRadians(25.0), osg::Vec3(0,0,1));
    msrh.makeRotate(osg::DegreesToRadians(_PhScAr[i].h), osg::Vec3(0,0,1));

    }


if (_TargetSystem.compare("Cave2") == 0)
    {
         //    height= 805 ;h=  26.0; width= 1432 ;  p= 0.0    ;originX= -802  ; originY= 1657  ;  r= -90.0 ;   name= 0 ;  originZ= 0   ;   screen= 0;
    int i =1;
   std::cout << " i, _PhScAr[i].originX  ,_PhScAr[i].originY _PhScAr[i].h "<< i<< " " << _PhScAr[i].originX  << " " << _PhScAr[i].originY << " " << _PhScAr[i].h << "\n";

   // mst.makeTranslate(osg::Vec3(_PhScAr[i].originX,_PhScAr[i].originY ,_PhScAr[i].originZ));
    mst.makeTranslate(osg::Vec3(-802,1657,450));
    ms.makeScale(osg::Vec3(40.0,25,1.0));
    msr.makeRotate(osg::DegreesToRadians(90.0), osg::Vec3(1,0,0));
    msrh.makeRotate(osg::DegreesToRadians(30.0), osg::Vec3(0,0,1));
    //msrh.makeRotate(osg::DegreesToRadians(_PhScAr[i].h), osg::Vec3(0,0,1));

    }
if (_TargetSystem.compare("StarCave") == 0)
    {
         //    height= 805 ;h=  26.0; width= 1432 ;  p= 0.0    ;originX= -802  ; originY= 1657  ;  r= -90.0 ;   name= 0 ;  originZ= 0   ;   screen= 0;
    int i =1;
   // std::cout << " i, _PhScAr[i].originX  ,_PhScAr[i].originY _PhScAr[i].h "<< i<< " " << _PhScAr[i].originX  << " " << _PhScAr[i].originY << " " << _PhScAr[i].h << "\n";

    mst.makeTranslate(osg::Vec3(_PhScAr[i].originX,_PhScAr[i].originY +150,450));
    // mst.makeTranslate(osg::Vec3(-802,1657,450));
    ms.makeScale(osg::Vec3(80.0,50,1.0));
    msr.makeRotate(osg::DegreesToRadians(90.0), osg::Vec3(1,0,0));
    msrh.makeRotate(osg::DegreesToRadians(25.0), osg::Vec3(0,0,1));
    msrh.makeRotate(osg::DegreesToRadians(_PhScAr[i].h), osg::Vec3(0,0,1));

    }
if (_TargetSystem.compare("Wave") == 0)
    {
         //    height= 805 ;h=  26.0; width= 1432 ;  p= 0.0    ;originX= -802  ; originY= 1657  ;  r= -90.0 ;   name= 0 ;  originZ= 0   ;   screen= 0;
    int i =1;
   // std::cout << " i, _PhScAr[i].originX  ,_PhScAr[i].originY _PhScAr[i].h "<< i<< " " << _PhScAr[i].originX  << " " << _PhScAr[i].originY << " " << _PhScAr[i].h << "\n";

    mst.makeTranslate(osg::Vec3(_PhScAr[i].originX,_PhScAr[i].originY +150,450));
    // mst.makeTranslate(osg::Vec3(-802,1657,450));
    ms.makeScale(osg::Vec3(80.0,50,1.0));
    msr.makeRotate(osg::DegreesToRadians(90.0), osg::Vec3(1,0,0));
    msrh.makeRotate(osg::DegreesToRadians(25.0), osg::Vec3(0,0,1));
    msrh.makeRotate(osg::DegreesToRadians(_PhScAr[i].h), osg::Vec3(0,0,1));

    }

    SlideMatrix.set ( ms*msr*msrh * mst);
 //   mtSlide1->setMatrix(SlideMatrix);
 //   mtSlide2->setMatrix(SlideMatrix);
  //  mtSlide3->setMatrix(SlideMatrix);
	resetSlidePosition();
 
    //atatch tidleSlide to matrix transform
    mtSlide1->addChild(tidleSlide1);
    mtSlide2->addChild(tidleSlide2);
    mtSlide3->addChild(tidleSlide3);
/*
    mtSlide4->addChild(tidleSlide4);
    mtSlide5->addChild(tidleSlide5);
    mtSlide6->addChild(tidleSlide6);
    mtSlide7->addChild(tidleSlide7);
    mtSlide8->addChild(tidleSlide8);
*/

    // atatch scaled model to switch 
    _EdSecSwitchSlid1->addChild(mtSlide1);
    _EdSecSwitchSlid2->addChild(mtSlide2);
    _EdSecSwitchSlid3->addChild(mtSlide3);
/*
    _EdSecSwitchSlid4->addChild(mtSlide4);
    _EdSecSwitchSlid5->addChild(mtSlide5);
    _EdSecSwitchSlid6->addChild(mtSlide6);
    _EdSecSwitchSlid7->addChild(mtSlide7);
    _EdSecSwitchSlid8->addChild(mtSlide8);
*/
    turnAllEduSlidsOff(); 

    // atatch switch to scene


    // addTidle screene.
    SceneObject * so = new SceneObject("EdSlide1",false,false,false,false,false);
    so->addChild(_EdSecSwitchSlid1);
    so->addChild(_EdSecSwitchSlid2);
    so->addChild(_EdSecSwitchSlid3);
/*
    so->addChild(_EdSecSwitchSlid4);
    so->addChild(_EdSecSwitchSlid5);
    so->addChild(_EdSecSwitchSlid6);
    so->addChild(_EdSecSwitchSlid7);
    so->addChild(_EdSecSwitchSlid8);
*/
    _EdSceneObject = so;

    PluginHelper::registerSceneObject(so,"ParticleDreams");

    so->setPosition(osg::Vec3(0,0,0));

    so->setScale(1);
    so->attachToScene();
    so->setNavigationOn(false);

}

void ParticleDreams::cleanupGeoEdSection()
{
    _EdSecSwitchSlid1 = NULL;
    _EdSecSwitchSlid2 = NULL;
    _EdSecSwitchSlid3 = NULL;
/*
    _EdSecSwitchSlid4 = NULL;
    _EdSecSwitchSlid5 = NULL;
    _EdSecSwitchSlid6 = NULL;
    _EdSecSwitchSlid7 = NULL;
    _EdSecSwitchSlid8 = NULL;
*/
    delete _EdSceneObject;
}

void ParticleDreams::turnAllEduSlidsOff()
	{
       _EdSecSwitchSlid1->setAllChildrenOff();
       _EdSecSwitchSlid2->setAllChildrenOff();
       _EdSecSwitchSlid3->setAllChildrenOff();
/*
       _EdSecSwitchSlid4->setAllChildrenOff();
       _EdSecSwitchSlid5->setAllChildrenOff();
       _EdSecSwitchSlid6->setAllChildrenOff();
       _EdSecSwitchSlid7->setAllChildrenOff();
       _EdSecSwitchSlid8->setAllChildrenOff();
*/
	}



void ParticleDreams::injectorSetMaxnumNum(int maxNumber)
	{
		//std::cout << "injMaxNumber " << maxNumber << "\n";
		h_injectorData[0][0][0]=maxNumber;
	}
	int ParticleDreams::injectorGetMaxnumNum(void)
	{
		
		int maxNumber = h_injectorData[0][0][0];
		//std::cout << "injMaxNumber " << maxNumber << "\n";
		return maxNumber;

	}
void ParticleDreams::injectorSetType (int injtNum,int type)
	{
		h_injectorData[injtNum][1][0]=type;
	}

void ParticleDreams::injectorSetDifaults(int injtNum)
	{
		for (int rownum =0;rownum < INJT_DATA_ROWS;rownum++)
		{ 
			h_injectorData[injtNum ][rownum][0]=0;
			h_injectorData[injtNum ][rownum][1]=0;
			h_injectorData[injtNum ][rownum][2]=0;
		}
	}

void ParticleDreams::injectorSetInjtRatio (int injtNum,float injtRatio)
	{
		h_injectorData[injtNum][1][1]=injtRatio;
	}
void ParticleDreams::injectorSetPosition (int injtNum,float x,float y, float z, axisIsUp up)
	{
			h_injectorData[injtNum][2][0]=x;h_injectorData[injtNum][2][1]=y;h_injectorData[injtNum][2][2]=z;
	}

void ParticleDreams::injectorSetVelocity (int injtNum,float vx,float vy, float vz, axisIsUp up)
	{
		h_injectorData[injtNum][3][0]=vx;h_injectorData[injtNum][3][1]=vy;h_injectorData[injtNum][3][2]=vz;
	}

void ParticleDreams::injectorSetSize (int injtNum,float x,float y, float z, axisIsUp up)
	{
		h_injectorData[injtNum][4][0]=x;h_injectorData[injtNum][4][1]=y;h_injectorData[injtNum][4][2]=z;//x,y,z size
	}

void ParticleDreams::injectorSetSpeedDist (int injtNum,float du,float dv, float dt, axisIsUp up)
	{
		h_injectorData[injtNum][5][0]=du;h_injectorData[injtNum][5][1]=dv;h_injectorData[injtNum][5][2]=dt;//t,u,v jiter v not implemented = speed 
	}

void ParticleDreams::injectorSetSpeedJitter (int injtNum,float du,float dv, float dt, axisIsUp up)
	{
		h_injectorData[injtNum][6][0]=du;h_injectorData[injtNum][6][1]=dv;h_injectorData[injtNum][6][2]=dt;//t,u,v jiter v not implemented = speed 
	}

void ParticleDreams::injectorSetSpeedCentrality (int injtNum,float du,float dv, float dt, axisIsUp up)
	{
						h_injectorData[injtNum][7][0]=du;h_injectorData[injtNum][7][1]=dv;h_injectorData[injtNum][7][2]=dt;//centrality of rnd distribution speed dt tu ~
	}


		int ParticleDreams::reflectorGetMaxnumNum(void)
		{
			int maxNumber= h_reflectorData[0][0][0];
			return maxNumber;
		}
		
		void ParticleDreams::reflectorSetMaxnumNum(int maxNumber)
		{
			h_reflectorData[0][0][0]=maxNumber;
		}
		void ParticleDreams::reflectorSetType (int reflNum,int type)
		{
			h_reflectorData[reflNum ][0][0] = type;// 0 os off, 1 is plain
		}
		void ParticleDreams::reflectorSetDifaults(int reflNum)
		{
			for (int rownum =0;rownum < REFL_DATA_ROWS;rownum++)
			{ 
				h_reflectorData[reflNum ][rownum][0]=0;
				h_reflectorData[reflNum ][rownum][1]=0;
				h_reflectorData[reflNum ][rownum][2]=0;
			}
 
		}
		void ParticleDreams::reflectorSetPosition (int reflNum,float x,float y, float z, axisIsUp up)
		{
			h_reflectorData[reflNum ][1][0]=x;    h_reflectorData[reflNum ][1][1]= y;h_reflectorData[reflNum ][1][2]=z;
		}	
		
		void ParticleDreams::reflectorSetNormal(int reflNum,float nx,float ny, float nz, axisIsUp up)
		{
			h_reflectorData[reflNum ][2][0]=nx;  h_reflectorData[reflNum ][2][1]=ny;    h_reflectorData[reflNum ][2][2]=nz;
		}
		
		void ParticleDreams::reflectorSetSize (int reflNum,float radius, axisIsUp up)
		{
			h_reflectorData[reflNum ][3][0]=radius; h_reflectorData[reflNum ][3][1]=0.00; h_reflectorData[reflNum ][3][2]=0; 
		}
		
		void ParticleDreams::reflectorSetDamping (int reflNum,float damping)
		{
			h_reflectorData[reflNum ][5][0]= damping;
		}
		void ParticleDreams::reflectorSetNoTraping (int reflNum,int noTraping)
		{
			h_reflectorData[reflNum ][5][1]= noTraping;
		}


