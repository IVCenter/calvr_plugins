#ifndef CVR_PARTICLE_DREAMS_H
#define CVR_PARTICLE_DREAMS_H

#define SCR2_PER_CARD  
#define OAS_SOUND
//#define OMICRON_SOUND
#ifdef OMICRON_SOUND
#include <omicron/SoundManager.h>
#endif
#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/CVRViewer.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrKernel/ComController.h>
#include "CudaParticle.h"
#include "PDObject.h"

#include <osg/Geometry>
#include <osg/Geode>
#include <osg/Shader>
#include <osg/Program>
#include <osg/Texture2D>
#include <osg/TextureRectangle>
#include <OpenThreads/Mutex>

#include <string>
#ifdef OAS_SOUND
	#include <OAS/OASClient.h>
#endif

class ParticleDreams : public cvr::CVRPlugin, public cvr::MenuCallback, public cvr::PerContextCallback
{
    public:
        ParticleDreams();
        virtual ~ParticleDreams();

        bool init();

        void menuCallback(cvr::MenuItem * item);
        void preFrame();
        bool processEvent(cvr::InteractionEvent * event);

	virtual void perContextCallback(int contextid, PerContextCallback::PCCType type) const;
		
    protected:
        void initPart();
        void initGeometry();
        void initSound();
        void initWater();

        void cleanupPart();
        void cleanupGeometry();
        void cleanupSound();

        void updateHand();
       void updateHead();
	void initGeoEdSection();
	       void cleanupGeoEdSection();
		osg::ref_ptr<osg::Switch> _EdSecSwitchSlid1;
		osg::ref_ptr<osg::Switch> _EdSecSwitchSlid2;
		osg::ref_ptr<osg::Switch> _EdSecSwitchSlid3;
		osg::ref_ptr<osg::Switch> _EdSecSwitchSlid4;
		osg::ref_ptr<osg::Switch> _EdSecSwitchSlid5;
		osg::ref_ptr<osg::Switch> _EdSecSwitchSlid6;
		osg::ref_ptr<osg::Switch> _EdSecSwitchSlid7;
		osg::ref_ptr<osg::Switch> _EdSecSwitchSlid8;

        cvr::SceneObject * _EdSceneObject;

	void turnAllEduSlidsOff();

        void setWater(bool b);

        void pdata_init_age(int mAge);
        void pdata_init_velocity(float vx,float vy,float vz);
		void pdata_init_rand();
        void copy_reflector(int source, int dest);
        void copy_injector( int sorce, int destination);
        int load6wallcaveWalls(int firstRefNum);
 
        void paint_on_walls_host();
        void sprial_fountens_host();
        void N_waterfalls_host();
        void painting_skys_host();
        void rain_host();

        void paint_on_walls_context(int contextid) const;
        void sprial_fountens_context(int contextid) const;
        void N_waterfalls_context(int contextid) const;
        void painting_skys_context(int contextid) const;
        void rain_context(int contextid) const;

        void paint_on_walls__clean_up();
        void sprial_fountens__clean_up();
        void N_waterfalls__clean_up();
        void painting_skys__clean_up();
        void rain__clean_up();
        void HeadReflectorsMake();

        cvr::SubMenu * _myMenu;
        cvr::MenuCheckbox * _enable;

		cvr::MenuRangeValue * _gravityRV;
		cvr::MenuRangeValue *_speedRV;
        cvr::MenuCheckbox * _rotateInjCB;
        cvr::MenuCheckbox * _reflectorCB;
        PDObject * _particleObject;
        osg::ref_ptr<osg::Geometry> _particleGeo;
        osg::ref_ptr<osg::Geode> _particleGeode;
        osg::ref_ptr<osg::Vec3Array> _positionArray;
        osg::ref_ptr<osg::Vec4Array> _colorArray;
        osg::ref_ptr<osg::DrawArrays> _primitive;

        bool _callbackAdded;
        bool _callbackActive;

        float * h_particleData;
        float h_injectorData[INJT_DATA_MUNB][INJT_DATA_ROWS][INJT_DATA_ROW_ELEM];
        float h_reflectorData[REFL_DATA_MUNB][REFL_DATA_ROWS][REFL_DATA_ROW_ELEM];
        float * h_debugData;
      	float *_old_refl_hits;// needs to have same length as d_debugData
        float * _refl_hits;// needs to have same length as d_debugData

        double modulator[4];
        double wandPos[3];
        double wandVec[3];
        double wandMat[16];
        double headVec[3];
        double headMat[16];
		osg::Vec3 headPos;
		osg::Vec3 navHeadPos;
		osg::Matrix SlideMatrix;
       	void resetSlidePosition();
 		osg::MatrixTransform * mtSlide1;
	    osg::MatrixTransform * mtSlide2 ;
	    osg::MatrixTransform * mtSlide3;


        size_t sizeDebug;
        int max_age;
        int disappear_age;
        int hand_id;
        int head_id;
        int draw_water_sky;
        float state;
        float gravity;
        float colorFreq;
        float alphaControl;
        float anim;
        int trigger,triggerold;
        int but4,but4old;
        int but3,but3old;
        int but2,but2old;
        int  hand_inject_or_reflect, hand_inject_or_reflectold;
 
        int but1,but1old;
		int skipTonextScene,skipTOnextSceneOld;
        int sceneName;
        int sceneOrder;
        int nextSean;
        int paint_on_walls_start;
        int sprial_fountens_start;
        int N_waterfalls_start;
        int painting_skys_start;
        int rain_start;
        int paint_on_wallsContextStart;
        int sprial_fountensContextStart;
        int N_waterfallsContextStart;
        int painting_skysContextStart;
        int rainContextStart;
        int name_paint_on_walls ;
		int name_sprial_fountens;
		int name_N_waterfalls;
		int name_painting_skys ;
		int name_rain ;
		float tidleSlideTime ;
		float n_waterfalls_timeInScene;
		float sprial_fountens_timeInScene;
		float paint_on_walls_timeInScene;
		float painting_skys_timeInScene;
		float rain_timeInScene;
		

        //int witch_scene;
        //int old_witch_scene;
        int sceneChange;
        //int firstScene ;

        float showFrameNo;
        float lastShowFrameNo;
	double showStartTime;
	double showTime;
	double lastShowTime;
        double startTime;
        double nowTime;
        double frNum;

        bool soundEnabled;
#ifdef OAS_SOUND
        oasclient::Sound chimes;
        oasclient::Sound pinkNoise;
        oasclient::Sound dan_texture_09;
        oasclient::Sound texture_12;
        oasclient::Sound short_sound_01a;
        oasclient::Sound short_sound_01a1;
        oasclient::Sound short_sound_01a2;
        oasclient::Sound short_sound_01a3;
        oasclient::Sound short_sound_01a4;
       	oasclient::Sound short_sound_01a5;
         oasclient::Sound texture_17_swirls3;
        oasclient::Sound rain_at_sea;
        oasclient::Sound dan_texture_13;
        oasclient::Sound dan_texture_05;
        oasclient::Sound dan_short_sound_04;
        oasclient::Sound dan_ambiance_2;
        oasclient::Sound dan_ambiance_1;
        oasclient::Sound dan_5min_ostinato;
        oasclient::Sound dan_10120603_Rez1;
        oasclient::Sound dan_mel_amb_slower;
        oasclient::Sound harmonicAlgorithm;
        oasclient::Sound dan_rain_at_sea_loop;
        oasclient::Sound dan_10122606_sound_spray;
        oasclient::Sound dan_10122608_sound_spray_low;
        oasclient::Sound dan_10120600_rezS3_rez2;
#endif
#ifdef OMICRON_SOUND
	omicron::SoundManager* _SoundMng;
	omicron::SoundEnvironment* _SoundEnv;
	//omicron::Sound* _harmonicAlgorithm;
	//omicron::SoundInstance* _harmonicAlgorithmInstance;
	//omicron::Sound* _dan_10122606_sound_spray;

      	omicron::Sound* _chimes;
        omicron::Sound* _pinkNoise;
        omicron::Sound* _dan_texture_09;
        omicron::Sound* _texture_12;
        omicron::Sound* _short_sound_01a;
        omicron::Sound* _short_sound_01a1;
        omicron::Sound* _short_sound_01a2;
        omicron::Sound* _short_sound_01a3;
        omicron::Sound* _short_sound_01a4;
       	omicron::Sound* _short_sound_01a5;
        omicron::Sound* _texture_17_swirls3;
        omicron::Sound* _rain_at_sea;
        omicron::Sound* _dan_texture_13;
        omicron::Sound* _dan_texture_05;
        omicron::Sound* _dan_short_sound_04;
        omicron::Sound* _dan_ambiance_2;
        omicron::Sound* _dan_ambiance_1;
        omicron::Sound* _dan_5min_ostinato;
        omicron::Sound* _dan_10120603_Rez1;
        omicron::Sound* _dan_mel_amb_slower;
        omicron::Sound* _harmonicAlgorithm;
        omicron::Sound* _dan_rain_at_sea_loop;
        omicron::Sound* _dan_10122606_sound_spray;
        omicron::Sound* _dan_10122608_sound_spray_low;
        omicron::Sound* _dan_10120600_rezS3_rez2;

       	omicron::SoundInstance* _chimesInstance;
        omicron::SoundInstance* _pinkNoiseInstance;
        omicron::SoundInstance* _dan_texture_09Instance;
        omicron::SoundInstance* _texture_12Instance;
        omicron::SoundInstance* _short_sound_01aInstance;
        omicron::SoundInstance* _short_sound_01a1Instance;
        omicron::SoundInstance* _short_sound_01a2Instance;
        omicron::SoundInstance* _short_sound_01a3Instance;
        omicron::SoundInstance* _short_sound_01a4Instance;
       	omicron::SoundInstance* _short_sound_01a5Instance;
        omicron::SoundInstance* _texture_17_swirls3Instance;
        omicron::SoundInstance* _rain_at_seaInstance;
        omicron::SoundInstance* _dan_texture_13Instance;
        omicron::SoundInstance* _dan_texture_05Instance;
        omicron::SoundInstance* _dan_short_sound_04Instance;
        omicron::SoundInstance* _dan_ambiance_2Instance;
        omicron::SoundInstance* _dan_ambiance_1Instance;
        omicron::SoundInstance* _dan_5min_ostinatoInstance;
        omicron::SoundInstance* _dan_10120603_Rez1Instance;
        omicron::SoundInstance* _dan_mel_amb_slowerInstance;
        omicron::SoundInstance* _harmonicAlgorithmInstance;
        omicron::SoundInstance* _dan_rain_at_sea_loopInstance;
        omicron::SoundInstance* _dan_10122606_sound_sprayInstance;
        omicron::SoundInstance* _dan_10122608_sound_spray_lowInstance;
        omicron::SoundInstance* _dan_10120600_rezS3_rez2Instance;


#endif

        float reflSoundUpdate(int refNum);
		void injSoundUpdate(int injNum);
		void ambient_sound_start_paint_on_walls();
		void ambient_sound_start_sprial_fountens();
		void ambient_sound_start_N_waterfalls();
		void ambient_sound_start_painting_skys();
		void ambient_sound_start_rain();

		void sound_stop_paint_on_walls();
		void sound_stop_sprial_fountens();
		void sound_stop_N_waterfalls();
		void sound_stop_painting_skys();
		void sound_stop_rain();
		void texture_swirls_setPosition(int injNum);

        
        std::string _dataDir;

        // water
        osg::ref_ptr<osg::Shader> _waterVert;
        osg::ref_ptr<osg::Shader> _waterFrag;
        osg::ref_ptr<osg::Program> _waterProgram;
        
        osg::ref_ptr<osg::Shader> _waterMirrorVert;
        osg::ref_ptr<osg::Shader> _waterMirrorFrag;
        osg::ref_ptr<osg::Program> _waterMirrorProgram;

        osg::ref_ptr<osg::Uniform> _waterTimeUni;
        osg::ref_ptr<osg::Uniform> _waterLightUni;
        osg::ref_ptr<osg::Uniform> _waterGlowUni;
        osg::ref_ptr<osg::Uniform> _waterFillUni;
        osg::ref_ptr<osg::Uniform> _waterNormUni;
        osg::ref_ptr<osg::Uniform> _waterReflUni;
        osg::ref_ptr<osg::Texture2D> _waterNormalTexture;
        osg::ref_ptr<osg::Texture2D> _waterSkyFillTexture;
        osg::ref_ptr<osg::Texture2D> _waterSkyGlowTexture;
        osg::ref_ptr<osg::Camera> _waterPreCamera;
        osg::ref_ptr<osg::Camera> _waterNestedCamera;
        osg::ref_ptr<osg::Camera> _waterPostCamera;
        osg::ref_ptr<osg::MatrixTransform> _waterMirrorXform;
        osg::ref_ptr<osg::Geode> _waterSkyGeode;
        osg::ref_ptr<osg::Geometry> _waterSkyGeometry;
        osg::ref_ptr<osg::TextureRectangle> _waterColorTexture;
        osg::ref_ptr<osg::Texture2D> _waterDepthTexture;
        double _skyTime;


        osg::ref_ptr<osg::Shader> _spriteVert;
        osg::ref_ptr<osg::Shader> _spriteFrag;
        osg::ref_ptr<osg::Program> _spriteProgram;
        osg::ref_ptr<osg::Texture2D> _spriteTexture;

        osg::ref_ptr<osg::Program> _injFaceProgram;
        osg::ref_ptr<osg::Program> _injLineProgram;

        osg::ref_ptr<osg::MatrixTransform> _handModelMT;
		osg::ref_ptr<osg::Switch> _reflectorObjSwitch;
		osg::ref_ptr<osg::Switch> _injecttorObjSwitch;
		osg::ref_ptr<osg::Switch> _refObjSwitchFace;
		osg::ref_ptr<osg::Switch> _injObjSwitchFace;
		osg::ref_ptr<osg::Switch> _refObjSwitchLine;
		osg::ref_ptr<osg::Switch> _injObjSwitchLine;

        mutable OpenThreads::Mutex _callbackLock;
        mutable std::map<int,bool> _callbackInit;
        mutable std::map<int,CUdeviceptr> d_debugDataMap;
        mutable std::map<int,CUdeviceptr> d_particleDataMap;
        mutable std::map<int,bool> _cudaContextSet;

        float _pointerHeading, _pointerPitch, _pointerRoll;
        float _headHeading, _headPitch, _headRoll;
		int loadPhysicalScreensArrayTourCaveCalit2();
		int loadPhysicalScreensArrayTourCaveCalit2_5lowerScr();
		int loadPhysicalScreensArrayTourCaveSaudi();
		int loadPhysicalScreensArrayCave2();
		int loadOneHalfPhysicalScreensArrayCave2();
		int loadPhysicalScreensArrayStarCave();
		int loadPhysicalScreensArrayStarCaveCenterRow();
		int loadPhysicalScreensArrayWave();
       int loadInjFountsFrScr(float dx,float dy,float dz,float speed);
       int loadReflFrScr();
       //
       void injectorSetMaxnumNum(int maxNumber);
       int injectorGetMaxnumNum(void);
		enum axisIsUp{axisUpX,axisUpY,axisUpZ} ;// transformatios not implimented yet
		void injectorSetType (int injtNumber,int type);
		void injectorSetDifaults(int injtNumber);
		void injectorSetInjtRatio (int injtNumber,float injtRatio);
		void injectorSetPosition (int injtNumber,float x,float y, float z, axisIsUp up);
		void injectorSetVelocity (int injtNumber,float vx,float vy, float vz, axisIsUp up);
		void injectorSetSize (int injtNumber,float x,float y, float z, axisIsUp up);
		void injectorSetSpeedDist (int injtNumber,float du,float dv, float dt, axisIsUp up);
		void injectorSetSpeedJitter (int injtNumber,float du,float dv, float dt, axisIsUp up);
		void injectorSetSpeedCentrality (int injtNumber,float du,float dv, float dt, axisIsUp up);

 		int reflectorGetMaxnumNum(void);
		void reflectorSetMaxnumNum(int maxNumber);
		void reflectorSetType (int refltNumber,int type);// 0 os off, 1 is plain
		void reflectorSetDifaults(int refltNumber);
		void reflectorSetPosition (int refltNumber,float x,float y, float z, axisIsUp up);
		void reflectorSetNormal(int refltNumber,float nx,float ny, float nz, axisIsUp up);
		void reflectorSetSize (int refltNumber,float radius, axisIsUp up);
		void reflectorSetDamping (int refltNumber,float damping);
		void reflectorSetNoTraping (int refltNumber,int noTraping);


		osg::Geometry* createQuad();
		osg::Geometry* _quad1;					
		struct _PhSc
 					{
						int index;
					 	float height;
						float h;
						float width;
						float p;
						float originX;
						float originY;
						float r;
						float originZ;
						int screen;
						float vx;
						float vy;
						float vz;

					};
	_PhSc * _PhScAr;
	float varGravity;

	std::string _TargetSystem;
	std::string _DisplaySystem;
	int targetFrameRate;

//	omicron::SoundManager* _SoundMng;
//	omicron::SoundEnvironment* _SoundEnv;
//	omicron::Sound* _harmonicAlgorithm;
//	omicron::SoundInstance* _harmonicAlgorithmInstance;
//	omicron::Sound* _dan_10122606_sound_spray;
};

#endif
