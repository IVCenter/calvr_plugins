#ifndef CVR_ALGEBRA_IN_MOTION_H
#define CVR_ALGEBRA_IN_MOTION_H

//#define SCR2_PER_CARD  

#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/CVRViewer.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuRangeValue.h>
#include "CudaParticle.h"
#include "PDObject.h"

#include <osg/Geometry>
#include <osg/Geode>
#include <osg/Shader>
#include <osg/Program>
#include <osg/Texture2D>
#include <OpenThreads/Mutex>

#include <string>

#include <OAS/OASClient.h>


class AlgebraInMotion : public cvr::CVRPlugin, public cvr::MenuCallback, public cvr::PerContextCallback
{
    public:
        AlgebraInMotion();
        virtual ~AlgebraInMotion();

        bool init();

        void menuCallback(cvr::MenuItem * item);
        void preFrame();
        bool processEvent(cvr::InteractionEvent * event);

		//ContextChabge fr 2 scr
		#ifdef SCR2_PER_CARD
        virtual void perContextCallback(int contextid) const;
        
        //ContestChange from 1 scr
        #else
		virtual void perContextCallback(int contextid, PerContextCallback::PCCType type) const;
        #endif
		
    protected:
        void initPart();
        void initGeometry();
        void initSound();

        void updateHand();

        void pdata_init_age(int mAge);
        void pdata_init_velocity(float vx,float vy,float vz);
		void pdata_init_rand();
        void copy_reflector(int source, int dest);
        void copy_injector( int sorce, int destination);
        int load6wallcaveWalls(int firstRefNum);
 
        void scene_data_0_host();
        void scene_data_1_host();
        void scene_data_2_host();
        void scene_data_3_host();
        void scene_data_4_host();

        void scene_data_0_context(int contextid) const;
        void scene_data_1_context(int contextid) const;
        void scene_data_2_context(int contextid) const;
        void scene_data_3_context(int contextid) const;
        void scene_data_4_context(int contextid) const;

        void scene_data_0_kill_audio();
        void scene_data_1_kill_audio();
        void scene_data_2_kill_audio();
        void scene_data_3_kill_audio();
        void scene_data_4_kill_audio();

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

        size_t sizeDebug;
        int max_age;
        int disappear_age;
        int hand_id;
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
        int but1,but1old;
		int skipTonextScene,skipTOnextSceneOld;
        int sceneNum;
        int sceneOrder;
        int nextSean;
        int scene0Start;
        int scene1Start;
        int scene2Start;
        int scene3Start;
        int scene4Start;
        int scene0ContextStart;
        int scene1ContextStart;
        int scene2ContextStart;
        int scene3ContextStart;
        int scene4ContextStart;
        int witch_scene;
        int old_witch_scene;
        int sceneChange;

        float showFrameNo;
        float lastShowFrameNo;
	double showStartTime;
	double showTime;
	double lastShowTime;
        double startTime;
        double nowTime;
        double frNum;

        bool soundEnabled;
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

        float reflSoundUpdate(int refNum);

		void injSoundUpdate(int injNum);

        
        std::string _dataDir;

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

        float _pointerHeading, _pointerPitch, _pointerRoll;
		int loadPhysicalScreensArrayTourCaveCalit2();
		int loadPhysicalScreensArrayTourCaveCalit2_5lowerScr();
		int loadPhysicalScreensArrayTourCaveSaudi();
		int loadPhysicalScreensArrayCave2();//not implemented
       int loadInjFountsFrScr(float dx,float dy,float dz,float speed);
       int loadReflFrScr();
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
	void initGeoEdSection();
		osg::ref_ptr<osg::Switch> _EdSecSwitchSlid1;
		osg::ref_ptr<osg::Switch> _EdSecSwitchSlid2;
		osg::ref_ptr<osg::Switch> _EdSecSwitchSlid3;
		osg::ref_ptr<osg::Switch> _EdSecSwitchSlid4;
		osg::ref_ptr<osg::Switch> _EdSecSwitchSlid5;
		osg::ref_ptr<osg::Switch> _EdSecSwitchSlid6;
		osg::ref_ptr<osg::Switch> _EdSecSwitchSlid7;
		osg::ref_ptr<osg::Switch> _EdSecSwitchSlid8;

	void turnAllEduSlidsOff();
/*	
	    osg::MatrixTransform * _EdAxisBallsXform	;
		osg::ref_ptr<osg::Switch> _EdSecSwitchSlid1;
		osg::ref_ptr<osg::Switch> _EdSecSwitchAxis;
		osg::ref_ptr<osg::Switch> _EdSecXballObjSwitch;
		osg::ref_ptr<osg::Switch> _EdSecYballObjSwitch;
		osg::ref_ptr<osg::Switch> _EdSecZballObjSwitch;
		osg::ref_ptr<osg::Switch> _EdSecXYZballObjSwitch;
		osg::MatrixTransform * _xBallXform;
		osg::MatrixTransform * _yBallXform;
		osg::MatrixTransform * _zBallXform;
		osg::MatrixTransform * _xyzBallXform;
		
		osg::ref_ptr<osg::Switch> _EdSecXboxObjSwitch;
		osg::ref_ptr<osg::Switch> _EdSecYboxObjSwitch;
		osg::ref_ptr<osg::Switch> _EdSecZboxObjSwitch;
		osg::MatrixTransform * _xboxXform;
		osg::MatrixTransform * _yboxXform;
		osg::MatrixTransform * _zboxXform;

		osg::ref_ptr<osg::Switch> _EdSecXarrowObjSwitch;
		osg::ref_ptr<osg::Switch> _EdSecYarrowObjSwitch;
		osg::ref_ptr<osg::Switch> _EdSecZarrowObjSwitch;
		osg::ref_ptr<osg::Switch> _EdSecXYZarrowObjSwitch;

		osg::MatrixTransform * _xArrowXform;
		osg::MatrixTransform * _yArrowXform;
		osg::MatrixTransform * _zArrowXform;
		osg::MatrixTransform * _xyzArrowXform;
	void   AddCylinderBetweenPoints(osg::Vec3   StartPoint, osg::Vec3   EndPoint, float radius, osg::Vec4   CylinderColor, osg::Group   *pAddToThisGroup) ;

	void   setsMatrixScalingUnitLengthZupObjectBetweenPoints(osg::Vec3   StartPoint, osg::Vec3   EndPoint, float xscale,float yscale, osg::Matrix * result) ;
*/
};

#endif
