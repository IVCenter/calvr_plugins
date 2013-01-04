#ifndef _VOLUME_
#define _VOLUME_

#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/FileHandler.h>
#include <cvrKernel/SceneObject.h>
#include <cvrMenu/MenuSystem.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/MenuText.h>

#include <osgDB/FileUtils>
#include <osgDB/FileNameUtils>

#include <osg/MatrixTransform>
#include <osg/ImageStream>
#include <osg/ImageSequence>
#include <osg/Uniform>
#include <osg/ClipPlane>

#include <osgVolume/Volume>

#include <string>
#include <vector>
#include <queue>

#include <virvo/vvfileio.h>
#include <virvo/vvvoldesc.h>
#include <virvo/vvtransfunc.h>
#include <virvo/vvtfwidget.h>

class Volume : public cvr::CVRPlugin, public cvr::MenuCallback ,public cvr::FileLoadCallback
{
    public:

        struct animationinfo
        {
            int id;
            int frame;
            double time;
        };
        
        struct volumeinfo
        {
            int id;
            std::string name;
            int x, y, z;
            float xMultiplier, yMultiplier, zMultiplier;
            int numberOfFrames;
            unsigned int r_offset;
            unsigned int numberOfBytesPerComponent;
            GLenum datatype;
            osg::ref_ptr<osg::ImageSequence> image;
            osg::ref_ptr<osgVolume::SwitchProperty> sp;
            osg::ref_ptr<osgVolume::VolumeTile> tile;
            osg::ref_ptr<osgVolume::IsoSurfaceProperty> isp;
            osg::ref_ptr<osgVolume::TransparencyProperty> tp;
            osg::ref_ptr<osgVolume::TransferFunctionProperty> tfp;
            osg::ref_ptr<osg::TransferFunction1D> tf;
            
            osg::Uniform * clipUniform;
            osg::Uniform * clipposUniform;
            osg::Uniform * clipnormUniform;
            osg::Uniform * isosurfaceValueMin;

            cvr::SceneObject* seedPoint;
            osg::Geode* points;
            osg::MatrixTransform* surface;
            osg::Vec3 seedLocation;

            vvTransFunc* defaultTransferFunc;
            float center;
            float width;
        }; 
            
        Volume();
    	virtual ~Volume();
	    bool init();
    	virtual bool loadFile(std::string file);
	    void menuCallback(cvr::MenuItem * item);
	    //bool processEvent(cvr::InteractionEvent * event);
	    void preFrame();

    protected:
   
        static int id;
    
        std::map<cvr::SceneObject*,cvr::MenuCheckbox*> _standardMap;
        std::map<cvr::SceneObject*,cvr::MenuCheckbox*> _lightMap;
        std::map<cvr::SceneObject*,cvr::MenuCheckbox*> _isosurfaceMap;
        std::map<cvr::SceneObject*,cvr::MenuCheckbox*> _maxintensityMap;
        std::map<cvr::SceneObject*,cvr::MenuCheckbox*> _transferFuncMap;
        std::map<cvr::SceneObject*,cvr::MenuRangeValue*> _isosurfaceValueMap;
        std::map<cvr::SceneObject*,cvr::MenuRangeValue*> _transparencyValueMap;
        std::map<cvr::SceneObject*,cvr::MenuButton*> _deleteMap;

        // animation controls
        std::map<cvr::SceneObject*,cvr::MenuCheckbox*> _playMap;
        std::map<cvr::SceneObject*,cvr::MenuRangeValue*> _speedMap;
        std::map<cvr::SceneObject*,cvr::MenuRangeValue*> _frameMap;

        // transfer function controls
        std::map<cvr::SceneObject*,cvr::MenuRangeValue*> _transferPositionMap;
        std::map<cvr::SceneObject*,cvr::MenuRangeValue*> _transferBaseWidthMap;
        std::map<cvr::SceneObject*,cvr::MenuCheckbox*> _transferDefaultMap;
        std::map<cvr::SceneObject*,cvr::MenuCheckbox*> _transferBrightMap;
        std::map<cvr::SceneObject*,cvr::MenuCheckbox*> _transferHueMap;
        std::map<cvr::SceneObject*,cvr::MenuCheckbox*> _transferGrayMap;

        std::map<cvr::SceneObject*,volumeinfo*> _volumeMap;

        // different load operators
        struct volumeinfo* loadXVF(std::string filename, int &x, int &y, int &z, float &xMultiplier, float &yMultiplier, float &zMultiplier, 
                                        int& sizeS, int& sizeT, int& sizeR, int& numberBytesPerComponent);

        // general functions
        osg::TransferFunction1D::ColorMap* computeColorMap(vvTransFunc* tf, float position = 0.0, float baseWidth = 0.0); 
        osg::TransferFunction1D::ColorMap* computeColorMap(int colorTable, float position, float baseWidth); 
        void setTransferFunction(volumeinfo* info, osg::TransferFunction1D::ColorMap* colorMap = NULL); 
        float convertValue(GLenum dataType, const char* data);
        void insertValue(GLenum dataType, char* data, float value);
        osg::Image* allocateVolume(int x, int y, int z, int &sizeS, int &sizeT, int &sizeR, int numberBytesPerComponent, GLenum &datatype); 
	    void clampToNearestValidPowerOfTwo(int& sizeX, int& sizeY, int& sizeZ, 
        int s_maximumTextureSize, int t_maximumTextureSize, int r_maximumTextureSize);
};

#endif
