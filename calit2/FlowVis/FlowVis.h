#ifndef FLOW_VIS_PLUGIN_H
#define FLOW_VIS_PLUGIN_H

#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/SceneObject.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuList.h>
#include <cvrMenu/MenuRangeValueCompact.h>
#include <cvrMenu/MenuRangeValue.h>

#include <osg/PrimitiveSet>
#include <osg/Array>
#include <osg/BoundingBox>
#include <osg/Program>
#include <osg/Texture1D>

#include <string>
#include <vector>

struct FileInfo
{
    std::string path;
    int start;
    int frames;
};

enum VTKAttribType
{
    VAT_SCALARS=0,
    VAT_VECTORS,
    VAT_UNKNOWN
};

enum VTKDataType
{
    VDT_INT=0,
    VDT_DOUBLE,
    VDT_UNKNOWN
};

struct VTKDataAttrib
{
    std::string name;
    VTKAttribType attribType;
    VTKDataType dataType;
    osg::ref_ptr<osg::IntArray> intData;
    osg::ref_ptr<osg::FloatArray> floatData;
    osg::ref_ptr<osg::Vec3Array> vecData;
    int intMin, intMax;
    float floatMin, floatMax;
};

struct VTKDataFrame
{
    osg::ref_ptr<osg::Vec3Array> verts;
    osg::ref_ptr<osg::DrawElementsUInt> indices;
    osg::ref_ptr<osg::DrawElementsUInt> surfaceInd;
    osg::ref_ptr<osg::IntArray> cellTypes;
    osg::BoundingBox bb;
    std::vector<VTKDataAttrib*> cellData;
    std::vector<VTKDataAttrib*> pointData;
};

enum FlowDataType
{
    FDT_VTK=0,
    FDT_UNKNOWN
};

struct FlowDataSet
{
    FileInfo info;
    FlowDataType type;
    std::vector<VTKDataFrame*> frameList;
    osg::ref_ptr<osg::Geode> geode;
    osg::ref_ptr<osg::Geometry> geometry;
    osg::ref_ptr<osg::StateSet> stateset;
    osg::ref_ptr<osg::Geometry> isoGeometry;
    std::map<std::string,std::pair<float,float> > attribRanges;
};

class FlowVis : public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:
        FlowVis();
        virtual ~FlowVis();

        bool init();
        void preFrame();
        void menuCallback(cvr::MenuItem * item);

    protected:
        FlowDataSet * parseVTK(std::string filePath, int start, int frames);
        VTKDataAttrib * parseVTKAttrib(FILE * file, std::string type, int count);
        void extractSurfaceVTK(VTKDataFrame * frame);
        void deleteVTKFrame(VTKDataFrame * frame);

        void initColorTable();

        cvr::SubMenu * _flowMenu;
        cvr::SubMenu * _loadMenu;
        cvr::MenuRangeValueCompact * _targetFPSRV;
        cvr::MenuRangeValue * _isoMaxRV;
        cvr::MenuButton * _removeButton;
        std::vector<cvr::MenuButton*> _loadButtons;
        std::vector<FileInfo*> _loadFiles;

        FlowDataSet * _loadedSet;
        cvr::SceneObject * _loadedObject;
        cvr::MenuList * _loadedAttribList;
        std::string _lastAttribute;
        int _currentFrame;
        double _animationTime;

        osg::ref_ptr<osg::Program> _normalProgram;
        osg::ref_ptr<osg::Program> _normalFloatProgram;
        osg::ref_ptr<osg::Program> _normalIntProgram;
        osg::ref_ptr<osg::Program> _isoProgram;

        osg::ref_ptr<osg::Uniform> _floatMinUni;
        osg::ref_ptr<osg::Uniform> _floatMaxUni;
        osg::ref_ptr<osg::Uniform> _intMinUni;
        osg::ref_ptr<osg::Uniform> _intMaxUni;
        osg::ref_ptr<osg::Uniform> _isoMaxUni;

        osg::ref_ptr<osg::Texture1D> _lookupColorTable;
};

struct SetBoundsCallback : public osg::Drawable::ComputeBoundingBoxCallback
{
    osg::BoundingBox computeBound(const osg::Drawable &) const
    {
        return bbox;
    }
    osg::BoundingBox bbox;
};

#endif
