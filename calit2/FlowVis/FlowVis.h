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

struct VortexCoreData
{
    osg::ref_ptr<osg::Vec3Array> verts;
    std::vector<osg::ref_ptr<osg::DrawArrays> > coreSegments;
    osg::ref_ptr<osg::FloatArray> coreStr;
    float min;
    float max;
};

struct SepAttLineData
{
    osg::ref_ptr<osg::Vec3Array> sverts;
    std::vector<osg::ref_ptr<osg::DrawArrays> > sSegments;
    osg::ref_ptr<osg::Vec3Array> averts;
    std::vector<osg::ref_ptr<osg::DrawArrays> > aSegments;
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
    osg::ref_ptr<osg::Vec4iArray> surfaceFacets;
    osg::ref_ptr<osg::IntArray> surfaceCells;
    osg::ref_ptr<osg::IntArray> cellTypes;
    VortexCoreData * vcoreData;
    SepAttLineData * sepAttData;
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
    float vcoreMin;
    float vcoreMax;
    std::map<std::string,std::pair<float,float> > attribRanges;
};

class FlowObject;

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

        void processWithFX(FlowDataSet * set);

        cvr::SubMenu * _flowMenu;
        cvr::SubMenu * _loadMenu;
        cvr::MenuButton * _removeButton;
        std::vector<cvr::MenuButton*> _loadButtons;
        std::vector<FileInfo*> _loadFiles;

        FlowDataSet * _loadedSet;
        FlowObject * _loadedObject;
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
