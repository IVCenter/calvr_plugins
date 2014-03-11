#include <string>
#include <vector>

#include <osg/BoundingBox>
#include <osg/Array>
#include <osg/PrimitiveSet>

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

struct FlowDataSet
{
    std::vector<VTKDataFrame*> frameList;
    std::map<std::string,std::pair<float,float> > attribRanges;
    osg::BoundingBox bb;
};

bool parseVTK(std::string file, FlowDataSet * set);
VTKDataAttrib * parseVTKAttrib(FILE * file, std::string type, int count);
void extractSurfaceVTK(VTKDataFrame * frame);
void deleteVTKFrame(VTKDataFrame * frame);

void processWithFX(FlowDataSet * set);

void writeBinaryFiles(FlowDataSet * set, std::string name);

