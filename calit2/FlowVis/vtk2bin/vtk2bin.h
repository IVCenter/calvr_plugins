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
    int coreDataOffset;
    int coreDataSize;
    float min;
    float max;
};

struct SepAttLineData
{
    osg::ref_ptr<osg::Vec3Array> sverts;
    std::vector<osg::ref_ptr<osg::DrawArrays> > sSegments;
    int sDataOffset;
    int sDataSize;
    osg::ref_ptr<osg::Vec3Array> averts;
    std::vector<osg::ref_ptr<osg::DrawArrays> > aSegments;
    int aDataOffset;
    int aDataSize;
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
    int dataOffset;
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
    int vertsDataOffset;
    int vertsDataSize;
    int indicesDataOffset;
    int indicesDataSize;
    int surfaceDataOffset;
    int surfaceDataSize;
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
void processFrameWithFX(FlowDataSet * set, int frame);

void writeBinaryFiles(FlowDataSet * set, std::string name);
void writeBinaryFrameFile(FlowDataSet * set, std::string name, int frame);
void writeBinaryMetaFile(FlowDataSet * set, std::string name);

