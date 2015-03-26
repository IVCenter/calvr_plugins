#include <osgDB/WriteFile>
#include <osg/Material>
#include <osg/Vec4>
#include <osg/StateSet>
#include <osg/PolygonMode>
#include <osgViewer/Viewer>
#include <osgUtil/SmoothingVisitor>
#include <osg/TriangleFunctor>
#include <osgUtil/Optimizer>
#include <osg/PointSprite>
#include <osg/BlendFunc>
#include <osg/AlphaFunc>
#include <osg/StateAttribute>
#include <osg/Point>
#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osg/Texture2D>
#include <osgDB/ReadFile>
#include <osg/ValueObject>

#include <sstream>
#include <iostream>
#include <iterator>
#include <fstream>
#include <limits>
#include <iomanip>
#include <exception>

#include <osg/ArgumentParser>
#include <osgDB/Archive>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/FileUtils>


using namespace std;

const unsigned int LEAFSIZE = 32768;

unsigned long totalNumberOfPoints = 0;
unsigned int numberOfLeafNodes = 0;
double averageDistance = 0.0;
unsigned int deepestNode = 0;
double deviationDistance = 0.0;

// percentage leaves to compute average
float percentage = 20.0f;

// shaders for spheres
static const char* sv = {
    "#version 150 compatibility\n"
    "#extension GL_ARB_gpu_shader5 : enable\n"
    "void main(void)\n"
    "{\n"
    "    gl_FrontColor = gl_Color;\n"
    "    gl_Position = gl_ModelViewMatrix * gl_Vertex;\n"
    "}\n"
};

static const char* sg = {
"#version 150 compatibility\n"
"#extension GL_EXT_geometry_shader4: enable\n"
"#extension GL_ARB_gpu_shader5 : enable\n"

"uniform float global_alpha;\n"
"uniform float point_size;\n"
"flat out vec4 eye_position;\n"
"flat out float sphere_radius;\n"

"void main(void)\n"
"{\n"
"	 vec4 distance = gl_ProjectionMatrix * gl_PositionIn[0];\n"

"    float ratio = clamp ( distance.z / distance.w, 0.0, 1.0);\n"
"    ratio = ratio * ratio;\n"

"    float scale = length(vec3(gl_ModelViewMatrix[0]));\n"

"    sphere_radius = scale * point_size;\n"
"    float halfsize = sphere_radius * 0.5;\n"

"    gl_FrontColor = gl_FrontColorIn[0];\n"
"    gl_FrontColor.a = global_alpha;\n"

"    eye_position = gl_PositionIn[0];\n"

"    gl_TexCoord[0].st = vec2(-1.0,-1.0);\n"
"    gl_Position = gl_PositionIn[0];\n"
"    gl_Position.xy += vec2(-halfsize, -halfsize);\n"
"    gl_Position = gl_ProjectionMatrix * gl_Position;\n"
"    EmitVertex();\n"

"    gl_TexCoord[0].st = vec2(-1.0,1.0);\n"
"    gl_Position = gl_PositionIn[0];\n"
"    gl_Position.xy += vec2(-halfsize, halfsize);\n"
"    gl_Position = gl_ProjectionMatrix * gl_Position;\n"
"    EmitVertex();\n"

"	gl_TexCoord[0].st = vec2(1.0,-1.0);\n"
"    gl_Position = gl_PositionIn[0];\n"
"    gl_Position.xy += vec2(halfsize, -halfsize);\n"
"    gl_Position = gl_ProjectionMatrix * gl_Position;\n"
"    EmitVertex();\n"

"    gl_TexCoord[0].st = vec2(1.0,1.0);\n"
"    gl_Position = gl_PositionIn[0];\n"
"    gl_Position.xy += vec2(halfsize, halfsize);\n"
"    gl_Position = gl_ProjectionMatrix * gl_Position;\n"
"    EmitVertex();\n"

"    EndPrimitive();\n"
"}\n"
};

static const char * sf = {
"#version 150 compatibility\n"
"#extension GL_ARB_gpu_shader5 : enable\n"

"flat in float sphere_radius;\n"
"flat in vec4 eye_position;\n"

"void main(void)\n"
"{\n"
"	 float x = gl_TexCoord[0].x;\n"
"    float y = gl_TexCoord[0].y;\n"
"    float zz = 1.0 - x*x - y*y;\n"

"    if (zz <= 0.0 )\n"
"        discard;\n"

"    float z = sqrt(zz);\n"
"    vec3 normal = vec3(x, y, z);\n"

"    float diffuse_value = max(dot(normal, vec3(0, 0, 1)), 0.0);\n"

"    vec4 pos = eye_position;\n"
"    pos.z += z*sphere_radius;\n"
"    pos = gl_ProjectionMatrix * pos;\n"

"    gl_FragDepth = (pos.z / pos.w + 1.0) / 2.0;\n"
"    gl_FragColor.rgb = gl_Color.rgb * diffuse_value;\n"
"    gl_FragColor.a = gl_Color.a;\n"
"}\n"
};


struct Point
{
    double pos[3];
    unsigned char color[3];
};

std::istream& operator >> (std::istream& i, Point & p)
{
    i.read((char*)&p, sizeof(Point));
    return i;
};

std::ostream& operator << (std::ostream& i, const Point& p)
{
    i.write((char*)&p, sizeof(Point));
    return i;
};


// global param that determines what axis a point list is sorted in
int sortParam = 0;

struct Cmp
{
	bool operator () (const Point& a, const Point& b) const
	{
		return (a.pos[sortParam] < b.pos[sortParam]);
	}

	static Point min_value()
	{
        Point p;
        p.pos[0] = p.pos[1] = p.pos[2] = std::numeric_limits<double>::min();
		return p;
	}

	static Point max_value()
	{
        Point p;
        p.pos[0] = p.pos[1] = p.pos[2] = std::numeric_limits<double>::max();
		return p;
	}
};

struct WriteFile
{
    std::ostream & out;

    WriteFile(std::ostream & o_):out(o_)
    {
    }

    void operator () ( const Point & p)
    {
        out << p; 
    }
};

// note make sure the sort param is set correctly before calling sort on the Point vector 
bool operator<(const Point &a, const Point &b)
{
    return (a.pos[sortParam] < b.pos[sortParam]);
};

typedef std::vector<Point> pvector;


class ArchivePagedLODSubgraphsVistor : public osg::NodeVisitor
{
private:
    std::string _archiveName;
    osg::ref_ptr<osgDB::Archive> _archive;
    bool _firstGeode;
    osg::Node* _baseNode;
    float _smallestArea, _overAllDensity, _rootDistance, _leafDistance;

public:
    ArchivePagedLODSubgraphsVistor(std::string archiveName, osg::Node* baseNode):
        osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN)
    {
        _baseNode = baseNode;
        _archiveName = archiveName;

        // create archive based on name
        _archive = osgDB::openArchive(std::string(_archiveName) + ".osga", osgDB::Archive::CREATE);

        // add key file first
        _archive->writeNode(*baseNode, std::string("base.ive"));

    }

    virtual void apply(osg::PagedLOD& plod)
    {
        // go through all the named children and write them out to disk.
        for(unsigned int i=0;i<plod.getNumChildren();++i)
        {
            osg::Node* child = plod.getChild(i);
            std::string filename = plod.getFileName(i);
            if (!filename.empty())
            {
                //osg::notify(osg::NOTICE)<<"Adding to archive "<<filename<<std::endl;
                _archive->writeNode(*child, filename);
            }
        }

        traverse(plod);
    }

	~ArchivePagedLODSubgraphsVistor()
	{
	}
};

// update the end to take into account the points that are removed for the leaf
osg::Geode* createPoints( unsigned long start, unsigned long & end, pvector & referencePoints,  osg::BoundingBoxd &remainingBox)
{
    // randomize points
    std::random_shuffle ( referencePoints.begin() + start, referencePoints.begin() + end );

    // set default leaf size
    unsigned int leafSize = LEAFSIZE;

    // smaller leaf
    bool smallLeaf = false;
    unsigned long stepSize = 1;
    
    // set the number of points that need to looped through
    unsigned long numPoints = end - start;

    // make sure have enough points for leaf
    if( numPoints < (2 * LEAFSIZE) )
    //if( numPoints < LEAFSIZE )
    {
        leafSize = numPoints;
        smallLeaf = true;
    }
    else // step size needs to be adjusted
    {
        stepSize = numPoints / (LEAFSIZE - 1);
    }

    // specify the size
	osg::Geode * points = new osg::Geode();
    osg::Geometry *pgeom = new osg::Geometry();
    osg::Vec3Array * vertices = new osg::Vec3Array(leafSize);
    osg::Vec4ubArray * colors = new osg::Vec4ubArray(leafSize);

    // shift remaining points forward (e.g. resultant remaining counter should be (end - start) - leafSize)
    unsigned long remainingCounter = start;
    unsigned int geomIndex = 0;

    // reset point bounds
    remainingBox.init();

    // temp boundingtest
    osg::BoundingBox box;

    // shift points forward and add sample of points to leaf node
    for(unsigned long index = 0; index < numPoints; index++)
    {
        Point p = referencePoints[index + start];

        if( smallLeaf || (((index % stepSize) == 0) && (geomIndex < leafSize)) )
        {
            // add to leaf
            vertices->at(geomIndex).set(p.pos[0], p.pos[1], p.pos[2]);

            // look up from the file via index
            colors->at(geomIndex).set(p.color[0], p.color[1], p.color[2], static_cast<unsigned char> (255));
            
            box.expandBy(p.pos[0], p.pos[1], p.pos[2]);

            geomIndex++;
        }
        else // push unused points forward in vector
        {
            // left over point push forward
            referencePoints[remainingCounter] = p;
            
            // expand bound to compute new bound
            remainingBox.expandBy(p.pos[0], p.pos[1], p.pos[2]);

            // increment counter
            remainingCounter++;
        }
    }
    
    // update end
    end = end - leafSize;

    // check distance in leaf (used for computing a point size for rendering)
    if( (vertices->size() != LEAFSIZE) &&  (vertices->size() > 1) && ( (rand() % 100) < percentage ) )
    {
        float distance = FLT_MAX;
        for(int i = 1; i < vertices->size(); i++)
        {
            float testDist = (vertices->at(0) - vertices->at(i)).length2();
            if( testDist < distance && testDist != 0.0 )
                distance = testDist;
        }

        numberOfLeafNodes++;
        averageDistance += pow(distance, 0.5);
        deviationDistance += distance;
    }

    pgeom->setVertexArray(vertices);
    pgeom->setColorArray(colors, osg::Array::BIND_PER_VERTEX);
    pgeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, vertices->size()));
    pgeom->setInitialBound(box);
    points->addDrawable(pgeom);

    totalNumberOfPoints += vertices->size();

	return points;
}

void sortPoints(unsigned long start, unsigned long end, int axis, pvector & referenceData, std::string name, osg::BoundingBoxd box, osg::Group *parent, bool split, int depth)
{

    // stop when resulting bounbing box is zero
    if( !box.valid() )
        return;

    // increment depth
    depth++;
        
    //check if depth is higher than depest point
    if( depth > deepestNode )
        deepestNode = depth;     

	// compute next sort param
	sortParam = axis;

    std::sort(referenceData.begin() + start, referenceData.begin() + end, Cmp());
	
    // find half way point
	unsigned long midPoint = ((end - start) / 2) + start;

    // this will split based on area not mid point (produce unbalanced tree, but should be better for frustrum culling)
    if( split )
    {
        double splitValue = ((box.corner(7)[axis] - box.corner(0)[axis]) * 0.5) + box.corner(0)[axis];
        Point p;
        p.pos[axis] = splitValue;
        pvector::iterator result = std::upper_bound(referenceData.begin() + start, referenceData.begin() + end, p, Cmp());
        midPoint = (result - referenceData.begin());
    }

    // create minBox dimensions
    osg::BoundingBox minBox(box.corner(0), box.corner(7));

    // adjust the bounds
    switch ( axis )
    {
        case 0:
            minBox.xMax() = referenceData[midPoint].pos[axis];
            break;
        case 1:
            minBox.yMax() = referenceData[midPoint].pos[axis];
            break;
        case 2:
            minBox.zMax() = referenceData[midPoint].pos[axis];
            break;
        default:
            break;
    }

    // add min plod
    osg::Vec3 min(minBox.corner(0));
    osg::Vec3 max(minBox.corner(7));

    // create minsize lod
    osg::PagedLOD* minplod = new osg::PagedLOD();
    minplod->setCenterMode(osg::LOD::USER_DEFINED_CENTER);
    minplod->setCenter( minBox.center() );
    minplod->setRadius( minBox.radius() );
    parent->addChild(minplod);

    osg::Group* minChild = new osg::Group();
    
    // add a bounding box
    //minChild->addChild(createBound(min, max, colors[axis]));

    // add a point cloud
    unsigned long updatedEnd = midPoint;

    // determine tighter bounds for remaining points 
    osg::BoundingBoxd minBounds;
    float pointSize = 0.0;
    minChild->addChild(createPoints(start, updatedEnd, referenceData, minBounds));
    minplod->addChild(minChild, 0.0, minBox.radius() * 15, name + "0.ive");
   
    //cerr << "Old way: " << (max - min).length() * 5 << " new way " << minBox.radius() * 15 << std::endl; 

    // check to see if the data needs to be split again
    //if( updatedEnd  !=  start )
    //{
        // use longest dimension to determine split
        double xSize = minBounds.xMax() - minBounds.xMin();
        double ySize = minBounds.yMax() - minBounds.yMin();
        double zSize = minBounds.zMax() - minBounds.zMin();
        int currentSplit = 0;
  
        if( ySize > zSize && ySize > xSize)
            currentSplit = 1;

        if( zSize > ySize && zSize > xSize)
            currentSplit = 2;

        // sort remaining leaf points
	    sortPoints(start, updatedEnd, currentSplit, referenceData, name + "0", minBounds, minChild, split, depth);
	    //sortPoints(start, updatedEnd, currentSplit, referenceData, name + "0", minBounds, minChild);
    //}
  
    // process the upper half 
    osg::BoundingBoxd maxBox(box.corner(0), box.corner(7));
    
    // adjust the bounds
    switch ( axis )
    {
        case 0:
            maxBox.xMin() = referenceData[midPoint].pos[axis];
            break;
        case 1:
            maxBox.yMin() = referenceData[midPoint].pos[axis];
            break;
        case 2:
            maxBox.zMin() = referenceData[midPoint].pos[axis];
            break;
        default:
            break;
    }

    // add max plod
    min.set(maxBox.corner(0));
    max.set(maxBox.corner(7));
   
    // create max side lod 
    osg::PagedLOD* maxplod = new osg::PagedLOD();
    maxplod->setCenterMode(osg::LOD::USER_DEFINED_CENTER);
    maxplod->setCenter( maxBox.center() );
    maxplod->setRadius( maxBox.radius() );
    parent->addChild(maxplod);

    osg::Group* maxChild = new osg::Group();

    //maxChild->addChild(createBound(min, max, colors[axis]));

    // add a point cloud
    updatedEnd = end;

    osg::BoundingBoxd maxBounds;
    maxChild->addChild(createPoints(midPoint, updatedEnd, referenceData, maxBounds));
    maxplod->addChild(maxChild, 0.0 , maxBox.radius() * 15, name + "1.ive");
    
    //cerr << "Old way: " << (max - min).length() * 5 << " new way " << maxBox.radius() * 15 << std::endl; 
  
    // check if more processing needs to occur 
    //if( updatedEnd != midPoint )
    //{
    
        // use longest dimension to determine split
        xSize = maxBounds.xMax() - maxBounds.xMin();
        ySize = maxBounds.yMax() - maxBounds.yMin();
        zSize = maxBounds.zMax() - maxBounds.zMin();
        currentSplit = 0;
  
        if( ySize > zSize && ySize > xSize)
            currentSplit = 1;

        if( zSize > ySize && zSize > xSize)
            currentSplit = 2;

        // sort remaining leaf points    
	    sortPoints(midPoint, updatedEnd, currentSplit, referenceData, name + "1", maxBounds, maxChild, split, depth);
	    //sortPoints(midPoint, updatedEnd, currentSplit, referenceData, name + "1", maxBounds, maxChild);
    //}
};

int main (int argc, char** argv)
{
  // pass and manage arguments
  osg::ArgumentParser arguments(&argc,argv);
  
  // argument set up Points2Archive test.xyz baseName 3 
  arguments.getApplicationUsage()->setApplicationName(arguments.getApplicationName());
  arguments.getApplicationUsage()->setCommandLineUsage(arguments.getApplicationName() + 
            " -i inputTextFile -o archiveName -input_format xyzrgbi -output_format xyzrgb -color_scale 1.0 -intensity_scale 0.165 -skip_header 7");

  // if user request help write it out to cout.
  if (arguments.read("-h") || arguments.read("--help"))
  {
    arguments.getApplicationUsage()->write(std::cout);
    return 1;
  }

  std::string inputFileName;
  std::string outputFileName("default");
  std::string inputFormat("xyz");
  std::string outputFormat("xyz");
  bool binaryInput = false;
  int skipHeader = 0;

  float colorScale = 1.0;
  float intensityScale = 1.0;

  float xShift = 0.0f;
  float yShift = 0.0f;

  while( arguments.read("-binary") )
  {
    binaryInput = true;
  }
  
  while (arguments.read("-xshift", xShift))
  {
  }    

  while (arguments.read("-yshift", yShift))
  {
  }    
  
  while (arguments.read("-i", inputFileName))
  {
  }    
  
  while (arguments.read("-o", outputFileName))
  {
  }    
  
  while (arguments.read("-output_format", outputFormat))
  {
  }    
  
  while (arguments.read("-input_format", inputFormat))
  {
  }    
  
  while (arguments.read("-color_scale", colorScale))
  {
  }    
  
  while (arguments.read("-intensity_scale", intensityScale))
  {
  }    
  
  while (arguments.read("-skip_header", skipHeader))
  {
  }    
  
  
  // create mapping look up
  std::vector<float> inputMapping;

  // check if any scales need to be set
  for(int i = 0; i < inputFormat.size();i++)
  {
      // check for color scale adjustment
      if(inputFormat.at(i) == 'i') 
      {
        inputMapping.push_back(intensityScale);
      }
      else if((inputFormat.at(i) == 'r') || (inputFormat.at(i) == 'g') || (inputFormat.at(i) == 'b'))
      {
         inputMapping.push_back(colorScale); 
      }
      else
      {
         inputMapping.push_back(1.0); 
      } 
  }

  // create output vector
  int outputMapping[6];

  // initalize to -1 (no data use default
  outputMapping[0] = outputMapping[1] = outputMapping[2] = outputMapping[3] = outputMapping[4] = outputMapping[5] = -1;
  
  for(int i = 0; i < outputFormat.size(); i++)
  {
      // find index mapping
      std::size_t found = inputFormat.find(outputFormat.at(i));
      if( found != std::string::npos )
      {
          outputMapping[i] = found;
      }
      else
      {
            std::cerr << "ERROR: No mapping found for: " << outputFormat.at(i) << " in input mapping\n";
            exit(1);
      }
  }

  // read in points and sort into bsp tree structure
  ifstream ifs(inputFileName.c_str());

  // root node to attach kd-tree
  osg::Group* group = new osg::Group();

  // set a CalVR node mask to disable special culling
  group->setNodeMask(group->getNodeMask() & ~0x1000000);

  // create shader for points
  osg::Program* prog = new osg::Program;
  prog->setName( "Sphere" );

  // load in shaders
  prog->addShader(new osg::Shader(osg::Shader::VERTEX, sv));
  prog->addShader(new osg::Shader(osg::Shader::FRAGMENT, sf));
  prog->addShader(new osg::Shader(osg::Shader::GEOMETRY, sg));
  prog->setParameter( GL_GEOMETRY_VERTICES_OUT_EXT, 4 );
  prog->setParameter( GL_GEOMETRY_INPUT_TYPE_EXT, GL_POINTS );
  prog->setParameter( GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP );

  // variables to hold data while reading 
  string value, values;
  stringstream ss;
  stringstream ssdouble;
  double dataValue;
  unsigned int color;
  int index;
  Point p;

  // if not binary create a binary representation
  if( !binaryInput )
  {
	  // open a binary file for writing all the points in
	  ofstream ofs (std::string(outputFileName).append(".bin").c_str(), ios::binary);
	  
	  // hold input data
	  double data[inputMapping.size()];

      // skip the first few lines on the configuration file (to deal with ply files)
      if( skipHeader > 0 )
      {
        for(int i =0; i < skipHeader; i++)
            getline( ifs, values );
      }

	  while( getline( ifs, values ) )
	  {
		  ss.str("");
		  ss.clear();
		  ss << values;

		  // reset index
		  index = 0;

		  // read in position
		  while( index < inputMapping.size() && ss >> value)
		  {
		      ssdouble << value;
		      ssdouble >> dataValue;
		      data[index] = dataValue * inputMapping.at(index);
		      ssdouble.str("");
		      ssdouble.clear();
		      index++;
		  }

		  // add point if valid
		  if(  index == inputMapping.size() )
		  {

		    // add point location
		    for(int i = 0; i < 3; i++)
		    {
		        p.pos[i] = data[outputMapping[i]];
            
                // check if a shift needs to occur    
                if( xShift != 0.0 && i == 0 )
                {
                    p.pos[i] += xShift;
                }
                else if( yShift != 0.0 && i == 1)
                {
                    p.pos[i] += yShift;
                }
		    }

		    // add color
		    for(int i = 0; i < 3; i++)
		    {
		        index = outputMapping[i+3];
		        if( index != -1 )
		        {
		            p.color[i] = static_cast<unsigned char>((int)data[index]);
		        }
		        else
		        {
		            p.color[i] = static_cast<unsigned char>(128);
		        }
		    }
            
		    // add point to binary file 
		    ofs << p;

		    //bounds.expandBy(osg::Vec3(p.pos[0], p.pos[1], p.pos[2])); 
		  }
	  }
	  ofs.close();
  }

  ifs.close();

  // try sorting
  std::fstream in(std::string(outputFileName).append(".bin").c_str(), std::ios::in | std::ios::binary);
  pvector v;
      
  std::copy(std::istream_iterator<Point>(in),
            std::istream_iterator<Point>(),
            std::back_inserter(v));
  
  std::cerr << "Number of Points read in is: " << v.size() << std::endl;

  // add base points ( these points are not initially sorted )
  unsigned long endIndex = v.size();

  // resulting bounds
  osg::BoundingBoxd bounds;
 
  // add base child (the resulting bounds will be used to determine first split axis)
  group->addChild(createPoints(0, endIndex, v, bounds));
  
  std::cerr << std::setprecision(std::numeric_limits<double>::digits10 + 1) << "Remaining Bounds: min " << bounds.xMin() << " " << bounds.yMin() << " " << bounds.zMin() <<std::endl;
  std::cerr << std::setprecision(std::numeric_limits<double>::digits10 + 1) << "Remaining Bounds: max " << bounds.xMax() << " " << bounds.yMax() << " " << bounds.zMax() <<std::endl;

  // use longest dimension to determine split
  double xSize = bounds.xMax() - bounds.xMin();
  double ySize = bounds.yMax() - bounds.yMin();
  double zSize = bounds.zMax() - bounds.zMin();
  int currentSplit = 0;
  
  if( ySize > zSize && ySize > xSize)
    currentSplit = 1;

  if( zSize > ySize && zSize > xSize)
    currentSplit = 2;

  // create a recursive sort function, need to update map for kd tree structure
  sortPoints(0, endIndex, currentSplit, v, std::string("base"), bounds, group, true, 2);

  // set stateset based on point size
  osg::StateSet* state = group->getOrCreateStateSet();
  state->setAttribute(prog);

  std::cerr << "Deepest node created " << deepestNode << std::endl;

  float minPointSize = (float)averageDistance / numberOfLeafNodes / deepestNode;
  state->addUniform(new osg::Uniform("point_size", minPointSize + (float) pow( pow( deviationDistance / deepestNode / (numberOfLeafNodes - 1), 0.5), 0.5 )));
  state->addUniform(new osg::Uniform("global_alpha",1.0f));
  state->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

  std::cerr << "Point Sorting Completed\n";
  std::cerr << "Total number of points in geodes is: " << totalNumberOfPoints << std::endl;

  // create archive
  ArchivePagedLODSubgraphsVistor archVisitor(outputFileName, group);
  group->accept(archVisitor);

  std::cerr << "Archive completed\n";
  return 0;
}
