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
#include <osg/Geometry>
#include <osg/Depth>
#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <osg/LineWidth>
#include <osg/BoundingBox>
#include <osgViewer/ViewerEventHandlers>

//#include <stxxl.h>

#include <sstream>
#include <iostream>
#include <iterator>
#include <fstream>
#include <exception>
#include <unordered_map>
#include <map>

#include <osg/ArgumentParser>
#include <osgDB/Archive>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/FileUtils>

#include <json/json.h>

using namespace std;

#define USE_STXXL 0

const double M_2PI = M_PI*2;

unsigned int counter = 0;

// used to hold color mapping lookup
osg::Vec3f sampleColorTable[12];
osg::Vec3f otuColorTable[12];
osg::Vec3f textColorTable[12];

struct IndexValue
{
    int index;
    float value;
};

// call back to attach and use to change color mapping for vertexs
struct DrawableUpdateCallback : public osg::Drawable::UpdateCallback
{
    bool _update;
    OpenThreads::Mutex _mutex;
    std::vector<IndexValue> * _changes;

    DrawableUpdateCallback() : _update(false) , _changes(NULL) {}

    void applyUpdate( std::vector<IndexValue> * changes)
    {
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex); 
        _changes = changes;
        _update = true;
    }

    // need to protect to update list is not changed which callback is in use
    virtual void update(osg::NodeVisitor*, osg::Drawable* drawable)
    {  
        // update the vertex color mapping 
       OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex); 
       if( _update && _changes )
       {
            osg::Geometry* geom = dynamic_cast<osg::Geometry* > (drawable);
            if( geom )
            {
                osg::Vec4Array* colors = dynamic_cast<osg::Vec4Array*> (geom->getColorArray());
                if ( colors )
                {
                    for(int i = 0; i < _changes->size(); i++)
                    {
                        osg::Vec4 *color = &colors->operator [](_changes->at(i).index);
                        color->set( (*color)[0], (*color)[1], (*color)[2], _changes->at(i).value );
                        //color[3] =  _changes->at(i).value;
                    }
                                                                                   
                    //geom->setColorArray(colors);
                }                                                                                        
            }   
            _update = false;
       }
    }
};



class ComputeBounds: public osg::NodeVisitor
{

	private:
		osg::BoundingBox m_bb;
		osg::Matrix m_curMatrix;

	public:

		ComputeBounds() :
		    osg::NodeVisitor(TRAVERSE_ALL_CHILDREN)
		{
			m_curMatrix = osg::Matrix();
			m_bb.init();
		}

		void apply(osg::Geode &geode)
		{
			for(unsigned int i = 0; i < geode.getNumDrawables(); i++)
			{
				//osg::BoundingBox bb = geode.getDrawable(i)->getBoundingBox();
				osg::BoundingBox bb = geode.getDrawable(i)->getBound();

				m_bb.expandBy(bb.corner(0) * m_curMatrix);
				m_bb.expandBy(bb.corner(1) * m_curMatrix);
				m_bb.expandBy(bb.corner(2) * m_curMatrix);
				m_bb.expandBy(bb.corner(3) * m_curMatrix);
				m_bb.expandBy(bb.corner(4) * m_curMatrix);
				m_bb.expandBy(bb.corner(5) * m_curMatrix);
				m_bb.expandBy(bb.corner(6) * m_curMatrix);
				m_bb.expandBy(bb.corner(7) * m_curMatrix);
			}
		}

		void apply(osg::Transform& node)
		{
			if(node.asMatrixTransform() || node.asPositionAttitudeTransform())
			{
				osg::Matrix prevMatrix = m_curMatrix;

				//m_curMatrix.preMult(node.asMatrixTransform()->getMatrix());
				node.computeLocalToWorldMatrix(m_curMatrix,this);

				traverse(node);

				m_curMatrix = prevMatrix;
			}
		}

		osg::BoundingBox & getBound()
		{
			return m_bb;
		}
};

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

// used for color map
struct PropertyMap
{
    std::string property;
    osg::Vec3f color;
};

// object to hold full color mapping data
struct ColorMap
{
    std::string metaName;
    PropertyMap mapping[12];
};

struct LineVertices
{
    unsigned int sInd;
    unsigned int oInd;
};

// structues to hold vertex mapping info
struct VertexMap
{
    unsigned int pointIndex;
    std::vector< LineVertices > * lineIndexs; // first point always sample
};

// forward declaration
struct OTU;
struct Sample;

struct Position
{
    double x;
    double y;
};

struct Point
{
    double pos[3];
    unsigned int name;
    unsigned int vertexIndex;
    std::string sName;

    Point()
    {
        pos[0] = pos[1] = pos[2] = 0.0;
        name = counter;
        counter++;
    };
};

struct OTUEdge
{
    OTU* otu;
    Sample* sample;
    float weight;
    float normalizedWeight;
    unsigned int vertexIndex[2];    // need to hold two points

    OTUEdge()
    {
        sample = NULL;
        otu = NULL;
        weight = normalizedWeight = 0.0;
    };

    OTUEdge(OTU* o, Sample* s, float w)
    {
      otu = o;
      sample = s;
      weight = w;  
    }; 
};

struct Sample
{
    Point p; // name in point
    std::vector< OTUEdge* >* edges;
    float highestWeight;

    Sample()
    {
		highestWeight = 0.0;
        p.pos[0] = p.pos[1] = p.pos[2] = 0.0;
        edges = new std::vector< OTUEdge* > ();
    };
};

struct OTU
{
    Point p; // NOTE can change to index to a list of all possible positions on the surface
    // name in point
    std::vector< OTUEdge* >* edges;
    float highestWeight;

    OTU()
    {
		highestWeight = 0.0;
        p.pos[0] = p.pos[1] = p.pos[2] = 0.0;
        edges = new std::vector< OTUEdge* >();
    };
};

// data structure for holding the bsp intersection testing
struct Bound;

struct Bound
{
    std::string boundName;
    osg::BoundingBox bb;
    std::vector<std::string> * names;
    Bound* left;
    Bound* right;

    Bound()
    {
        bb.init();
        names = new std::vector< std::string >();
    };
};


osg::StateSet* makeStateSet(float size)
{
	osg::StateSet *set = new osg::StateSet();
    
	/// Setup cool blending
	set->setMode(GL_BLEND, osg::StateAttribute::ON);
	osg::BlendFunc *fn = new osg::BlendFunc();
	fn->setFunction(osg::BlendFunc::SRC_ALPHA, osg::BlendFunc::DST_ALPHA);
	set->setAttributeAndModes(fn, osg::StateAttribute::ON);
	
	/// Setup the point sprites
	osg::PointSprite *sprite = new osg::PointSprite();
	set->setTextureAttributeAndModes(0, sprite, osg::StateAttribute::ON);
	
	/// Give some size to the points to be able to see the sprite
	osg::Point *point = new osg::Point();
	point->setSize(size);
	set->setAttribute(point);
	
	/// Disable depth test to avoid sort problems and Lighting
	set->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF);
	set->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
	
	/// The texture for the sprites
	osg::Texture2D *tex = new osg::Texture2D();
	tex->setImage(osgDB::readRefImageFile("/home/pweber/development/DividePoints/particle.rgb"));
	set->setTextureAttributeAndModes(0, tex, osg::StateAttribute::ON);
    return set;
}

osg::Geode* createBound( osg::Vec3 min, osg::Vec3 max, osg::Vec4ub color)
{
    osg::Geometry * geometry = new osg::Geometry();
    osg::Vec3Array * verts = new osg::Vec3Array(0);
    osg::Vec4ubArray * colors = new osg::Vec4ubArray(1);
    osg::DrawArrays * primitive = new osg::DrawArrays(osg::PrimitiveSet::LINES,
            0,0);
    geometry->setVertexArray(verts);
    geometry->setColorArray(colors);
    geometry->setColorBinding(osg::Geometry::BIND_OVERALL);
    geometry->addPrimitiveSet(primitive);
    (*colors)[0] = color;
    primitive->setCount(24);

    verts->push_back(min);
    verts->push_back(osg::Vec3(max[0],min[1],min[2]));
    verts->push_back(osg::Vec3(max[0],min[1],min[2]));
    verts->push_back(osg::Vec3(max[0],min[1],max[2]));
    verts->push_back(osg::Vec3(max[0],min[1],max[2]));
    verts->push_back(osg::Vec3(min[0],min[1],max[2]));
    verts->push_back(osg::Vec3(min[0],min[1],max[2]));
    verts->push_back(min);
    verts->push_back(osg::Vec3(min[0],max[1],min[2]));
    verts->push_back(osg::Vec3(max[0],max[1],min[2]));
    verts->push_back(osg::Vec3(max[0],max[1],min[2]));
    verts->push_back(max);
    verts->push_back(max);
    verts->push_back(osg::Vec3(min[0],max[1],max[2]));
    verts->push_back(osg::Vec3(min[0],max[1],max[2]));
    verts->push_back(osg::Vec3(min[0],max[1],min[2]));
    verts->push_back(min);
    verts->push_back(osg::Vec3(min[0],max[1],min[2]));
    verts->push_back(osg::Vec3(max[0],min[1],min[2]));
    verts->push_back(osg::Vec3(max[0],max[1],min[2]));
    verts->push_back(osg::Vec3(min[0],min[1],max[2]));
    verts->push_back(osg::Vec3(min[0],max[1],max[2]));
    verts->push_back(osg::Vec3(max[0],min[1],max[2]));
    verts->push_back(max);

    //geometry->setInitialBound(osg::BoundingBox(min, max));
    osg::Geode * bounds = new osg::Geode();
    bounds->addDrawable(geometry);

    osg::StateSet * stateset = bounds->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

    return bounds;
}

osg::Geode* createLines(std::vector<osg::Vec3>* lines, std::vector<osg::Vec4>* colors)
{
        // do lame copy
        osg::Vec3Array* veca = new osg::Vec3Array(); 
        osg::Vec4Array* colora = new osg::Vec4Array(); 
       
        for(int i = 0; i < lines->size(); i++)
        {
            veca->push_back(lines->at(i));
            colora->push_back(colors->at(i));
        }

        if( lines )
        {
            delete lines;
            lines = NULL;
        }
        
        if( colors )
        {
            delete colors;
            colors = NULL;
        }
   
        std::cerr << "added primitiveset\n";

        // create last geode 
        osg::Geode* geode = new osg::Geode();
        osg::Geometry* linesGeom = new osg::Geometry();
        linesGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES,0,veca->size()));
        linesGeom->setColorArray(colora, osg::Array::BIND_PER_VERTEX);
        linesGeom->setVertexArray(veca);
        geode->addDrawable(linesGeom);
        return geode;
};

// divide up the lines // alternate bound split (between 0 and 1)
void sortLines(std::vector<osg::Vec3>* lines, std::vector<osg::Vec4>* colors, std::string name, osg::BoundingBox box, osg::Group* parent, unsigned int axisSplit, int depth)
{

    // check if depth is zero then stop
    if( depth == 0 )
    {
        parent->addChild(createLines(lines, colors));
        //parent->addChild(createBound(box.corner(0), box.corner(7), osg::Vec4ub(0,255,0, 255)));
        return;
    }

    // create a group with multi geodes that dont over lap
    osg::BoundingBox boxMin, boxMax;

    double splitValue = box.center()[axisSplit];

    osg::Vec3 minAxis = box.corner(0);
    osg::Vec3 maxAxis = box.corner(7);

    osg::Vec3 halfMin = maxAxis;
    halfMin[axisSplit] = splitValue;
    
    osg::Vec3 halfMax = minAxis;
    halfMax[axisSplit] = splitValue;
    
    
    // determine new axis split
    unsigned int newAxis = (axisSplit + 1) % 2;
     
    boxMin.set(minAxis, halfMin);
    boxMax.set(halfMax, maxAxis);

    std::vector<osg::Vec3>* minLines = new std::vector<osg::Vec3>();
    std::vector<osg::Vec4>* minColors = new std::vector<osg::Vec4>();
    
    std::vector<osg::Vec3>* maxLines = new std::vector<osg::Vec3>();
    std::vector<osg::Vec4>* maxColors = new std::vector<osg::Vec4>();
    
    std::vector<osg::Vec3>* remainder = new std::vector<osg::Vec3>();
    std::vector<osg::Vec4>* remainderC = new std::vector<osg::Vec4>();
    
    // to create over lapping lines
    //osg::Vec3Array* remainder = new osg::Vec3Array();
    //osg::Vec4Array* remainderC = new osg::Vec4Array();

    for(int i = 0; i < lines->size(); i = i + 2)
    {
        // test which lines fit in which box
        if( boxMin.contains(lines->at(i)) && boxMin.contains(lines->at(i+1)) )
        { 
            minLines->push_back(lines->at(i));
            minLines->push_back(lines->at(i + 1));
            minColors->push_back(colors->at(i));
            minColors->push_back(colors->at(i + 1));
        }
        else if( boxMax.contains(lines->at(i)) && boxMax.contains(lines->at(i+1)) )
        {
            maxLines->push_back(lines->at(i));
            maxLines->push_back(lines->at(i + 1));
            maxColors->push_back(colors->at(i));
            maxColors->push_back(colors->at(i + 1));
        }
        else  // spans both boxes
        {
            remainder->push_back(lines->at(i));
            remainder->push_back(lines->at(i + 1));
            remainderC->push_back(colors->at(i));
            remainderC->push_back(colors->at(i + 1));
        }
    }

    // free up memory
    if( lines )
    {
        delete lines;
        lines = NULL;
    }
    
    if( colors )
    {
        delete colors;
        colors = NULL;
    }

	// collect all nodes that span across both bounds
    if( remainder->size() )
    {
        //osg::Geode* geode = new osg::Geode();
        //osg::Geometry* linesGeom = new osg::Geometry();
        //linesGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES,0,remainder->size()));
        //linesGeom->setColorArray(remainderC, osg::Array::BIND_PER_VERTEX);
        //linesGeom->setVertexArray(remainder);
        //geode->addDrawable(linesGeom);
        //parent->addChild(geode);
        std::cerr << "Level " << depth << " has " << remainder->size() / 2 << " lines\n";

        parent->addChild(createLines(remainder, remainderC));
        //parent->addChild(createBound(box.corner(0), box.corner(7), osg::Vec4ub(0,255,0,255)));
    }

    // decrement depth
    depth--;

    if( minLines->size() )
	{
		// create minsize lod
    	osg::PagedLOD* minplod = new osg::PagedLOD();
        //minplod->setRangeMode(osg::LOD::PIXEL_SIZE_ON_SCREEN);
    	minplod->setCenterMode(osg::LOD::USER_DEFINED_CENTER);
    	minplod->setCenter( boxMin.center() );
    	minplod->setRadius( boxMin.radius() );
    	parent->addChild(minplod);

		osg::Group* minChild = new osg::Group();
        //minChild->addChild(sortLines(minLines, minColors, name + "0", boxMin, minChild, newAxis, depth));
        sortLines(minLines, minColors, name + "0", boxMin, minChild, newAxis, depth);
        //minChild->addChild(createLines(minLines, minColors));
		minplod->addChild(minChild, 0.0, boxMin.radius(), name + "0.ive");
	}

    if( maxLines->size() )
	{   
		 // create max side lod 
    	osg::PagedLOD* maxplod = new osg::PagedLOD();
        //maxplod->setRangeMode(osg::LOD::PIXEL_SIZE_ON_SCREEN);
    	maxplod->setCenterMode(osg::LOD::USER_DEFINED_CENTER);
    	maxplod->setCenter( boxMax.center() );
    	maxplod->setRadius( boxMax.radius() );
    	parent->addChild(maxplod);

    	osg::Group* maxChild = new osg::Group(); 
    	//maxChild->addChild(sortLines(maxLines, maxColors, name + "1", boxMax, maxChild, newAxis, depth));
    	sortLines(maxLines, maxColors, name + "1", boxMax, maxChild, newAxis, depth);
        //maxChild->addChild(createLines(maxLines, maxColors));
		maxplod->addChild(maxChild, 0.0 , boxMax.radius(), name + "1.ive");
	}
};

/*
osg::Group* createAnimationStructure(std::unordered_map<std::string, Sample* >* samples, std::unordered_map<std::string, OTU* >* otus, osg::BoundingBox bounds, float highWeight, std::unordered_map<std::string, std::pair<std::string, int> >* mappingValues )
{

    // TODO adjust time mapping values
    //std::string path = "/home/calvr/Philip/Layout/";
    std::string path = "/home/pweber/development/LayoutTest/";

    // load in shader to render points
    osg::Program* prog = new osg::Program;
    prog->setName( "Sphere" );
    prog->addShader(osg::Shader::readShaderFile(osg::Shader::VERTEX, path + "AnimSphere.vert"));
    prog->addShader(osg::Shader::readShaderFile(osg::Shader::FRAGMENT, path + "AnimSphere.frag"));
    prog->addShader(osg::Shader::readShaderFile(osg::Shader::GEOMETRY, path + "AnimSphere.geom"));
    prog->setParameter( GL_GEOMETRY_VERTICES_OUT_EXT, 4 );
    prog->setParameter( GL_GEOMETRY_INPUT_TYPE_EXT, GL_POINTS );
    prog->setParameter( GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP );

    // load in shader to render lines
    osg::Program* lprog = new osg::Program;
    lprog->setName( "Line" );
    lprog->addShader(osg::Shader::readShaderFile(osg::Shader::VERTEX, path + "AnimLine.vert"));
    lprog->addShader(osg::Shader::readShaderFile(osg::Shader::FRAGMENT, path + "AnimLine.frag"));
    lprog->addShader(osg::Shader::readShaderFile(osg::Shader::GEOMETRY, path + "AnimLine.geom"));
    lprog->setParameter( GL_GEOMETRY_VERTICES_OUT_EXT, 2 );
    lprog->setParameter( GL_GEOMETRY_INPUT_TYPE_EXT, GL_TRIANGLE_FAN );
    lprog->setParameter( GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_LINE_STRIP );
    
    // text shader
    osg::Program* tprog = new osg::Program;
    tprog->setName( "Text" );
    tprog->addShader(osg::Shader::readShaderFile(osg::Shader::VERTEX, path + "AnimText.vert"));
    tprog->addShader(osg::Shader::readShaderFile(osg::Shader::FRAGMENT, path + "AnimText.frag"));
    tprog->addShader(osg::Shader::readShaderFile(osg::Shader::GEOMETRY, path + "AnimText.geom"));
    tprog->setParameter( GL_GEOMETRY_VERTICES_OUT_EXT, 3 );
    tprog->setParameter( GL_GEOMETRY_INPUT_TYPE_EXT, GL_TRIANGLES );
    tprog->setParameter( GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP );

    osg::Matrix mat;
    mat.makeRotate(M_PI_2, osg::Vec3(1, 0, 0));
    std::cerr << "Bounding box: " << bounds.xMin() << " " << bounds.xMax() << " " << bounds.yMin() << " " << bounds.yMax() << " " << bounds.zMin() << " " << bounds.zMax() << std::endl;

    osg::MatrixTransform* base = new osg::MatrixTransform();
    base->setMatrix(mat);
    
    base->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::PROTECTED);

    // default size for graph nodes ( should be able to set independent encode in color TODO )
    float sampleradius = 0.1;
    
    // add uniforms for spheres for samples
    osg::Geode* samp = new osg::Geode();
    osg::StateSet* state = samp->getOrCreateStateSet();
    state->setAttribute(prog);
    state->addUniform(new osg::Uniform("point_size", sampleradius));
    state->addUniform(new osg::Uniform("global_alpha",1.0f));
    state->addUniform(new osg::Uniform("time",0.0f));
    state->addUniform(new osg::Uniform("destination",osg::Vec3f(0.0, 0.0, 0.0)));
   
    // add color palette for sample 
    state->addUniform(new osg::Uniform(osg::Uniform::FLOAT_VEC3, "colorTable", 12));
    for(int i = 0; i < 12; i++)
    {
        state->getUniform("colorTable")->setElement(i, sampleColorTable[i]);
    }
    
    // adding uniforms for state for text
    osg::ref_ptr<osgText::Font> font = osgText::readRefFontFile("/home/calvr/CalVR/resources/arial.ttf");
    osg::Vec4 layoutColor(0.5f,0.5f,0.5f,1.0f);

    // set default character size
    float layoutCharacterSize = sampleradius * 0.2f;

    // create geode for Samples
    osg::Geode* samplenames = new osg::Geode();
    state = samplenames->getOrCreateStateSet();
    state->setRenderBinDetails( -1, "RenderBin");
    state->setAttribute(tprog);
    state->addUniform(new osg::Uniform("minWeight",0.0f)); // only draw lines above 20 weight
    state->addUniform(new osg::Uniform("destination", osg::Vec3(0.0, 0.0, 0.0))); 
    state->addUniform(new osg::Uniform("time", 0.0f)); 
  
    state->addUniform(new osg::Uniform(osg::Uniform::FLOAT_VEC3, "colorTable", 12));
    for(int i = 0; i < 12; i++)
    {
        state->getUniform("colorTable")->setElement(i, textColorTable[i]);
    }
    
    // use time uniform to determine movement and line alpha transition
    // use uniform for start and end position (pass to text, line and sphere)

    // between time steps need to set new start and end positions, also load in different primitive sets

}
*/

osg::Group* createTestStructure(std::unordered_map<std::string, Sample* >* samples , std::unordered_map<std::string, OTU* >* otus, osg::BoundingBox bounds, float highWeight, std::unordered_map<std::string, std::pair<std::string, int> >* mappingValues )
{
    std::string path = "/home/calvr/Philip/LayoutTest/";
    //std::string path = "/home/pweber/development/LayoutTest/";

    // load in shader to render points
    osg::Program* prog = new osg::Program;
    prog->setName( "Sphere" );
    prog->addShader(osg::Shader::readShaderFile(osg::Shader::VERTEX, path + "Sphere.vert"));
    prog->addShader(osg::Shader::readShaderFile(osg::Shader::FRAGMENT, path + "Sphere.frag"));
    prog->addShader(osg::Shader::readShaderFile(osg::Shader::GEOMETRY, path + "Sphere.geom"));
    prog->setParameter( GL_GEOMETRY_VERTICES_OUT_EXT, 4 );
    prog->setParameter( GL_GEOMETRY_INPUT_TYPE_EXT, GL_POINTS );
    prog->setParameter( GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP );

    // load in shader to render lines
    osg::Program* lprog = new osg::Program;
    lprog->setName( "Line" );
    lprog->addShader(osg::Shader::readShaderFile(osg::Shader::VERTEX, path + "Line.vert"));
    lprog->addShader(osg::Shader::readShaderFile(osg::Shader::FRAGMENT, path + "Line.frag"));
    lprog->addShader(osg::Shader::readShaderFile(osg::Shader::GEOMETRY, path + "Line.geom"));
    lprog->setParameter( GL_GEOMETRY_VERTICES_OUT_EXT, 2 );
    lprog->setParameter( GL_GEOMETRY_INPUT_TYPE_EXT, GL_TRIANGLE_FAN );
    lprog->setParameter( GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_LINE_STRIP );
    
    // text shader
    osg::Program* tprog = new osg::Program;
    tprog->setName( "Text" );
    tprog->addShader(osg::Shader::readShaderFile(osg::Shader::VERTEX, path + "Text.vert"));
    tprog->addShader(osg::Shader::readShaderFile(osg::Shader::FRAGMENT, path + "Text.frag"));
    tprog->addShader(osg::Shader::readShaderFile(osg::Shader::GEOMETRY, path + "Text.geom"));
    tprog->setParameter( GL_GEOMETRY_VERTICES_OUT_EXT, 3 );
    tprog->setParameter( GL_GEOMETRY_INPUT_TYPE_EXT, GL_TRIANGLES );
    tprog->setParameter( GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP );

    osg::Matrix mat;
    mat.makeRotate(M_PI_2, osg::Vec3(1, 0, 0));
    std::cerr << "Bounding box: " << bounds.xMin() << " " << bounds.xMax() << " " << bounds.yMin() << " " << bounds.yMax() << " " << bounds.zMin() << " " << bounds.zMax() << std::endl;

    osg::MatrixTransform* base = new osg::MatrixTransform();
    base->setMatrix(mat);

    //rootMat.setTrans(osg::Vec3( bounds.xMin() + (0.5 * (bounds.xMax() - bounds.xMin())), 0.0, bounds.zMin() + (0.5 * (bounds.zMax() - bounds.zMin())) ));

    //osg::Group* base = new osg::Group();
    base->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::PROTECTED);

    // default size for graph nodes ( should be able to set independent encode in color TODO )
    float sampleradius = 0.1;
    float oturadius = 0.03;

/*
    osg::Vec3 colorTable[12];
    colorTable[0] = osg::Vec3(0.89412,0.10196,0.109804);
    colorTable[1] = osg::Vec3(0.21569,0.49412,0.72157);
    colorTable[2] = osg::Vec3(0.302,0.6863,0.2902);
    colorTable[3] = osg::Vec3(0.59608,0.305882,0.63922);
    colorTable[4] = osg::Vec3(0.96078,0.96078,0.8627);
    colorTable[5] = colorTable[6] = colorTable[6] = colorTable[7] = colorTable[8] = colorTable[9] = colorTable[10] = colorTable[11] = osg::Vec3(1.0, 1.0, 1.0);
*/    

    // add spheres for samples
    osg::Geode* samp = new osg::Geode();
    osg::StateSet* state = samp->getOrCreateStateSet();
    state->setAttribute(prog);
    state->addUniform(new osg::Uniform("point_size", sampleradius));
    state->addUniform(new osg::Uniform("global_alpha",1.0f));
    state->addUniform(new osg::Uniform("minEdges",0)); 
    

    // TODO need to add the colorTable uniform to lines so can map to the same colors as sample
    // Create a seperate uniform so it will allow line color to map to different colors if we so choosen
    state->addUniform(new osg::Uniform(osg::Uniform::FLOAT_VEC3, "colorTable", 12));
    for(int i = 0; i < 12; i++)
    {
        state->getUniform("colorTable")->setElement(i, sampleColorTable[i]);
    }
    
    // add sphere for otus 
    osg::Geode* otusg = new osg::Geode();
    state = otusg->getOrCreateStateSet();
    state->setAttribute(prog);
    state->addUniform(new osg::Uniform("point_size", oturadius));
    state->addUniform(new osg::Uniform("global_alpha",0.8f)); 
    state->addUniform(new osg::Uniform("minEdges",0));  // at least 40 edges
    state->addUniform(new osg::Uniform("minWeight",0.0f)); // only draw lines above 20 weight
    //state->addUniform(new osg::Uniform("color",osg::Vec3f(0.0, 0.0, 1.0))); 
    
    //osg::Uniform* oColorTable = new osg::Uniform(osg::Uniform::FLOAT_VEC3, "colorTable[0]", 12);
    //state->addUniform(oColorTable); 
    state->addUniform(new osg::Uniform(osg::Uniform::FLOAT_VEC3, "colorTable", 12));
    for(int i = 0; i < 12; i++)
    {
        //sColorTable->setElement(i, otuColorTable[i]);
        state->getUniform("colorTable")->setElement(i, otuColorTable[i]);
    }

    // read in font to use
    //osg::ref_ptr<osgText::Font> font = osgText::readRefFontFile("/home/pweber/development/calvr/resources/arial.ttf");
    osg::ref_ptr<osgText::Font> font = osgText::readRefFontFile("/home/calvr/CalVR/resources/arial.ttf");
    osg::Vec4 layoutColor(0.5f,0.5f,0.5f,1.0f);

    // set default character size
    float layoutCharacterSize = sampleradius * 0.2f;

    // create geode for Samples
    osg::Geode* samplenames = new osg::Geode();
    state = samplenames->getOrCreateStateSet();
    state->setRenderBinDetails( -1, "RenderBin");
    state->setAttribute(tprog);
    state->addUniform(new osg::Uniform("minWeight",0.0f)); // only draw lines above 20 weight
  
    state->addUniform(new osg::Uniform(osg::Uniform::FLOAT_VEC3, "colorTable", 12));
    for(int i = 0; i < 12; i++)
    {
        state->getUniform("colorTable")->setElement(i, textColorTable[i]);
    }
    
    // create geode for otus (NOTED added a text shader to try and do quick disabling of text based on minWeight param)
    osg::Geode* otunames = new osg::Geode();
    state = otunames->getOrCreateStateSet();
    state->setRenderBinDetails( -1, "RenderBin");
    state->setAttribute(tprog);
    state->addUniform(new osg::Uniform("minWeight",0.0f)); // only draw lines above 20 weight
  
    state->addUniform(new osg::Uniform(osg::Uniform::FLOAT_VEC3, "colorTable", 12));
    for(int i = 0; i < 12; i++)
    {
        state->getUniform("colorTable")->setElement(i, textColorTable[i]);
    }

    // TODO use colors to encode graph information
    osg::Geometry* sampleGeom = new osg::Geometry();
    osg::Vec3Array *sampleVertices = new osg::Vec3Array();
    osg::Vec4Array* sampleColors = new osg::Vec4Array();
	
    osg::Geometry* otuGeom = new osg::Geometry();
    osg::Vec3Array *otuVertices = new osg::Vec3Array();
    osg::Vec4Array* otuColors = new osg::Vec4Array();

    // use to create default bounding box
    osg::BoundingBox bb;

    // values for highest weights ( these values should be collected when reading in the data TODO )
    double highestWeight = 0;
    unsigned int highestEdges = 0;
    
    // add stateset properties here
	
    // try not using a group
    //osg::Group* group = new osg::Group();

    osg::Geode* child = new osg::Geode();
    osg::Geometry* childGeom = new osg::Geometry();
    osg::Vec3Array* lines = new osg::Vec3Array();
    osg::Vec4Array* colors = new osg::Vec4Array();

    // starting position in array
    int start = 0;

    //std::string key;

    // TODO after apply color mapping as a test
    Sample* sample = NULL;
    for(auto it = samples->begin(); it != samples->end(); ++it)
    {

        // data to go in the data map
        //SampleData* data = new SampleData;
        //data->sampleIndex = sampleVertices->size(); // set index position 

        // access sampleName
	
	    // need to convert all keys to lower

        // disable color mappign for now
	    //key = it->first;
	    //std::transform(key.begin(), key.end(), key.begin(), ::tolower);

        std::pair< std::string, int> appearance(it->first, 11);
        //if( mappingValues->find(key) != mappingValues->end() )
        //    appearance = mappingValues->find(key)->second;
	    //else
	    //    std::cerr << "Could not find an appearance for : " << it->first << std::endl;

/*        
	// temp crap to color samples correctly
        if( it->first.compare(0, 2, "Sm") == 0 )
            appearance.second = 0.0;
        else if( it->first.compare(0, 2, "CD") == 0 )
            appearance.second = 1.0;
        else if( it->first.compare(0, 2, "UC") == 0 )
            appearance.second = 2.0;
        else if( it->first.compare(0, 2, "HE") == 0 )
            appearance.second = 3.0;
*/

        // access sample
        sample = it->second;

        // set vertexIndex
        sample->p.vertexIndex = sampleVertices->size();
        sample->p.sName = it->first;

        //TODO adjust radius based on some feature
        osg::Vec3 origin(sample->p.pos[0], sample->p.pos[1],sample->p.pos[2]);
        sampleVertices->push_back(origin);

        
        // add sample name
        osgText::Text* text = new osgText::Text;
        text->setUseVertexBufferObjects(true);
        text->setFont(font);

	layoutColor[2] = sample->highestWeight; 
	layoutColor[3] = 0.0; 
        text->setColor(layoutColor);

        // TODO need to create mapping for sample names
        // need to adjust text without rebuilding image
        // adjust text size in shader, using a uniform
        text->setCharacterSize(layoutCharacterSize);
        text->setPosition(osg::Vec3(sample->p.pos[0],sample->p.pos[1],sample->p.pos[2]));
        text->setAxisAlignment(osgText::Text::XY_PLANE);
        text->setAlignment(osgText::Text::CENTER_TOP);
        text->setFontResolution(40,40);
        text->setText(appearance.first.c_str());
        samplenames->addDrawable(text);
	
        float sampleTotalWeight = 0.0;

        // origin for fan lines
        unsigned int fanOrigin = lines->size();

        // create a triangle fan
        lines->push_back(origin);
        colors->push_back(osg::Vec4(0.0, 0.0, 0.0, 0.0));

        for(unsigned int i = 0; i < sample->edges->size(); i++)
        {
            sampleTotalWeight += sample->edges->at(i)->weight; // actual weight

            // set vertexIndex of Edge
            sample->edges->at(i)->vertexIndex[0] = fanOrigin;
            sample->edges->at(i)->vertexIndex[1] = lines->size();

            // used to line sorting (normalizing weight from 0.0 to 1.0)
            lines->push_back(osg::Vec3(sample->edges->at(i)->otu->p.pos[0], sample->edges->at(i)->otu->p.pos[1], sample->edges->at(i)->otu->p.pos[2]));
            colors->push_back(osg::Vec4(sample->edges->at(i)->weight / highWeight, 0.0, sample->edges->at(i)->weight, (float)appearance.second));
            
            // add edge index to data
            //data->edgeIndex->push_back(i);

            // expand bounding box
            bb.expandBy(lines->back());
        }

        int fanSize = sample->edges->size() + 1;

        if( sample->edges->size() > highestEdges )
            highestEdges = sample->edges->size();

        // refers to total weight of a sample (all edge weights)
        if( sampleTotalWeight > highestWeight )
            highestWeight = sampleTotalWeight;

        //std::cerr << "Color look up is: " << appearance.second << std::endl;
        sampleColors->push_back(osg::Vec4(sample->edges->size(), sampleTotalWeight, sample->highestWeight, (float)appearance.second));
        childGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_FAN,start,fanSize));
        
        // calculate new drawarray start
        start += fanSize;
    }
        
    childGeom->setVertexArray(lines);
    childGeom->setColorArray(colors, osg::Array::BIND_PER_VERTEX);
    child->addDrawable(childGeom);
    //group->addChild(child);

    std::cerr << "Highest Sample Weight " << highestWeight << std::endl;
    std::cerr << "Highest Number of edges " << highestEdges << std::endl;
    
    // divide up in the different patient groups so I can set colors using stateset    
    sampleGeom->setVertexArray(sampleVertices);
    sampleGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS,0,sampleVertices->size()));
    sampleGeom->setColorArray(sampleColors, osg::Array::BIND_PER_VERTEX);
    samp->addDrawable(sampleGeom);

    OTU* otu = NULL;

    //TODO need to compute otu weight to the highest connection, not just the total so can then remove otus at the same time as
    //line connections
    
    // create otu objects
    for( auto it = otus->begin(); it != otus->end(); ++it)
    {
	    //key = it->first;
	    //std::transform(key.begin(), key.end(), key.begin(), ::tolower);
       
        // disable color mapping for now 
	    // possible replacement mappings
        std::pair< std::string, int> appearance(it->first, 11);
        //if( mappingValues->find(key) != mappingValues->end() )
        //    appearance = mappingValues->find(key)->second;

        otu = it->second;

        // set vertexIndex
        otu->p.vertexIndex = otuVertices->size();
        otu->p.sName = it->first;

        // start with existing location
        osg::Vec3d otuPos(otu->p.pos[0], otu->p.pos[1], otu->p.pos[2]);
	otuVertices->push_back(otuPos);
        //otuColors->push_back(osg::Vec4(otu->edges->size(), 0.0, 0.0, 4.0 ));
        otuColors->push_back(osg::Vec4(otu->edges->size(), 0.0, otu->highestWeight, (float)appearance.second ));
        
        // add sample name
        osgText::Text* text = new osgText::Text;
        text->setUseVertexBufferObjects(true);
        text->setFont(font);
	
	// set weight in layout color
	layoutColor[2] = otu->highestWeight; 
	layoutColor[3] = 1.0; 
        text->setColor(layoutColor);
        
	    text->setCharacterSize(layoutCharacterSize * 0.5);
        text->setPosition(osg::Vec3(otu->p.pos[0], otu->p.pos[1], otu->p.pos[2]));
        text->setAxisAlignment(osgText::Text::XY_PLANE);
        text->setAlignment(osgText::Text::CENTER_TOP);
        text->setFontResolution(40,40);
        text->setText(it->first.c_str());
        //text->setText(appearance.first.c_str());
        otunames->addDrawable(text);
    }

    otuGeom->setVertexArray(otuVertices);
    otuGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, otuVertices->size()));
    otuGeom->setColorArray(otuColors, osg::Array::BIND_PER_VERTEX);
    otusg->addDrawable(otuGeom);		

    // stateset properties to render the lines
    //state = group->getOrCreateStateSet();
    state = child->getOrCreateStateSet();
    state->setAttribute(lprog);
    state->addUniform(new osg::Uniform("global_alpha",1.0f)); 
    state->addUniform(new osg::Uniform("minWeight",0.0f)); // only draw lines above 20 weight
    //state->addUniform(new osg::Uniform("color",osg::Vec3f(1.0, 1.0, 1.0))); 
    state->addUniform(new osg::Uniform("simplify", 1)); // render only ever 16th line  
    state->addUniform(new osg::Uniform("colorEdgesToSample", false));  
    
    // add same color palete for lines as samples
    state->addUniform(new osg::Uniform(osg::Uniform::FLOAT_VEC3, "colorTable", 12));
    for(int i = 0; i < 12; i++)
    {
        //sColorTable->setElement(i, sampleColorTable[i]);
        state->getUniform("colorTable")->setElement(i, sampleColorTable[i]);
    }

    state->setMode(GL_BLEND, osg::StateAttribute::ON);

    // add line width state
    osg::LineWidth* lineWidth = new osg::LineWidth();
    lineWidth->setWidth(4.0f);
    state->setAttributeAndModes(lineWidth, osg::StateAttribute::ON);

    // TODO might have to adjust
    osg::Depth* depth = new osg::Depth();
    depth->setWriteMask(false);
    state->setAttributeAndModes(depth, osg::StateAttribute::ON);
    std::cerr << "Finish line sort\n";

    base->addChild(samp);
    //base->addChild(group); // contains edges
    base->addChild(child); // contains edges
    base->addChild(otusg);
    
    // add text
    base->addChild(samplenames);
    base->addChild(otunames);

    // compute bound of structure
    ComputeBounds test;
    base->accept(test);
    osg::BoundingBox testBox = test.getBound();

    std::cerr << "Bounding box final: " << testBox.xMin() << " " << testBox.xMax() << " " << testBox.yMin() << " " << testBox.yMax() << " " 
                                        << testBox.zMin() << " " << testBox.zMax() << std::endl;

    return base;
}

void createJsonMappingFile(std::string fileName, std::unordered_map<std::string, Sample* >* samples , std::unordered_map<std::string, OTU* >* otus)
{
    // loop through and create json with vertex mappings
    Json::Value root;
  
    float weight = 0;
   
    int indexNumber = 0; 
    // loop through samples adding data
    for( auto it = samples->begin(); it != samples->end(); ++it)
    {
        root["Samples"][indexNumber]["sampleName"] = it->first; 
        root["Samples"][indexNumber]["index"] = it->second->p.vertexIndex; 
        
        for(int j = 0; j < it->second->edges->size(); j++)
        {
            // need name of samples this otu is connected too
            root["Samples"][indexNumber]["links"][j]["otuName"] = it->second->edges->at(j)->otu->p.sName; 
            root["Samples"][indexNumber]["links"][j]["sampIndex"] = it->second->edges->at(j)->vertexIndex[0]; 
            root["Samples"][indexNumber]["links"][j]["otuIndex"] = it->second->edges->at(j)->vertexIndex[1]; 
        
            // update weight
            if( it->second->edges->at(j)->weight > weight )
                weight = it->second->edges->at(j)->weight;
        }
        indexNumber++;
    }


    // reset counter
    indexNumber = 0;

    // loop through otus adding data
    for( auto it = otus->begin(); it != otus->end(); ++it)
    {    
        root["Otus"][indexNumber]["otuName"] = it->first; 
        root["Otus"][indexNumber]["index"] = it->second->p.vertexIndex; 

        for(int j = 0; j < it->second->edges->size(); j++)
        {
            // need name of samples this otu is connected too
            root["Otus"][indexNumber]["links"][j]["sampleName"] = it->second->edges->at(j)->sample->p.sName; 
            root["Otus"][indexNumber]["links"][j]["otuIndex"] = it->second->edges->at(j)->vertexIndex[1]; 
            root["Otus"][indexNumber]["links"][j]["sampleIndex"] = it->second->edges->at(j)->vertexIndex[0]; 
        }
        indexNumber++;
    }
  
    // adding this means done need to cause if original mapping data is normalized to 0-1 
    root["HighestWeight"] = weight;
    
    // try writing json linkage file
    ofstream ofs(fileName.c_str());
    ofs << root;
    ofs.close();
}

// TODO need to test create meta data for observations and be able to group them
// Be able to assign names for samples and otus by using mapping

// need to create a timeseries mapping file
void createJsonTimeSeriesFile(std::string fileName, std::unordered_map<std::string, Sample* >* samples , std::unordered_map<std::string, OTU* >* otus)
{
    // create (cluster based on time, so Key is time, then SampleName and position) 


    // update observation mapping and adjust connections 

}

void outputOpenORDFormat(std::unordered_map<std::string, Sample* >* samples, std::unordered_map<unsigned int, OTU* >* otus, std::string filename)
{
    ofstream ofs (filename.c_str());

    stringstream ss;
    for( auto it = samples->begin(); it != samples->end(); ++it)
    {

        // clear old stream
        ss.str("");
        ss.clear();

        Sample* sample = it->second;
        std::vector< OTUEdge* >* edges = sample->edges;
        for(unsigned int i = 0; i < edges->size(); i++)
        {
            ss << sample->p.name;
            ss << '\t';
            ss << edges->at(i)->otu->p.name;
            ss << '\t';
            ss << edges->at(i)->weight;
            ss << '\n';
        }
        // write stream 
        ofs << ss.rdbuf(); 
    }

    ofs.close(); 

/*
    // create map    
    ofstream ofsm (std::string("map_").append(filename.c_str()));
    for( auto it = samples->begin(); it != samples->end(); ++it)
    {
        Sample* sample = it->second;
        ofsm << sample->p.name;
        ofsm << " " << it->first << '\n';
    }
    for( auto it = otus->begin(); it != otus->end(); ++it)
    {
        OTU* otu = it->second;
        ofsm << otu->p.name;
        ofsm << " " << it->first << '\n';
    }
    ofsm.close();
*/
};

//just need to do once
void computeNormalizedWeights(std::unordered_map<std::string, Sample* >* samples, double largestWeight)
{
    for( auto it = samples->begin(); it != samples->end(); ++it)
    {
        Sample* sample = it->second;

        std::vector< OTUEdge* >* edges = sample->edges;
        for(unsigned int i = 0; i < edges->size(); i++)
        {
            edges->at(i)->normalizedWeight = (float)edges->at(i)->weight / largestWeight;
        }
    }
};


void loadSpecificMetaData(std::string metaHeader, std::string metaDataFile, std::map<std::string, int> & types, std::map<std::string, string > & sampleMapping)
{
    std::cerr << "Start!!\n";

    // open json meta
    ifstream ifs(metaDataFile);

    Json::Reader reader;
    Json::Value obj;
    reader.parse(ifs, obj);

    for(int i = 0; i < obj.size(); i++)
    {
        std::string name = obj[i]["#SampleID"].asString();
        std::string data = obj[i][metaHeader].asString();
        
        // register type
        if( types.find(data) != types.end() )
            types[data] += 1;
        else
            types[data] = 0;

        // use map to remove repeats
        sampleMapping[name] = data;

        //if( name.compare("10317.000068259") == 0 )
        //    std::cerr << name << " " << data << " exists\n";
    }

    ifs.close();
    std::cerr << "End!!\n";
}

// update graph color via mapping look up
void applySampleMapping(osg::Geode* sampleNode, osg::Geode* edgeNode, std::string colorMapName,
                         std::map<std::string, ColorMap> & colorMapping, // property to color
                         std::map<std::string, string> & sampleMapping, // sample to property
                         std::map<std::string, VertexMap > & vertexMapping) // name to vertex look up

{
   std::cerr << "Applying Sample Mapping\n";
   // need to access uniform and update it  (this should exist lazy not checking)  


   // create quick index map to use
   std::map<std::string, int> quickLookup;

   // load specific color map
   if( colorMapping.find(colorMapName) != colorMapping.end())
   {
       for( int i = 0; i < 12; i++)
       {
           //std::cerr << "Setting color: " << colorMapping[colorMapName].mapping[i].color[0] << " "
           //                               << colorMapping[colorMapName].mapping[i].color[1] << " "
           //                               << colorMapping[colorMapName].mapping[i].color[2] << std::endl;
           sampleNode->getStateSet()->getUniform("colorTable")->setElement(i, osg::Vec3f(colorMapping[colorMapName].mapping[i].color));
           edgeNode->getStateSet()->getUniform("colorTable")->setElement(i, osg::Vec3f(colorMapping[colorMapName].mapping[i].color));
            
           // to make setting actual indexes fast
           quickLookup[colorMapping[colorMapName].mapping[i].property] = i;
       }
   }

   std::cerr << "Updated uniforms\n";
  
   // create two vectors to pass to update visitor to adjust color parameters
   std::vector< IndexValue > sampleUpdates; 
   std::vector< IndexValue > lineUpdates; 

   // TODO create drawable visitor to modify color vertices (can later extend to move vertices for animation)
   
   // need name to vertex map to create list to use for modification
   // temp value for insert
   IndexValue iValue; 
   
   // loop through all samples and look up mapping value
   std::map<std::string, std::string>::iterator it = sampleMapping.begin();
   for(; it != sampleMapping.end(); ++it)
   {
        std::string sampName = it->first;
        std::string optionName = it->second;

        if( vertexMapping.find(sampName) != vertexMapping.end() )
        {    
            VertexMap vert = vertexMapping[sampName];
        
            int colorInt = 11;
            if( quickLookup.find(optionName) != quickLookup.end() )
                colorInt = quickLookup[optionName];
            else
                std::cerr << "No option found for " << optionName << std::endl;

            //std::cerr << "Point Index: " << vert.pointIndex << " Color: " << colorInt << std::endl;

            // color doesnt exist set lines and samples to use default color (num 11 in look up table)
            // use name of sample to find corresponding index and set its value 
            iValue.index = vert.pointIndex;
            iValue.value = colorInt; // default to last value in lookup
            sampleUpdates.push_back(iValue);

            // add values to lineUpdates
            for(int j = 0; j < vert.lineIndexs->size(); j++)
            {
                iValue.index = vert.lineIndexs->at(j).sInd;
                iValue.value = colorInt;
                lineUpdates.push_back(iValue);
            }
        }
        //else
        //{
        //    std::cerr << "If doesnt exist it is not in the graph\n";
        //}
    }

    // TODO apply maps to allow graph to be colored (will temp test just by modifiying the geode now)
    osg::Geometry* geom = sampleNode->getDrawable(0)->asGeometry();
    osg::Vec4Array* colors = dynamic_cast<osg::Vec4Array*> (geom->getColorArray());
   
    if ( colors )
    {
        std::cerr << "Adjusting samples\n"; 
        for(int i = 0; i < sampleUpdates.size(); i++)
        {
            osg::Vec4 *color = &colors->operator [](sampleUpdates.at(i).index);
            color->set( (*color)[0], (*color)[1], (*color)[2], sampleUpdates.at(i).value );
        }

        geom->setColorArray(colors);
    } 
   
    // update edges 
    geom = edgeNode->getDrawable(0)->asGeometry();
    colors = dynamic_cast<osg::Vec4Array*> (geom->getColorArray());
   
    if ( colors )
    { 
        std::cerr << "Adjusting edges\n"; 
        for(int i = 0; i < lineUpdates.size(); i++)
        {
            osg::Vec4 *color = &colors->operator [](lineUpdates.at(i).index);
            color->set( (*color)[0], (*color)[1], (*color)[2], lineUpdates.at(i).value );
        }

        geom->setColorArray(colors);
    } 
}


// load in mapping meta data
void loadJsonVertexMappingFile(std::string fileName, std::map<std::string, VertexMap> & vertexSampleLookup)
{
   // read in vert mapping file       
   fstream ifs(fileName);
   
   Json::Reader reader;
   Json::Value obj;
   reader.parse(ifs, obj);

   Json::Value sampleList = obj["Samples"];
   
   // get list of mapping elements           
   for(int i = 0; i < sampleList.size(); i++)
   {
        int sampleIndex = obj["Samples"][i]["index"].asInt();
		std::string sampleName = obj["Samples"][i]["sampleName"].asString();

        // add mapping
		vertexSampleLookup[sampleName].pointIndex = sampleIndex;
		vertexSampleLookup[sampleName].lineIndexs = new std::vector<LineVertices >();

        Json::Value links = obj["Samples"][i]["links"];
        for( int j = 0; j < links.size(); j++)
        {
            LineVertices lv;
            lv.sInd = obj["Samples"][i]["links"][j]["sampIndex"].asInt();
            lv.oInd = obj["Samples"][i]["links"][j]["otuIndex"].asInt();
            vertexSampleLookup[sampleName].lineIndexs->push_back(lv);
        }

        //if( sampleName.compare("10317.000068259") == 0 )
        //{
        //    std::cerr << "Added: " << sampleName << std::endl;
        //    exit(1);
        //}
    }
} 

void loadColorMappingFile(std::string vertMapFile, std::map< std::string, ColorMap > & vertColorMapping)
{
   // read in vert mapping file       
   fstream ifs(vertMapFile);
   
   Json::Reader reader;
   Json::Value obj;
   reader.parse(ifs, obj);
        
   osg::Vec3 defaultColor(1.0, 1.0, 1.0);
   
   // get list of mapping elements           
   for(int i = 0; i < obj.size(); i++)
   {
        std::string mapName = obj[i]["MapName"].asString();
        vertColorMapping[mapName].metaName = obj[i]["MetaProperty"].asString();
       
        std::cerr << "loaded: " << mapName << std::endl;
      
        // NOTE: this does not set 12 colors by default 
        Json::Value cm = obj[i]["Properties"];
        for( int j = 0; j < cm.size(); j++)
        {
            vertColorMapping[mapName].mapping[j].property = obj[i]["Properties"][j]["Property"].asString();  
            vertColorMapping[mapName].mapping[j].color.set( obj[i]["Properties"][j]["Color"][0].asFloat(),
                                                              obj[i]["Properties"][j]["Color"][1].asFloat(),
                                                              obj[i]["Properties"][j]["Color"][2].asFloat());

            if( vertColorMapping[mapName].mapping[j].property.compare("default") == 0)
                defaultColor.set(vertColorMapping[mapName].mapping[j].color);
        }

        // TODO set the rest of the colors using hte default color
        for(int j = cm.size(); j < 12; j++)
        {
            vertColorMapping[mapName].mapping[j].color.set(defaultColor);
        }
    }
} 

int main (int argc, char** argv)
{
  // pass and manage arguments
  osg::ArgumentParser arguments(&argc,argv);
  
  // argument set up DividePoint test.xyz baseName 3 
  arguments.getApplicationUsage()->setApplicationName(arguments.getApplicationName());
  arguments.getApplicationUsage()->setCommandLineUsage(arguments.getApplicationName() + 
            " -ie inputEdgeTextFile -ip inputPositionsTextFile -o archiveName");

  // if user request help write it out to cout.
  if (arguments.read("-h") || arguments.read("--help"))
  {
    arguments.getApplicationUsage()->write(std::cout);
    return 1;
  }

  // initialize the sample and otu color lookups
  for(int i = 0; i < 12; i++)
  {
    //sampleColorTable[i] = osg::Vec3( 189.0 / 255.0, 189.0 / 255.0, 189.0 / 255.0 );
    
    // TODO make light gray instead of red
    //otuColorTable[i] = osg::Vec3( 240.0 / 255.0, 240.0 / 255.0, 240.0 / 255.0 );
    
    
    sampleColorTable[i] = osg::Vec3( 27.0 / 255.0, 158.0 / 255.0, 119.0 / 255.0 );
    otuColorTable[i] = osg::Vec3( 217.0 / 255.0, 95.0 / 255.0, 2.0 / 255.0 );
    textColorTable[i] = osg::Vec3( 211.0 / 255.0, 211.0 / 255.0, 211.0 / 255.0 );
  }
  
  // hack set 0 to a darker grey for text  
  textColorTable[0] = osg::Vec3( 128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0 );
  
  // hack to set second color for otus
  otuColorTable[1] = osg::Vec3( 117.0 / 255.0, 112.0 / 255.0, 179.0 / 255.0 );

  std::string inputEdgeFileName;
  //std::string inputNodeFileName;
  std::string inputPositionsFileName;
  std::string outputFileName("default");
  bool binaryInput = false;

  while( arguments.read("-binary") )
  {
    binaryInput = true;
  }
  
  while (arguments.read("-ie", inputEdgeFileName))
  {
  }    
  
  //while (arguments.read("-in", inputNodeFileName))
  //{
  //}    
  
  while (arguments.read("-ip", inputPositionsFileName))
  {
  }    
  
  while (arguments.read("-o", outputFileName))
  {
  }    
  
  // NOTE THIS look up has mapping for all node data (sample and otu)
  // reason that data is complined so left it combined 
  // if file exists read and create appearance mappings based on mapping file
  std::unordered_map<std::string, std::pair< std::string, int> >* m_namecolor = new std::unordered_map<std::string, std::pair<std::string, int > >();
 
  
  // read in points and sort into bsp tree structure
  ifstream ifs(inputEdgeFileName.c_str());

  // root node to attach kd-tree
  //osg::Group* group = new osg::Group();

  // set a CalVR node mask to disable special culling
  //group->setNodeMask(group->getNodeMask() & ~0x1000000);

  //std::map<unsigned int, std::string> lookup;
  
  // bound of the graph
  osg::BoundingBox bound;

  // TODO colormapping should be handled in here

  // map containing position information
  std::unordered_map<std::string, Position >* m_positions = new std::unordered_map<std::string, Position >();

  // loads in all position locations

  ifstream ifsp;
  if( !inputPositionsFileName.empty() )
  {
        ifsp.open(inputPositionsFileName);

        // read in input positions

        std::string pvalues, value;
        std::stringstream ss, ssnumber;
        std::string key;
        Position position;
        
        int index = 0;
        while( getline(ifsp, pvalues) )
        {
            // reset counter
            index = 0;
		  
            ss.str("");
		    ss.clear();
            ss << pvalues;
            
            std::getline(ss, key, '\t');
            std::getline(ss, value, '\t');
            ssnumber.str("");
	    ssnumber.clear();
            ssnumber << value;
            ssnumber >> position.x;          
            std::getline(ss, value, '\t');
            ssnumber.str("");
	    ssnumber.clear();
            ssnumber << value;
            ssnumber >> position.y;          

            //std::cerr << "key: " << key << " Pos: " << position.x << " " << position.y << std::endl;

            bound.expandBy(osg::Vec3(position.x, 0.0, position.y));

            // add position
            m_positions->insert(std::pair<std::string, Position >(key, position));
        }
  }
  
  /*
        ifsp.open(inputPositionsFileName);

        std::string values;

        // read in lookup map file
        ifstream ifsm(std::string("map_").append(inputPositionsFileName));

        while( getline( ifsm, values ) )
        {
            std::string value;
            unsigned int key;
            stringstream ss;
            ss << values;
            ss >> key;
            ss >> value;
            lookup[key] = value;        
        }
        ifsm.close();

  }
  */

  // variables to hold data while reading 
  string value, values;
  stringstream ss;
  stringstream ssdouble;
  int index;
  float highestWeight = 0.0;
  unsigned int connections =0;

  // create maps
  std::unordered_map<std::string, Sample* >* m_samples = new std::unordered_map<std::string, Sample* >();
  std::unordered_map<std::string, OTU* >* m_otus = new std::unordered_map<std::string, OTU* >();

  // hold data for sample time series map
  //std::unordered_map<std::string, Sample* >* m_samples_time = new std::unordered_map<std::string, Sample* >();
  //std::unordered_map<std::string, OTU* >* m_otus_time = new std::unordered_map<std::string, OTU* >();

  // if not binary create a binary representation
  //if( !binaryInput )
  //{
	  // open a binary file for writing all the points in
	  //ofstream ofs (std::string(outputFileName).append(".ConBin").c_str(), ios::binary);
	 
      // first line has header 
      // getline( ifs, values );

      std::string sampleName;
      std::string otuName;
      float weight;

      // line number used to be able to look up meta data in the file
      //unsigned int lineNumber = 1;

      // parse values in file now
      while( getline( ifs, values ) )
      {
	      ss.str("");
	      ss.clear();
	      ss << values;

          // read in values
          std::getline(ss, sampleName, '\t');
          std::getline(ss, otuName, '\t');
          std::getline(ss, value, '\t');
	      ssdouble.str("");
	      ssdouble.clear();
          ssdouble << value;
          ssdouble >> weight;
          
          // make sure objects exist
          if( m_samples->find(sampleName) == m_samples->end() )
          {
              Sample* temp = new Sample();
              m_samples->insert(std::pair<std::string, Sample* >(sampleName, temp));   
         
              //std::cerr << "Sample name is " << sampleName << " look up name is " << lookup.find(0)->second << std::endl;
             
              // try and look up a position
              if( m_positions->find(sampleName) != m_positions->end() )
              {
                    Position p = m_positions->find(sampleName)->second;
                    temp->p.pos[0] = p.x;
                    temp->p.pos[1] = p.y;
              }
              else
              {
                std::cerr << "Error sample position not found: " << sampleName << std::endl;
              }
          }
          
          if( m_otus->find(otuName) == m_otus->end() )
          {
              OTU* temp =  new OTU();
              m_otus->insert(std::pair<std::string, OTU* >(otuName, temp));   
             
              // try and look up a position
              if( m_positions->find(otuName) != m_positions->end() )
              {
                    Position p = m_positions->find(otuName)->second;
                    temp->p.pos[0] = p.x;
                    temp->p.pos[1] = p.y;
              }
              else
              {
                std::cerr << "Error otu position not found: " << otuName << std::endl;
                Position p = m_positions->find(sampleName)->second;
                temp->p.pos[0] = p.x;
                temp->p.pos[1] = p.y - 0.05;
              }
          }
         
          //add data to maps
          OTU* currentOTU = m_otus->find(otuName)->second;
          Sample* currentSample = m_samples->find(sampleName)->second;
        
          // create an edge connection 
          OTUEdge* currentEdge = new OTUEdge(currentOTU, currentSample, weight);
          
          // add edge to maps 
          currentSample->edges->push_back(currentEdge);
	      if( currentSample->highestWeight < weight )
	        currentSample->highestWeight = weight;

          currentOTU->edges->push_back(currentEdge);
	      if( currentOTU->highestWeight < weight )
	        currentOTU->highestWeight = weight;

          connections++;
         
          // only use visible weights 
          if( weight > highestWeight )
            highestWeight = weight;
          
          // increment line number
          //lineNumber++;    
           
	  }
	  //ofs.close();
  //}

  ifs.close();

  std::cerr << "Read in file complete\n";
  std::cerr << "Highest weight: " << highestWeight << std::endl;
  std::cerr << "Number of OTUs: " << m_otus->size() << std::endl;
  std::cerr << "Number of Samples: " << m_samples->size() << std::endl;
  std::cerr << "Number of connections: " << connections << std::endl;

/*
  // TODO convert to json file
  // write out properties to be used, also list the mapping files used
  ofstream oss((outputFileName + ".meta").c_str());
  oss << highestWeight;
  oss << '\n';
  oss << m_samples->size();
  oss << '\n';
  oss << m_otus->size();
  oss << '\n';
  oss << connections;
  oss << '\n';
  oss << sampleMappingFileName;
  oss << '\n';
  oss << otuMappingFileName;
  oss << '\n';
  oss.close();

  // write out special format
  //outputOpenORDFormat(m_samples, m_otus, std::string("BigTest.int"));

  // compute initial position on sphere for OTUS
  //createInitialSpherePositions(m_otus);
  createInitialCirclePositions(m_otus);

  std::cerr << "Initial sphere positions computed\n";

  // compute normalized weights
  computeNormalizedWeights(m_samples, heighestWeight);

  std::cerr << "Normalized weights computed\n";

  for(int i = 0; i < 100; i++)
  {
    // compute location for samples
    computeSamplePositions(m_samples);

    //std::cerr << "Sample weights computed\n";

    //recompute OTU locations
    recomputeOTUPositions(m_otus);

    //std::cerr << "OTU positions recomputed\n";
  }
*/

  std::cerr << "Map start\n";

  // create graphics to render (need to pass in color mapping data) 
  osg::Group* test = createTestStructure(m_samples, m_otus, bound, highestWeight, m_namecolor);
 
  // bsp the points to speed up intersection testing

  // write out meta mapping for look up
 
  
  // todo need to create a meta mapping file for the data structure so graphics can be updated
  std::cerr << "Map completed\n";

  //ArchivePagedLODSubgraphsVistor blah("test", test);
  //test->accept(blah);

  osgDB::writeNodeFile(*test, outputFileName + ".osgb");
  std::cerr << "Wrote out file\n";
  //return 0;

  // critical!!!!!!!!!! TODO
  //TODO read in mapping from testJson.json and read in a mapping file description and read in metaFile info and try and upadte the graph

/*
  // return list of names of elements and index to the types data (color mapping is limited to 12 colors)
  std::map<std::string, int> colorMapping; // int is frequency
  std::map< std::string, string > sampMapping;
  
  // color mapping returned is specific to this data set
  loadSpecificMetaData("FLOSSING_FREQUENCY", "/home/pweber/data/networkData/ag-cleaned.json", colorMapping, sampMapping);

  //function to load in vertex mapping file (TODO color mapping file corrupt)
  std::map< std::string, ColorMap > vertColorMapping;
  std::string vertMapFileName = "/home/pweber/data/networkData/vertTestMap.json";
  loadColorMappingFile(vertMapFileName, vertColorMapping);
  // color mapping read in

  // test print out found options (test output)
  //std::map<std::string, int>::iterator it = colorMapping.begin();
  //for(;it != colorMapping.end(); ++it) 
  //  std::cerr << "Color param name: " << it->first << " frequency " << it->second << std::endl;
*/
  std::cerr << "Start mapping file\n";
  
  // write out vertex mapping file // this should be done at time of creation
  createJsonMappingFile(outputFileName + "_vert.json", m_samples, m_otus);
  
  std::cerr << "Finished mapping file\n";
  return 0;
/*
  // TODO read in json mapping file
  // sampleName to index and list of vertex data
  std::map<std::string, VertexMap > vertexSampleLookup;
  loadJsonVertexMappingFile("testJson.json", vertexSampleLookup); 

  std::cerr << "Before color update\n";

  // TODO create visitor to find nodes that need to be updated (cleaner and safer)
  osg::Geode* sGeode = test->asGroup()->getChild(0)->asGeode();
  osg::Geode* eGeode = test->asGroup()->getChild(1)->asGeode();
  if( sGeode && eGeode )
  { 
      applySampleMapping(sGeode, eGeode, "Flossing-oral",
                           vertColorMapping, // property to color
                           sampMapping, // sample to property
                           vertexSampleLookup); // name to vertex look up
    std::cerr << "Test color changes applied\n";
  }  
  std::cerr << "After color update\n";
*/
  // TODO in network program need to used json meta file to produce color mapping

  osgViewer::Viewer viewer;
  viewer.addEventHandler(new osgViewer::StatsHandler());
  viewer.setSceneData(test);
  return viewer.run();

/*
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

  std::cerr << "Deepest node found " << deepestNode << std::endl;

  float minPointSize = (float)averageDistance / numberOfLeafNodes / deepestNode;
  state->addUniform(new osg::Uniform("point_size", minPointSize + (float) pow( pow( deviationDistance / deepestNode / (numberOfLeafNodes - 1), 0.5), 0.5 )));
  state->addUniform(new osg::Uniform("global_alpha",1.0f));
  state->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

  std::cerr << "Point Sorting Completed\n";
  std::cerr << "Total number of points in geodes is: " << totalNumberOfPoints << std::endl;

  // update uniform
  //UpdatePointSizeVisitor pvisitor(averageDistance, deviationDistance, numberOfLeafNodes);
  //group->accept(pvisitor);

  // create archive
  ArchivePagedLODSubgraphsVistor archVisitor(outputFileName, group);
  group->accept(archVisitor);

  std::cerr << "Archive completed\n";
*/
  return 0;
}
