#include "ArtifactVis.h"
#include "vvtokenizer.h"
#include "../OssimPlanet/OssimPlanet.h"

#include <iostream>

#include <config/ConfigManager.h>
#include <kernel/PluginHelper.h>
#include <kernel/SceneManager.h>
#include <menu/MenuSystem.h>

#include <osg/CullFace>
#include <osg/Matrix>
#include <osg/ShapeDrawable>

#include <osgDB/ReadFile>

using namespace osg;
using namespace std;
using namespace cvr;

CVRPLUGIN(ArtifactVis)

ArtifactVis::ArtifactVis()
{

}

bool ArtifactVis::init()
{
    std::cerr << "ArtifactVis init\n";

    root = new osg::MatrixTransform();

    avMenu = new SubMenu("ArtifactVis", "ArtifactVis");
    avMenu->setCallback(this);

    showCheckbox = new MenuCheckbox("Show Artifacts", false);
    showCheckbox->setCallback(this);
    avMenu->addItem(showCheckbox);

    MenuSystem::instance()->addMenuItem(avMenu);
    SceneManager::instance()->getObjectsRoot()->addChild(root);

    _my_own_root = new LOD();
    _my_sphere_root = new Group();

    _LODmaxRange = ConfigManager::getFloat("Plugins.ArtifactVis.MaxVisibleRange", 30.0);

    _my_own_root->addChild( _my_sphere_root, 0, 0 );

    std::cerr << "ArtifactVis init done.\n";
    return true;
}


ArtifactVis::~ArtifactVis()
{
}

void ArtifactVis::menuCallback(MenuItem* menuItem)
{
    if (menuItem == showCheckbox)
    {
        if (showCheckbox->getValue())
        {
            // load artifacts and send them to OssimPlanet (once)
            static bool load = true;
            if (load)
            {
                readArtifactsFile(ConfigManager::getEntry("Plugin.ArtifactVis.Database"));
                //listArtifacts(); // uncomment this line to Debug
                displayArtifacts(_my_sphere_root);

                if(!OssimPlanet::addModel(_my_sphere_root,
                //if(!OssimPlanet::addModel(_my_own_root,
                    ConfigManager::getFloat("Plugins.ArtifactVis.Site.Latitude",0),
                    ConfigManager::getFloat("Plugins.ArtifactVis.Site.Longitude",0),
		    osg::Vec3(0.6, 0.6, 0.6), 0.0, 0.0, 0.0, 135.0))
                {
                    std::cerr<<"Could not add artifacts to OssimPlanet. Adding to objects root instead.\n";
                    PluginHelper::getObjectsRoot()->addChild( _my_sphere_root );
                    //PluginHelper::getObjectsRoot()->addChild( _my_own_root );
                }
		else
	        {
                    std::cerr<<"Added artifacts to OssimPlanet.\n";
                }
                load = false;
            }

            // enable spheres
            _my_own_root->setRange(0,0,_LODmaxRange);
        }
        else
        {
            // disable spheres
            _my_own_root->setRange(0,0,0);
        }
    }
}

void ArtifactVis::readArtifactsFile(std::string filename)
{
  cerr << "Reading artifacts file: " << filename << endl;

  vvTokenizer::TokenType ttype;
  FILE* fp = fopen(filename.c_str(), "rb");
  if (fp==NULL)
  {
    cerr << "Cannot read file: " << filename << endl;
    return;
  }
  vvTokenizer* tokenizer = new vvTokenizer(fp);
  tokenizer->setEOLisSignificant(true);
  tokenizer->setCaseConversion(vvTokenizer::VV_UPPER);
  tokenizer->setParseNumbers(true);
  tokenizer->setAlphaCharacter(' ');
  tokenizer->setWhitespaceCharacter('\t');
  tokenizer->setCommentCharacter('#');
  while ((ttype = tokenizer->nextToken()) != vvTokenizer::VV_EOF)
  {
    Artifact* newArtifact = new Artifact();
    
    // EDM:
    if (tokenizer->ttype != vvTokenizer::VV_NUMBER) 
    {
      cerr << "Error: expected EDM in line " << tokenizer->getLineNumber() << endl;
      exit(1);
    }
    newArtifact->edm = int(tokenizer->nval);
    
    // DC:
    tokenizer->nextToken();
    if (tokenizer->ttype != vvTokenizer::VV_WORD) 
    {
      cerr << "Error: expected DC Code in line " << tokenizer->getLineNumber() << endl;
      exit(1);
    }    
    string dcString(tokenizer->sval);
    newArtifact->dc = dcString;

    //Add this descriptor to the list, if it does not exist yet

    vector<std::string>::iterator dc_item = _descriptor_list.begin();
    int foundit=0;
    for (; dc_item < _descriptor_list.end(); dc_item++)
    {
      if ((*dc_item).compare(dcString)==0)
        {
         foundit=1;
         break;
        }
    }

    if (foundit==0){
      _descriptor_list.push_back(dcString);
      _descriptor_list_colors.push_back(Vec4((float)rand()/(float)RAND_MAX,(float)rand()/(float)RAND_MAX,(float)rand()/(float)RAND_MAX,1.00));
      //cerr << dcString << endl;
    }


    // Locus:
    tokenizer->nextToken();
    if (tokenizer->ttype != vvTokenizer::VV_NUMBER)
    {
      cerr << "Error: expected Locus in line " << tokenizer->getLineNumber() << endl;
      cerr << "Read: " << tokenizer->sval << endl;
      exit(1);
    } 

    newArtifact->locus = int(tokenizer->nval);

    // Basket:
    tokenizer->nextToken();
    if (tokenizer->ttype != vvTokenizer::VV_NUMBER)
    {
      cerr << "Error: expected Basket in line " << tokenizer->getLineNumber() << endl;
      exit(1);
    }
    newArtifact->basket = int(tokenizer->nval);

    // Square:
    tokenizer->nextToken();
    string square(tokenizer->sval);
    newArtifact->square = square;

    // Date:
    tokenizer->nextToken();
    string date(tokenizer->sval);
    newArtifact->date = date;

    // Area:
    tokenizer->nextToken();
    if (tokenizer->ttype != vvTokenizer::VV_WORD)
    {
      cerr << "Error: expected Area in line " << tokenizer->getLineNumber() << ", " << tokenizer->sval << endl;
    }
    newArtifact->area = (tokenizer->sval[0]);
    
    // Site:
    tokenizer->nextToken();
    if (tokenizer->ttype != vvTokenizer::VV_WORD) 
    {
      cerr << "Error: expected Site in line " << tokenizer->getLineNumber() << endl;
      exit(1);
    }    
    string siteString(tokenizer->sval);
    newArtifact->site = siteString;

    // Pos[0]
    tokenizer->nextToken();
    if (tokenizer->ttype != vvTokenizer::VV_NUMBER)
    {
      cerr<< "Error: expected Northing in line " << tokenizer->getLineNumber() << endl;
      exit(1);
    }
    newArtifact->pos[0] = tokenizer->nval;

    // Pos[1]
    tokenizer->nextToken();
    if (tokenizer->ttype != vvTokenizer::VV_NUMBER)
    {
      cerr<< "Error: expected Easting in line " << tokenizer->getLineNumber() << endl;
      exit(1);
    }
    newArtifact->pos[1] = tokenizer->nval;

    // Pos[2]
    tokenizer->nextToken();
    if (tokenizer->ttype != vvTokenizer::VV_NUMBER)
    {
      cerr<< "Error: expected Elevation in line " << tokenizer->getLineNumber() << endl;
      exit(1);
    }
    newArtifact->pos[2] = tokenizer->nval;

    // done with line:
    _artifacts.push_back(newArtifact);
    tokenizer->nextLine();      // won't need this once all elements in line are read
  }
  delete tokenizer;
  fclose(fp); 
}

void ArtifactVis::listArtifacts()
{
  cerr << "Listing " << _artifacts.size() << " elements:" << endl;
  vector<Artifact*>::iterator item = _artifacts.begin();
  for (; item < _artifacts.end(); item++)
  {
    cerr << "edm: " << (*item)->edm << ", dc: " << (*item)->dc << ", basket: " << (*item)->basket << ", northing: " << (*item)->pos[0] << ", easting: " << (*item)->pos[1] << ", elevation: " << (*item)->pos[2] << endl;
  }

    vector<std::string>::iterator dc_item = _descriptor_list.begin();
 
    int ind=-1;
 for (; dc_item < _descriptor_list.end(); dc_item++)
  {
    ind++;
    cerr <<ind<< "  dc: " << (*dc_item)<<endl;
  }

   cerr<<"Num descriptors = "<<_descriptor_list.size()<<endl;
}

void ArtifactVis::displayArtifacts(Group * root_node)
{
    const double M_TO_MM = 1000.0f;

    cerr << "Creating " << _artifacts.size() << " artifacts...";
    vector<Artifact*>::iterator item = _artifacts.begin();

    Vec3f offset = Vec3f(
        ConfigManager::getFloat("Plugin.ArtifactVis.Offset.X",0),
        ConfigManager::getFloat("Plugin.ArtifactVis.Offset.Y",0),
        ConfigManager::getFloat("Plugin.ArtifactVis.Offset.Z",0));

    float tessellation = ConfigManager::getFloat("Plugin.ArtifactVis.Tessellation",.2);

    int artCount = _artifacts.size();
    for (int objCount = 0; item < _artifacts.end();item++)
    {
        //cerr<<"Creating object "<<++objCount<<" out of"<<artCount<<endl;

        // Create translation node:
        Vec3f position((*item)->pos[0], (*item)->pos[1], (*item)->pos[2]);
        Vec3f pos = position+offset;

        Matrixd trans;
        trans.makeTranslate(position + offset);
        Matrixd scale;
        scale.makeScale(M_TO_MM, M_TO_MM, M_TO_MM);
        Matrixd rot1;
        rot1.makeRotate(osg::DegreesToRadians(-90.0), 0, 1, 0);
        Matrixd rot2;
        rot2.makeRotate(osg::DegreesToRadians(90.0), 1, 0, 0);
        Matrixd mirror;
        mirror.makeScale(1, -1, 1);

        // This is needed to correctly render the spheres
        MatrixTransform* mirror2Node = new MatrixTransform();
        mirror2Node->setMatrix(mirror);
      /*  Matrixd transPlanet;
        Matrixd rotator2; // rotator2 rotates world about x-axis
        Matrixd rotator3; // rotator3 rotates world about y-axis
        transPlanet.makeTranslate(Vec3f(0.0, 0.0, -5000.0));
        rotator2.makeRotate(osg::DegreesToRadians(-40.0), 1, 0, 0);
        rotator3.makeRotate(osg::DegreesToRadians(-135.0), 0, 0, 1);
        _planet->postMult(transPlanet);
        _planet->postMult(rotator2);
        _planet->postMult(rotator3); */
        vector<std::string>::iterator dc_item = _descriptor_list.begin();
        int index=0;
        for (; dc_item < _descriptor_list.end(); dc_item++)
        {
            if ((*dc_item).compare((*item)->dc)==0)
                break;
            index++;
        }

        Node* g = createObject(index, (*item)->edm,tessellation, pos);

        Group* gg=g->asGroup();

        if (!gg)
            (*item)->geode=g;
        else
        {
            cerr<<"Num children "<<gg->getNumChildren()<<endl;
            (*item)->geode=gg->getChild(0);
        }

        MatrixTransform* objTrans = new MatrixTransform();
        objTrans->setMatrix(mirror);
        objTrans->postMult(trans);
        objTrans->postMult(scale);
        objTrans->postMult(mirror);
        objTrans->postMult(rot2);
        objTrans->postMult(rot1);

        objTrans->addChild(g);
        root_node->addChild(objTrans);
    }

    cerr << "done" << endl;

    StateSet * ss=root_node->getOrCreateStateSet();
  
    osg::CullFace * cf=new osg::CullFace();
    cf->setMode(osg::CullFace::BACK);
  
    ss->setAttributeAndModes( cf, osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);
}

Node* ArtifactVis::createObject(int index, int edm, float tessellation, Vec3f & pos)
{
    Matrix scale;
    scale.makeScale(0.005f, 0.005f, 0.005f);    
    MatrixTransform * scaleNode = new MatrixTransform();
    scaleNode->setMatrix(scale);

    Geode* geode = new Geode();
    const double radius = 0.05f;
    vector<Vec4>::iterator dc_color = _descriptor_list_colors.begin();
    dc_color+=index;
    
    // setDetailRatio is a factor to multiply the default values for
    // numSegments (40) and numRows (10). 
    // They won't go below the minimum values of MIN_NUM_SEGMENTS = 5, MIN_NUM_ROWS = 3
    TessellationHints * hints = new TessellationHints();
    hints->setDetailRatio(tessellation);
    Vec3 center(0.0f, 0.0f, 0.0f);
    Vec3 norm(0.0f, 0.0f, 1.0f);

    Sphere* sphereShape = new Sphere(center, radius); 
    ShapeDrawable * shapeDrawable = new ShapeDrawable(sphereShape);
    shapeDrawable->setTessellationHints(hints);
    shapeDrawable->setColor((*dc_color));//(*dc_color));
    geode->addDrawable(shapeDrawable); 

    StateSet * stateSet = geode->getOrCreateStateSet();
    stateSet->setMode(GL_BLEND, StateAttribute::ON);
    stateSet->setRenderingHint( osg::StateSet::TRANSPARENT_BIN );

    Material * mat =new Material();	
    mat->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    //Specifying the yellow colour of the object
    mat->setDiffuse(Material::FRONT,(*dc_color));
    //Specifying the yellow colour of the object
    //Vec4 color_sp(1,0,0,1);
    //mat->setSpecular(Material::FRONT_AND_BACK,color_sp);
    //Attaching the newly defined state set object to the node state set
    stateSet->setAttribute(mat);
    
    shapeDrawable->setUseDisplayList(false);    // allow shape updates

    return geode;
}
