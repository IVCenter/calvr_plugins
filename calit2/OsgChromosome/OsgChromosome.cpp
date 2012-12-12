#include "OsgChromosome.h"
#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/SceneObject.h>
#include <cvrKernel/PluginManager.h>
#include <cvrMenu/MenuSystem.h>
#include <cvrMenu/MenuItem.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include "vvtokenizer.h"
#include <osg/Matrix>
#include <osgDB/ReadFile>
#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <osg/Material>

using namespace osg;
using namespace std;
using namespace cvr;

CVRPLUGIN(OsgChromosome)

ClipNode * root;

OsgChromosome::OsgChromosome()
{
    printf("OsgChromosome called.\n");
}

bool OsgChromosome::init()
{
    printf("OsgChromosome::init() called\n");

    basepath = ConfigManager::getEntry("Plugin.OsgChromosome.Basepath");

    mymenu = new SubMenu("OsgChromosome","OsgChromosome");
    mymenu->setCallback(this);
    //cvr::PluginHelper::addRootMenuItem(mymenu);
    MenuSystem::instance()->addMenuItem(mymenu);

    clear = new MenuButton("Clear");
    clear->setCallback(this);
    cvr::PluginHelper::addRootMenuItem(clear);

    chr1 = new cvr::MenuButton("Chromosome 1");
    chr1->setCallback(this);
    mymenu->addItem(chr1);

    chr2 = new cvr::MenuButton("Chromosome 2");
    chr2->setCallback(this);
    mymenu->addItem(chr2);
    
    chr3 = new cvr::MenuButton("Chromosome 3");
    chr3->setCallback(this);
    mymenu->addItem(chr3);

    chr4 = new cvr::MenuButton("Chromosome 4");
    chr4->setCallback(this);
    mymenu->addItem(chr4);

    chr5 = new cvr::MenuButton("Chromosome 5");
    chr5->setCallback(this);
    mymenu->addItem(chr5);

    chr6 = new cvr::MenuButton("Chromosome 6");
    chr6->setCallback(this);
    mymenu->addItem(chr6);

    chr7 = new cvr::MenuButton("Chromosome 7");
    chr7->setCallback(this);
    mymenu->addItem(chr7);

    chr8 = new cvr::MenuButton("Chromosome 8");
    chr8->setCallback(this);
    mymenu->addItem(chr8);
    
    chr9 = new cvr::MenuButton("Chromosome 9");
    chr9->setCallback(this);
    mymenu->addItem(chr9);
    
    chr10 = new cvr::MenuButton("Chromosome 10");
    chr10->setCallback(this);
    mymenu->addItem(chr10);
    
    chr11 = new cvr::MenuButton("Chromosome 11");
    chr11->setCallback(this);
    mymenu->addItem(chr11);
    
    chr12 = new cvr::MenuButton("Chromosome 12");
    chr12->setCallback(this);
    mymenu->addItem(chr12);
    
    chr13 = new cvr::MenuButton("Chromosome 13");
    chr13->setCallback(this);
    mymenu->addItem(chr13);
    
    chr14 = new cvr::MenuButton("Chromosome 14");
    chr14->setCallback(this);
    mymenu->addItem(chr14);
    
    chr15 = new cvr::MenuButton("Chromosome 15");
    chr15->setCallback(this);
    mymenu->addItem(chr15);
    
    chr16 = new cvr::MenuButton("Chromosome 16");
    chr16->setCallback(this);
    mymenu->addItem(chr16);
    
    chr17 = new cvr::MenuButton("Chromosome 17");
    chr17->setCallback(this);
    mymenu->addItem(chr17);
    
    chr18 = new cvr::MenuButton("Chromosome 18");
    chr18->setCallback(this);
    mymenu->addItem(chr18);
    
    chr19 = new cvr::MenuButton("Chromosome 19");
    chr19->setCallback(this);
    mymenu->addItem(chr19);
    
    chr20 = new cvr::MenuButton("Chromosome 20");
    chr20->setCallback(this);
    mymenu->addItem(chr20);
    
    chr21 = new cvr::MenuButton("Chromosome 21");
    chr21->setCallback(this);
    mymenu->addItem(chr21);
    
    chr22 = new cvr::MenuButton("Chromosome 22");
    chr22->setCallback(this);
    mymenu->addItem(chr22);
    
    chr23 = new cvr::MenuButton("Chromosome 23");
    chr23->setCallback(this);
    mymenu->addItem(chr23);

    //root = SceneManager::instance()->getObjectsRoot();

    return true;
}

void OsgChromosome::menuCallback(cvr::MenuItem * item) {
        if (item == chr1) { 
            //root->removeChildren(1, root->getNumChildren());
            renderChromosome((char *)(basepath + "hIMR90_Hind3_3D_chr1.txt").c_str());
            /*MenuItem * viewall = mymenu->getChild(6);
            MenuCallback * call = viewall->getCallback();
            call->menuCallback(viewall);*/
            }
        else if (item == chr2) { 
            //root->removeChildren(1, root->getNumChildren());
            renderChromosome((char *) (basepath + "hIMR90_Hind3_3D_chr2.txt").c_str());
            }
        else if (item == chr3) { 
            //root->removeChildren(1, root->getNumChildren());
            renderChromosome((char *)(basepath + "hIMR90_Hind3_3D_chr3.txt").c_str());
            }
        else if (item == chr4) { 
            //root->removeChildren(1, root->getNumChildren());
            renderChromosome((char *)(basepath + "hIMR90_Hind3_3D_chr4.txt").c_str());
            }
        else if (item == chr5) { 
            //root->removeChildren(1, root->getNumChildren());
            renderChromosome((char *)(basepath + "hIMR90_Hind3_3D_chr5.txt").c_str());
            }
        else if (item == chr6) { 
            //root->removeChildren(1, root->getNumChildren());
            renderChromosome((char *)(basepath + "hIMR90_Hind3_3D_chr6.txt").c_str());
            }
        else if (item == chr7) { 
            //root->removeChildren(1, root->getNumChildren());
            renderChromosome((char *)(basepath + "hIMR90_Hind3_3D_chr7.txt").c_str());
            }
        else if (item == chr8) { 
            //root->removeChildren(1, root->getNumChildren());
            renderChromosome((char *)(basepath + "hIMR90_Hind3_3D_chr8.txt").c_str());
            }
        else if (item == chr9) { 
            //root->removeChildren(1, root->getNumChildren());
            renderChromosome((char *)(basepath + "hIMR90_Hind3_3D_chr9.txt").c_str());
            }
        else if (item == chr10) { 
            //root->removeChildren(1, root->getNumChildren());
            renderChromosome((char *)(basepath + "hIMR90_Hind3_3D_chr10.txt").c_str());
            }
        else if (item == chr11) { 
            //root->removeChildren(1, root->getNumChildren());
            renderChromosome((char *)(basepath + "hIMR90_Hind3_3D_chr11.txt").c_str());
            }
        else if (item == chr12) { 
            //root->removeChildren(1, root->getNumChildren());
            renderChromosome((char *)(basepath + "hIMR90_Hind3_3D_chr12.txt").c_str());
            }
        else if (item == chr13) { 
            //root->removeChildren(1, root->getNumChildren());
            renderChromosome((char *)(basepath + "hIMR90_Hind3_3D_chr13.txt").c_str());
            }
        else if (item == chr14) { 
            //root->removeChildren(1, root->getNumChildren());
            renderChromosome((char *)(basepath + "hIMR90_Hind3_3D_chr14.txt").c_str());
            }
        else if (item == chr15) { 
            //root->removeChildren(1, root->getNumChildren());
            renderChromosome((char *)(basepath + "hIMR90_Hind3_3D_chr15.txt").c_str());
            }
        else if (item == chr16) { 
            //root->removeChildren(1, root->getNumChildren());
            renderChromosome((char *)(basepath + "hIMR90_Hind3_3D_chr16.txt").c_str());
            }
        else if (item == chr17) { 
            //root->removeChildren(1, root->getNumChildren());
            renderChromosome((char *)(basepath + "hIMR90_Hind3_3D_chr17.txt").c_str());
            }
        else if (item == chr18) { 
            //root->removeChildren(1, root->getNumChildren());
            renderChromosome((char *)(basepath + "hIMR90_Hind3_3D_chr18.txt").c_str());
            }
        else if (item == chr19) { 
            //root->removeChildren(1, root->getNumChildren());
            renderChromosome((char *)(basepath + "hIMR90_Hind3_3D_chr19.txt").c_str());
            }
        else if (item == chr20) { 
            //root->removeChildren(1, root->getNumChildren());
            renderChromosome((char *)(basepath + "hIMR90_Hind3_3D_chr20.txt").c_str());
            }
        else if (item == chr21) { 
            //root->removeChildren(1, root->getNumChildren());
            renderChromosome((char *)(basepath + "hIMR90_Hind3_3D_chr21.txt").c_str());
            }
        else if (item == chr22) { 
            //root->removeChildren(1, root->getNumChildren());
            renderChromosome((char *)(basepath + "hIMR90_Hind3_3D_chr22.txt").c_str());
            }
        else if (item == chr23) { 
            //root->removeChildren(1, root->getNumChildren());
            renderChromosome((char *)(basepath + "hIMR90_Hind3_3D_chr23.txt").c_str());
            }
        else if (item == clear) {
            //root->removeChildren(1, root->getNumChildren());
            }

        /*MenuItem * viewall = mymenu->getChild(6);
        MenuCallback * call = viewall->getCallback();
        call->menuCallback(viewall); */
}

void OsgChromosome::renderChromosome(char * file){
    Sphere * mysphere;
	ShapeDrawable * mydrawable;
	Geode* mygeode = new Geode();
    osg::ref_ptr<osg::Material> pMaterial;
	ifstream myReadFile;
    SceneObject * so = new SceneObject(file, true, true, true, true, false);
    PluginHelper::registerSceneObject(so);
    so->attachToScene();
    so->setNavigationOn(true);
    so->addMoveMenuItem();
    so->addNavigationMenuItem();
    
    // open file to read
    myReadFile.open(file);
 	string chr, start, end, comp, x, y, z, output;
    Vec3 prev, curr;
    Vec4 color;
    bool isFirst = true;
 	if (myReadFile.is_open()) {
        printf("is open\n");
            // get rid of top line  
	    getline(myReadFile, output);
            // read each table entry 
        while (!myReadFile.eof()) {
            myReadFile >> chr;
            myReadFile >> start;
            myReadFile >> end;
            myReadFile >> comp;
            myReadFile >> x;
            myReadFile >> y;
            myReadFile >> z;
            curr = Vec3(atof(x.c_str()), atof(y.c_str()), atof(z.c_str()));
            mysphere = new Sphere(curr, .02);

            mydrawable = new ShapeDrawable(mysphere);
            mygeode = new Geode();
            mygeode->addDrawable(mydrawable);
            so->addChild(mygeode);
        	//root->addChild(mygeode);

            // determine the color of the segment
            pMaterial = new osg::Material;
            if (comp.compare("A") == 0)
                color = Vec4(1,0,0,1);
            else if (comp.compare("B") == 0)
                color = Vec4(0,0,1,1);
            else
                color = Vec4(0,1,0,1);

            // set the color of the sphere
	        pMaterial->setDiffuse( osg::Material::FRONT, color);
            mygeode->getOrCreateStateSet()->setAttribute( pMaterial, osg::StateAttribute::OVERRIDE );

            // draw the cylinder bewtween spheres
            if (!isFirst) {
                //mygeode = new Geode();
                AddCylinderBetweenPoints(prev, curr, (float) .02, color, so/*(Group *) root*/);
                //root->addChild(mygeode);   
            }
            prev = curr;
            isFirst = false;
        }
        printf("get attached is %d, number of children is %d\n", so->getAttached(), so->getNumChildObjects());
    }
    myReadFile.close();
}

void OsgChromosome::AddCylinderBetweenPoints(Vec3 & StartPoint, Vec3 & EndPoint, float radius, Vec4 CylinderColor, SceneObject * pAddToThisGroup)
{
 	osg::Vec3	center;
 	float		height;

	osg::ref_ptr<osg::Cylinder> cylinder;
	osg::ref_ptr<osg::ShapeDrawable> cylinderDrawable;
	osg::ref_ptr<osg::Material> pMaterial;
	osg::ref_ptr<osg::Geode> geode;
	
	height = (StartPoint- EndPoint).length();
 	center = osg::Vec3( (StartPoint.x() + EndPoint.x()) / 2,  (StartPoint.y() + EndPoint.y()) / 2,  (StartPoint.z() + EndPoint.z()) / 2);
 
	// This is the default direction for the cylinders to face in OpenGL
	osg::Vec3	z = osg::Vec3(0,0,1);

	// Get diff between two points you want cylinder along
	osg::Vec3 p = (StartPoint - EndPoint);

	// Get CROSS product (the axis of rotation)
	osg::Vec3	t = z ^  p;

	// Get angle. length is magnitude of the vector
	double angle = acos( (z * p) / p.length());
  
	//	Create a cylinder between the two points with the given radius
    cylinder = new osg::Cylinder(center,radius,height);
	cylinder->setRotation(osg::Quat(angle, osg::Vec3(t.x(), t.y(), t.z())));
	
	//	A geode to hold our cylinder
	geode = new osg::Geode;
	cylinderDrawable = new osg::ShapeDrawable(cylinder );
    geode->addDrawable(cylinderDrawable);
 
	//	Set the color of the cylinder that extends between the two points.
	pMaterial = new osg::Material;
	pMaterial->setDiffuse( osg::Material::FRONT, CylinderColor);
	geode->getOrCreateStateSet()->setAttribute( pMaterial, osg::StateAttribute::OVERRIDE );

	//	Add the cylinder between the two points to an existing group  
	pAddToThisGroup->addChild(geode);
}


OsgChromosome::~OsgChromosome()
{
}
