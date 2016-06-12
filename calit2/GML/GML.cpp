//#include <strstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <iterator>
#include <math.h>

// OSG:
#include <osg/Node>
#include <osg/Switch>
#include <osg/CullFace>
#include <osg/Sequence>
#include <osg/Geode>
#include <osg/LineWidth>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/PolygonMode>
#include <osg/Vec3>
#include <osg/MatrixTransform>
#include <osgUtil/SceneView>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgText/Text>
#include <osg/ShapeDrawable>
#include <osg/LineWidth>
#include <osg/PointSprite>
#include <osg/AlphaFunc>
#include <osg/PositionAttitudeTransform>
#include <osg/Billboard>
#include <map>

#include "GML.h"

using namespace std;
using namespace cvr;

CVRPLUGIN(GML)

//constructor
GML::GML() : FileLoadCallback("gml")
{

}


//Menu Items Listener
void GML::menuCallback(cvr::MenuItem * item)
{
    if(item == forceDirectedLayout)
    {
	  makeForceDirectedLayoutLocations();
    }
}



bool GML::loadFile(std::string filename)
{
  ifstream ifs( filename.c_str() );
  
  int nodeindex = 0;
  numEdges = 0;

  std::map<string, Node* > nodes;
  nodelist = new std::vector<Node*>();
  std::map<string, string> lookup;
 
  while(! ifs.eof() )
  {
	string s;
  	getline( ifs, s );

	// separate line into substrings	
	stringstream ss(s);
	string buf;	
	vector<string> tokens; // Create vector to hold our words

 	while (ss >> buf)
       		tokens.push_back(buf);
	
	

	// check the keyword
	if( tokens.size() != 0 && tokens[0].compare("node") == 0)
	{
		int brackets = 0;

		// check for brackets
		for(int i = 0; i < tokens.size(); i++)
			if(tokens[i].compare("[") == 0)
				brackets++;

		Node *n = new Node();

  		osg::Vec3 initPoint;

		// keep reading subloop until closing tag ] is found
		while(! ifs.eof())
		{
			// get line from input
  			getline( ifs, s );
	
			// separate line into substrings	
			stringstream ss(s);
			string buf;	
			vector<string> tokens; // Create vector to hold our words

    			while (ss >> buf)
        			tokens.push_back(buf);


			for(int i = 0; i < tokens.size(); i++)
			{
				if(tokens[i].compare("[") == 0)
					brackets++;
				if(tokens[i].compare("]") == 0)
					brackets--;
			}
	
			if(tokens[0].compare("id") == 0)
			{
				stringstream ssi;
                                ssi << nodeindex;


                                lookup[tokens[1]] = ssi.str();

                                // set id
                                n->setIndex(tokens[1]);

                                // add new node to tree
                                nodes[ssi.str()] = n;

                                // add to nodelist
                                nodelist->push_back(n);
			}
			else if(tokens[0].compare("label") == 0)
			{
				n->setName(tokens[1]);
			}
                     
			else if(tokens[0].compare("x") == 0)
			{
				//n->setX(atof(tokens[1].c_str()));
				initPoint[0] = atof(tokens[1].c_str());
			}
			else if(tokens[0].compare("y") == 0)
			{
				//n->setY(atof(tokens[1].c_str()));
				initPoint[1] = atof(tokens[1].c_str());
			}
			else if(tokens[0].compare("z") == 0)
			{
				//n->setZ(atof(tokens[1].c_str()));
				initPoint[2] = atof(tokens[1].c_str());
			}
			else if(tokens[0].compare("w") == 0)
			{
				//n->setW(atof(tokens[1].c_str()));
			}
			else if(tokens[0].compare("h") == 0)
			{
				n->setH(atof(tokens[1].c_str()));
			}
			else if(tokens[0].compare("fill") == 0)
			{
				unsigned int hex = 0;
				sscanf(tokens[1].substr(tokens[1].size() - 3,2).c_str(), "%X", &hex);
				n->setR((float)hex / 255.0);
				sscanf(tokens[1].substr(tokens[1].size() - 5,2).c_str(), "%X", &hex);
				n->setG((float)hex / 255.0);
				sscanf(tokens[1].substr(tokens[1].size() - 7,2).c_str(), "%X", &hex);
				n->setB((float)hex / 255.0);
			}

			//check if finished subloop
			if(brackets == 0)
				break; // exit subloop
		}

		n->setIndice(nodeindex);
                
		// increment nodeindex;
                nodeindex++;

		// set inital point to move too
		n->savePoint(initPoint.x(), initPoint.y(), initPoint.z());
	}
	else if( tokens.size() != 0 && tokens[0].compare("edge") == 0)
	{
		// add a edge to the list
		Edge *e = new Edge();
		int brackets = 0;

		for(int i = 0; i < tokens.size(); i++)
			if(tokens[i].compare("[") == 0)
				brackets++;

		// keep reading subloop until closing tag ] is found
		while(! ifs.eof())
		{
  			getline( ifs, s );

			// separate line into substrings        
                        stringstream ss(s);
                        string buf;
                        vector<string> tokens; // Create vector to hold our words

                        while (ss >> buf)
                                tokens.push_back(buf);


			for(int i = 0; i < tokens.size(); i++)
			{
				if(tokens[i].compare("[") == 0)
					brackets++;
				if(tokens[i].compare("]") == 0)
					brackets--;
					
			}

                        if(tokens[0].compare("source") == 0)
                        {

				string key;

                                // use look up table to mind corressponding mapping
                                map<string, string>::iterator itk;
                                itk = lookup.find(tokens[1]);
                                if(itk != lookup.end())
                                {
                                        key = (*itk).second;
                                }
                                else
                                {
                                        printf("key should exist\n");
                                }

				// look up source and add pointer here and also add the edge to its list
				map<string, Node* >::iterator it;
				it = nodes.find(key);
				if(it != nodes.end())
				{
					(*it).second->addEdge(e);
					e->setSource((*it).second);
				}				
                        }
                        else if(tokens[0].compare("target") == 0)
                        {
				string key;

                                // use look up table to mind corressponding mapping
                                map<string, string>::iterator itk;
                                itk = lookup.find(tokens[1]);
                                if(itk != lookup.end())
                                {
                                        key = (*itk).second;
                                }
                                else
                                {
                                        printf("key should exist\n");
                                }

				map<string, Node* >::iterator it;
				it = nodes.find(key);
				if(it != nodes.end())
				{
					(*it).second->addEdge(e);
					e->setTarget((*it).second);
				}				
                        }
                        else if(tokens[0].compare("label") == 0)
                        {
                                e->setName(tokens[1]);
                        }
                        else if(tokens[0].compare("width") == 0)
                        {
                                //e->setW(atof(tokens[1].c_str()));
                        }
                        else if(tokens[0].compare("fill") == 0)
                        {
				unsigned int hex = 0;
                                sscanf(tokens[1].substr(tokens[1].size() - 3,2).c_str(), "%X", &hex);
                                e->setR((float)hex / 255.0);
                                sscanf(tokens[1].substr(tokens[1].size() - 5,2).c_str(), "%X", &hex);
                                e->setG((float)hex / 255.0);
                                sscanf(tokens[1].substr(tokens[1].size() - 7,2).c_str(), "%X", &hex);
                                e->setB((float)hex / 255.0);
                        }

			// Check the source Arrow and target Arrow
			else if(tokens[0].compare("source_arrow") == 0)
			{
				e->setSourceArrow(atoi(tokens[1].c_str()));
			}
			else if(tokens[0].compare("target_arrow") == 0)
			{
				e->setTargetArrow(atoi(tokens[1].c_str()));	
			}

			//check if finished subloop
			if(brackets == 0)
				break; // exit subloop

			numEdges += 2;
		}
	}
	
  }
  ifs.close();

  verticesP = new osg::Vec3Array(nodeindex);
  verticesE = new osg::Vec3Array(numEdges);
  colorsNodes = new osg::Vec4Array(nodeindex);
  colorsEdges = new osg::Vec4Array(numEdges);

  // create geometry and geodes to hold the data
  nodeGeom = new osg::Geometry();
  nodeGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS,0, verticesP->size()));
  osg::VertexBufferObject* vboP = nodeGeom->getOrCreateVertexBufferObject();
  vboP->setUsage (GL_STREAM_DRAW);

  nodeGeom->setUseDisplayList (false);
  nodeGeom->setUseVertexBufferObjects(true);
  nodeGeom->setVertexArray(verticesP);
  nodeGeom->setColorArray(colorsNodes);
  nodeGeom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

  lineGeom = new osg::Geometry();
  osg::VertexBufferObject* vboE = lineGeom->getOrCreateVertexBufferObject();
  vboE->setUsage (GL_STREAM_DRAW);
  lineGeom->setUseDisplayList (false);
  lineGeom->setUseVertexBufferObjects(true);
  lineGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES,0,(int)verticesE->size()));
  lineGeom->setVertexArray(verticesE);
  lineGeom->setColorArray(colorsEdges);
  lineGeom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

  // point shader
  osg::StateSet *state = nodeGeom->getOrCreateStateSet();
  osg::Program* pgm1 = new osg::Program;
  pgm1->setName( "Sphere" );
  pgm1->addShader(osg::Shader::readShaderFile(osg::Shader::VERTEX, osgDB::findDataFile("/home/covise/CalVR/plugins/calit2/GML/shaders/Sphere.vert")));
  pgm1->addShader(osg::Shader::readShaderFile(osg::Shader::FRAGMENT, osgDB::findDataFile("/home/covise/CalVR/plugins/calit2/GML/shaders/Sphere.frag")));
  pgm1->addShader(osg::Shader::readShaderFile(osg::Shader::GEOMETRY, osgDB::findDataFile("/home/covise/CalVR/plugins/calit2/GML/shaders/Sphere.geom")));
  pgm1->setParameter( GL_GEOMETRY_VERTICES_OUT_EXT, 4 );
  pgm1->setParameter( GL_GEOMETRY_INPUT_TYPE_EXT, GL_POINTS );
  pgm1->setParameter( GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP );
  state->setAttribute(pgm1);


  // tube shader
  osg::ref_ptr<osg::Program> pgm2 = new osg::Program;
  pgm2->setName( "Cylinder" );
  pgm2->addShader(osg::Shader::readShaderFile(osg::Shader::VERTEX, osgDB::findDataFile("/home/covise/CalVR/plugins/calit2/GML/shaders/cylinder.vert")));
  pgm2->addShader(osg::Shader::readShaderFile(osg::Shader::FRAGMENT, osgDB::findDataFile("/home/covise/CalVR/plugins/calit2/GML/shaders/cylinder.frag")));
  pgm2->addShader(osg::Shader::readShaderFile(osg::Shader::GEOMETRY, osgDB::findDataFile("/home/covise/CalVR/plugins/calit2/GML/shaders/cylinder.geom")));
  pgm2->setParameter( GL_GEOMETRY_VERTICES_OUT_EXT, 100 );
  pgm2->setParameter( GL_GEOMETRY_INPUT_TYPE_EXT, GL_LINES );
  pgm2->setParameter( GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP );

  // get line state
  state = lineGeom->getOrCreateStateSet();
  state->setAttribute(pgm2);
  edgeScale = new osg::Uniform("edgeScale", 1.0f);
  state->addUniform(edgeScale);

  state = nodeGeom->getOrCreateStateSet();
  state->setAttribute(pgm1);
  objectScale = new osg::Uniform("objectScale", PluginHelper::getObjectScale());
  //nodeScale = new osg::Uniform("nodeScale", 1.0f);
  //state->addUniform(nodeScale);
  state->addUniform(objectScale);
  
  geode = new osg::Geode();
  geode->addDrawable(lineGeom);
  geode->addDrawable(nodeGeom);
  
  printf("Initalization finished\n");
 
  // attach graph to group
  group->addChild(geode); 

  // initialize data
  initializeVertices();

  // create connectivity for the graph (needed for tree layout)
  createConnectivity();

  makeForceDirectedLayoutLocations();
  //movePoints(); // will move to default position if there is a default layout (generally there isnt)

  return true; 
}

// just need to modify colors
void GML::updateColor(Node* node, bool active)
{
   if(!node)
	return;
	
   if(active)
   {
      colorsNodes->at(node->getIndice()) = osg::Vec4(1.0 - node->getR(), 1.0 - node->getG(), 1.0 - node->getB(), node->getW());
      
      int index = 0;
      vector<Edge *> edges = node->getEdges();
      for(int i = 0; i < (int)edges.size(); i++)
      {
	if( !edges.at(i)->isFalseEdge())
	{
	        Edge* edge = edges.at(i);
		
		// need to find
		if( node != edges.at(i)->getSource() )
		{
			Node* childNode = edges.at(i)->getSource();
			vector<Edge* > attachedEdges =  childNode->getEdges();
			
			int ind = 0;	
			// find where edge is in edge vector
			for(int j = 0; j < (int)attachedEdges.size();j++)
			{
				if(edges.at(i) == attachedEdges.at(j))
				{
					colorsEdges->at(childNode->getColorStartIndex() + ind) = osg::Vec4(1.0 - edge->getR(), 1.0 - edge->getG(), 1.0 - edge->getB(), colorsEdges->at(node->getColorStartIndex() + index).w());		
					break;
				}
				
				// increment if source is equal to node (to adjust offset in color array)
				if( childNode == attachedEdges.at(j)->getSource() )
					ind +=2;
			}
		}
		else
		{
			colorsEdges->at(node->getColorStartIndex() + index) = osg::Vec4(1.0 - edge->getR(), 1.0 - edge->getG(), 1.0 - edge->getB(), colorsEdges->at(node->getColorStartIndex() + index).w());
			index += 2;
		}	
	}
      }
   }
   else
   {
      colorsNodes->at(node->getIndice()) = osg::Vec4(node->getR(), node->getG(), node->getB(), node->getW());
      int index = 0;
      vector<Edge *> edges = node->getEdges();
      for(int i = 0; i < (int)edges.size(); i++)
      {
	if( !edges.at(i)->isFalseEdge())
        {
                Edge* edge = edges.at(i);

                // need to find
                if( node != edges.at(i)->getSource() )
                {
                        Node* childNode = edges.at(i)->getSource();
                        vector<Edge* > attachedEdges =  childNode->getEdges();

                        int ind = 0;    
                        // find where edge is in edge vector
                        for(int j = 0; j < (int)attachedEdges.size();j++)
                        {
                                if(edges.at(i) == attachedEdges.at(j))
                                {
                                        colorsEdges->at(childNode->getColorStartIndex() + ind) = osg::Vec4(edge->getR(), edge->getG(), edge->getB(), colorsEdges->at(childNode->getColorStartIndex() + ind).w());               
                                        break;
                                }

                                // increment if source is equal to node (to adjust offset in color array)
                                if( childNode == attachedEdges.at(j)->getSource() )
                                        ind +=2;
                        }
                }
                else
                {
                        colorsEdges->at(node->getColorStartIndex() + index) = osg::Vec4(edge->getR(), edge->getG(), edge->getB(), colorsEdges->at(node->getColorStartIndex() + index).w());
                        index += 2;
                }
	   }
        }
   }
   
   verticesE->dirty();
   colorsEdges->dirty();
   colorsNodes->dirty();
}

// call when file is loaded
void GML::initializeVertices()
{
  for(int i = 0; i < (int)nodelist->size(); i++)
  {
        Node * node = nodelist->at(i);
       
	// set vertices
	osg::Vec3 vec = osg::Vec3(node->getX(), node->getY(), node->getZ());
	verticesP->at(i) = vec;
 
	// add the base color
        colorsNodes->at(i) = osg::Vec4(node->getR(), node->getG(), node->getB(), node->getW());
  }

  // need to break because node data needs to be initialized first
  int index = 0;
  for(int i = 0; i < (int)nodelist->size(); i++)
  {
        Node * node = nodelist->at(i);

	//edge color start index
	node->setColorStartIndex(index);

        vector<Edge *> edges = node->getEdges();
        for(int j = 0; j < (int)edges.size(); j++)
        {
                Edge* edge = edges.at(j);
                Node* target = edge->getTarget();
		osg::Vec4 arrowInfo(0.0, 0.0, 0.0, 0.0);

		if( node == edge->getSource() && !edge->isFalseEdge())
		{
                	verticesE->at(index) = verticesP->at(i);
                	verticesE->at(index + 1) = verticesP->at(target->getIndice());

                	colorsEdges->at(index) = osg::Vec4(edge->getR(), edge->getG(), edge->getB(), edge->getW()); // radius of edge

			// note: if sizes of the nodes exist then an arrow needs to be generated for that edge
			// second color parameter contains information (size of source node, size of target node, 0.0, 0.0)
			arrowInfo[0] = node->getW();
			arrowInfo[1] = target->getW();
			arrowInfo[2] = (float)edge->getSourceArrow();
			arrowInfo[3] = (float)edge->getTargetArrow();
 
                	colorsEdges->at(index + 1) = arrowInfo;
			index += 2;
		}          
        }
  }

  verticesP->dirty();
  verticesE->dirty();
  colorsNodes->dirty();
  colorsEdges->dirty();
  nodeGeom->dirtyBound();
  group->computeBound();
}

// call when graph is modified
void GML::updateVertices()
{
  for(int i = 0; i < (int)nodelist->size(); i++)
  {
        Node * node = nodelist->at(i);
	osg::Vec3 vec = osg::Vec3(node->getX(), node->getY(), node->getZ());

	// need to also set the node values
        verticesP->at(i) = vec;
  }

  // need to break up because node data needs to initalized first 
  int index = 0;
  for(int i = 0; i < (int)nodelist->size(); i++)
  {        
        Node * node = nodelist->at(i);

        vector<Edge *> edges = node->getEdges();
        for(int j = 0; j < (int)edges.size(); j++)
        {
                Edge* edge = edges.at(j);
                Node* target = edge->getTarget();

		if( node == edge->getSource() && !edge->isFalseEdge())
		{
                	verticesE->at(index) = verticesP->at(i);
                	verticesE->at(index + 1) = verticesP->at(target->getIndice());
			index += 2;          
		}
        }
  }

  verticesP->dirty();
  verticesE->dirty();
  nodeGeom->dirtyBound();
  lineGeom->dirtyBound();
  group->computeBound();
}

void GML::createText(string label, osg::Vec3 position, float fontsize)
{
/*
   osgText::Text* labelText = new osgText::Text();
   labelText->setDataVariance(osg::Object::DYNAMIC);
   labelText->setDrawMode(osgText::Text::TEXT);
   labelText->setColor(osg::Vec4(1,1,1,1));

   labelText->setAlignment(osgText::Text::CENTER_BASE_LINE);
   labelText->setCharacterSize(fontsize);
   labelText->setLayout(osgText::Text::LEFT_TO_RIGHT);
   labelText->setAxisAlignment(osgText::Text::XY_PLANE);

   osg::ref_ptr<osg::Geode> textNode = new osg::Geode();
   textNode->addDrawable(labelText);

   // TODO add use shader to recompute billboard positions correctly

   cvr::Billboard *billBoard = new cvr::Billboard();
   //osg::Vec3 zaxis(0,0,1);
   //osg::Vec3 zaxis(0,1,0);
   //billBoard->setAxis(zaxis);  // need to rotate to the viewer at all times
   //osg::Vec3 normal(0,-1,0);
   osg::Vec3 normal(0,0,1);
   billBoard->setNormal(normal);
   billBoard->addChild(textNode.get());

   labelText->setText(label);

   // create matrixtransform for the position
   osg::MatrixTransform* mat = new osg::MatrixTransform();
   osg::Matrix location;
   location.setTrans(position);
   mat->setMatrix(location);
   mat->addChild(billBoard);

   // add billboard 
   group->addChild(mat);
*/
}

void GML::createConnectivity()
{
	// default attach node
	Node* attach = nodelist->at(0);
		
	for(int i = 0; i < (int)nodelist->size(); i++)
	{
		Node * node = nodelist->at(i);
		if(!node->isSeen())
		{
			node->setSeen();

			//check if new node needs to be connected to graph
			if(node != attach)
			{	
				Edge* edge = new Edge(true);
                                attach->addEdge(edge);
                                node->addEdge(edge);
                                edge->setSource(attach);
                                edge->setTarget(node);
			}
		
			// internal loop that traverses all the children of this subgraph	
			queue< Node* > lastNodes;
			lastNodes.push(node);
			while((int)lastNodes.size())
			{
				Node* current = lastNodes.front();
                		lastNodes.pop();

				vector<Edge *> edges = current->getEdges();
        			for(int j = 0; j < (int)edges.size(); j++)
        			{
				
                			Edge* edge = edges.at(j);
                			if( !edge->isFalseEdge() && !edge->isSeen())
                			{

						if(!edge->getTarget()->isSeen())
						{
							lastNodes.push(edge->getTarget());
							edge->getTarget()->setSeen();
						}
						if(!edge->getSource()->isSeen())
						{
							lastNodes.push(edge->getSource());
							edge->getSource()->setSeen();
						}

						// set seen flag
						edge->visited();
					}
				}
			}
                }
        }
}


// need to adjust the reader to add the additional information for using igraph layout
void GML::makeForceDirectedLayoutLocations()
{
  srand( 13 );

  igraph_t graph;
  igraph_matrix_t results;

  // read data into igraph layout algorithm
  int numnodes = (int) nodelist->size();
 
  //TODO need to compute number of edges
  printf("Number of nodes %d num edges %d\n", numnodes, numEdges);

  igraph_matrix_init(&results, numnodes, 3);

  igraph_vector_t edges;

  // Number Edges are indEdges size
  igraph_vector_init(&edges, numEdges);

  // reset index
  int index = 0;
  for (int i = 0; i < numnodes; i++)
  {
       	Node * currentNode = nodelist->at(i);
       	int sourceId = currentNode->getIndice();

       	// loop through edge list adding the edge (skip first edge as it was using in initalization of the vector)
       	for(int j = 0; j < (int)currentNode->getEdges().size(); j++)
       	{
		
               	Edge * edge = currentNode->getEdges().at(j);
		if(currentNode == edge->getSource() && !edge->isFalseEdge()) // second for loop conditions
		{		
               		VECTOR(edges)[index] = sourceId;
               		index++;
               		VECTOR(edges)[index] = edge->getTarget()->getIndice();
               		index++;
		}
       	}
		
	// generate random positions
	MATRIX(results, i, 0) = (((double)rand() / RAND_MAX) * 2) -1;	
	MATRIX(results, i, 1) = (((double)rand() / RAND_MAX) * 2) -1;	
	MATRIX(results, i, 2) = (((double)rand() / RAND_MAX) * 2) -1;	
  }

  igraph_create(&graph, &edges, 0, 0);
  igraph_vector_destroy(&edges);

  printf("before layout\n");

  // execute layout algorithm
  igraph_layout_fruchterman_reingold_3d(&graph, &results, 500, numnodes,
                        numnodes * numnodes * numnodes * 0.2, 1.5,
                        numnodes * numnodes * numnodes * numnodes * 0.2,
                        true, NULL);

  printf("After layout\n");
  
  // set vertices
  for(int i = 0; i < igraph_matrix_nrow(&results); i++)
  {
    	nodelist->at(i)->savePoint(MATRIX(results,i,0),MATRIX(results,i,1), MATRIX(results,i,2));
  }

  igraph_matrix_destroy(&results);

  
  //update vertices for layout (should of already been initalized in gml reader)
  movePoints();
}


// push onto vector and pop off when visiting them
void GML::makeLocations(Node * rootNode)
{
   
    // reset all edges on nodes before traversal
    resetNodes();

    // set activeNode
    activeNode = rootNode;

    // first node is the root node
    bool isroot = true;

    // create sphere placement
    DataParams param;
    param.xmin = 0;
    param.xmax = 180;
    param.ymin = 0;
    param.ymax = 360;
    param.lastradius = 0;
    param.lastaddition = 0;

    queue< std::pair< Node*, DataParams > > lastNodes;
    lastNodes.push( std::pair<Node* , DataParams >(rootNode, param));

    while(lastNodes.size())
    {
	
	std::pair<Node*, DataParams > last = lastNodes.front();
	lastNodes.pop();
	Node * current = last.first;
	DataParams par = last.second;

    	static float DtoR = M_PI / 180.0;
    	float tempradius, tempaddition;
    	float xn = par.xmin;
    	float xx = par.xmax;
    	float yn = par.ymin;
    	float yx = par.ymax;

	if(isroot)
    	{
		tempradius = 1000; // make it configurable base radius
		tempaddition = 1000;
		isroot = false;

		// save the resultant point
		//current->savePoint(basePosition.getX(), basePosition.get(Y),0.0, 0.0);
    	}
    	else
    	{
		tempaddition = (par.lastaddition * 0.9); //decay factor make configurable
		tempradius = par.lastradius + tempaddition;
		
		float px = (xn + ((xx - xn) / 2.0f)) * DtoR;
		float py = (yn + ((yx - yn) / 2.0f)) * DtoR;
		
		//save the resultant point (relative to the root node)
		current->savePoint(rootNode->getX() + (tempradius * sin(px) * cos(py)), rootNode->getY() + (tempradius * sin(px) * sin(py)), rootNode->getZ() + (tempradius * cos(px)));
    	}

	// get number of children that need to be traversed
	int numberOfChildren = current->getNumOfActiveEdges();
    	if( !numberOfChildren ) 
    	{
		continue;
    	}

 	float sqr = sqrt(numberOfChildren);
    	int sqri = (int)sqr;
    	if(sqr > (float)sqri)
    	{
		sqri++;
    	}

    	float difx = (par.xmax - par.xmin) / ((float) sqri);
    	float dify = (par.ymax - par.ymin) / ((float) sqri);
    	float tempxmin = par.xmin;
    	float tempxmax = par.xmin + difx;
    	float tempymin = par.ymin;
    	float tempymax = par.ymin + dify;


    	for(int i = 0; i < sqri; i++)
    	{
		tempxmin = par.xmin;
		tempxmax = par.xmin + difx;

		for(int j = 0; j < sqri; j++)
		{

			std::pair<Node* ,Edge* > node = current->getFirstActiveNode();
			if(node.first == NULL)
				continue;

			// create param data
			DataParams param;
			param.xmin = tempxmin;
			param.xmax = tempxmax;
			param.ymin = tempymin;
			param.ymax = tempymax;
			param.lastradius = tempradius;
			param.lastaddition = tempaddition;

			// add node to list of nodes to traverse with param information
			lastNodes.push(std::pair<Node *, DataParams > (node.first, param));

	    		tempxmax += difx;
	    		tempxmin += difx;
		}

		tempymax += dify;
		tempymin += dify;
    	}
    }

    // update edge vertices to match
    movePoints();
}



void GML::resetNodes()
{
	for(int i = 0; i < (int)nodelist->size(); i++)
	{
		Node* node = nodelist->at(i);
		node->resetEdges();
		node->resetSeen();
	}
}

void GML::clearGraph()
{
	// remove all nodes attached to the scene graph
        SceneManager::instance()->getObjectsRoot()->removeChild(group);

   	// replace removed node
	group = new osg::Group();
	SceneManager::instance()->getObjectsRoot()->addChild(group);
}

void GML::updateMovement()
{
	// if not moving exit
	if(!pointsMoving)
		return;

	bool done = true;
	for(int i = 0; i < (int)nodelist->size(); i++)
	{
		Node * node = nodelist->at(i);
		Lerpable<osg::Vec3> * lerp = node->getLerp();
		lerp->moveToTarget();
		osg::Vec3 temp = lerp->getValue();
		node->setLastPoint(temp);
		node->setX(temp.x());
		node->setY(temp.y());
		node->setZ(temp.z());

        	if(lerp->hasChanged())
        	{
            		done = false;
        	}
	}

	if(done)
        {
                pointsMoving = false;
        }

        // update the vertices
	updateVertices();
}

void GML::movePoints()
{
	// points still transitioning
	if(pointsMoving)
		return;

	pointsMoving = true;

	for(int i = 0; i < (int) nodelist->size(); i++)
	{
		Node * node = nodelist->at(i);
		Lerpable<osg::Vec3> * lerp = node->getLerp();
		lerp->setImmediateValue(node->getLastPoint());
		lerp->setValue(node->getPoint());
		lerp->setSpeed(0.20);
	}
}

// TODO shift to gpu in pre process.... as test is slow the more nodes you have (only intersect test with visible data)
void GML::intersectionTesting()
{
  if( geode != NULL)
  {
	// intersection testing
  	osg::Matrix w2o = PluginHelper::getWorldToObjectTransform();
  	osg::Vec3 pointerStart = PluginHelper::getHandMat(0).getTrans() * w2o, pointerEnd;
  	pointerEnd.set(0.0f, 10000.0f, 0.0f);
  	pointerEnd = pointerEnd * PluginHelper::getHandMat(0) * w2o;

  	Node* inter = NULL;
  	for(int i = 0; i < (int)nodelist->size(); i++)
  	{
		Node* node = nodelist->at(i);
        	osg::Vec3 point = verticesP->at(node->getIndice());
        	float dist = ((pointerEnd - pointerStart) ^ (pointerStart - point)).length() / (pointerEnd - pointerStart).length();
        	if(dist < colorsNodes->at(node->getIndice()).w())
        	{
            		inter = node;

			if( mouseOver != inter ) // check if nodes match
			{
				//reset previous mousedOver
				if(mouseOver != NULL)
				{
					mouseOver->resetMouseOver();
					
					//reset color
					updateColor(mouseOver, false);
				}

				// set the newly intersected node to mouseOver
				mouseOver = inter;
				mouseOver->setMouseOver();

				// TODO pass NODE do function call
        			updateColor(mouseOver, true);
			}
		}
        }

	// nothing intersected
	if( inter == NULL )
	{
		//disable any mouse over
		if(mouseOver != NULL)
		{
			mouseOver->resetMouseOver();
       			updateColor(mouseOver, false);
		}
		mouseOver = NULL;
	}
  }
}

bool GML::processEvent(InteractionEvent * event)
{
    TrackedButtonInteractionEvent * tie = event->asTrackedButtonEvent();
    if(!tie)
    {
	return false;
    }

    if(tie->getHand() == 0 && tie->getButton() == 0 && tie->getInteraction() == BUTTON_DOWN)
    {
	// check for current intersected node
	if( mouseOver != NULL && activeNode != mouseOver)
	{
	    activeNode = mouseOver;
	    makeLocations(activeNode);
	}
    }
    return false;
}

// intialize
bool GML::init()
{
  cerr << "GML::GML" << endl;

  // enable osg debugging
  //osg::setNotifyLevel( osg::INFO );

  // init variables
  pointsMoving = false;
  activeNode = NULL;
  mouseOver = NULL;
  objectScale = NULL;

  geode = NULL;
  group = new osg::Group();
  SceneManager::instance()->getObjectsRoot()->addChild(group);

  GMLLayoutMenuItem = new SubMenu("GML Network Layout", "GML Network Layout");
  GMLLayoutMenuItem->setCallback(this);
  
  forceDirectedLayout = new MenuButton("Force Directed Layout");
  forceDirectedLayout ->setCallback(this);
  
  GMLLayoutMenuItem ->addItem(forceDirectedLayout);
 
  MenuSystem::instance()->addMenuItem(GMLLayoutMenuItem);

  return true;
}

// this is called if the plugin is removed at runtime
GML::~GML()
{
   fprintf(stderr,"GML::~GML\n");
}

void GML::preFrame()
{
  if( objectScale != NULL )
    objectScale->set(PluginHelper::getObjectScale());

  //intersectionTesting();
  
  updateMovement();
}

// NODE FUNCTIONS

Node::Node() 
{
	_x = 0.0;
	_y = 0.0;
	_z = 0.0;
        _r = 1.0;
	_g = 0.0;
	_b = 0.0;
	//_w = 20.0;
	_w = 500.0;
	mouseOver = false;
	lerp = new Lerpable<osg::Vec3>();
}

Node::~Node() {}


void Node::setName(string label)
{
	_label = label;
}

string Node::getName()
{
	return _label;
}

void Node::addEdge(Edge* edge)
{
	if( edge != NULL )
		_edges.push_back(edge);
}

int Node::getNumOfActiveEdges()
{
	int num = 0;
	for(int i = 0; i < _edges.size(); i++)
	{
		if( !_edges[i]->hasBeenSeen() && _edges[i]->getSource() != _edges[i]->getTarget())
			num++;
	}
	return num;
}

std::pair< Node* ,Edge* > Node::getFirstActiveNode()
{
	for(int i = 0; i < _edges.size(); i++)
	{
		Edge* edge = _edges[i];
		if( !edge->hasBeenSeen() && edge->getSource() != edge->getTarget())
		{
			// set that the edge has been seen
			edge->visited();
			if(this != edge->getSource())
				return std::pair< Node*, Edge* > (edge->getSource(), edge);
			else // this can return a loop edge
				return std::pair< Node*, Edge* > (edge->getTarget(), edge);
		}
	}
	return std::pair<Node*, Edge* > (NULL, NULL);
}

void Node::resetEdges()
{
	for(int i = 0; i < _edges.size(); i++)
	{
		_edges[i]->reset();
	}
}

// EDGE FUNCTIONS
Edge::Edge() 
{
	_r = 0.0;
	_g = 1.0;
	_b = 0.0;
	_w = 100.0;
	_seen = false;
	_falseEdge = false;
};

Edge::Edge(bool edge) 
{
	_r = 1.0;
	_g = 1.0;
	_b = 1.0;
	_w = 100.0;
	_seen = false;
	_falseEdge = edge;
};

Edge::~Edge() {};
