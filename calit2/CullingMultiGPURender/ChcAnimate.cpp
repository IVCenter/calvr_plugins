#include "ChcAnimate.h"

#include <cvrConfig/ConfigManager.h>

#include <cstring>

ChcAnimate::ChcAnimate(std::string filename, int numcontexts) : numContexts(numcontexts)
{
   //init starting frame
   currentFrame = 0;
   lastFrame = -1;

   _paused = false;

   pmap = new std::map<int, Geometry* >();

   // set up context lists
   for(int i = 0; i < numContexts; i++)
   {
	plist.push_back(new std::vector<int>());	
   }

   _cudaCopy = cvr::ConfigManager::getBool("Plugin.CullingMultiGPURender.CudaCopy",false);
   
   // load in all meta data
   loadFrameMetaData(filename);

   FetchQueue::getInstance()->setMaxFrames((int)frameSetup.size());

   // initalize all geometry on for first frame
   std::map<int, Geometry*>::iterator it;
   for(it = pmap->begin(); it != pmap->end(); ++it)
   {
       it->second->SetVisible(false);   
   }

   traverser = NULL;
}

std::map<int, Geometry* > * ChcAnimate::getGeometryMap()
{
   return pmap; 
}

std::vector<int> * ChcAnimate::getPartList(int gpuIndex)
{
    return plist.at(gpuIndex);
}

void ChcAnimate::setFrame(int frame)
{
    currentFrame = frame % (int)frameSetup.size();
}

void ChcAnimate::advance()
{
    currentFrame++;
    currentFrame = currentFrame % (int)frameSetup.size();
}

int ChcAnimate::getFrame()
{
    return currentFrame;
}

void ChcAnimate::play()
{
    _paused = false;
}

void ChcAnimate::pause()
{
    _paused = true;
}

bool ChcAnimate::getPaused()
{
    return _paused;
}

void ChcAnimate::update()
{
    if(!_paused)
    {
	advance();
    }
}

int ChcAnimate::getNextFrame()
{
    return (currentFrame+1) % (int)frameSetup.size();
}

void ChcAnimate::setNextFrame()
{
	//printf("nextframe called\n");
	if( traverser == NULL)
	{
	    traverser = new RenderTraverser(pmap);
	}

	// frame start
	Geometry::ResetStats();
	if(currentFrame != lastFrame)
	{
	    traverser->SetFrame(currentFrame);
	    traverser->SetHierarchy(frameSetup.at(currentFrame));
	}

	lastFrame = currentFrame;

	traverser->SetViewpoint(eyePos);
	traverser->SetProjViewMatrix(eyeProjView);
	traverser->PreRender();

	//printf("Number of parts that should be rendered %d\n", traverser->GetNumRenderedGeometry());
        //printf("Number of traversed nodes %d\n", traverser->GetNumTraversedNodes());
	//printf("Number of query culled nodes %d\n", traverser->GetNumQueryCulledNodes());
	//printf("Number of frustrum culled nodes %d\n", traverser->GetNumFrustumCulledNodes());
}

void ChcAnimate::turnOffGeometry()
{
    std::map<int, Geometry*>::iterator it;
    for(it = pmap->begin(); it != pmap->end(); ++it)
    {
	it->second->SetVisible(false);
    }
}

void ChcAnimate::postRender()
{
	float totalcopy = 0;
	float totalload = 0;
	float totaldata = 0;
	// compute totalCopy time
	/*std::map<int, Geometry*>::iterator it;
	for(it = pmap->begin(); it != pmap->end(); ++it)
	{
	    if( it->second->isDrawn() )
	    {
		totaldata = totaldata + it->second->getDataSize();
	    }
	}

	printf("Data tranfered from disk: %f megabytes\n", totaldata / 1000000.0);*/


	traverser->PostRender();

	//printf("Post Number of parts that should be rendered %d\n", traverser->GetNumRenderedGeometry());
        //printf("Post Number of traversed nodes %d\n", traverser->GetNumTraversedNodes());
	//printf("Post Number of query culled nodes %d\n", traverser->GetNumQueryCulledNodes());
	//printf("Post Number of frustrum culled nodes %d\n", traverser->GetNumFrustumCulledNodes());
}

void ChcAnimate::postRenderPerThread(int context)
{
    for(int i = 0; i < plist[context]->size(); i++)
    {
	(*pmap)[plist[context]->at(i)]->processPostDrawn();
    }
}

void ChcAnimate::updateViewParameters(double* eyepos, double* eyeprojview)
{
    memcpy(eyePos, eyepos, sizeof(double) * 3);
    memcpy(eyeProjView, eyeprojview, sizeof(double) * 16);     
}

// read in all the frame meta data
void ChcAnimate::loadFrameMetaData(std::string filename)
{
    std::cerr << "Loading frame meta data." << std::endl;
	// load in all the geometry first (indices and vertices
	int numberParts = 0;
	int partNumber = 0;

	int numIndices = 0;
	unsigned int * indices = NULL;
	Geometry * temp = NULL;
	ifstream myind ((filename + "Indices.bin").c_str(), ios::binary);
	if(myind.fail())
	{
	    std::cerr << "Error opening file: " << filename + "Indices.bin" << std::endl;
	}
	else
	{
	    std::cerr << "Reading file: " << filename + "Indices.bin" << std::endl;;
	}
        myind.read((char*)&numberParts, sizeof(int));
        for(int i = 0; i < numberParts; i++)
        {
                // calculate where description data is kept in file
                int offseti = sizeof(int) + (sizeof(int) * 3 * i);
                myind.seekg(offseti);

                myind.read((char*) &partNumber, sizeof(int));
	        // read out size and then offset
		
	        myind.read((char*) &numIndices, sizeof(int));
	        indices  = new unsigned int[numIndices];
	        myind.read((char*) &offseti, sizeof(int));
	        // get part number and number of vertices from file, calc fileoffset first
	        myind.seekg(offseti);
	        myind.read((char*)indices, sizeof(int) * numIndices);

		

		// init geometry and add to map
		temp = new Geometry(partNumber, filename, numIndices, indices, _cudaCopy);
		//temp = new Geometry(partNumber, filename, _cudaCopy);
		// add geometry to map
		pmap->insert(std::pair<int, Geometry*> (partNumber, temp));
		// add to context lists
		plist.at( i % numContexts )->push_back(partNumber);

	}
	myind.close();

	int numVertexs = 0;
	int offset = 0;
	ifstream myvert ((filename + "0data.bin").c_str(), ios::binary);
	if(myvert.fail())
	{
	    std::cerr << "Unable to open file: " << filename + "0data.bin" << std::endl;
	}
	//myvert.open(filename.c_str(), std::ios::binary | std::ios::in);
	myvert.read((char*)&numberParts, sizeof(int));
	std::cerr << "Got " << numberParts << " parts" << std::endl;
	for(int i = 0; i < numberParts; i++)
	{
	       // calculate where description data is kept in file
	       int offseti = sizeof(int) + (sizeof(int) * 3 * i);
	       myvert.seekg(offseti);
               myvert.read((char*) &partNumber, sizeof(int));

               // read out size and then offset
               myvert.read((char*) &numVertexs, sizeof(int));
               myvert.read((char*) &offset, sizeof(int));

	       // look up geometry in map and set vertex data
	       std::map<int, Geometry*>::iterator it;
	       it = pmap->find(partNumber);
	       if( it != pmap->end())
	       {
		   //std::cerr << "Setting numVerts " << numVertexs << std::endl;
		    (*it).second->SetVertices(numVertexs, offset);
	       }
	}
	myvert.close();


	// loop through and read in as many frame structures as they exist
	while(true)
	{
		HierarchyNode* node = NULL;
		
		string name(filename);
		stringstream ss;
		ss << (int)frameSetup.size();
		name.append(ss.str()).append("layout.bin");

        	// now read binary structure file
        	ifstream myfile (name.c_str(), ios::binary);
        	if (myfile.is_open())
        	{
                	node = DecodeState(myfile,node,0, (int)frameSetup.size(), filename);
                	myfile.close();
			frameSetup.push_back(node);
        	}
		else
		{
		    return;
		}
	}		
}

HierarchyNode* ChcAnimate::DecodeState(ifstream &in, HierarchyNode* parent, int level, int frameNum, std::string filename)
{
        // check first character if null then return NULL
        int nochild = -1;
        in.read((char*)&nochild, sizeof(int));

        //check if null
        if(nochild == 0)
        {
                return NULL;
        }
        else
        {
                // read in bounds
		Vector3 lower, upper;
                in.read((char *)&lower, sizeof(double) * 3);
                in.read((char *)&upper, sizeof(double) * 3);

		HierarchyNode *node = NULL;	
		if(parent == NULL) // root node
			node = new HierarchyNode();
		else
                	node = new HierarchyNode(lower,upper,parent, level);

		node->SetFrame(frameNum);

		//read in bounding volume
		double bound[6];
		in.read((char*) bound, sizeof(double) * 6);

                // read in number of geometry
		int partNumber = 0;
                int size = 0;
                in.read((char*)&size, sizeof(int));

                for(int i =0; i < size; i++)
                {
			in.read((char*)&partNumber, sizeof(int)); //d3plot partnumber

			Geometry* temp = NULL;
			std::map<int, Geometry*>::iterator it;
			it = pmap->find(partNumber);
			if(it != pmap->end())
			{
				temp = it->second;
			}
			else
			{
				printf("Part loaded in wrong spot\n");

				// TODO need to read in index table for each new geometry
				temp = new Geometry(partNumber, filename, _cudaCopy);

				// add geometry to map
				pmap->insert(std::pair<int, Geometry*> (partNumber, temp));

				// add to context lists
				plist.at( i % numContexts )->push_back(partNumber);
			}
	
                        node->AddGeometry(temp);
                }
		
		//set boundingVolume
		memcpy((void*)&node->GetBoundingVolume(), (void*)bound, sizeof(double) * 6);
		
                node->SetLeftChild(DecodeState(in, node, level+1, frameNum, filename));
                node->SetRightChild(DecodeState(in, node, level+1, frameNum, filename));

                return node;
        }
}



