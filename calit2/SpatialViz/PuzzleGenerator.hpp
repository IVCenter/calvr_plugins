// STD
#include <iostream>
#include <vector>
#include <queue>

// OSG
#include <osg/Vec3d>

// PhysX
#include <foundation/Px.h>

using namespace std;
using namespace osg;
using namespace physx;

class PuzzleGenerator{
public:
    static void createTetrisPiece(int, int, vector<Vec3>&);
    static void createPuzzleCube(int, vector<Vec3>&);
    static void create5x5(int, vector<Vec3>&, vector<Vec3>&, vector<Vec3>&);
    static void createLabyrinth(float, float, vector<PxVec3>&, vector<PxVec3>&);
};

//---------------------------------------- TETRIS PIECES ----------------------------------------//

void PuzzleGenerator::createTetrisPiece(int size, int numPieces, vector<Vec3> &cubeLocations) {
    
    // size = dimension of the cube we want the tetris piece to fit in
    // numPieces = number of small cubes to make up the tetris piece
    // cubeLocations is the vector to store the center of the locations of the cubes
    
    // Initiallize and setup cube for maze generation
	int*** cube = new int**[size];
	for (int i = 0; i < size; i++) {
		
		cube[i] = new int*[size];
		for (int j = 0; j < size; j++) {
			
			cube[i][j] = new int[size];
			for (int k = 0; k < size; k++) {
			    
			    int temp = rand() % (999);
			    cube[i][j][k] = temp + 1;       // values range from 1 to 1000 (0 means visited)
			}
		}
	}
    
    // PRINT OUT THE MATRIX
	/*for (int currLayer = 0; currLayer < size; currLayer++){
	    cerr << "layer " << currLayer << endl;
	    for (int x = 0; x < size; x++) {
	        for (int y = 0; y < size; y++) {
	            cerr << setw(2) << cube[x][y][currLayer] << " : " ;
	        }cerr << endl;
	    }cerr << endl << endl;
	}*/
	
	// Setup for iteration of prim's algorithm
	priority_queue< pair<int, Vec3 > > q = priority_queue< pair<int, Vec3 > >   ();
	
	// mark the first corner as visited
	cube[0][0][0] = 0;
	
	// defining the position to pass into the queue
	Vec3 pos1 = Vec3( 1, 0, 0 );
	Vec3 pos2 = Vec3( 0, 1, 0 );
	Vec3 pos3 = Vec3( 0, 0, 1 );
	
		
	// adding the potential wall to the queue
	//                 value        position of wall
	q.push( make_pair(cube[1][0][0], pos1));    // right
	q.push( make_pair(cube[0][1][0], pos2));    // down
	q.push( make_pair(cube[0][0][1], pos3));    // up a level
	
	// accumulate the cube locations
    cubeLocations.push_back(Vec3(0,0,0));
    while (!q.empty() && cubeLocations.size() < numPieces) {

		// Look at the top(greatest) element
		pair<int, Vec3 > edge = q.top();
		q.pop();
		
		//cerr << "popped: " << edge.second[0] << " : " << edge.second[1] << " : " << edge.second[2] << ", pVal = " << edge.first << endl;
		        
        int x = edge.second[0];
		int y = edge.second[1];
		int z = edge.second[2];
		
		// if you have not yet visited that location 
		if (cube[x][y][z] != 0) {
		
		    // mark as visited
		    cube[x][y][z] = 0;
		    
		    // add the cube to cubeLocations
		    cubeLocations.push_back(Vec3(x,y,z));
		    
		    // push the neighbors into the queue
		    // right and left neighbors
		    if (x-1 >= 0 && cube[x-1][y][z]) {
		        //cube[x][y][z] = 0;
		        q.push(make_pair(cube[x-1][y][z], Vec3(x-1, y, z)));
		    }
		    if (x+1 < size && cube[x+1][y][z]) {
		        //cube[x][y][z] = 0;
		        q.push(make_pair(cube[x+1][y][z], Vec3(x+1, y, z)));
		    }
		    // up and down neighbors
		    if (y-1 >= 0 && cube[x][y-1][z]) {
		        //cube[x][y][z] = 0;
		        q.push(make_pair(cube[x][y-1][z], Vec3(x, y-1, z)));
		    }
		    if (y+1 < size && cube[x][y+1][z]) {
		        //cube[x][y][z] = 0;
		        q.push(make_pair(cube[x][y+1][z], Vec3(x, y+1, z)));
		    }
		    // upstairs and downstairs neighbors
		    if (z-1 >= 0 && cube[x][y][z-1]) {
		        //cube[x][y][z] = 0;
		        q.push(make_pair(cube[x][y][z-1], Vec3(x, y, z-1)));
		    }
		    if (z+1 < size && cube[x][y][z+1]) {
		        //cube[x][y][z] = 0;
		        q.push(make_pair(cube[x][y][z+1], Vec3(x, y, z+1)));
		    }
		}
    }
    
    // delete cube
    for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			delete(cube[i][j]);
		}delete(cube[i]);
	}delete(cube);
    
}

//------------------------------------------ MAZE CUBE ------------------------------------------//

void PuzzleGenerator::createPuzzleCube(int size, vector<Vec3>& positions) {
    // generates the positions for the cubes to create the mazeCube of dimensions: size x size
    // these positions are returned in the vector of Vec3's if (size < 2) return;

    if (size < 2) return;
    int cleanliness = 50;       // 0 - 100
	int straightener = 50;  // 0 - charMax
	srand((unsigned)time(0));

	// Initiallize and setup cube for maze generation
	char*** cube = new char**[size];
	for (int i = 0; i < size; i++) {
		cube[i] = new char*[size];
		for (int j = 0; j < size; j++) {
			cube[i][j] = new char[size];
			for (int k = 0; k < size; k++) {
				if ((i == 0 || j == 0 || k == 0 || i == size - 1 || j == size - 1 || k == size - 1)) {
					cube[i][j][k] = rand() % (CHAR_MAX - 1) + 2;
				} else {
					cube[i][j][k] = 1;
				}
			}
		}
	}
		
	// Setup for iteration of prim's algorithm
	priority_queue< pair<char, Vec3 > > q = priority_queue< pair<char, Vec3 > > ();
    
	cube[0][0][0] = 0;
	// defining the position to pass into the queue
    Vec3 pos1 = Vec3( 1, 0, 0 );
    Vec3 pos2 = Vec3( 0, 1, 0 );
    Vec3 pos3 = Vec3( 0, 0, 1 );
	q.push(make_pair(cube[1][0][0], pos1));
	q.push(make_pair(cube[0][1][0], pos2));
	q.push(make_pair(cube[0][0][1], pos3));

    while (!q.empty()) {

		// Look at the top(greatest) element
		pair<char, Vec3 > edge = q.top();
		q.pop();

		int x = edge.second[0];
		int y = edge.second[1];
		int z = edge.second[2];
		
		// Check if pixel creates a loop
		int direction = 1;
		if (x > 0 && cube[x - 1][y][z] == 0) direction *= 10;
		if (x < size - 1 && cube[x + 1][y][z] == 0) direction *= 60;
		if (y > 0 && cube[x][y - 1][z] == 0) direction *= 20;
		if (y < size - 1 && cube[x][y + 1][z] == 0) direction *= 50;
		if (z > 0 && cube[x][y][z - 1] == 0) direction *= 30;
		if (z < size - 1 && cube[x][y][z + 1] == 0) direction *= 40;

		if (direction > 60) continue;
		
		// Set pixel to visited
		cube[x][y][z] = 0;

		// Push neighbors, while adjusting weights so that straight is encouraged
		if (x > 0 && cube[x - 1][y][z] > 1) {
			Vec3 temp = Vec3(x - 1, y, z);
			if (direction == 60) cube[x - 1][y][z] = (cube[x - 1][y][z] > CHAR_MAX - straightener) ? CHAR_MAX : cube[x - 1][y][z] + straightener;
			q.push(make_pair(cube[x - 1][y][z], temp));
		}
		if (x < size - 1 && cube[x + 1][y][z] > 1) {
			Vec3 temp = Vec3( x + 1, y, z);
			if (direction == 10) cube[x + 1][y][z] = (cube[x + 1][y][z] > CHAR_MAX - straightener) ? CHAR_MAX : cube[x + 1][y][z] + straightener;
			q.push(make_pair(cube[x + 1][y][z], temp));
		}
		if (y > 0 && cube[x][y - 1][z] > 1) {
			Vec3 temp = Vec3( x, y - 1, z);
			if (direction == 50) cube[x][y - 1][z] = (cube[x][y - 1][z] > CHAR_MAX - straightener) ? CHAR_MAX : cube[x][y - 1][z] + straightener;
			q.push(make_pair(cube[x][y - 1][z], temp));
		}
		if (y < size - 1 && cube[x][y + 1][z] > 1) {
			Vec3 temp = Vec3( x, y + 1, z );
			if (direction == 20) cube[x][y + 1][z] = (cube[x][y + 1][z] > CHAR_MAX - straightener) ? CHAR_MAX : cube[x][y + 1][z] + straightener;
			q.push(make_pair(cube[x][y + 1][z], temp));
		}
		if (z > 0 && cube[x][y][z - 1] > 1) {
			Vec3 temp = Vec3( x, y, z - 1 );
			if (direction == 40) cube[x][y][z - 1] = (cube[x][y][z - 1] > CHAR_MAX - straightener) ? CHAR_MAX : cube[x][y][z - 1] + straightener;
			q.push(make_pair(cube[x][y][z - 1], temp));
		}
		if (z < size - 1 && cube[x][y][z + 1] > 1) {
			Vec3 temp = Vec3( x, y, z + 1 );
			if (direction == 30) cube[x][y][z + 1] = (cube[x][y][z + 1] > CHAR_MAX - straightener) ? CHAR_MAX : cube[x][y][z + 1] + straightener;
			q.push(make_pair(cube[x][y][z + 1], temp));
		}
    }
    
    // Clean up pointless dead ends
	for (int x = 0; x < size; x++) {
		for (int y = 0; y < size; y++) {
			for (int z = 0; z < size; z++) {
				if ((x == 0 || y == 0 || z == 0 || x == size - 1
					|| y == size - 1 || z == size - 1) && cube[x][y][z] == 0) {

					int direction = 1;
					if (x > 0 && cube[x - 1][y][z] == 0) direction *= 10;
					if (x < size - 1 && cube[x + 1][y][z] == 0) direction *= 60;
					if (y > 0 && cube[x][y - 1][z] == 0) direction *= 20;
					if (y < size - 1 && cube[x][y + 1][z] == 0) direction *= 50;
					if (z > 0 && cube[x][y][z - 1] == 0) direction *= 30;
					if (z < size - 1 && cube[x][y][z + 1] == 0) direction *= 40;

					if (direction > 60) continue;

					if (rand() % 100 > cleanliness) {
						int count = 0;

						switch (direction) {
						case 10:
							if (y > 0 && cube[x - 1][y - 1][z] == 0) count++;
							if (y < size - 1 && cube[x - 1][y + 1][z] == 0) count++;
							if (z > 0 && cube[x - 1][y][z - 1] == 0) count++;
							if (z < size - 1 && cube[x - 1][y][z + 1] == 0) count++;

							break;
						case 60:
							if (y > 0 && cube[x + 1][y - 1][z] == 0) count++;
							if (y < size - 1 && cube[x + 1][y + 1][z] == 0) count++;
							if (z > 0 && cube[x + 1][y][z - 1] == 0) count++;
							if (z < size - 1 && cube[x + 1][y][z + 1] == 0) count++;

							break;
						case 20:
							if (x > 0 && cube[x - 1][y - 1][z] == 0) count++;
							if (x < size - 1 && cube[x + 1][y - 1][z] == 0) count++;
							if (z > 0 && cube[x][y - 1][z - 1] == 0) count++;
							if (z < size - 1 && cube[x][y - 1][z + 1] == 0) count++;

							break;
						case 50:
							if (x > 0 && cube[x - 1][y + 1][z] == 0) count++;
							if (x < size - 1 && cube[x + 1][y + 1][z] == 0) count++;
							if (z > 0 && cube[x][y + 1][z - 1] == 0) count++;
							if (z < size - 1 && cube[x][y + 1][z + 1] == 0) count++;

							break;
						case 30:
							if (x > 0 && cube[x - 1][y][z - 1] == 0) count++;
							if (x < size - 1 && cube[x + 1][y][z - 1] == 0) count++;
							if (y > 0 && cube[x][y - 1][z - 1] == 0) count++;
							if (y < size - 1 && cube[x][y + 1][z - 1] == 0) count++;

							break;
						case 40:
							if (x > 0 && cube[x - 1][y][z + 1] == 0) count++;
							if (x < size - 1 && cube[x + 1][y][z + 1] == 0) count++;
							if (y > 0 && cube[x][y - 1][z + 1] == 0) count++;
							if (y < size - 1 && cube[x][y + 1][z + 1] == 0) count++;

							break;
						default:
							break;
						}

						if (count > 1) cube[x][y][z] = 2;

					}

				}
			}
		}
	}
		
	// Load walls
	//vector<Vec3> positions = vector<Vec3>();
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < size; k++) {
				if ( (i == 0 || j == 0 || k == 0 || i == size - 1 || j == size - 1 || k == size - 1) && cube[i][j][k] > 1) {
	                Vec3 temp( (float)i, (float)j, (float)k );
					positions.push_back(temp);
				} 
			}
			delete(cube[i][j]);
		}
		delete(cube[i]);
	}
	delete(cube);
}

//------------------------------------------ 5x5 PUZZLE -----------------------------------------//

void PuzzleGenerator::create5x5(int puzzleSize, vector<Vec3>& verticalWalls, vector<Vec3>& horizontalWalls, vector<Vec3>& floors) {
    // puzzleSize is the dimension of the final puzzle, the three vectors of Vec3's store the different types of walls to draw

    float puzzleDim = 0.25;
    float height = 0.05;    float radius = 0.0375;
    float width = 0.001;
    float spacing = 0.1;                        // to have the maze an even 5x5 grid
    PxVec3 position = PxVec3(0.0f, 0.0f, 0.0f);
    
    
    //int puzzleSize = 5;           int layers = 2;
    int prioritySize = 2 * (puzzleSize - 1);
    
    // Initiallize and setup cube for maze generation
	int*** cube = new int**[prioritySize+1];
	for (int i = 0; i <= prioritySize; i++) {
		
		cube[i] = new int*[prioritySize+1];
		for (int j = 0; j <= prioritySize; j++) {
			
			cube[i][j] = new int[prioritySize+1];
			for (int k = 0; k <= prioritySize; k++) {
			    
			    // give the spots that represent grid locations 0 (meaning they have not yet been visited)
			    if ((i % 2 == 0 && j % 2 == 0 && k % 2 == 0) || (i % 2 == 1 && j % 2 == 1)) {
					cube[i][j][k] = 0;
				}
				//  k = even => wall layer          
			    else if (k % 2 == 0) {
				    int temp = rand() % (100);
				    cube[i][j][k] = temp;
				}
				// k = odd => floor layer   i and j are even (grid locations)
				else if ( (k % 2 == 1) && (i % 2 == 0 && j % 2 == 0) ){
				    int temp = rand() % (20);
				    cube[i][j][k] = temp;
				}
				else {
				    cube[i][j][k] = 0;
				}
			}
		}
	}
	
	// PRINT OUT THE MATRIX
	/*for (int currLayer = 0; currLayer <= prioritySize; currLayer++){
	    cerr << "layer " << currLayer << endl;
	    for (int x = 0; x <= prioritySize; x++) {
	        for (int y = 0; y <= prioritySize; y++) {
	            cerr << setw(2) << cube[x][y][currLayer] << " : " ;
	        }cerr << endl;
	    }cerr << endl << endl;
	}*/
   
    // Setup for iteration of prim's algorithm
	priority_queue< pair<int, Vec3 > > q = priority_queue< pair<int, Vec3 > >   ();

    // mark the first corner as visited
	cube[0][0][0] = 1;
	
	// defining the position to pass into the queue
	Vec3 pos1 = Vec3( 1, 0, 0 );
	Vec3 pos2 = Vec3( 0, 1, 0 );
	Vec3 pos3 = Vec3( 0, 0, 1 );
	
		
	// adding the potential wall to the queue
	//                 value        position of wall
	q.push( make_pair(cube[1][0][0], pos1));    // right
	q.push( make_pair(cube[0][1][0], pos2));    // down
	q.push( make_pair(cube[0][0][1], pos3));    // up a level
    
    
    bool verticalWall, horizontalWall, floor;

    while (!q.empty()) {

		// Look at the top(greatest) element
		pair<int, Vec3 > edge = q.top();
		q.pop();
		
		//cerr << "popped: " << edge.second[0] << " : " << edge.second[1] << " : " << edge.second[2] << ", pVal = " << edge.first << endl;
		        
        // Check if 
		//      the node we are trying to get to has already been visited or not
		//      a boundry check
		// based on this 
		//      -> increment the appropiate x, y or z value to the next block you are trying to connect 
		
		
		
		int x = edge.second[0];
		int y = edge.second[1];
		int z = edge.second[2];
		
		// get the two grid locations 
		Vec3 grid1, grid2;  	bool visit1, visit2;
		
		// store the wall location to potentially add to the walls vector to draw
		Vec3 wallLocation;
		
		// if the row is odd, the col is even, and z is even ... horizontal wall
		if (x % 2 == 1 && y % 2 == 0 && z % 2 == 0) {
            //cerr << "\t-> horizontal wall" << " (x,y) = " << x << ", " << y << " -> (l, w) = " << y/2 << ", " << x/2+1 << endl;
            horizontalWall = true;          verticalWall = false;       floor = false;
            
            grid1 = grid2 = edge.second;
            grid1[0]--;
            grid2[0]++;
            
            // store the indices for drawing the walls 
            wallLocation[0] = y/2;      // l
            wallLocation[1] = x/2+1;    // w
            wallLocation[2] = z/2;      // level
            
        }
        // if the row is even, the col is odd, and z is even ... vertical wall 
		else if (x % 2 == 0 && y % 2 == 1 && z % 2 == 0) {
		    //cerr << "\t-> vertical wall" << " (x,y) = " << x << ", " << y << " -> (l, w) = " << x/2 << ", " << y/2+1 << endl;
		    horizontalWall = false;          verticalWall = true;       floor = false;
		    
		    grid1 = grid2 = edge.second;
		    grid1[1]--;
		    grid2[1]++;
		    
		    // store the indices for drawing the walls 
            wallLocation[0] = x/2;      // l
            wallLocation[1] = y/2+1;    // w
            wallLocation[2] = z/2;      // level
           
		}
		// if x and y (row/col) are even, and z is odd ... floor tile 
		else if (x % 2 == 0 && y % 2 == 0 && z % 2 == 1) {
		    //cerr << "\t-> floor" << " (x,y) = " << x << ", " << y << " -> (l, w) = " << x/2 << ", " << y/2 << " level = " << z/2+1 << endl;
		    horizontalWall = false;          verticalWall = false;       floor = true;
		    
		    grid1 = grid2 = edge.second;
		    grid1[2]--;
		    grid2[2]++;
		    
		    // store the indices for drawing the floor
		    wallLocation[0] = x/2;      // l
		    wallLocation[1] = y/2;      // w
		    wallLocation[2] = z/2+1;    // level
		}
		else continue;
		
		// have we visited the two grid locations?
		visit1 = (int)cube[(int)grid1[0]][(int)grid1[1]][(int)grid1[2]];
		visit2 = (int)cube[(int)grid2[0]][(int)grid2[1]][(int)grid2[2]];
		
		//cerr << "\t-> connecting: " << endl;
        //cerr << "\t" << grid1[0] << " : " << grid1[1] << " : " << grid1[2] << ", visited? " << visit1 << endl;
        //cerr << "\t" << grid2[0] << " : " << grid2[1] << " : " << grid2[2] << ", visited? " << visit2 << endl;


        // check if we both of those grid locations are occupied or not
        if (!( visit1 && visit2)){
            // if we have not visited both the grid locations yet, knock down the wall(s) and mark them as visited
            
            // push on the neighbors for the unvisited node and mark that node as visited
            int x,y,z;              // the grid location (+- 1 is the wall location +-2 is the neighboring grid location
            
            // if we haven't visited grid1
            if (!visit1){
                
                // mark grid1 as visited
                cube[(int)grid1[0]][(int)grid1[1]][(int)grid1[2]] = 1;
                
                x = (int)grid1[0];
                y = (int)grid1[1];
                z = (int)grid1[2];
                
                // push on grid1's neighbors
                // go up/down
                // if we haven't visited the up neighbor
                if (x-2 >= 0 && !cube[x-2][y][z]) {
                   Vec3 temp = Vec3(x-1, y, z);
                    q.push(make_pair(cube[x-1][y][z], temp ));
                }
                // if we haven't visited the down neighbor
                if ( x+2 <= prioritySize && !cube[x+2][y][z]) {
                    Vec3 temp = Vec3(x+1, y, z);
                    q.push(make_pair(cube[x+1][y][z], temp ));
                }
                
                // go left/right
                // if we haven't visited the left neighbor
                if (y-2 >= 0 && !cube[x][y-2][z]) {
                    Vec3 temp = Vec3(x, y-1, z);
                    q.push(make_pair(cube[x][y-1][z], temp ));
                }
                // if we haven't visited the right neighbor
                if (y+2 <= prioritySize && !cube[x][y+2][z]) {
                    Vec3 temp = Vec3(x, y+1, z);
                    q.push(make_pair(cube[x][y+1][z], temp ));
                }
                
                // go upstairs/downstairs
                // if we haven't visited the upstairs neighbor
                if (z - 2 >= 0 && !cube[x][y][z-2] ) {
                    Vec3 temp = Vec3(x, y, z-1);
                    q.push( make_pair( cube[x][y][z-1], temp ) );
                }
                
                // if we haven't visited the upstairs neighbor
                if( z+2 <= prioritySize && !cube[x][y][z+2]) {
                    Vec3 temp = Vec3(x, y, z+1);
                    q.push( make_pair( cube[x][y][z+1], temp ) );
                }
                
            }
            // if we haven't visited grid2
            if (!visit2) {
                
                // mark grid2 as visited
                cube[(int)grid2[0]][(int)grid2[1]][(int)grid2[2]] = 1;
                
                // push on grid2's neighbors
                x = (int)grid2[0];
                y = (int)grid2[1];
                z = (int)grid2[2];
                
                // push on grid1's neighbors
                
                // go up/down
                // if we haven't visited the up neighbor 
                if (x - 2 >= 0 && !cube[x-2][y][z]) {
                    Vec3 temp = Vec3(x-1, y, z);
                    q.push(make_pair(cube[x-1][y][z], temp ));
                }
                // if we haven't visited the down neighbor
                if (x + 2 <= prioritySize && !cube[x+2][y][z]) {
                    Vec3 temp = Vec3(x+1, y, z);
                    q.push(make_pair(cube[x+1][y][z], temp ));
                }
                
                // go left/right
                // if we haven't visited the left neighbor
                if (y-2 >= 0 && !cube[x][y-2][z]) {
                    Vec3 temp = Vec3(x, y-1, z);
                    q.push(make_pair(cube[x][y-1][z], temp ));
                }
               // if we haven't visited the right neighbor
                if (y+2 <= prioritySize && !cube[x][y+2][z]) {
                    Vec3 temp = Vec3(x, y+1, z);
                    q.push(make_pair(cube[x][y+1][z], temp ));
                }
                
                // go upstairs/downstairs
                // if we haven't visited the upstairs neighbor
                if (z - 2 >= 0 && !cube[x][y][z-2] ) {
                    Vec3 temp = Vec3(x, y, z-1);
                    q.push( make_pair( cube[x][y][z-1], temp ) );
                }
                
               // if we haven't visited the upstairs neighbor
                if( z+2 <= prioritySize && !cube[x][y][z+2]) {
                    Vec3 temp = Vec3(x, y, z+1);
                    q.push( make_pair( cube[x][y][z+1], temp ) );
                }
            }
        }
        else {
            //cerr << "Both have been visited" << endl;
            if (verticalWall) {
                verticalWalls.push_back(wallLocation);
            }
            if (horizontalWall) {
                horizontalWalls.push_back(wallLocation);
            }
            if (floor) {
                floors.push_back(wallLocation);
            }
        }
    }
    
    // deleteCube
    for (int i = 0; i <= prioritySize; i++) {
		for (int j = 0; j <= prioritySize; j++) {
			delete(cube[i][j]);
		}delete(cube[i]);
	}delete(cube);
}

//-------------------------------------- LABYRINTH PUZZLE ---------------------------------------//

void PuzzleGenerator::createLabyrinth(float boxHeight, float floorHeight, vector<PxVec3>& positions, vector<PxVec3>& dimensions) {

    // create vectors with the positions and dimensions
    
    // the surrounding box:
    positions.push_back(PxVec3(0.0f, floorHeight+boxHeight, .20f));         dimensions.push_back(PxVec3(0.15, boxHeight*2, 0.005));
    positions.push_back(PxVec3(0.0f, floorHeight+boxHeight, 0.0f));         dimensions.push_back(PxVec3(0.15, boxHeight*2, 0.005));
    positions.push_back(PxVec3(-0.155f, floorHeight+boxHeight, 0.1f));      dimensions.push_back(PxVec3(0.005, boxHeight*2, 0.105));
    positions.push_back(PxVec3(0.155f, floorHeight+boxHeight, 0.1f));       dimensions.push_back(PxVec3(0.005, boxHeight*2, 0.105));
    
    // maze portions
    // left to right - horizontal boxes
    positions.push_back(PxVec3(-0.135, floorHeight, 0.16));                 dimensions.push_back(PxVec3(0.02, boxHeight, 0.0015));
    positions.push_back(PxVec3(-0.135, floorHeight, 0.08));                 dimensions.push_back(PxVec3(0.02, boxHeight, 0.0015));    
    positions.push_back(PxVec3(-0.105, floorHeight, 0.12));                 dimensions.push_back(PxVec3(0.0225, boxHeight, 0.0015));
    positions.push_back(PxVec3(-0.105, floorHeight, 0.04));                 dimensions.push_back(PxVec3(0.0225, boxHeight, 0.0015));    
    positions.push_back(PxVec3(-0.055, floorHeight, 0.16));                 dimensions.push_back(PxVec3(0.0325, boxHeight, 0.0015));
    positions.push_back(PxVec3(-0.02, floorHeight, 0.12));                  dimensions.push_back(PxVec3(0.03, boxHeight, 0.0015));
    positions.push_back(PxVec3(-0.02, floorHeight, 0.08));                  dimensions.push_back(PxVec3(0.03, boxHeight, 0.0015));
    positions.push_back(PxVec3(0.07, floorHeight, 0.04));                   dimensions.push_back(PxVec3(0.02, boxHeight, 0.0015));
    positions.push_back(PxVec3(0.11, floorHeight, 0.08));                   dimensions.push_back(PxVec3(0.02, boxHeight, 0.0015));
    positions.push_back(PxVec3(0.05, floorHeight, 0.08));                   dimensions.push_back(PxVec3(0.02, boxHeight, 0.0015));
    positions.push_back(PxVec3(0.09, floorHeight, 0.155));                  dimensions.push_back(PxVec3(0.03, boxHeight, 0.0015));
    positions.push_back(PxVec3(0.13, floorHeight, 0.12));                   dimensions.push_back(PxVec3(0.02, boxHeight, 0.0015));
    
    // top down - vertical boxes
    positions.push_back(PxVec3(-0.025, floorHeight, 0.175));                dimensions.push_back(PxVec3(0.0015, boxHeight, 0.0175));
    positions.push_back(PxVec3(0.09, floorHeight, 0.175));                  dimensions.push_back(PxVec3(0.0015, boxHeight, 0.0175));
    positions.push_back(PxVec3(-0.085, floorHeight, 0.14));                 dimensions.push_back(PxVec3(0.0015, boxHeight, 0.0175));
    positions.push_back(PxVec3(-0.085, floorHeight, 0.06));                 dimensions.push_back(PxVec3(0.0015, boxHeight, 0.0175));
    positions.push_back(PxVec3(-0.055, floorHeight, 0.02));                 dimensions.push_back(PxVec3(0.0015, boxHeight, 0.0175));
    positions.push_back(PxVec3(-0.02, floorHeight, 0.06));                  dimensions.push_back(PxVec3(0.0015, boxHeight, 0.0175));
    positions.push_back(PxVec3(0.025, floorHeight, 0.02));                  dimensions.push_back(PxVec3(0.0015, boxHeight, 0.0175));
    positions.push_back(PxVec3(0.0325, floorHeight, 0.13));                 dimensions.push_back(PxVec3(0.001, boxHeight, 0.0475));
    positions.push_back(PxVec3(0.0875, floorHeight, 0.07));                 dimensions.push_back(PxVec3(0.0015, boxHeight, 0.0275));
}
