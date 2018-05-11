#include <vector>
#include <array>
#include <queue>
#include <time.h>

using namespace std;
//using namespace physx;

class MazeCube {
public:
	static vector<array<float, 3>> generateMazeCube(int size) {
		if (size < 2) return vector<array<float, 3>>();

		size = 2 * size - 1;

		// This commented section can be used to make the runtime O(n^2), but them it would take much more code
		/*int** latSurface = new int*[size];
		for (int i = 0; i < size; i++) {
			latSurface[i] = new int[4 * (size - 1)];
		}

		int** topSurface = new int*[size - 2];
		for (int i = 0; i < size; i++) {
			topSurface[i] = new int[size - 2];
		}

		int** botSurface = new int*[size - 2];
		for (int i = 0; i < size; i++) {
			topSurface[i] = new int[size - 2];
		}*/
		srand((unsigned)time(0));

		// Initiallize and setup cube for maze generation
		char*** cube = new char**[size];
		for (int i = 0; i < size; i++) {
			cube[i] = new char*[size];
			for (int j = 0; j < size; j++) {
				cube[i][j] = new char[size];
				for (int k = 0; k < size; k++) {
					if ((i == 0 || j == 0 || k == 0 || i == size - 1 || j == size - 1 || k == size - 1)	&& (i % 2 == 0 && j % 2 == 0 && k % 2 == 0)) {
						cube[i][j][k] = 0;
					} else {
						cube[i][j][k] = rand() % (CHAR_MAX - 1) + 2;
					}
				}
			}
		}

		// Setup for iteration of prim's algorithm
		priority_queue<pair<char, array<int, 3>>> q = priority_queue<pair<char, array<int, 3>>>();

		cube[0][0][0] = 1;
		// defining the position to pass into the queue
		array<int, 3> arr1{ { 1, 0, 0 } };
		array<int, 3> arr2{ { 0, 1, 0 } };
		array<int, 3> arr3{ { 0, 0, 1 } };
		
		// adding the potential walls to the queue
		//                 value        position of wall
		q.push(make_pair(cube[1][0][0], arr1));
		q.push(make_pair(cube[0][1][0], arr2));
		q.push(make_pair(cube[0][0][1], arr3));

		while (!q.empty()) {

			// Look at the top(greatest) element
			auto edge = q.top();
			q.pop();
            
            // getting the coordinate of the nearby walls
			int x = edge.second[0];
			int y = edge.second[1];
			int z = edge.second[2];

			// Check if node is visited (if cube[x +- 1][y][z] != 0), and find direction of edge
			if (x > 0 && cube[x - 1][y][z] == 0)                    x--;
			else if (x < size - 1 && cube[x + 1][y][z] == 0)        x++;
			else if (y > 0 && cube[x][y - 1][z] == 0)               y--;
			else if (y < size - 1 && cube[x][y + 1][z] == 0)        y++;
			else if (z > 0 && cube[x][y][z - 1] == 0)               z--;
			else if (z < size - 1 && cube[x][y][z + 1] == 0)        z++;
			else continue;

			//cout << x << " " << y << " " << z << ":" << (int)cube[x][y][z] << endl;

			// Set the edge to visited
			cube[edge.second[0]][edge.second[1]][edge.second[2]] = 1;       // breaking down the wall
			cube[x][y][z] = 1;                                              // mark as visited

			// Push neighbors
			if (x > 1 && cube[x - 2][y][z] == 0) {
				array<int, 3> arr{ {x - 1, y, z} };
				q.push(make_pair(cube[x - 1][y][z], arr));
			}
			if (x < size - 2 && cube[x + 2][y][z] == 0) {
				array<int, 3> arr{ { x + 1, y, z } };
				q.push(make_pair(cube[x + 1][y][z], arr));
			}
			if (y > 1 && cube[x][y - 2][z] == 0) {
				array<int, 3> arr{ { x, y - 1, z } };
				q.push(make_pair(cube[x][y - 1][z], arr));
			}
			if (y < size - 2 && cube[x][y + 2][z] == 0) {
				array<int, 3> arr{ { x, y + 1, z } };
				q.push(make_pair(cube[x][y + 1][z], arr));
			}
			if (z > 1 && cube[x][y][z - 2] == 0) {
				array<int, 3> arr{ { x, y, z - 1 } };
				q.push(make_pair(cube[x][y][z - 1], arr));
			}
			if (z < size - 2 && cube[x][y][z + 2] == 0) {
				array<int, 3> arr{ { x, y, z + 1 } };
				q.push(make_pair(cube[x][y][z + 1], arr));
			}
		}

		// Load walls
		vector<array<float, 3>> positions = vector<array<float, 3>>();
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				for (int k = 0; k < size; k++) {
					if ((i == 0 || j == 0 || k == 0 || i == size - 1
						|| j == size - 1 || k == size - 1) && cube[i][j][k] > 1) {
						array<float, 3> arr{ {(float)i, (float)j, (float)k} };
						positions.push_back(arr);
					}
				}
				delete(cube[i][j]);
			}
			delete(cube[i]);
		}
		delete(cube);

		return positions;
	}




    // noisy one got rid of block if did not create a loop
	static vector<array<float, 3>> generateAltMazeCube(int size) {
		if (size < 2) return vector<array<float, 3>>();

		srand((unsigned)time(0));

		// Initiallize and setup cube for maze generation
		char*** cube = new char**[size];
		for (int i = 0; i < size; i++) {
			cube[i] = new char*[size];
			for (int j = 0; j < size; j++) {
				cube[i][j] = new char[size];
				for (int k = 0; k < size; k++) {
					if ((i == 0 || j == 0 || k == 0 || i == size - 1
						|| j == size - 1 || k == size - 1)) {
						cube[i][j][k] = rand() % (CHAR_MAX - 1) + 2;
					} else {
						cube[i][j][k] = 1;
					}
				}
			}
		}
		\
		// Setup for iteration of prim's algorithm
		priority_queue<pair<char, array<int, 3>>> q = priority_queue<pair<char, array<int, 3>>>();

		cube[0][0][0] = 0;
		array<int, 3> arr1{ { 1, 0, 0 } };
		array<int, 3> arr2{ { 0, 1, 0 } };
		array<int, 3> arr3{ { 0, 0, 1 } };
		q.push(make_pair(cube[1][0][0], arr1));
		q.push(make_pair(cube[0][1][0], arr2));
		q.push(make_pair(cube[0][0][1], arr3));

		while (!q.empty()) {

			// Look at the top(greatest) element
			auto edge = q.top();
			q.pop();

			int x = edge.second[0];
			int y = edge.second[1];
			int z = edge.second[2];

			// Check if pixel creates a loop
			int count = 0;
			if (x > 0 && cube[x - 1][y][z] == 0) count++;
			if (x < size - 1 && cube[x + 1][y][z] == 0) count++;
			if (y > 0 && cube[x][y - 1][z] == 0) count++;
			if (y < size - 1 && cube[x][y + 1][z] == 0) count++;
			if (z > 0 && cube[x][y][z - 1] == 0) count++;
			if (z < size - 1 && cube[x][y][z + 1] == 0) count++;

			if (count > 1) continue;

			// Set pixel to visited
			cube[x][y][z] = 0;

			// Push neighbors
			if (x > 0 && cube[x - 1][y][z] > 1) {
				array<int, 3> arr{ { x - 1, y, z } };
				q.push(make_pair(cube[x - 1][y][z], arr));
			}
			if (x < size - 1 && cube[x + 1][y][z] > 1) {
				array<int, 3> arr{ { x + 1, y, z } };
				q.push(make_pair(cube[x + 1][y][z], arr));
			}
			if (y > 0 && cube[x][y - 1][z] > 1) {
				array<int, 3> arr{ { x, y - 1, z } };
				q.push(make_pair(cube[x][y - 1][z], arr));
			}
			if (y < size - 1 && cube[x][y + 1][z] > 1) {
				array<int, 3> arr{ { x, y + 1, z } };
				q.push(make_pair(cube[x][y + 1][z], arr));
			}
			if (z > 0 && cube[x][y][z - 1] > 1) {
				array<int, 3> arr{ { x, y, z - 1 } };
				q.push(make_pair(cube[x][y][z - 1], arr));
			}
			if (z < size - 1 && cube[x][y][z + 1] > 1) {
				array<int, 3> arr{ { x, y, z + 1 } };
				q.push(make_pair(cube[x][y][z + 1], arr));
			}
		}

		// Load walls
		vector<array<float, 3>> positions = vector<array<float, 3>>();
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				for (int k = 0; k < size; k++) {
					if ((i == 0 || j == 0 || k == 0 || i == size - 1
						|| j == size - 1 || k == size - 1) && cube[i][j][k] > 1) {
						array<float, 3> arr{ { (float)i, (float)j, (float)k } };
						positions.push_back(arr);
					}
				}
				delete(cube[i][j]);
			}
			delete(cube[i]);
		}
		delete(cube);

		return positions;
	}




    // USE THIS ONE
	static vector<array<float, 3>> generateAltMazeCube2(int size) {
		if (size < 2) return vector<array<float, 3>>();

		int cleanliness = 50;   // 0 - 100
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
		priority_queue<pair<char, array<int, 3>>> q = priority_queue<pair<char, array<int, 3>>>();

		cube[0][0][0] = 0;
		array<int, 3> arr1{ { 1, 0, 0 } };
		array<int, 3> arr2{ { 0, 1, 0 } };
		array<int, 3> arr3{ { 0, 0, 1 } };
		q.push(make_pair(cube[1][0][0], arr1));
		q.push(make_pair(cube[0][1][0], arr2));
		q.push(make_pair(cube[0][0][1], arr3));

		while (!q.empty()) {

			// Look at the top(greatest) element
			auto edge = q.top();
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

			//cout << x << " " << y << " " << z << ":" << (int)cube[x][y][z] << endl;

			// Set pixel to visited
			cube[x][y][z] = 0;

			// Push neighbors, while adjusting weights so that straight is encouraged
			if (x > 0 && cube[x - 1][y][z] > 1) {
				array<int, 3> arr{ { x - 1, y, z } };
				if (direction == 60) cube[x - 1][y][z] = (cube[x - 1][y][z] > CHAR_MAX - straightener) ? CHAR_MAX : cube[x - 1][y][z] + straightener;
				q.push(make_pair(cube[x - 1][y][z], arr));
			}
			if (x < size - 1 && cube[x + 1][y][z] > 1) {
				array<int, 3> arr{ { x + 1, y, z } };
				if (direction == 10) cube[x + 1][y][z] = (cube[x + 1][y][z] > CHAR_MAX - straightener) ? CHAR_MAX : cube[x + 1][y][z] + straightener;
				q.push(make_pair(cube[x + 1][y][z], arr));
			}
			if (y > 0 && cube[x][y - 1][z] > 1) {
				array<int, 3> arr{ { x, y - 1, z } };
				if (direction == 50) cube[x][y - 1][z] = (cube[x][y - 1][z] > CHAR_MAX - straightener) ? CHAR_MAX : cube[x][y - 1][z] + straightener;
				q.push(make_pair(cube[x][y - 1][z], arr));
			}
			if (y < size - 1 && cube[x][y + 1][z] > 1) {
				array<int, 3> arr{ { x, y + 1, z } };
				if (direction == 20) cube[x][y + 1][z] = (cube[x][y + 1][z] > CHAR_MAX - straightener) ? CHAR_MAX : cube[x][y + 1][z] + straightener;
				q.push(make_pair(cube[x][y + 1][z], arr));
			}
			if (z > 0 && cube[x][y][z - 1] > 1) {
				array<int, 3> arr{ { x, y, z - 1 } };
				if (direction == 40) cube[x][y][z - 1] = (cube[x][y][z - 1] > CHAR_MAX - straightener) ? CHAR_MAX : cube[x][y][z - 1] + straightener;
				q.push(make_pair(cube[x][y][z - 1], arr));
			}
			if (z < size - 1 && cube[x][y][z + 1] > 1) {
				array<int, 3> arr{ { x, y, z + 1 } };
				if (direction == 30) cube[x][y][z + 1] = (cube[x][y][z + 1] > CHAR_MAX - straightener) ? CHAR_MAX : cube[x][y][z + 1] + straightener;
				q.push(make_pair(cube[x][y][z + 1], arr));
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
		vector<array<float, 3>> positions = vector<array<float, 3>>();
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				for (int k = 0; k < size; k++) {
					if ((i == 0 || j == 0 || k == 0 || i == size - 1
						|| j == size - 1 || k == size - 1) && cube[i][j][k] > 1) {
						array<float, 3> arr{ { (float)i, (float)j, (float)k } };
						positions.push_back(arr);
					}
				}
				delete(cube[i][j]);
			}
			delete(cube[i]);
		}
		delete(cube);

		return positions;
	}
};
