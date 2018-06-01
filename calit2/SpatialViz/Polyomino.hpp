#include <vector>
//#include <array>
#include <queue>
#include <time.h>
#include <stack>
#include <bitset>

#include <osg/Vec3d>

using namespace osg;
using namespace std;

class Polyomino {
public:

	// Generate a polyomino of size n
	static vector<bool> generatePolyomino(int n) {
		if (n < 2) return vector<bool>();

		// Arbitrary bounder so that long predictable shapes are not formed
		int size = (int)sqrt(n) + 1;

		// Initialize the setup arrays
		bool*** polyomino = new bool**[size];
		char*** cube = new char**[size];
		for (int x = 0; x < size; x++) {
			polyomino[x] = new bool*[size];
			cube[x] = new char*[size];
			for (int y = 0; y < size; y++) {
				polyomino[x][y] = new bool[size];
				cube[x][y] = new char[size];
				for (int z = 0; z < size; z++) {
					polyomino[x][y][z] = false;

					// Create some noise in the companion array
					cube[x][y][z] = rand() % CHAR_MAX;
				}
			}
		}

		// Create the queue for prim's algorithm
		priority_queue<pair<char, Vec3> > q = priority_queue<pair<char, Vec3> >();

		// Start from the middle of the cube
		polyomino[size / 2][size / 2][size / 2] = true;
		if (size / 2 > 0) {
			Vec3 arr1( size / 2 - 1, size / 2, size / 2 );
			Vec3 arr2(size / 2, size / 2 - 1, size / 2);
			Vec3 arr3(size / 2, size / 2, size / 2 - 1 );
			q.push(make_pair(cube[size / 2 - 1][size / 2][size / 2], arr1));
			q.push(make_pair(cube[size / 2][size / 2 - 1][size / 2], arr2));
			q.push(make_pair(cube[size / 2][size / 2][size / 2 - 1], arr3));
		}
		if (size / 2 < size - 1) {
			Vec3 arr1(size / 2 + 1, size / 2, size / 2);
			Vec3 arr2(size / 2, size / 2 + 1, size / 2);
			Vec3 arr3(size / 2, size / 2, size / 2 + 1);
			q.push(make_pair(cube[size / 2 + 1][size / 2][size / 2], arr1));
			q.push(make_pair(cube[size / 2][size / 2 + 1][size / 2], arr2));
			q.push(make_pair(cube[size / 2][size / 2][size / 2 + 1], arr3));
		}

		// Iterate while the size of the piece is less than the desired size
		int count = 1;
		while (count < n) {

			// Look at the top(greatest) element
			pair<char, Vec3> cell = q.top();
			q.pop();

			int x = cell.second[0];
			int y = cell.second[1];
			int z = cell.second[2];

			// Check if the segment is already in the piece
			if (!polyomino[x][y][z]) {

				// Put the segment in the polyomino
				count++;
				polyomino[x][y][z] = true;

				// Add the neighboring unvisited segments
				if (x > 0 && !polyomino[x - 1][y][z]) {
					Vec3 arr(x - 1, y, z);
					q.push(make_pair(cube[x - 1][y][z], arr));
				}
				if (x < size - 1 && !polyomino[x + 1][y][z]) {
					Vec3 arr( x + 1, y, z );
					q.push(make_pair(cube[x + 1][y][z], arr));
				}
				if (y > 0 && !polyomino[x][y - 1][z]) {
					Vec3 arr( x, y - 1, z );
					q.push(make_pair(cube[x][y - 1][z], arr));
				}
				if (y < size - 1 && !polyomino[x][y + 1][z]) {
					Vec3 arr( x, y + 1, z);
					q.push(make_pair(cube[x][y + 1][z], arr));
				}
				if (z > 0 && !polyomino[x][y][z - 1]) {
					Vec3 arr( x, y, z - 1 );
					q.push(make_pair(cube[x][y][z - 1], arr));
				}
				if (z < size - 1 && !polyomino[x][y][z + 1]) {
					Vec3 arr(x, y, z + 1 );
					q.push(make_pair(cube[x][y][z + 1], arr));
				}
			}
		}

		// Prepare the return vector
		vector<bool> ret = vector<bool>();

		for (int x = 0; x < size; x++) {
			for (int y = 0; y < size; y++) {
				for (int z = 0; z < size; z++) {
					ret.push_back(polyomino[x][y][z]);
				}
			}
		}

		//TODO: check to see that no space is enclosed

		// Clear the allocated memory
		for (int x = 0; x < size; x++) {
			for (int y = 0; y < size; y++) {
				delete(polyomino[x][y]);
				delete(cube[x][y]);
			}
			delete(polyomino[x]);
			delete(cube[x]);
		}
		delete(polyomino);
		delete(cube);

		return setPolyomino(ret);
	}



	// Shift all the values to a corner so that it can be compared
	static vector<bool> setPolyomino(vector<bool> base) {
		if (base.size() < 8) return vector<bool>();

		// Retrieve the bounding size
		int size = (int)cbrt(base.size());
		
		// Find the minimum values in each dimension
		int minx = -1;
		int miny = -1;
		int minz = -1;
		
		for (int x = 0; x < size; x++) {
			for (int y = 0; y < size; y++) {
				for (int z = 0; z < size; z++) {
					if (base[x * size * size + y * size + z]) {
						minx = x;
						break;
					}
				}
				if (minx != -1) break;
			}
			if (minx != -1) break;
		}

		for (int y = 0; y < size; y++) {
			for (int x = 0; x < size; x++) {
				for (int z = 0; z < size; z++) {
					if (base[x * size * size + y * size + z]) {
						miny = y;
						break;
					}
				}
				if (miny != -1) break;
			}
			if (miny != -1) break;
		}

		for (int z = 0; z < size; z++) {
			for (int y = 0; y < size; y++) {
				for (int x = 0; x < size; x++) {
					if (base[x * size * size + y * size + z]) {
						minz = z;
						break;
					}
				}
				if (minz != -1) break;
			}
			if (minz != -1) break;
		}

		// Create a new return vector with the values shifted.
		vector<bool> ret = vector<bool>(size * size * size);
		for (int x = minx; x < size; x++) {
			for (int y = miny; y < size; y++) {
				for (int z = minz; z < size; z++) {
					ret[(x - minx) * size * size + (y - miny) * size + z - minz] = base[x * size * size + y * size + z];
				}
			}
		}

		return ret;
	}



	static vector<bool> polyominoId(vector<bool> base) {

		int size = (int)cbrt(base.size());

		vector<bool> id = setPolyomino(base);
		vector<bool> rotatedPoly;

		for (int i = 0; i < 24; i++) {
			rotatedPoly = polyonimoOrientation(base, i);
			rotatedPoly = setPolyomino(rotatedPoly);
			if (id < rotatedPoly) {
				id = rotatedPoly;
			}
		}

		return id;
	}


	
	// Get a polyomino in the desired orientation (enumerated from 0 to 23)
	static vector<bool> polyonimoOrientation(vector<bool> base, int rot) {
		
		// retrieve the size of the bounding box
		int size = (int)cbrt(base.size());
		
		// Create an array for notational simplicity
		bool*** rotated = new bool**[size];
		for (int x = 0; x < size; x++) {
			rotated[x] = new bool*[size];
			for (int y = 0; y < size; y++) {
				rotated[x][y] = new bool[size];
				for (int z = 0; z < size; z++) {
					rotated[x][y][z] = false;
				}
			}
		}

		// et tu Brute
		switch (rot) {
		case 1:
			for (int x = 0, ox = 0; x < size; x++, ox++) {
				for (int y = size - 1, oy = 0; y >= 0; y--, oy++) {
					for (int z = size - 1, oz = 0; z >= 0; z--, oz++) {
						rotated[x][y][z] = base[ox * size * size + oy * size + oz];
					}
				}
			}
			break;
		case 2:
			for (int x = 0, ox = 0; x < size; x++, ox++) {
				for (int z = size - 1, oy = 0; z >= 0; z--, oy++) {
					for (int y = 0, oz = 0; y < size; y++, oz++) {
						rotated[x][y][z] = base[ox * size * size + oy * size + oz];
					}
				}
			}
			break;
		case 3:
			for (int x = 0, ox = 0; x < size; x++, ox++) {
				for (int z = 0, oy = 0; z < size; z++, oy++) {
					for (int y = size - 1, oz = 0; y >= 0; y--, oz++) {
						rotated[x][y][z] = base[ox * size * size + oy * size + oz];
					}
				}
			}
			break;
		case 4:
			for (int x = size - 1, ox = 0; x >= 0; x--, ox++) {
				for (int y = size - 1, oy = 0; y >= 0; y--, oy++) {
					for (int z = 0, oz = 0; z < size; z++, oz++) {
						rotated[x][y][z] = base[ox * size * size + oy * size + oz];
					}
				}
			}
			break;
		case 5:
			for (int x = size - 1, ox = 0; x >= 0; x--, ox++) {
				for (int y = 0, oy = 0; y < size; y++, oy++) {
					for (int z = size - 1, oz = 0; z >= 0; z--, oz++) {
						rotated[x][y][z] = base[ox * size * size + oy * size + oz];
					}
				}
			}
			break;
		case 6:
			for (int x = size - 1, ox = 0; x >= 0; x--, ox++) {
				for (int z = 0, oy = 0; z < size; z++, oy++) {
					for (int y = 0, oz = 0; y < size; y++, oz++) {
						rotated[x][y][z] = base[ox * size * size + oy * size + oz];
					}
				}
			}
			break;
		case 7:
			for (int x = size - 1, ox = 0; x >= 0; x--, ox++) {
				for (int z = size - 1, oy = 0; z >= 0; z--, oy++) {
					for (int y = size - 1, oz = 0; y >= 0; y--, oz++) {
						rotated[x][y][z] = base[ox * size * size + oy * size + oz];
					}
				}
			}
			break;
		case 8:
			for (int y = 0, ox = 0; y < size; y++, ox++) {
				for (int x = size - 1, oy = 0; x >= 0; x--, oy++) {
					for (int z = 0, oz = 0; z < size; z++, oz++) {
						rotated[x][y][z] = base[ox * size * size + oy * size + oz];
					}
				}
			}
			break;
		case 9:
			for (int y = 0, ox = 0; y < size; y++, ox++) {
				for (int x = 0, oy = 0; x < size; x++, oy++) {
					for (int z = size - 1, oz = 0; z >= 0; z--, oz++) {
						rotated[x][y][z] = base[ox * size * size + oy * size + oz];
					}
				}
			}
			break;
		case 10:
			for (int y = 0, ox = 0; y < size; y++, ox++) {
				for (int z = 0, oy = 0; z < size; z++, oy++) {
					for (int x = 0, oz = 0; x < size; x++, oz++) {
						rotated[x][y][z] = base[ox * size * size + oy * size + oz];
					}
				}
			}
			break;
		case 11:
			for (int y = 0, ox = 0; y < size; y++, ox++) {
				for (int z = size - 1, oy = 0; z >= 0; z--, oy++) {
					for (int x = size - 1, oz = 0; x >= 0; x--, oz++) {
						rotated[x][y][z] = base[ox * size * size + oy * size + oz];
					}
				}
			}
			break;
		case 12:
			for (int y = size - 1, ox = 0; y >= 0; y--, ox++) {
				for (int x = 0, oy = 0; x < size; x++, oy++) {
					for (int z = 0, oz = 0; z < size; z++, oz++) {
						rotated[x][y][z] = base[ox * size * size + oy * size + oz];
					}
				}
			}
			break;
		case 13:
			for (int y = size - 1, ox = 0; y >= 0; y--, ox++) {
				for (int x = size - 1, oy = 0; x >= 0; x--, oy++) {
					for (int z = size - 1, oz = 0; z >= 0; z--, oz++) {
						rotated[x][y][z] = base[ox * size * size + oy * size + oz];
					}
				}
			}
			break;
		case 14:
			for (int y = size - 1, ox = 0; y >= 0; y--, ox++) {
				for (int z = 0, oy = 0; z < size; z++, oy++) {
					for (int x = size - 1, oz = 0; x >= 0; x--, oz++) {
						rotated[x][y][z] = base[ox * size * size + oy * size + oz];
					}
				}
			}
			break;
		case 15:
			for (int y = size - 1, ox = 0; y >= 0; y--, ox++) {
				for (int z = size - 1, oy = 0; z >= 0; z--, oy++) {
					for (int x = 0, oz = 0; x < size; x++, oz++) {
						rotated[x][y][z] = base[ox * size * size + oy * size + oz];
					}
				}
			}
			break;
		case 16:
			for (int z = 0, ox = 0; z < size; z++, ox++) {
				for (int x = 0, oy = 0; x < size; x++, oy++) {
					for (int y = 0, oz = 0; y < size; y++, oz++) {
						rotated[x][y][z] = base[ox * size * size + oy * size + oz];
					}
				}
			}
			break;
		case 17:
			for (int z = 0, ox = 0; z < size; z++, ox++) {
				for (int x = size - 1, oy = 0; x >= 0; x--, oy++) {
					for (int y = size - 1, oz = 0; y >= 0; y--, oz++) {
						rotated[x][y][z] = base[ox * size * size + oy * size + oz];
					}
				}
			}
			break;
		case 18:
			for (int z = 0, ox = 0; z < size; z++, ox++) {
				for (int y = 0, oy = 0; y < size; y++, oy++) {
					for (int x = size - 1, oz = 0; x >= 0; x--, oz++) {
						rotated[x][y][z] = base[ox * size * size + oy * size + oz];
					}
				}
			}
			break;
		case 19:
			for (int z = 0, ox = 0; z < size; z++, ox++) {
				for (int y = size - 1, oy = 0; y >= 0; y--, oy++) {
					for (int x = 0, oz = 0; x < size; x++, oz++) {
						rotated[x][y][z] = base[ox * size * size + oy * size + oz];
					}
				}
			}
			break;
		case 20:
			for (int z = size - 1, ox = 0; z >= 0; z--, ox++) {
				for (int x = size - 1, oy = 0; x >= 0; x--, oy++) {
					for (int y = 0, oz = 0; y < size; y++, oz++) {
						rotated[x][y][z] = base[ox * size * size + oy * size + oz];
					}
				}
			}
			break;
		case 21:
			for (int z = size - 1, ox = 0; z >= 0; z--, ox++) {
				for (int x = 0, oy = 0; x < size; x++, oy++) {
					for (int y = size - 1, oz = 0; y >= 0; y--, oz++) {
						rotated[x][y][z] = base[ox * size * size + oy * size + oz];
					}
				}
			}
			break;
		case 22:
			for (int z = size - 1, ox = 0; z >= 0; z--, ox++) {
				for (int y = 0, oy = 0; y < size; y++, oy++) {
					for (int x = 0, oz = 0; x < size; x++, oz++) {
						rotated[x][y][z] = base[ox * size * size + oy * size + oz];
					}
				}
			}
			break;
		case 23:
			for (int z = size - 1, ox = 0; z >= 0; z--, ox++) {
				for (int y = size - 1, oy = 0; y >= 0; y--, oy++) {
					for (int x = size - 1, oz = 0; x >= 0; x--, oz++) {
						rotated[x][y][z] = base[ox * size * size + oy * size + oz];
					}
				}
			}
			break;
		default:
			for (int x = 0, ox = 0; x < size; x++, ox++) {
				for (int y = 0, oy = 0; y < size; y++, oy++) {
					for (int z = 0, oz = 0; z < size; z++, oz++) {
						rotated[x][y][z] = base[ox * size * size + oy * size + oz];
					}
				}
			}
			break;
		}

		// prepare the return
		vector<bool> ret = vector<bool>();

		for (int x = 0; x < size; x++) {
			for (int y = 0; y < size; y++) {
				for (int z = 0; z < size; z++) {
					ret.push_back(rotated[x][y][z]);
				}
			}
		}

		for (int x = 0; x < size; x++) {
			for (int y = 0; y < size; y++) {
				delete(rotated[x][y]);
			}
			delete(rotated[x]);
		}
		delete(rotated);

		return ret;
	}


	// Creates a vector of similar polyominos, but one of them will be the same
	static vector<vector<bool> > generateSimilarities(vector<bool> base, vector<vector<bool> > currSimilarities, int t) {
		// get the size of the bounding region
		int size = (int)cbrt(base.size());
		
		// loop until vector is full
		while (currSimilarities.size() < t) {
			//cout << (int)currSimilarities.size() << endl;
			// Make sure a solution is present
			bool foundMatch = false;
			for (int i = 0; i < currSimilarities.size(); i++) {
				if (polyominoId(currSimilarities[i]) == polyominoId(base)) foundMatch = true;
			}
			if (!foundMatch) {
				//cout << "push s" << endl;
				currSimilarities.push_back(base);
				continue;
			}

			int r = rand() % 100;
			if (r < 60) {
				// mirror
				vector<bool> mirror = vector<bool>(size * size * size);
				for (int x = 0; x < size; x++) {
					for (int y = 0; y < size; y++) {
						for (int z = 0; z < size; z++) {
							mirror[(size - x - 1) * size * size + y * size + z] = base[x * size * size + y * size + z];
						}
					}
				}

				if (polyominoId(mirror) != polyominoId(base)) {
					while (true) {
						vector<bool> temp = polyonimoOrientation(mirror, rand() % 24);
						bool in = false;
			               for (int i = 0; i < currSimilarities.size(); i++) {
							if (setPolyomino(currSimilarities[i]) == setPolyomino(temp)) continue;
						}
						if (!in) {
							//cout << "push m" << endl;
							currSimilarities.push_back(temp);
							break;
						}
					}
				}
			} else if (r < 80) {
				// add
				vector<bool> added = vector<bool>(size * size * size);
				vector<vector<int> > candidates = vector<vector<int> >();
				int maxNeighbors = 0;
				for (int x = 0; x < size; x++) {
					for (int y = 0; y < size; y++) {
						for (int z = 0; z < size; z++) {
							added[x * size * size + y * size + z] = base[x * size * size + y * size + z];

							if (added[x * size * size + y * size + z]) continue;

							int count = 0;
							if (x > 0 && base[(x - 1) * size * size + y * size + z]) {
								count++;
							}
							if (x < size - 1 && base[(x + 1) * size * size + y * size + z]) {
								count++;
							}
							if (y > 0 && base[x * size * size + (y - 1) * size + z]) {
								count++;
							}
							if (y < size - 1 && base[x * size * size + (y + 1) * size + z]) {
								count++;
							}
							if (z > 0 && base[x * size * size + y * size + (z - 1)]) {
								count++;
							}
							if (z < size - 1 && base[x * size * size + y * size + (z + 1)]) {
								count++;
							}

							if (count > maxNeighbors) {
								maxNeighbors = count;
								candidates.clear();
							}

							if (count == maxNeighbors) {
								vector<int> coords = vector<int>();
								coords.push_back(x);
								coords.push_back(y);
								coords.push_back(z);
								candidates.push_back(coords);
							}
						}
					}
				}

				if (candidates.size() > 0) {
					vector<int> newBlock = candidates[rand() % candidates.size()];
					added[newBlock[0] * size * size + newBlock[1] * size + newBlock[2]] = true;
						
					while (true) {
						vector<bool> temp = polyonimoOrientation(added, rand() % 24);
						bool in = false;
			            for (int i = 0; i < currSimilarities.size(); i++) {
							if (setPolyomino(currSimilarities[i]) == setPolyomino(temp)) continue;
						}
						if (!in) {
							//cout << "push a" << endl;
							currSimilarities.push_back(temp);
							break;
						}
					}
				}
			} else {
				// remove
				vector<bool> removed = vector<bool>(size * size * size);
				vector<vector<int> > candidates = vector<vector<int> >();
				for (int x = 0; x < size; x++) {
					for (int y = 0; y < size; y++) {
						for (int z = 0; z < size; z++) {
							removed[x * size * size + y * size + z] = base[x * size * size + y * size + z];

							if (!removed[x * size * size + y * size + z]) continue;

							int count = 0;
							if (x > 0 && base[(x - 1) * size * size + y * size + z]) {
								count++;
							}
							if (x < size - 1 && base[(x + 1) * size * size + y * size + z]) {
								count++;
							}
							if (y > 0 && base[x * size * size + (y - 1) * size + z]) {
								count++;
							}
							if (y < size - 1 && base[x * size * size + (y + 1) * size + z]) {
								count++;
							}
							if (z > 0 && base[x * size * size + y * size + (z - 1)]) {
								count++;
							}
							if (z < size - 1 && base[x * size * size + y * size + (z + 1)]) {
								count++;
							}

							if (count > 1) {
								vector<int> coords = vector<int>();
								coords.push_back(x);
								coords.push_back(y);
								coords.push_back(z);
								candidates.push_back(coords);
							}
						}
					}
				}

				while (candidates.size() > 0) {
					int selected = rand() % candidates.size();
					vector<int> newBlock = candidates[selected];
					removed[newBlock[0] * size * size + newBlock[1] * size + newBlock[2]] = false;
					
					// Check connectedness
					if (!checkConnected(removed)) {
						removed[newBlock[0] * size * size + newBlock[1] * size + newBlock[2]] = true;
						candidates.erase(candidates.begin() + selected);
						continue;
					}

					bool set = false;
					while (true) {
						vector<bool> temp = polyonimoOrientation(removed, rand() % 24);
						bool in = false;
			            for (int i = 0; i < currSimilarities.size(); i++) {
							if (setPolyomino(currSimilarities[i]) == setPolyomino(temp)) in = true;
						}
						if (!in) {
							//cout << "push r" << endl;
							currSimilarities.push_back(temp);
							set = true;
							break;
						}
					}
					if (set) break;
				}
			}
		}

		return currSimilarities;
	}



	// Check connectivity of trues using BFS
	static bool checkConnected(vector<bool> base) {
		int size = (int)cbrt(base.size());
		int i = 0;
		
		for (; i < base.size(); i++) {
			if (base[i]) break;
		}
		int count = 0;
		
		vector<bool> base2 = vector<bool>(base.size());
		for (int j = 0; j < base.size(); j++) {
			base2[j] = base[j];
		}
		
		queue<int> q = queue<int>();
		q.push(i);
		while (!q.empty()) {
			i = q.front();
			int x = i / size / size;
			int y = i / size % size;
			int z = i % size;
			q.pop();
			
			base2[i] = false;
			
			if (x > 0 && base2[(x - 1) * size * size + y * size + z]) {
				q.push((x - 1) * size * size + y * size + z);
			}
			if (x < size - 1 && base2[(x + 1) * size * size + y * size + z]) {
				q.push((x + 1) * size * size + y * size + z);
			}
			if (y > 0 && base2[x * size * size + (y - 1) * size + z]) {
				q.push(x * size * size + (y - 1) * size + z);
			}
			if (y < size - 1 && base2[x * size * size + (y + 1) * size + z]) {
				q.push(x * size * size + (y + 1) * size + z);
			}
			if (z > 0 && base2[x * size * size + y * size + (z - 1)]) {
				q.push(x * size * size + y * size + (z - 1));
			}
			if (z < size - 1 && base2[x * size * size + y * size + (z + 1)]) {
				q.push(x * size * size + y * size + (z + 1));
			}
		}

		for (i = 0; i < base2.size(); i++) {
			if (base2[i]) return false;
		}

		return true;
	}


	static vector<vector<vector<float> > > generatePolyominoQuiz(int n, int nChoices) {
		vector<vector<vector<float> > > objList = vector<vector<vector<float> > >();
		
		vector<bool> polyomino = generatePolyomino(n);
		int size = (int)cbrt(polyomino.size());
		vector< vector<bool> > options = Polyomino::generateSimilarities(polyomino, vector<vector<bool> >(), nChoices);
		//random_shuffle(options.begin(), options.end());
		
		//cout << (int)options.size() << endl;
		
		vector<vector<float> > piece = vector<vector<float> >();
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				for (int k = 0; k < size; k++) {
					if (polyomino[i * size * size + j * size + k]) {
						vector<float> temp = vector<float>();
						temp.push_back((float)i);
						temp.push_back((float)j);
						temp.push_back((float)k);
						piece.push_back(temp);
					}
				}
			}
		}
		objList.push_back(piece);
		
		
		for (int o = 0; o < options.size(); o++) {
			vector<vector<float> > choice = vector<vector<float> >();
			for (int i = 0; i < size; i++) {
				for (int j = 0; j < size; j++) {
					for (int k = 0; k < size; k++) {
						if (options[o][i * size * size + j * size + k]) {
							vector<float> temp = vector<float>();
							temp.push_back((float)i);
							temp.push_back((float)j);
							temp.push_back((float)k);
							choice.push_back(temp);
						}
					}
				}
			}
			objList.push_back(choice);
		}

		return objList;
	}
};
