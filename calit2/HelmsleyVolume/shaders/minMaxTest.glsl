////////////////////////////////////////
// minMax.comp
// computes the min/max values for the volume
////////////////////////////////////////

#version 430

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;// 64 threads

layout(std430, binding = 0) buffer minMaxBuffer {
	uint[2] minMaxData;
};
// to count the number of pixels shown in the mask

void main() {
	
	atomicMin(minMaxData[0], 1);
}