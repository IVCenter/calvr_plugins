////////////////////////////////////////
// minMax.comp
// computes the min/max values for the volume
////////////////////////////////////////

#version 460

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;	// 64 threads

// input Dicom volume and mask
layout(r16ui, binding = 0) uniform uimage3D volume;
//layout(r8ui, binding = 1) uniform uimage3D mask;

// to calculate the min/max value
layout(std430, binding = 2) buffer minMaxBuffer {
	uint[2] minMaxData;
};
// to count the number of pixels shown in the mask
layout(std430, binding = 3) buffer pixelCountBuffer {
	uint pixelCount;
};

uniform uint NUM_BINS;		// number of gray values in the Volume 
uniform uvec3 offset;		// start of the region to apply CLAHE to
uniform uvec3 volumeDims;	// size of the section of the volume we are applying CLAHE to 
uniform bool useMask;		// if we are performing CLAHE on just the Masked region 


void main() {

	// calculate the max and min for the volume 
	uvec3 index = gl_GlobalInvocationID.xyz;
	uint val = imageLoad(volume, ivec3(index + offset)).x;
	uint maskVal = imageLoad(volume, ivec3(index + offset)).y;	//was  imageLoad(mask, ivec3(index + offset)).x

	minMaxData[0] = 1;
	minMaxData[1] = 1;
//	atomicMin(minMaxData[0], val);
//	atomicMax(minMaxData[1], val);

	// if we are not within the volume of interest -> return 
	if ( index.x >= volumeDims.x || index.y >= volumeDims.y || index.z >= volumeDims.z ) {
		return;
	}
	// if we are using the mask but are masked out -> return 
	if ( useMask && maskVal == 0 ) {
		return;
	}


	// count the number of pixels contributing to the hist (used for the masked version)
	atomicAdd(pixelCount, 1);
}