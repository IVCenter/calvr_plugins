#version 460

#pragma multi_compile LIGHT_DIRECTIONAL LIGHT_SPOT LIGHT_POINT
#pragma multi_compile MASK

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
layout(rg16, binding = 0) uniform image3D volume;
layout(rg16, binding = 1) uniform image3D baked;

//uniform float Exposure;
//uniform float Threshold;
//uniform float Density;

uniform float ContrastBottom;
uniform float ContrastTop;

uniform float OpacityCenter;
uniform float OpacityWidth;
uniform float OpacityMult;

uniform vec3 WorldScale;
uniform vec3 TexelSize;

uniform float LightDensity;
uniform vec3 LightPosition;
uniform vec3 LightDirection;
uniform float LightAngle;
uniform float LightAmbient;
uniform float LightIntensity;

vec2 Sample(ivec3 p) {
	vec2 s = imageLoad(volume, p).rg;

	#ifdef INVERT
	s.r = 1 - s.r;
	#endif

	s.r = (s.r - ContrastBottom) / (ContrastTop - ContrastBottom);

	#ifndef MASK
	s.g = s.r;
	#endif

	s.g = 1 - (abs(OpacityCenter - s.r) / OpacityWidth);
	s.g *= OpacityMult;

	//s.g = max(0.0, (s.g - Threshold) / (1.0 - Threshold)); // subtractive for soft edges
	//s.g *= Density;

	//s.r *= Exposure;
	return s;
}

float Light(vec3 p) {
	#define ldt .002
	#define ls 10
	#define ils2 .01

	#if defined(LIGHT_DIRECTIONAL)

	vec3 uvwldir = -normalize(LightDirection / WorldScale) / TexelSize;
	float ld = 0.0;
	for (uint i = 1; i < ls; i++)
		ld += Sample(ivec3(p + uvwldir * ldt * i)).g;

	return exp(-ld * LightDensity * ldt) * LightIntensity + LightAmbient; // extinction = e^(-x)

	#elif defined(LIGHT_SPOT) || defined(LIGHT_POINT)

	vec3 wp = (p * TexelSize - vec3(.5)) * WorldScale;
	vec3 ldir = wp - LightPosition;
	float dist = length(ldir);
	ldir /= dist;

	vec3 lp = (LightPosition / WorldScale + vec3(.5)) / TexelSize;

	// sum density towards the light source
	float ld = 0.0;
	for (uint i = 1; i <= ls; i++)
		ld += Sample(ivec3(mix(p, lp, float(i) / float(ls)))).g;

	dist = 75.0 * dist + 1.0;

	return exp(-ld * LightDensity * ldt) * // extinction = e^(-x)
		LightIntensity / (dist * dist) // LightDistanceAttenuation
	#ifdef LIGHT_SPOT
		* clamp(10.0 * (max(0.0, dot(LightDirection, -ldir)) - LightAngle), 0.0, 1.0) // LightAngleAttenuation
	#endif
		+ LightAmbient;
	
	#else

	return 1.0;

	#endif
}

void main() {
	ivec3 index = ivec3(gl_GlobalInvocationID.xyz);

	vec2 s = Sample(index);
	s.r *= Light(vec3(index));
	imageStore(baked, index, vec4(s, 0.0, 0.0));
}