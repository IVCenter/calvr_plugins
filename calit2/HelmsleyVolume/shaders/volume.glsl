#version 460

#pragma import_defines ( COLOR_FUNCTION, ORGANS_ONLY, LIGHT_DIRECTIONAL, LIGHT_SPOT,LIGHT_POINT )
#pragma import_defines ( COLON, BLADDER, KIDNEYS, SPLEEN )

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
layout(rg16, binding = 0) uniform image3D volume;
layout(rgba8, binding = 1) uniform image3D baked;


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

vec3 hsv2rgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec4 Sample(ivec3 p) {
	vec2 ra = imageLoad(volume, p).rg;
	vec4 s = vec4(0,0,0,0);

	#ifdef INVERT
		ra.r = 1 - ra.r;
	#endif

	ra.r = (ra.r - ContrastBottom) / (ContrastTop - ContrastBottom);
	ra.r = max(0, min(1, ra.r));


	s.a = 1 - (abs(OpacityCenter - ra.r) / OpacityWidth);
	s.a *= OpacityMult;

	#ifdef COLOR_FUNCTION
		s.rgb = COLOR_FUNCTION
	#else
		s.rgb = vec3(ra.r);
	#endif

	#ifdef COLON
		if(ra.g > 0.01)
		{
			s.rgb = vec3(ra.rr, 0);
		}
	#endif

	#ifdef ORGANS_ONLY
		s.a *= ra.g;
	#endif


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

	vec4 s = Sample(index);
	s.rgb *= Light(vec3(index));
	imageStore(baked, index, s);
}