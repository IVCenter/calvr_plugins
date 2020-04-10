#version 460

#pragma import_defines ( COLOR_FUNCTION, ORGANS_ONLY, LIGHT_DIRECTIONAL, LIGHT_SPOT,LIGHT_POINT )
#pragma import_defines ( COLON, BLADDER, KIDNEY, SPLEEN, BODY )
#pragma import_defines ( COLON_RGB, BLADDER_RGB, KIDNEY_RGB, SPLEEN_RGB, BODY_RGB )
#pragma import_defines ( CONTRAST_ABSOLUTE )


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
uniform float OpacityTopWidth;
uniform float OpacityMult = 1.0;

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

	if(ra.r < ContrastBottom || ra.r > ContrastTop)
	{
		ra.r = 0;
	}
	#ifndef CONTRAST_ABSOLUTE
	ra.r = (ra.r - ContrastBottom) / (ContrastTop - ContrastBottom);
	#endif
	ra.r = max(0, min(1, ra.r));


//	s.a = 1 - (abs(OpacityCenter - ra.r) / OpacityWidth);


	
	s.a = smoothstep((OpacityCenter-OpacityTopWidth) - (OpacityWidth/2.0), OpacityCenter - OpacityTopWidth, ra.r);
	if(s.a == 1.0){
		s.a = 1.0 - smoothstep(OpacityCenter+OpacityTopWidth, OpacityCenter+ OpacityTopWidth + (OpacityWidth/2.0), ra.r);
	}
	s.a *= OpacityMult;

	#ifdef COLOR_FUNCTION
		s.rgb = COLOR_FUNCTION
	#else
		s.rgb = vec3(ra.r);
	#endif

	
	//TODO: change to floatBitsToUint
	uint bitmask = uint(ra.g * 65535.0);


	#ifdef ORGANS_ONLY
		float alpha = 0.0;
		s.rgb = vec3(ra.r);
	#else
		float alpha = s.a;
	#endif

	#ifdef BLADDER
		if(bitmask == 1)
		{
				s.rgb = vec3(ra.rr, 0);
				alpha = s.a;
			#ifdef BLADDER_RGB
				s.rgb = BLADDER_RGB
				s.rgb*=ra.r;
			#endif
		}
	#endif

	#ifdef KIDNEY
		if(bitmask == 2)
		{
			s.rgb = vec3(0, ra.r, 0);
			alpha = s.a;
			#ifdef KIDNEY_RGB
				s.rgb = KIDNEY_RGB
				s.rgb*=ra.r;
			#endif
		}
	#endif

	#ifdef COLON
		if(bitmask == 4)
		{
			
			s.rgb = vec3(ra.r, 0, 0);
			alpha = s.a;
			#ifdef COLON_RGB
				s.rgb = COLON_RGB		
				s.rgb*=ra.r;
			#endif
		}
	
	#endif

	#ifdef SPLEEN
		if(bitmask == 8)
		{
			s.rgb = vec3(0, ra.rr);
			alpha = s.a;
			#ifdef SPLEEN_RGB
				s.rgb = SPLEEN_RGB		
				s.rgb*=ra.r;
			#endif
		}
	#endif

	

	
	

	s.a = alpha;


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