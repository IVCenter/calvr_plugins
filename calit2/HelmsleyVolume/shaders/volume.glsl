#version 460

#pragma import_defines ( COLOR_FUNCTION, ORGANS_ONLY, LIGHT_DIRECTIONAL, LIGHT_SPOT,LIGHT_POINT )
#pragma import_defines ( COLON, BLADDER, KIDNEY, SPLEEN, ILLEUM, AORTA, BODY, VEIN )
#pragma import_defines ( COLON_RGB, BLADDER_RGB, KIDNEY_RGB, SPLEEN_RGB, BODY_RGB, ILLEUM_RGB, AORTA_RGB )
#pragma import_defines ( CONTRAST_ABSOLUTE )
#pragma import_defines (  EDGE_DETECTION )
#pragma import_defines (  CLAHE )


layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
layout(rg16, binding = 0) uniform image3D volume;
layout(rgba16, binding = 1) uniform image3D baked;
 layout(rg16, binding = 5) uniform image3D CLAHEvolume;




uniform float ContrastBottom;
uniform float ContrastTop;
uniform float Brightness = 0.0;

uniform float TrueContrast = 1.0;
uniform float ContrastCenter = .5;


uniform float OpacityCenter[10];
uniform float OpacityWidth[10];
uniform float OpacityTopWidth[10];
uniform float OpacityMult[10];
uniform float Lowest[10];
uniform float TriangleCount = 1.0;

uniform vec3 leftColor = vec3(1,0,0);
uniform vec3 rightColor = vec3(1,1,1);

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

vec3 custom(vec3 c) {
	vec3 color = mix(leftColor, rightColor, c);
	return color;
}

vec4 Sample(ivec3 p) {
	vec4 s = vec4(0,0,0,0);
	#ifndef CLAHE
		vec2 ra = imageLoad(volume, p).rg;
		
	#else
		vec2 ra = imageLoad(CLAHEvolume, p).rg;
		s.rgb = vec3(ra.r);
		s.a = 1.0f;
		return s;
	#endif
	
	uint bitmask = uint(imageLoad(volume, p).rg.g * 65535.0);
	const uint organCount = 7;
	vec3 organColors[organCount] = 
	vec3[organCount](vec3(0.992, 0.968, 0.843), vec3(0, 0.278, 1), 
					  vec3(0.752, 0.635, 0.996),  vec3(1, 0.874, 0.109),
					vec3(0.968, 0.780, 1), vec3(1, 0, 0), vec3(0, .8, 1));

	uint organMask = 0;
	#ifdef BLADDER
	organMask |= 1;
	#endif
	#ifdef KIDNEY
	organMask |= 2;
	#endif
	#ifdef COLON
	organMask |= 4;
	#endif
	#ifdef SPLEEN
	organMask |= 8;
	#endif
	#ifdef ILLEUM
	organMask |= 16;
	#endif
	#ifdef AORTA
	organMask |= 32;
	#endif
	#ifdef VEIN
	organMask |= 64;
	#endif
	//etc...

	bitmask &= organMask;
	//ORGANS
	for (int i = 0; i < organCount; i++){
	  if (bool(bitmask & (1 << i))) {
		s.rgb = organColors[i];
		//Contrast
		ra.r = mix(ContrastBottom, ContrastTop, ra.r);
		ra.r = clamp(ra.r + (Brightness - 0.5), 0.0, 1.0);
		ra.r = (ra.r - ContrastCenter) * TrueContrast + ContrastCenter + (Brightness);
		s.rgb *= ra.rrr;
		s.a = 1.0;
		return s;
	  }
	}

	#ifndef ORGANS_ONLY
		return s;
	#endif
	//FULLVOLUME
	//Opacity
	float highestOpacity = 0.0;
	for(int i = 0; i < TriangleCount; i++){
		s.a = smoothstep((OpacityCenter[i]-OpacityTopWidth[i]) - (OpacityWidth[i] - OpacityTopWidth[i]), OpacityCenter[i] - OpacityTopWidth[i], ra.r);
		if(s.a == 1.0){
			s.a = 1.0 - smoothstep(OpacityCenter[i]+OpacityTopWidth[i], OpacityCenter[i]+ OpacityTopWidth[i] + (OpacityWidth[i] - OpacityTopWidth[i]), ra.r);
		}
		if(s.a != 0.0){
			s.a *= 1/pow(2, ((1-OpacityMult[i])*10));
			float lowestLimit = 1/pow(2, ((1-Lowest[i])*10)); 
			s.a = max(s.a, lowestLimit);
			if(s.a >= highestOpacity)
				highestOpacity = s.a;
		}
	}
	s.a = highestOpacity;	

	//Contrast
	if(ra.r < ContrastBottom) 
	{
		ra.r = 0;
	}
	if (ra.r > ContrastTop){
		ra.r = 1;
	}
	ra.r = smoothstep(ContrastBottom, ContrastTop, ra.r);
	ra.r = clamp(ra.r, 0.0, 1.0);

	#ifdef COLOR_FUNCTION
		s.rgb = COLOR_FUNCTION
	#else
		s.rgb = vec3(ra.r);
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

float sobel(ivec3 index){


	float intensity = 1.0;
//	//xyz sobel
	ivec3 i0 =  ivec3(index.x-1, index.y+1, index.z);
	ivec3 i1 =  ivec3(index.x, index.y+1, index.z);
	ivec3 i2 =  ivec3(index.x+1, index.y+1, index.z);
	ivec3 i3 =  ivec3(index.x-1, index.y, index.z);

	ivec3 i5 =  ivec3(index.x+1, index.y, index.z);
	ivec3 i6 =  ivec3(index.x-1, index.y-1, index.z);
	ivec3 i7 =  ivec3(index.x, index.y-1, index.z);
	ivec3 i8 =  ivec3(index.x+1, index.y-1, index.z);

	ivec3 indices[8] = {i0,i1,i2,i3,i5,i6,i7,i8};

	///////////////z///////////
	ivec3 zi0 =  ivec3(index.x-1, index.y, index.z+1);
	ivec3 zi1 =  ivec3(index.x, index.y, index.z+1);
	ivec3 zi2 =  ivec3(index.x+1, index.y, index.z+1);
	ivec3 zi3 =  ivec3(index.x-1, index.y, index.z-1);
	ivec3 zi4 =  ivec3(index.x, index.y, index.z-1);
	ivec3 zi5 =  ivec3(index.x+1, index.y, index.z-1);

	ivec3 zindices[6] = {zi0, zi1, zi2, zi3, zi4, zi5};
	float zvalues[6];
	for(int i = 0; i < 6; i++){
		zvalues[i] = (imageLoad(volume, zindices[i]).r);
	}
	float gzValues[6] = {1.0,2.0,1.0,-1.0,-2.0,-1.0};
	float gz = 0.0;
	for(int i = 0; i < 6; i++){
		gz += gzValues[i] * zvalues[i];
	}
		///////////////z///////////

	float values[8];

	for(int i = 0; i < 8; i++){
		values[i] = (imageLoad(volume, indices[i]).r);
	}
	
	float gxValues[8] = {1.0,0.0,-1.0,2.0,-2.0,1.0,0.0,-1.0};
	float gyValues[8] = {1.0,2.0,1.0,0.0,0.0,-1.0,-2.0,-1.0};
	

	float gx = 0.0;
	float gy = 0.0;

	for(int i = 0; i < 8; i++){
		gx += gxValues[i] * values[i];
		gy += gyValues[i] * values[i];
	}
	

	float g = sqrt(pow(gx,2) + pow(gy,2) + pow(gz,2));
	
	float mx = sqrt(48);

	g /= mx;

	return g;
}

void main() {
	ivec3 index = ivec3(gl_GlobalInvocationID.xyz);

	vec4 s = Sample(index);

	s.rgb *= Light(vec3(index));

	#ifdef EDGE_DETECTION
		sobel(index);
		s.a *= sobel(index);
	#endif


	imageStore(baked, index, s);
}