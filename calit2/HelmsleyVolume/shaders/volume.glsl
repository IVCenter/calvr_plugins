#version 460

#pragma import_defines ( COLOR_FUNCTION, ORGANS_ONLY, LIGHT_DIRECTIONAL, LIGHT_SPOT,LIGHT_POINT )
#pragma import_defines ( COLON, BLADDER, KIDNEY, SPLEEN, ILLEUM, AORTA, BODY )
#pragma import_defines ( COLON_RGB, BLADDER_RGB, KIDNEY_RGB, SPLEEN_RGB, BODY_RGB, ILLEUM_RGB, AORTA_RGB )
#pragma import_defines ( CONTRAST_ABSOLUTE )


layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
layout(rg16, binding = 0) uniform image3D volume;
layout(rgba8, binding = 1) uniform image3D baked;


//uniform float Exposure;
//uniform float Threshold;
//uniform float Density;

uniform float ContrastBottom;
uniform float ContrastTop;
uniform float Brightness;

uniform float OpacityCenter[10];
uniform float OpacityWidth[10];
uniform float OpacityTopWidth[10];
uniform float OpacityMult[10];
uniform float Lowest[10];
uniform float TriangleCount = 1.0;

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

	



	//Opacity
	float highestOpacity = 0.0;
	for(int i = 0; i < TriangleCount; i++){
		s.a = smoothstep((OpacityCenter[i]-OpacityTopWidth[i]) - (OpacityWidth[i] - OpacityTopWidth[i]), OpacityCenter[i] - OpacityTopWidth[i], ra.r);
		if(s.a == 1.0){
			s.a = 1.0 - smoothstep(OpacityCenter[i]+OpacityTopWidth[i], OpacityCenter[i]+ OpacityTopWidth[i] + (OpacityWidth[i] - OpacityTopWidth[i]), ra.r);
		}
		if(s.a != 0.0){
			s.a *= 1/pow(2, ((1-OpacityMult[i])*10));
			float lowestLimit = 1/pow(2, ((1-Lowest[i])*10)); //Non-Linear Opacity Multiplier
			s.a = max(s.a, lowestLimit);
			if(s.a >= highestOpacity)
				highestOpacity = s.a;
		}
			
	}
	s.a = highestOpacity;
	//Contrast


	vec2 organRA = ra; //removes contrast from organs

	if(ra.r < ContrastBottom) 
	{
		ra.r = 0;
	}
	if (ra.r > ContrastTop){
		ra.r = 1;
	}
	ra.r = smoothstep(ContrastBottom, ContrastTop, ra.r);
	ra.r = clamp(ra.r + (Brightness - 0.5), 0.0, 1.0);

	

	#ifdef COLOR_FUNCTION
		s.rgb = COLOR_FUNCTION
	#else
		s.rgb = vec3(ra.r);
	#endif

	#ifdef ORGANS_ONLY
		float alpha = 0.0;
	#else
		float alpha = s.a;
	#endif
	//TODO: change to floatBitsToUint
	uint bitmask = uint(ra.g * 65535.0);


	

	#ifdef BLADDER
		if(bitmask == 1)	
		{
				float adjusted = mix(.5,1.0,organRA.r);
				s.rgb = vec3(0.235, 0.003, 0.631);
				s.rgb *= adjusted;
				//alpha = s.a;
				alpha = 1.00;
//			#ifdef BLADDER_RGB
//				s.rgb = BLADDER_RGB
//				s.rgb*=organRA.r;
//			#endif
		}
	#endif

	#ifdef KIDNEY
		if(bitmask == 2)
		{
			float adjusted = mix(.5,1.0,organRA.r);
			s.rgb = vec3(0, 0.278, 1);
			s.rgb *= adjusted;
			alpha = 1.00;
//			#ifdef KIDNEY_RGB
//				s.rgb = KIDNEY_RGB
//				s.rgb*=organRA.r;
//			#endif
		}
	#endif

	#ifdef COLON
		if(bitmask == 4)
		{
			
			//s.rgb = vec3(organRA.r, 0, 0);
			float adjusted = mix(.7,1.0,organRA.r);
			s.rgb = vec3(0.450, 0.090, 1);
			s.rgb *= adjusted;
			alpha = 1.00;
//			#ifdef COLON_RGB
//				s.rgb = COLON_RGB		
//				s.rgb*=organRA.r;
//			#endif
		}
	
	#endif

	#ifdef SPLEEN
		if(bitmask == 8)
		{
			float adjusted = mix(.5,1.0,organRA.r);
			s.rgb = vec3(1, 0.874, 0.109);
			s.rgb *= adjusted;
			alpha = 1.00;
//			#ifdef SPLEEN_RGB
//				s.rgb = SPLEEN_RGB		
//				s.rgb*=organRA.r;
//			#endif
		}
	#endif

	#ifdef ILLEUM
		if(bitmask == 16)
		{
			float adjusted = mix(.99,1.0,organRA.r);
			s.rgb = vec3(0.968, 0.780, 1);
			s.rgb *= adjusted;
			alpha = 1.00;
//			#ifdef ILLEUM_RGB
//				s.rgb = ILLEUM_RGB		
//				s.rgb*=organRA.r;
//			#endif
		}
	#endif

	#ifdef AORTA
		if(bitmask == 32)
		{
			float adjusted = mix(.5,1.0,organRA.r);
			s.rgb = vec3(0.984, 0.109, 0.372);
			s.rgb *= adjusted;
			alpha = 1.00;
//			#ifdef AORTA_RGB
//				s.rgb = AORTA_RGB		
//				s.rgb*=organRA.r;
//			#endif
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