#version 460

#pragma import_defines ( COLOR_FUNCTION )



uniform float OpacityCenter[10];
uniform float OpacityWidth[10];
uniform float OpacityTopWidth[10];
uniform float OpacityMult[10];
uniform float Lowest[10];
uniform float TriangleCount = 1.0;

uniform float ContrastBottom;
uniform float ContrastTop;
uniform float Brightness;

out vec4 FragColor;

in vs_out {
	vec2 uv;
} i;

vec3 hsv2rgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    vec2 ra = i.uv;
	vec4 col = vec4(0.0,0.0,0.0,0.0);

	float highestOpacity = 0.0;
	for(int i = 0; i < TriangleCount; i++){
		col.a = smoothstep((OpacityCenter[i]-OpacityTopWidth[i]) - (OpacityWidth[i] - OpacityTopWidth[i]), OpacityCenter[i] - OpacityTopWidth[i], ra.r);
		if(col.a == 1.0){
			col.a = 1.0 - smoothstep(OpacityCenter[i]+OpacityTopWidth[i], OpacityCenter[i]+ OpacityTopWidth[i] + (OpacityWidth[i] - OpacityTopWidth[i]), ra.r);
		}
		if(col.a != 0.0){
			col.a *= OpacityMult[i];
			if(col.a >= highestOpacity){
				highestOpacity = col.a;
			}
		}
	
//			float lowestLimit = 1/pow(2, ((1-Lowest[i])*10));
//			col.a *= 1/pow(2, ((1-OpacityMult[i])*10));	//Non-Linear Opacity Multiplier
//			if(col.a != 0.0)
//				col.a = max(col.a, lowestLimit);
	}
	col.a = highestOpacity;	
   
   
   
	col.rgb = vec3(1.0);


	
	FragColor = vec4(col);
}