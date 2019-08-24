#version 460

#pragma multi_compile SAMPLECOUNT

out vec4 FragColor;

in vs_out {
	vec3 rd;
	vec3 ro;
	vec3 sp;
	mat4 WorldToObject;
	//mat4 ViewToObject;
} i;

uniform float StepSize;

uniform vec2 InvResolution;

uniform mat4 osg_ProjectionMatrixInverse;
uniform mat4 osg_ViewMatrixInverse;
uniform mat4 osg_ViewMatrix;

uniform vec3 PlanePoint;
uniform vec3 PlaneNormal;

uniform sampler3D Volume;
uniform sampler2D DepthTexture;

vec2 RayCube(vec3 ro, vec3 rd, vec3 extents) {
    vec3 tMin = (-extents - ro) / rd;
    vec3 tMax = (extents - ro) / rd;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    return vec2(max(max(t1.x, t1.y), t1.z), min(min(t2.x, t2.y), t2.z));
}
float RayPlane(vec3 ro, vec3 rd, vec3 planep, vec3 planen) {
	float d = dot(planen, rd);
	float t = dot(planep - ro, planen);
	return d > 1e-5 ? (t / d) : (t > 0 ? 1e5 : -1e5);
}

// depth texture to object-space ray depth
float DepthTextureToObjectDepth(vec3 ro, vec3 screenPos) {
	vec4 clip = vec4(screenPos.xy / screenPos.z, 0.0, 1.0);

	clip.z = textureLod(DepthTexture, clip.xy * .5 + .5, 0.0).r * 2.0 - 1.0;

    vec4 viewSpacePosition = osg_ProjectionMatrixInverse * clip;
    viewSpacePosition /= viewSpacePosition.w;

	return 1;
	//return length((i.ViewToObject * viewSpacePosition).xyz - ro);
}

vec4 Sample(vec3 p) {
	vec4 s;

	vec2 ra = textureLod(Volume, p, 0.0).rg;
	s.rgb = vec3(ra.r);
	s.a = ra.g;
	s.a = (dot((p - .5) - PlanePoint, PlaneNormal) < 0) ? 0 : s.a;

	return s;
}

void main() {
	//FragColor = textureLod(DepthTexture, i.sp.xy/i.sp.z, 0.0);
	//FragColor.a = 1.0;
	//return;


	vec3 ro = i.ro;//vec3(i.WorldToObject * vec4(osg_ViewMatrixInverse[3].xyz, 1));
	vec3 rd = normalize(i.rd.xyz);

	//FragColor = vec4(ro, 1);
	//return;

	vec2 intersect = RayCube(ro, rd, vec3(.5));
	intersect.x = max(0, intersect.x);
	intersect.x = max(intersect.x, RayPlane(ro, rd, PlanePoint, PlaneNormal));
	
	// depth buffer intersection
	//float z = DepthTextureToObjectDepth(ro, i.sp);
	//intersect.y = min(intersect.y, z);
	//if (intersect.y < intersect.x) discard;

	ro += .5; // cube has a radius of .5, transform to UVW space

	vec4 sum = vec4(0);

	uint steps = 0;
	float pd = 0;
	for (float t = intersect.x; t < intersect.y;) {
		if (sum.a > .98 || steps > 750) break;

		vec3 p = ro + rd * t;
		vec4 col = Sample(p);

		if (col.a > .01){
			if (pd < .01) {
				// first time entering volume, binary subdivide to get closer to entrance point
				float t0 = t - StepSize * 4;
				float t1 = t;
				float tm;
				#define BINARY_SUBDIV tm = (t0 + t1) * .5; p = ro + rd * tm; if (Sample(p).a > .01) t1 = tm; else t0 = tm;
				BINARY_SUBDIV
				BINARY_SUBDIV
				BINARY_SUBDIV
				BINARY_SUBDIV
				#undef BINARY_SUBDIV
				t = tm;
				col = Sample(p);
			}

			col.rgb *= col.a;
			sum += col * (1 - sum.a);
		}

		steps++; // only count steps through the volume

		pd = col.a;
		t += col.a > .01 ? StepSize : StepSize * 4; // step farther if not in dense part
	}

	if(sum.a <= 0.01) discard;

	#ifdef SAMPLECOUNT
	FragColor = vec4(mix(vec3(.2, .2, 1.0), vec3(1.0, .2, .2), float(steps) / 750.0), 1.0);
	#else
	sum.a = clamp(sum.a, 0.0, 1.0);
	FragColor = sum;
	#endif
}